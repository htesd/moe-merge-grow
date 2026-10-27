"""Structural edits on a MoETransformer + AdamW state. Call them between optimizer steps
(after opt.step() and opt.zero_grad()). `opt` may be None (e.g. offline analysis)."""

import copy
import math

import torch
import torch.nn.functional as F


def _state(opt, p):
    if opt is None:
        return None
    st = opt.state.get(p)
    return st if st else None


def _clone_state(st):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in st.items()}


def _copy_router_row_state(opt, router, src, dst):
    st = _state(opt, router)
    if st is None:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        st[key][dst] = st[key][src]


@torch.no_grad()
def align_units(pa, pb, match_scale=True):
    """Hidden-unit transform that maps slot b onto slot a.

    A SwiGLU hidden unit i is (gate_i, up_i, down_i). Function-preserving symmetries per unit:
    permuting units, and (up_i, down_i) -> (c * up_i, down_i / c) for any c != 0 (incl. sign).
    Returns (perm, scale): new unit i of b := old unit perm[i], with up scaled by scale[i].
    """
    from scipy.optimize import linear_sum_assignment

    a, b = pa.float(), pb.float()
    na, nb = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    cos_g = na[0] @ nb[0].T
    cos_ud = na[1] @ nb[1].T + na[2] @ nb[2].T
    sim = cos_g + cos_ud.abs() / 2
    _, col = linear_sum_assignment(sim.cpu().numpy(), maximize=True)
    perm = torch.as_tensor(col, dtype=torch.long, device=pa.device)
    ar = torch.arange(perm.numel(), device=pa.device)
    sign = torch.sign(cos_ud[ar, perm])
    sign[sign == 0] = 1.0
    if not match_scale:
        return perm, sign
    # choose |c| so that b's up/down norm ratio matches a's
    nu_a, nd_a = a[1].norm(dim=-1), a[2].norm(dim=-1)
    nu_b, nd_b = b[1][perm].norm(dim=-1), b[2][perm].norm(dim=-1)
    mag = ((nu_a * nd_b) / (nd_a * nu_b).clamp_min(1e-20)).sqrt().clamp(1e-2, 1e2)
    return perm, sign * mag


@torch.no_grad()
def transform_units(p, st, perm, scale):
    """Apply a function-preserving hidden-unit transform to slot tensor p and its Adam state.
    up' = c*up -> grad/c, so m/c and v/c^2; down' = down/c -> m*c and v*c^2."""
    sc = scale.to(p.dtype)[:, None]
    p.copy_(p[:, perm])
    p[1].mul_(sc)
    p[2].div_(sc)
    if st is not None:
        m, v = st["exp_avg"], st["exp_avg_sq"]
        m.copy_(m[:, perm])
        v.copy_(v[:, perm])
        m[1].div_(sc)
        m[2].mul_(sc)
        v[1].div_(sc * sc)
        v[2].mul_(sc * sc)


@torch.no_grad()
def align_slot(model, opt, ref, s, match_scale=True):
    """Re-express slot s in the hidden-unit basis of slot ref (function unchanged)."""
    pa, pb = model.pool.slots[ref], model.pool.slots[s]
    perm, scale = align_units(pa, pb, match_scale)
    transform_units(pb, _state(opt, pb), perm, scale)
    return float((pa - pb).norm() / pa.norm())


@torch.no_grad()
def merge_slots(model, opt, keep, drop, alpha=0.5, align=True, match_scale=True):
    """Tie every logical expert of slot `drop` to slot `keep` and free `drop`.

    Weights: keep <- alpha * keep + (1 - alpha) * aligned(drop).
    Adam: the tied slot will receive g_keep + g_drop, so m <- m_a + m_b and
    v <- (sqrt(v_a) + sqrt(v_b))^2 (exact when the two gradients are aligned).
    Same-layer merges become one router group (logit = logsumexp of the rows).
    """
    pool = model.pool
    pa, pb = pool.slots[keep], pool.slots[drop]
    sa, sb = _state(opt, pa), _state(opt, pb)
    if align:
        perm, scale = align_units(pa, pb, match_scale)
        transform_units(pb, sb, perm, scale)
    rel_dist = float((pa - pb).norm() / pa.norm())
    pa.mul_(alpha).add_(pb, alpha=1.0 - alpha)
    if sb is not None:
        if sa is not None:
            sa["exp_avg"].add_(sb["exp_avg"])
            sa["exp_avg_sq"].copy_((sa["exp_avg_sq"].sqrt() + sb["exp_avg_sq"].sqrt()).square())
        else:
            opt.state[pa] = _clone_state(sb)
    model.redirect_slot(drop, keep)
    if opt is not None:
        opt.state.pop(pb, None)
    pool.release(drop)
    model.rebuild()
    return {"rel_dist_after_align": rel_dist}


@torch.no_grad()
def _copy_slot(model, opt, src):
    pool = model.pool
    t = pool.alloc()
    pool.slots[t].copy_(pool.slots[src])
    if opt is not None:
        opt.state.pop(pool.slots[t], None)
        st = _state(opt, pool.slots[src])
        if st is not None:
            opt.state[pool.slots[t]] = _clone_state(st)
    return t


@torch.no_grad()
def split_expert(model, opt, layer_idx, rep, noise=0.0):
    """Clone a logical expert into a free slot (Net2Net-style widening of the layer).

    Single-row expert: the router row is duplicated and both rows get -log(2) offset, so the
    pre-top-k mixture is unchanged. Multi-row group (an earlier same-layer merge): half of the
    rows move to the new slot, i.e. copy-on-write un-merge with exact softmax mass.
    `noise` (relative to the slot std) breaks the symmetry; it is antithetic when the slot
    is not shared with other layers.
    """
    m = model.blocks[layer_idx].moe
    pool = model.pool
    members = m.groups()[rep]
    s = int(m.rep_slot[rep])
    t = _copy_slot(model, opt, s)
    if noise > 0:
        eps = torch.randn_like(pool.slots[s]) * (noise * float(pool.slots[s].std()))
        pool.slots[t].add_(eps)
        if len(model.slot_users(s)) == 1:
            pool.slots[s].sub_(eps)
    if len(members) > 1:
        moved = [r for r in members if r != rep][: len(members) // 2]
        new_rep = moved[0]
        for r in moved:
            m.row_group[r] = new_rep
        m.rep_slot[new_rep] = t
    else:
        r = m.alloc_row()
        m.router[r] = m.router[rep]
        _copy_router_row_state(opt, m.router, rep, r)
        m.row_offset[rep] -= math.log(2.0)
        m.row_offset[r] = m.row_offset[rep]
        m.rep_slot[r] = t
    model.rebuild()
    return t


@torch.no_grad()
def untie_expert(model, opt, layer_idx, rep):
    """Cross-layer copy-on-write: give this layer's logical expert a private copy of its slot."""
    m = model.blocks[layer_idx].moe
    s = int(m.rep_slot[rep])
    if len(model.slot_users(s)) < 2:
        raise ValueError("slot is not shared")
    t = _copy_slot(model, opt, s)
    m.rep_slot[rep] = t
    model.rebuild()
    return t


@torch.no_grad()
def add_fresh_expert(model, opt, layer_idx, donor_rep, share=0.5):
    """Add a randomly initialised expert (down projection zero) that starts with `share` of
    the donor's routing mass, avoiding router cold start. Fresh Adam state (step 0).
    Not function-preserving: the donor's contribution on its tokens drops by `share`."""
    m = model.blocks[layer_idx].moe
    pool = model.pool
    t = pool.alloc()
    pool.init_slot(t, zero_down=True)
    if opt is not None:
        opt.state.pop(pool.slots[t], None)
    members = m.groups()[donor_rep]
    r = m.alloc_row()
    m.router[r] = m.router[donor_rep]
    _copy_router_row_state(opt, m.router, donor_rep, r)
    base = float(m.row_offset[donor_rep])
    for x in members:
        m.row_offset[x] += math.log(1.0 - share)
    m.row_offset[r] = base + math.log(share)
    m.rep_slot[r] = t
    model.rebuild()
    return t


@torch.no_grad()
def drop_expert(model, opt, layer_idx, rep):
    """Remove a logical expert (pruning baseline). Its slot is freed if nothing else uses it."""
    m = model.blocks[layer_idx].moe
    members = m.groups()[rep]
    s = int(m.rep_slot[rep])
    for x in members:
        m.row_active[x] = False
        m.row_group[x] = x
        m.row_offset[x] = 0.0
    m.rep_slot[rep] = -1
    model.rebuild()
    if not model.slot_users(s):
        if opt is not None:
            opt.state.pop(model.pool.slots[s], None)
        model.pool.release(s)


@torch.no_grad()
def soft_pull(model, pairs, strength):
    """Decoupled pull of aligned slot pairs toward their mean; call after opt.step().
    Each call shrinks the pair distance by a factor (1 - strength)."""
    for a, b in pairs:
        pa, pb = model.pool.slots[a], model.pool.slots[b]
        delta = (pa - pb) * (strength / 2)
        pa.sub_(delta)
        pb.add_(delta)
