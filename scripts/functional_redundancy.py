#!/usr/bin/env python
"""Functional redundancy of a trained slot-pool MoE (final_model.pt written by scripts/train.py).

A unit is functionally redundant when removing it, after the best least-squares compensation by the
units that stay, hardly changes the loss, whatever its weights look like. Calibration tokens give,
per layer, the gate-weighted Gram matrix P_e = sum_t p_t^2 a_t a_t^T of every expert's hidden
activations a (silu(gate x) * up x) and the Fisher of the MoE output G = E[g g^T], g = d loss / d y.

Deleting a set D of hidden units of an expert and refitting the down rows of the others is a linear
least-squares problem: w_K += P_KK^-1 P_KD w_D, with residual tr(W_D^T (P^-1)_DD^-1 W_D G) (sum over
calibration tokens). Its greedy version is Optimal Brain Surgeon on the down projection; half of
the residual divided by the number of tokens is the Fisher prediction of the loss increase.

Part 1, neurons: delete a fraction q of all hidden units, chosen by
    random | weight magnitude | weight similarity (best triple cosine inside the expert) |
    output energy | best single partner (activation correlation) | LS-OBS in L2 | LS-OBS in G
  always followed by the same least-squares compensation (plus two rules without it).
Part 2, expert pairs: merge M pairs per layer, chosen by
    random | router cosine | weight soft alignment | rerouting cost on data,
  with drop | reroute (keep one expert's weights) | aligned weight average |
  functional merge (keep the best f of the pair's 2f hidden units, least-squares down projection).
Part 3, single pairs: Delta loss of individual merges against every predictor (rank correlation).
Every configuration is scored by the real change of the validation loss.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmg.model import ModelConfig, MoETransformer  # noqa: E402
from mmg.moe import _dispatch  # noqa: E402
from mmg.ops import align_units, transform_units  # noqa: E402

NEURON_RULES = ("random", "magnitude", "weight_sim", "energy", "best_partner", "obs_l2", "obs_fisher")
PAIR_RULES = ("random", "router_cos", "weight_softalign", "reroute_cost")
PAIR_OPS = ("drop", "reroute", "avg_aligned", "functional")


def log(rec):
    print("REC " + json.dumps(rec), flush=True)


def autocast(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MoETransformer(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(dev)
    model.compute_dtype = torch.bfloat16 if dev.type == "cuda" else None
    model.eval()
    model.rebuild()
    return model


def val_batches(tokens, start, n_batches, batch_size, seq_len, dev):
    span = seq_len + 1
    end = start + n_batches * batch_size * span
    if end > len(tokens):
        raise ValueError(f"val tokens: need {end}, have {len(tokens)}")
    arr = np.asarray(tokens[start:end], dtype=np.int64).reshape(n_batches, batch_size, span)
    out = []
    for a in arr:
        t = torch.from_numpy(a.copy()).to(dev)
        out.append((t[:, :-1].contiguous(), t[:, 1:].contiguous()))
    return out, end


@torch.no_grad()
def evaluate(model, batches):
    dev = batches[0][0].device
    tot = torch.zeros((), device=dev, dtype=torch.float64)
    for x, y in batches:
        with autocast(dev):
            loss, _ = model(x, y)
        tot += loss.double()
    return float(tot) / len(batches)


def layer_weights(model, li):
    """[E, 3, f, d] float32 copy of a layer's experts (gate, up, down^T), in router order."""
    slots = model.blocks[li].moe._s["slots"]
    return torch.stack([model.pool.slots[s].detach().float() for s in slots])


# ---------------------------------------------------------------- calibration statistics


def gram_update(s, x, local, top_p, Wl):
    E = Wl.shape[0]
    order, tok, pos_sorted, _, _, rows = _dispatch(local, E, None)
    cap = rows // E
    xp = x.new_zeros(E * cap, x.shape[1]).index_copy(0, pos_sorted, x.index_select(0, tok)).view(E, cap, -1)
    gp = x.new_zeros(E * cap).index_copy(0, pos_sorted, top_p.reshape(-1)[order]).view(E, cap, 1)
    h = F.silu(torch.bmm(xp, Wl[:, 0].transpose(1, 2))) * torch.bmm(xp, Wl[:, 1].transpose(1, 2)) * gp
    s["P"] += torch.bmm(h.transpose(1, 2), h).double()
    flat = local.reshape(-1)
    s["mass"].index_add_(0, flat, top_p.reshape(-1).double())
    s["count"] += torch.bincount(flat, minlength=E).double()


def collect(model, batches, store_batches):
    """One forward + backward per batch. Per layer: P [E, f, f], G [d, d], routing mass and count,
    and for the first store_batches the MoE inputs with their routing (for expert-pair analysis)."""
    cfg = model.cfg
    dev = model.embed.weight.device
    d, f = cfg.d_model, cfg.d_ff
    W = [layer_weights(model, li) for li in range(cfg.n_layers)]
    E = W[0].shape[0]
    st = []
    for _ in range(cfg.n_layers):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)  # noqa: E731
        st.append({"P": z(E, f, f), "G": z(d, d), "mass": z(E), "count": z(E), "x": [], "idx": [], "gate": []})
    flag = {"store": False}

    def pre_hook(li):
        def hook(mod, args):
            x = args[0].detach().reshape(-1, d)
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
                x = x.float()
                top_p, top_r = mod.route_logits(x).softmax(-1).topk(min(cfg.top_k, E), dim=-1)
                local = mod._s["local_of_row"][top_r]
                gram_update(st[li], x, local, top_p, W[li])
                if flag["store"]:
                    st[li]["x"].append(x.to(torch.bfloat16))
                    st[li]["idx"].append(local.to(torch.int16))
                    st[li]["gate"].append(top_p)
        return hook

    def post_hook(li):
        def hook(mod, args, out):
            y = out[0]
            if y.requires_grad:
                def grad_hook(g):
                    g = g.detach().reshape(-1, d).float()
                    st[li]["G"] += (g.T @ g).double()
                y.register_hook(grad_hook)
        return hook

    handles = []
    for li, blk in enumerate(model.blocks):
        handles.append(blk.moe.register_forward_pre_hook(pre_hook(li)))
        handles.append(blk.moe.register_forward_hook(post_hook(li)))
    n_tok = n_store = 0
    try:
        for bi, (x, y) in enumerate(batches):
            flag["store"] = bi < store_batches
            with autocast(dev):
                loss, _ = model(x, y)
            (loss.float() * y.numel()).backward()  # per-token loss sum: g_t of every token
            model.zero_grad(set_to_none=True)
            n_tok += y.numel()
            n_store += y.numel() if flag["store"] else 0
    finally:
        for h in handles:
            h.remove()
    for s in st:
        s["G"] /= n_tok
        if s["x"]:
            s["x"], s["idx"], s["gate"] = torch.cat(s["x"]), torch.cat(s["idx"]).long(), torch.cat(s["gate"])
    return st, n_tok, n_store


def fisher_factor(G):
    """R with R R^T = G, so |v R|^2 = v^T G v."""
    lam, V = torch.linalg.eigh(G)
    return V * lam.clamp_min(0).sqrt()


def ridge_inverse(P, rel):
    lam = rel * P.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-12
    eye = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    return torch.linalg.inv(P + lam[..., None, None] * eye)


# ---------------------------------------------------------------- least-squares elimination


def obs_eliminate(Pinv, B, n_remove):
    """Greedy backward elimination of features of least-squares fits, batched over dim 0.

    Pinv [b, m, m] = (Phi^T Phi + ridge)^-1, B [b, m, o] = coefficients in the output metric.
    Dropping feature i and refitting the rest raises the residual by |B_i|^2 / Pinv_ii and changes
    B by -Pinv[:, i] B_i / Pinv_ii. Returns removal order, marginal costs [b, n_remove] and the
    alive mask after n_remove steps."""
    Pinv, B = Pinv.clone(), B.clone()
    b, m, _ = Pinv.shape
    dev = Pinv.device
    alive = torch.ones(b, m, dtype=torch.bool, device=dev)
    ar = torch.arange(b, device=dev)
    order = torch.empty(b, n_remove, dtype=torch.long, device=dev)
    marg = torch.empty(b, n_remove, dtype=Pinv.dtype, device=dev)
    for s in range(n_remove):
        diag = torch.diagonal(Pinv, dim1=1, dim2=2)
        cost = ((B * B).sum(-1) / diag.clamp_min(1e-300)).masked_fill(~alive, float("inf"))
        c, i = cost.min(1)
        order[:, s], marg[:, s] = i, c
        col = Pinv[ar, :, i]
        piv = Pinv[ar, i, i].clamp_min(1e-300)
        B -= col[:, :, None] * (B[ar, i, :] / piv[:, None])[:, None, :]
        Pinv -= col[:, :, None] * col[:, None, :] / piv[:, None, None]
        alive[ar, i] = False
        B[ar, i, :] = 0
        Pinv[ar, i, :] = 0
        Pinv[ar, :, i] = 0
    return order, marg, alive


def refit_after_removal(Pinv, B, D, G=None):
    """Least-squares refit of one fit after removing the feature set D (bool [m]):
    B_K <- B_K - Pinv_KD Pinv_DD^-1 B_D. Returns new B (rows of D zero) and the residual increase
    tr(B_D^T Pinv_DD^-1 B_D G) (G=None: identity)."""
    Bn = torch.zeros_like(B)
    if not bool(D.any()):
        return B.clone(), 0.0
    K = ~D
    sol = torch.linalg.solve(Pinv[D][:, D], B[D])
    if bool(K.any()):
        Bn[K] = B[K] - Pinv[K][:, D] @ sol
    BG = B[D] if G is None else B[D] @ G
    return Bn, float((sol * BG).sum())


# ---------------------------------------------------------------- part 1: neurons


def neuron_rule(rule, L, gen):
    """('score', [E, f]) lower = delete first, or ('prefix', order [E, f], marginal [E, f])."""
    W, P = L["W"], L["P"]
    E, _, f, _ = W.shape
    dev = W.device
    if rule == "random":
        return "score", torch.rand(E, f, generator=gen, dtype=torch.float64).to(dev)
    if rule == "magnitude":
        return "score", W.norm(dim=-1).prod(1).double()
    if rule == "weight_sim":
        g, u, w = (F.normalize(W[:, r], dim=-1) for r in range(3))
        s = (torch.bmm(g, g.transpose(1, 2)) + (torch.bmm(u, u.transpose(1, 2)) + torch.bmm(w, w.transpose(1, 2))).abs()) / 3
        s.diagonal(dim1=1, dim2=2).fill_(-9.0)
        return "score", -s.max(-1).values.double()
    wGw = ((L["Wd"] @ L["G"]) * L["Wd"]).sum(-1)  # w_i^T G w_i
    diag = P.diagonal(dim1=1, dim2=2)
    if rule == "energy":
        return "score", diag * wGw
    if rule == "best_partner":
        rho2 = P * P / (diag[:, :, None] * diag[:, None, :]).clamp_min(1e-300)
        rho2.diagonal(dim1=1, dim2=2).fill_(0.0)
        return "score", diag * (1 - rho2.max(-1).values.clamp(0, 1)) * wGw
    B = L["Wd"] if rule == "obs_l2" else L["Wd"] @ L["R"]
    order, marg, _ = obs_eliminate(L["Pinv"], B, f)
    return "prefix", order, marg


def select_global(items, frac):
    """Deletion masks [E, f] per layer for the globally cheapest `frac` of all hidden units."""
    kind = items[0][0]
    if kind == "score":
        flat = torch.cat([it[1].reshape(-1) for it in items])
    else:
        flat = torch.cat([torch.cummax(it[2], dim=1).values.reshape(-1) for it in items])
    n_del = int(round(frac * flat.numel()))
    chosen = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    chosen[torch.argsort(flat, stable=True)[:n_del]] = True
    out, o = [], 0
    for it in items:
        n = it[1].numel()
        c = chosen[o : o + n].view(it[1].shape)
        o += n
        if kind == "score":
            out.append(c)
        else:
            order = it[1]
            n_e = c.sum(1)
            rank = torch.empty_like(order).scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
            out.append(rank < n_e[:, None])
    return out


def select_allocation(items, frac, mode):
    """Deletion masks [E, f] per layer. mode: 'global' = cheapest units anywhere in the model,
    'layer' = the same fraction in every layer, 'expert' = the same number in every expert."""
    if mode == "global":
        return select_global(items, frac)
    if mode == "layer":
        return [select_global([it], frac)[0] for it in items]
    if mode != "expert":
        raise ValueError(mode)
    out = []
    for it in items:
        if it[0] == "score":
            s = it[1]
            idx = torch.argsort(s, dim=1, stable=True)[:, : int(round(frac * s.shape[1]))]
        else:
            idx = it[1][:, : int(round(frac * it[1].shape[1]))]
        out.append(torch.zeros(idx.shape[0], it[1].shape[1], dtype=torch.bool, device=idx.device).scatter_(1, idx, True))
    return out


@torch.no_grad()
def apply_neuron_deletion(model, lays, masks, compensate, n_tok):
    """Zero the gate/up rows of the deleted hidden units; set the down rows to the least-squares
    refit (compensate) or just zero them. Returns the Fisher prediction of the loss increase."""
    inc = 0.0
    for li, D in enumerate(masks):
        L = lays[li]
        slots = model.blocks[li].moe._s["slots"]
        for e in torch.nonzero(D.any(1)).flatten().tolist():
            De = D[e]
            W = L["Wd"][e]
            if compensate:
                Wn, c = refit_after_removal(L["Pinv"][e], W, De, L["G"])
            else:
                Wn = W.clone()
                Wn[De] = 0
                c = float(((L["P"][e][De][:, De] @ W[De]) * (W[De] @ L["G"])).sum())
            inc += c
            p = model.pool.slots[slots[e]]
            p[0][De] = 0
            p[1][De] = 0
            p[2].copy_(Wn.to(p.dtype))
    return inc / (2 * n_tok)


# ---------------------------------------------------------------- part 2: expert pairs


def weight_softalign(W, chunk=2048):
    """[E, E] mean over the neurons of a of their best triple similarity in b, symmetrised."""
    E, _, f, d = W.shape
    g, u, w = (F.normalize(W[:, r].reshape(E * f, d), dim=-1) for r in range(3))
    q_exp = torch.arange(E, device=W.device).repeat_interleave(f)
    out = torch.zeros(E, E, device=W.device)
    for a in range(0, E * f, chunk):
        b = min(E * f, a + chunk)
        s = (g[a:b] @ g.T + (u[a:b] @ u.T + w[a:b] @ w.T).abs()) / 3
        out.index_add_(0, q_exp[a:b], s.view(b - a, E, f).max(-1).values)
        del s
    out /= f
    return ((out + out.T) / 2).double()


def router_cos(model, li):
    m = model.blocks[li].moe
    r = F.normalize(m.router.detach().float()[m._s["reps_t"]], dim=-1)
    return (r @ r.T).double()


def expert_outputs(X, W):
    """Outputs of every expert of W [E, 3, f, d] on tokens X [T, d] -> [T, E, d]."""
    E, _, f, d = W.shape
    h = F.silu(X @ W[:, 0].reshape(E * f, d).T) * (X @ W[:, 1].reshape(E * f, d).T)
    return torch.einsum("tef,efd->ted", h.view(-1, E, f), W[:, 2])


def reroute_cost(W, s, R, n_store, per_expert, gen):
    """D[l, k] = Fisher prediction of the loss increase when expert l's assignments are served by
    expert k: 1/(2N) sum_{t routed to l} p_t^2 |E_l(x_t) - E_k(x_t)|_G^2 (sampled)."""
    E = W.shape[0]
    D = torch.zeros(E, E, dtype=torch.float64, device=W.device)
    Rf = R.float()
    for l in range(E):
        rows, cols = (s["idx"] == l).nonzero(as_tuple=True)
        cnt = rows.numel()
        if cnt == 0:
            continue
        sel = torch.randperm(cnt, generator=gen)[:per_expert].to(rows.device)
        t, pl = rows[sel], s["gate"][rows[sel], cols[sel]]
        O = expert_outputs(s["x"][t].float(), W)
        diff = (O[:, l : l + 1] - O) * pl[:, None, None]
        D[l] = ((diff @ Rf) ** 2).sum(-1).sum(0).double() * (cnt / sel.numel())
    return D / (2 * n_store)


def pair_cost_matrix(rule, C, mass, gen):
    """Symmetric [E, E] cost (lower = merge first) for a pair-selection rule."""
    E = mass.numel()
    if rule == "random":
        c = torch.rand(E, E, generator=gen, dtype=torch.float64).to(mass.device)
        return (c + c.T) / 2
    if rule in ("router_cos", "weight_softalign"):
        return -C
    rem_is_row = mass[:, None] <= mass[None, :]  # the less used expert is removed
    return torch.where(rem_is_row, C, C.T)  # C[rem, keep]


def greedy_pairs(cost, mass, n_pairs):
    """Greedy matching by ascending cost: [(keep, remove)] with keep = the more used expert."""
    E = cost.shape[0]
    iu = np.triu_indices(E, 1)
    c = cost.cpu().numpy()[iu]
    m = mass.cpu().numpy()
    used = np.zeros(E, dtype=bool)
    out = []
    for i in np.argsort(c, kind="stable"):
        a, b = int(iu[0][i]), int(iu[1][i])
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        out.append((a, b) if m[a] >= m[b] else (b, a))
        if len(out) == n_pairs:
            break
    return out


def functional_merge(W, s, R, keep, rem, ridge, n_store):
    """Merged expert for routing group {keep, rem} (gate = p_keep + p_rem): the f hidden units out
    of both experts' 2f that best reproduce p_k E_k + p_r E_r on their tokens, chosen by LS-OBS in
    the Fisher metric, with a least-squares down projection. Returns weights [3, f, d] and the
    Fisher prediction of the loss increase."""
    f = W.shape[2]
    mk, mr = s["idx"] == keep, s["idx"] == rem
    pk, pr = (s["gate"] * mk).sum(1), (s["gate"] * mr).sum(1)
    u = (pk > 0) | (pr > 0)
    if not bool(u.any()):
        return W[keep].clone(), 0.0
    X, pk, pr = s["x"][u].float(), pk[u], pr[u]
    ak = F.silu(X @ W[keep, 0].T) * (X @ W[keep, 1].T)
    ar_ = F.silu(X @ W[rem, 0].T) * (X @ W[rem, 1].T)
    Y = (pk[:, None] * (ak @ W[keep, 2]) + pr[:, None] * (ar_ @ W[rem, 2])).double()
    Phi = (torch.cat([ak, ar_], 1) * (pk + pr)[:, None]).double()
    Pinv = ridge_inverse(Phi.T @ Phi, ridge)
    B = Pinv @ (Phi.T @ Y)
    base = float((((Y - Phi @ B) @ R) ** 2).sum())
    _, marg, alive = obs_eliminate(Pinv[None], (B @ R)[None], f)
    keep_mask = alive[0]
    Bn, _ = refit_after_removal(Pinv, B, ~keep_mask)
    gate = torch.cat([W[keep, 0], W[rem, 0]])[keep_mask]
    up = torch.cat([W[keep, 1], W[rem, 1]])[keep_mask]
    out = torch.stack([gate, up, Bn[keep_mask].float()])
    return out, (base + float(marg.sum())) / (2 * n_store)


@torch.no_grad()
def apply_pairs(model, li, pairs, op, W, mass, func_cache, func_fn):
    """Merge (keep, remove) pairs of layer li; slots are private to the layer (no cross-layer ties)."""
    m = model.blocks[li].moe
    reps, slots = m._s["reps"], m._s["slots"]
    groups = m.groups()
    pool = model.pool
    pred = 0.0
    for keep, rem in pairs:
        rk, rr = reps[keep], reps[rem]
        sk, sr = slots[keep], slots[rem]
        if op == "drop":
            for x in groups[rr]:
                m.row_active[x] = False
                m.row_group[x] = x
                m.row_offset[x] = 0.0
        else:
            pa = pool.slots[sk]
            if op == "avg_aligned":
                pb = pool.slots[sr]
                perm, scale = align_units(pa, pb)
                transform_units(pb, None, perm, scale)
                a = float(mass[keep] / (mass[keep] + mass[rem]).clamp_min(1e-12))
                pa.mul_(a).add_(pb, alpha=1.0 - a)
            elif op == "functional":
                key = (li, keep, rem)
                if key not in func_cache:
                    func_cache[key] = func_fn(keep, rem)
                w, c = func_cache[key]
                pa.copy_(w.to(pa.dtype))
                pred += c
            for x in groups[rr]:
                m.row_group[x] = rk
        m.rep_slot[rr] = -1
        pool.release(sr)
    return pred


def obs_prune(model, calib_batches, frac, ridge=1e-4, compensate=True):
    """Delete the globally cheapest `frac` of all expert hidden units (LS-OBS in the Fisher metric)
    with least-squares compensation. Returns the per-layer deleted fractions."""
    st, n_tok, _ = collect(model, calib_batches, 0)
    lays = []
    for li in range(model.cfg.n_layers):
        W = layer_weights(model, li)
        s = st[li]
        lays.append({"W": W, "Wd": W[:, 2].double(), "P": s["P"], "Pinv": ridge_inverse(s["P"], ridge),
                     "G": s["G"], "R": fisher_factor(s["G"])})
    items = [neuron_rule("obs_fisher", L, None) for L in lays]
    masks = select_global(items, frac)
    apply_neuron_deletion(model, lays, masks, compensate, n_tok)
    model.rebuild()
    return [round(float(m.float().mean()), 4) for m in masks]


# ---------------------------------------------------------------- driver


def spearman(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="final_model.pt of scripts/train.py")
    ap.add_argument("--val", required=True, help="val.bin (uint16 tokens)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--calib-batches", type=int, default=128)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--store-batches", type=int, default=64, help="calibration batches kept for part 2/3")
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--single-eval-batches", type=int, default=32)
    ap.add_argument("--fractions", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--pairs", default="16,64,128", help="merged pairs per layer")
    ap.add_argument("--single-layers", default="2,5")
    ap.add_argument("--single-per-rule", type=int, default=12)
    ap.add_argument("--reroute-samples", type=int, default=256)
    ap.add_argument("--ridge", type=float, default=1e-4, help="relative ridge of the Gram matrices")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    parts = {int(p) for p in args.parts.split(",") if p}
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    model = load_model(args.model, dev)
    cfg = model.cfg
    Lyr = cfg.n_layers
    tokens = np.memmap(args.val, dtype=np.uint16, mode="r")
    eval_b, eval_end = val_batches(tokens, 0, args.eval_batches, args.eval_bs, cfg.seq_len, dev)
    calib_span = args.calib_batches * args.calib_bs * (cfg.seq_len + 1)
    calib_start = len(tokens) - calib_span
    if calib_start < eval_end:
        raise ValueError("calibration and evaluation tokens overlap")
    calib_b, _ = val_batches(tokens, calib_start, args.calib_batches, args.calib_bs, cfg.seq_len, dev)
    single_b = eval_b[: args.single_eval_batches]
    orig = {k: v.clone() for k, v in model.state_dict().items()}

    def restore():
        model.load_state_dict(orig)

    base = evaluate(model, eval_b)
    base_again = evaluate(model, eval_b)
    base_single = evaluate(model, single_b)
    log({"part": 0, "base_val_loss": base, "repeat_diff": base_again - base, "base_single": base_single,
         "eval_tokens": args.eval_batches * args.eval_bs * cfg.seq_len, "d_model": cfg.d_model,
         "layers": Lyr, "experts": cfg.experts_per_layer, "d_ff": cfg.d_ff, "top_k": cfg.top_k})

    st, n_tok, n_store = collect(model, calib_b, args.store_batches)
    del calib_b
    lays = []
    for li in range(Lyr):
        W = layer_weights(model, li)
        s = st[li]
        lays.append({"W": W, "Wd": W[:, 2].double(), "P": s["P"], "Pinv": ridge_inverse(s["P"], args.ridge),
                     "G": s["G"], "R": fisher_factor(s["G"]), "mass": s["mass"]})
    log({"part": 0, "calib_tokens": n_tok, "stored_tokens": n_store, "elapsed_s": round(time.time() - t0),
         "dead_experts": [int((s["count"] == 0).sum()) for s in st],
         "fisher_trace": [round(float(s["G"].trace()), 6) for s in st]})

    results = []

    def record(rec):
        rec["elapsed_s"] = round(time.time() - t0)
        results.append(rec)
        log(rec)

    if 1 in parts:
        fracs = [float(x) for x in args.fractions.split(",")]
        for rule in NEURON_RULES:
            items = [neuron_rule(rule, lays[li], gen) for li in range(Lyr)]
            variants = [True] + ([False] if rule in ("energy", "obs_fisher") else [])
            for q in fracs:
                masks = select_global(items, q)
                per_layer = [round(float(m.float().mean()), 4) for m in masks]
                for comp in variants:
                    restore()
                    pred = apply_neuron_deletion(model, lays, masks, comp, n_tok)
                    v = evaluate(model, eval_b)
                    record({"part": 1, "rule": rule, "compensate": comp, "frac": q, "val_loss": v,
                            "delta": v - base, "pred_delta": pred, "frac_per_layer": per_layer})
        restore()

    if parts & {2, 3}:
        Ms = [int(x) for x in args.pairs.split(",")]
        try:
            import scipy  # noqa: F401
            ops = PAIR_OPS
        except ImportError:
            ops = tuple(o for o in PAIR_OPS if o != "avg_aligned")
        crit = {"router_cos": [], "weight_softalign": [], "reroute_cost": []}
        for li in range(Lyr):
            crit["router_cos"].append(router_cos(model, li))
            crit["weight_softalign"].append(weight_softalign(lays[li]["W"]))
            crit["reroute_cost"].append(reroute_cost(lays[li]["W"], st[li], lays[li]["R"], n_store,
                                                     args.reroute_samples, gen))
        log({"part": 2, "criteria_ready_s": round(time.time() - t0)})
        func_cache = {}

        def func_fn(li):
            return lambda keep, rem: functional_merge(lays[li]["W"], st[li], lays[li]["R"], keep, rem,
                                                      args.ridge, n_store)

    if 2 in parts:
        for rule in PAIR_RULES:
            plist = []
            for li in range(Lyr):
                C = crit[rule][li] if rule != "random" else None
                cost = pair_cost_matrix(rule, C, lays[li]["mass"], gen)
                plist.append(greedy_pairs(cost, lays[li]["mass"], max(Ms)))
            for M in Ms:
                for op in ops:
                    restore()
                    pred = 0.0
                    for li in range(Lyr):
                        pairs = plist[li][:M]
                        pred += apply_pairs(model, li, pairs, op, lays[li]["W"], lays[li]["mass"], func_cache, func_fn(li))
                        if op == "reroute":
                            D = crit["reroute_cost"][li]
                            pred += float(sum(D[r, k] for k, r in pairs))
                    model.rebuild()
                    v = evaluate(model, eval_b)
                    record({"part": 2, "rule": rule, "op": op, "pairs_per_layer": M,
                            "experts_removed_frac": M / cfg.experts_per_layer, "val_loss": v, "delta": v - base,
                            "pred_delta": pred if op in ("reroute", "functional") else None})
        restore()

    if 3 in parts:
        n = args.single_per_rule
        for li in [int(x) for x in args.single_layers.split(",") if x != ""]:
            if li >= Lyr:
                continue
            mass = lays[li]["mass"]
            E = mass.numel()
            iu = torch.triu_indices(E, E, 1, device=mass.device)
            cand = []
            for rule in PAIR_RULES:
                cost = pair_cost_matrix(rule, crit[rule][li] if rule != "random" else None, mass, gen)
                for i in torch.argsort(cost[iu[0], iu[1]])[:n].tolist():
                    a, b = int(iu[0, i]), int(iu[1, i])
                    pair = (a, b) if mass[a] >= mass[b] else (b, a)
                    if pair not in [c[0] for c in cand]:
                        cand.append((pair, rule))
            for (keep, rem), rule in cand:
                rec = {"part": 3, "layer": li, "keep": keep, "remove": rem, "picked_by": rule,
                       "router_cos": float(crit["router_cos"][li][keep, rem]),
                       "weight_softalign": float(crit["weight_softalign"][li][keep, rem]),
                       "reroute_cost": float(crit["reroute_cost"][li][rem, keep]),
                       "mass_removed": float(mass[rem] / mass.sum())}
                for op in ("reroute", "functional"):
                    restore()
                    pred = apply_pairs(model, li, [(keep, rem)], op, lays[li]["W"], mass, func_cache, func_fn(li))
                    model.rebuild()
                    rec[f"delta_{op}"] = evaluate(model, single_b) - base_single
                    if op == "functional":
                        rec["functional_pred"] = pred
                record(rec)
        restore()
        p3 = [r for r in results if r["part"] == 3]
        corr = {}
        for op in ("reroute", "functional"):
            y = [r[f"delta_{op}"] for r in p3]
            corr[op] = {k: spearman([r[k] for r in p3], y)
                        for k in ("router_cos", "weight_softalign", "reroute_cost", "functional_pred", "mass_removed")}
        record({"part": 3, "spearman_vs_delta": corr, "n_pairs": len(p3)})

    with open(os.path.join(args.out, "results.json"), "w") as fo:
        json.dump({"base_val_loss": base, "calib_tokens": n_tok, "results": results}, fo)
    log({"done": True, "elapsed_s": round(time.time() - t0)})


if __name__ == "__main__":
    main()
