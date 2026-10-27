import torch
import torch.nn.functional as F
from torch import nn


GROUP_ALIGN = 16  # rows per expert group are padded to this for torch._grouped_mm


def grouped_mm_available(x, w):
    return (
        hasattr(torch, "_grouped_mm")
        and x.is_cuda
        and w.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(x.device) >= (9, 0)
    )


def _dispatch(local, n, align):
    """Sort assignments by expert. Returns (order, tok, pos_sorted, pos, ends, rows): pos_sorted /
    pos give each assignment's row in a buffer where expert e owns rows [ends[e-1], ends[e]),
    in sorted and in (token, choice) order. align=None pads every expert to the max count."""
    T, k = local.shape
    dev = local.device
    flat_local = local.reshape(-1)  # assignment a = t * k + j
    order = torch.argsort(flat_local, stable=True)
    sorted_local = flat_local[order]
    tok = torch.div(order, k, rounding_mode="floor")
    counts = torch.bincount(sorted_local, minlength=n)
    if align is None:
        size = torch.full_like(counts, int(counts.max()))
    else:
        size = (counts + align - 1) // align * align
    ends = torch.cumsum(size, 0)
    rank = torch.arange(T * k, device=dev) - (torch.cumsum(counts, 0) - counts)[sorted_local]
    pos_sorted = (ends - size)[sorted_local] + rank
    pos = torch.empty_like(pos_sorted).index_copy_(0, order, pos_sorted)
    return order, tok, pos_sorted, pos, ends, int(ends[-1])


def expert_forward_grouped(x, local, gate, w):
    """Same as expert_forward, but with torch._grouped_mm (bf16, sm90+): no padding to max count."""
    T, d = x.shape
    k = local.shape[1]
    n = w.shape[0]
    _, tok, pos_sorted, pos, ends, rows = _dispatch(local, n, GROUP_ALIGN)
    offs = ends.to(torch.int32)
    xs = x.to(w.dtype).index_select(0, tok)
    xp = xs.new_zeros(rows, d).index_copy(0, pos_sorted, xs)
    # _grouped_mm wants a column-major right operand: [n, K, N] with stride(-2) == 1
    g = torch._grouped_mm(xp, w[:, 0].transpose(1, 2), offs=offs)
    u = torch._grouped_mm(xp, w[:, 1].transpose(1, 2), offs=offs)
    down = w[:, 2].transpose(1, 2).contiguous().transpose(1, 2)  # [n, d_ff, d] column-major
    yp = torch._grouped_mm(F.silu(g) * u, down, offs=offs)
    y = yp.index_select(0, pos).view(T, k, d)
    return torch.bmm(gate.to(y.dtype).unsqueeze(1), y).squeeze(1).float()


def expert_forward(x, local, gate, weights):
    """Dropless top-k expert computation with one padded bmm per projection.

    x:       [T, d] token activations
    local:   [T, k] index into `weights` for every (token, choice) assignment
    gate:    [T, k] gate value for every assignment
    weights: [n, 3, d_ff, d] tensor (or list of n [3, d_ff, d]) holding gate, up, down^T
    Returns [T, d] in float32.
    """
    w = weights if torch.is_tensor(weights) else torch.stack(weights)
    T, d = x.shape
    k = local.shape[1]
    n = w.shape[0]
    _, tok, pos_sorted, pos, _, rows = _dispatch(local, n, None)
    cap = rows // n

    # index_select / index_copy only: their backward is index_add / index_select. Advanced
    # indexing with a shared padding row makes the backward serialise on duplicate indices.
    xs = x.to(w.dtype).index_select(0, tok)
    xp = xs.new_zeros(n * cap, d).index_copy(0, pos_sorted, xs).view(n, cap, d)
    g = torch.bmm(xp, w[:, 0].transpose(1, 2))
    u = torch.bmm(xp, w[:, 1].transpose(1, 2))
    yp = torch.bmm(F.silu(g) * u, w[:, 2])  # [n, cap, d]
    y = yp.reshape(n * cap, d).index_select(0, pos).view(T, k, d)
    # gate-weighted sum over the k choices as one batched matmul (no atomics, no fp32 copies)
    return torch.bmm(gate.to(y.dtype).unsqueeze(1), y).squeeze(1).float()


class MoELayer(nn.Module):
    """Router + logical-expert table for one layer. Expert weights live in the shared ExpertPool.

    Router rows are preallocated (max_rows). Rows are grouped; a group is one logical expert whose
    logit is logsumexp over its rows, so merging two logical experts in the same layer keeps the
    softmax mass exactly (p_a + p_b). Each group's representative row maps to one physical slot.
    """

    def __init__(self, d_model, max_rows, top_k, router_std=0.02):
        super().__init__()
        self.d_model = d_model
        self.max_rows = max_rows
        self.top_k = top_k
        self.router = nn.Parameter(torch.randn(max_rows, d_model) * router_std)
        self.register_buffer("row_active", torch.zeros(max_rows, dtype=torch.bool))
        self.register_buffer("row_group", torch.arange(max_rows))
        self.register_buffer("row_offset", torch.zeros(max_rows))
        self.register_buffer("rep_slot", torch.full((max_rows,), -1, dtype=torch.long))
        self._s = None
        self.last_counts = None
        self.use_grouped_mm = True

    def groups(self):
        """{representative row: [member rows]} for active rows."""
        grp = self.row_group.tolist()
        out = {}
        for r in self.row_active.nonzero().flatten().tolist():
            out.setdefault(grp[r], []).append(r)
        return out

    def rebuild(self):
        dev = self.router.device
        groups = self.groups()
        reps = sorted(groups)
        slots = [int(self.rep_slot[r]) for r in reps]
        for r in reps:
            if r not in groups[r]:
                raise RuntimeError(f"representative row {r} is not in its own group")
        if slots and min(slots) < 0:
            raise RuntimeError("logical expert without a physical slot")
        if len(set(slots)) != len(slots):
            raise RuntimeError("a layer may reference a slot once; merge rows instead")
        local_of_row = torch.full((self.max_rows,), -1, dtype=torch.long)
        for i, r in enumerate(reps):
            local_of_row[r] = i
        multi = [(r, m) for r, m in groups.items() if len(m) > 1]
        nonrep = torch.zeros(self.max_rows, dtype=torch.bool)
        for r, members in multi:
            for x in members:
                if x != r:
                    nonrep[x] = True
        self._s = {
            "reps": reps,
            "slots": slots,
            "reps_t": torch.tensor(reps, dtype=torch.long, device=dev),
            "local_of_row": local_of_row.to(dev),
            "multi": [(r, torch.tensor(m, dtype=torch.long, device=dev)) for r, m in multi],
            "multi_reps": torch.tensor([r for r, _ in multi], dtype=torch.long, device=dev),
            "nonrep": nonrep.to(dev),
            "group_size": torch.tensor([len(groups[r]) for r in reps], dtype=torch.float32, device=dev),
            "n_rows": sum(len(m) for m in groups.values()),
        }

    def alloc_row(self):
        free = (~self.row_active).nonzero().flatten()
        if free.numel() == 0:
            raise RuntimeError("no free router rows; raise max_rows")
        r = int(free[0])
        self.row_active[r] = True
        self.row_group[r] = r
        self.row_offset[r] = 0.0
        return r

    def route_logits(self, x):
        s = self._s
        logits = F.linear(x.float(), self.router.float()) + self.row_offset
        logits = logits.masked_fill(~self.row_active, float("-inf"))
        if s["multi"]:
            lse = torch.stack(
                [torch.logsumexp(logits.index_select(1, m), dim=1) for _, m in s["multi"]], dim=1
            )
            logits = logits.index_copy(1, s["multi_reps"], lse)
            logits = logits.masked_fill(s["nonrep"], float("-inf"))
        return logits

    def forward(self, x, pool, compute_dtype=None):
        if self._s is None or self._s["reps_t"].device != x.device:
            self.rebuild()
        s = self._s
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        T = x.shape[0]
        n = len(s["reps"])
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.route_logits(x)
            probs = logits.softmax(dim=-1)
            k = min(self.top_k, n)
            top_p, top_r = probs.topk(k, dim=-1)
        local = s["local_of_row"][top_r]
        # one stack + one cast per layer instead of n small casts
        w = torch.stack([pool.slots[i] for i in s["slots"]]).to(compute_dtype or x.dtype)
        if self.use_grouped_mm and grouped_mm_available(x, w):
            y = expert_forward_grouped(x, local, top_p, w)
        else:
            y = expert_forward(x, local, top_p, w)

        counts = torch.bincount(local.reshape(-1), minlength=n).float()
        frac = counts / (T * k)
        p_mean = probs.mean(0).index_select(0, s["reps_t"])
        target = s["group_size"] / s["n_rows"]
        lb = (frac * p_mean / target).sum()
        z = torch.logsumexp(logits, dim=-1).square().mean()
        self.last_counts = counts.detach()
        return y.reshape(shape), lb, z
