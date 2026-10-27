#!/usr/bin/env python
"""Weight-space redundancy statistics of the FFN / MoE experts of a HuggingFace checkpoint.

Unit   = one SwiGLU FFN: a dense layer MLP, a shared expert or a routed expert.
Neuron = one hidden unit of a unit: (gate row, up row, down column), three d_model vectors.
Neurons live in the residual-stream basis shared by every layer, so they can be compared across
experts and layers without alignment. Their only symmetry, (up, down) -> (c*up, down/c), is
removed by using cosines with the best joint sign:

    sim(a, b) = (cos(g_a, g_b) + |cos(u_a, u_b) + cos(w_a, w_b)|) / 3      in [-1/3, 1]

Query neurons (all of them, or whole units sampled to fit --query-gb) are compared with every
neuron of the model in one pass over the layers. Recorded per query neuron: best match per
candidate layer and inside its own unit. Recorded per (query unit, candidate unit): mean best
match ("soft alignment"), fraction of neurons matched above --cov-threshold ("coverage") and,
for equal shapes, the same-index similarity, i.e. what an unaligned matrix comparison sees.
A randomly rotated copy of some queries gives the chance level for the same candidate count.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BINS = np.linspace(-0.4, 1.0, 141)
NEG = -10.0
KINDS = ("dense", "shared", "routed")

_FFN = r"(?:mlp|block_sparse_moe|moe|feed_forward|ffn)"
_PROJ = r"(gate_proj|up_proj|down_proj|w1|w2|w3)"
ROUTED_RE = re.compile(rf"^(.*?)layers\.(\d+)\.{_FFN}\.experts\.(\d+)\.{_PROJ}\.weight$")
SHARED_RE = re.compile(rf"^(.*?)layers\.(\d+)\.{_FFN}\.(?:shared_experts?|shared_mlp)\.{_PROJ}\.weight$")
DENSE_RE = re.compile(rf"^(.*?)layers\.(\d+)\.{_FFN}\.{_PROJ}\.weight$")
FUSED_RE = re.compile(rf"^(.*?)layers\.(\d+)\.{_FFN}\.experts\.(gate_up_proj|down_proj)$")
PROJ = {"gate_proj": "gate", "w1": "gate", "up_proj": "up", "w3": "up", "down_proj": "down", "w2": "down"}


class Checkpoint:
    def __init__(self, path):
        from safetensors import safe_open

        self._open = safe_open
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {}
            for fn in sorted(n for n in os.listdir(path) if n.endswith(".safetensors")):
                with safe_open(os.path.join(path, fn), framework="pt") as f:
                    for k in f.keys():
                        self.weight_map[k] = fn
        self._handles = {}

    def _h(self, name):
        fn = self.weight_map[name]
        if fn not in self._handles:
            self._handles[fn] = self._open(os.path.join(self.path, fn), framework="pt")
        return self._handles[fn]

    def shape(self, name):
        return tuple(self._h(name).get_slice(name).get_shape())

    def get(self, name):
        t = self._h(name).get_tensor(name)
        if t.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
            s = self.get(name + "_scale_inv").float()  # DeepSeek-V3 style 128x128 block scales
            t = t.float()
            bm, bn = math.ceil(t.shape[0] / s.shape[0]), math.ceil(t.shape[1] / s.shape[1])
            s = s.repeat_interleave(bm, 0)[: t.shape[0]].repeat_interleave(bn, 1)[:, : t.shape[1]]
            t = t * s
        return t


def discover_units(ckpt, hidden, n_layers):
    """FFN units of the language model: the tensor-name prefix (before 'layers.') with the most
    matches wins, which drops vision towers / MTP modules living under other prefixes."""
    per, fused = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))
    for n in ckpt.weight_map:
        m = ROUTED_RE.match(n)
        if m:
            per[m.group(1)][int(m.group(2))].setdefault(("routed", int(m.group(3))), {})[PROJ[m.group(4)]] = n
            continue
        m = SHARED_RE.match(n)
        if m:
            per[m.group(1)][int(m.group(2))].setdefault(("shared", 0), {})[PROJ[m.group(3)]] = n
            continue
        m = DENSE_RE.match(n)
        if m:
            per[m.group(1)][int(m.group(2))].setdefault(("dense", 0), {})[PROJ[m.group(3)]] = n
            continue
        m = FUSED_RE.match(n)
        if m:
            fused[m.group(1)][int(m.group(2))][m.group(3)] = n
    prefixes = set(per) | set(fused)
    if not prefixes:
        return {}
    prefix = max(prefixes, key=lambda p: sum(len(v) for v in per[p].values()) + len(fused[p]))
    per, fused = per[prefix], fused[prefix]
    layers = {}
    for l in sorted(set(per) | set(fused)):
        if n_layers and l >= n_layers:  # e.g. multi-token-prediction modules
            continue
        units = []
        for (kind, eid), projs in sorted(per[l].items(), key=lambda kv: (KINDS.index(kv[0][0]), kv[0][1])):
            if set(projs) == {"gate", "up", "down"}:
                f, d = ckpt.shape(projs["gate"])
                if hidden and d != hidden:
                    continue
                units.append({"kind": kind, "id": eid, "names": projs, "f": f})
        if {"gate_up_proj", "down_proj"} <= set(fused[l]):
            gu, dn = ckpt.get(fused[l]["gate_up_proj"]), ckpt.get(fused[l]["down_proj"])
            if gu.shape[1] == hidden:  # [E, d, 2f] -> [E, 2f, d]
                gu = gu.transpose(1, 2)
            if dn.shape[2] == hidden:  # [E, f, d] -> [E, d, f]
                dn = dn.transpose(1, 2)
            f = gu.shape[1] // 2
            units += [{"kind": "routed", "id": e, "tensors": (gu[e, :f], gu[e, f:], dn[e]), "f": f}
                      for e in range(gu.shape[0])]
        if units:
            layers[l] = units
    return layers


def load_unit(ckpt, unit, dev, dt):
    """Unit-normalised gate rows, up rows and down columns, each [f, d] in dtype dt."""
    if "tensors" in unit:
        g, u, w = unit["tensors"]
    else:
        g, u, w = (ckpt.get(unit["names"][k]) for k in ("gate", "up", "down"))
    return [F.normalize(t.to(dev, torch.float32), dim=-1).to(dt) for t in (g, u, w.t())]


def sim_block(Ga, Ua, Wa, Gc, Uc, Wc):
    s = Ga @ Gc.T
    t = Ua @ Uc.T
    t.addmm_(Wa, Wc.T)
    return s.add_(t.abs_()).div_(3)


def hist_summary(v):
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    hist, _ = np.histogram(np.clip(v, BINS[0], BINS[-1] - 1e-6), bins=BINS)
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 4),
        "quantiles": {str(q): round(float(np.quantile(v, q)), 4) for q in (0.5, 0.9, 0.99, 0.999)},
        "frac_above": {str(t): round(float((v > t).mean()), 5) for t in THRESHOLDS},
        "hist": hist.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--query-gb", type=float, default=24.0, help="memory for query neurons")
    ap.add_argument("--chunk-elems", type=float, default=1.5e9, help="elements per similarity block")
    ap.add_argument("--null-queries", type=int, default=4096)
    ap.add_argument("--cov-threshold", type=float, default=0.7)
    ap.add_argument("--max-layers", type=int, default=0, help="debug: only the first N FFN layers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    name = args.name or os.path.basename(os.path.normpath(args.model_dir))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    tcfg = cfg.get("text_config", cfg)
    hidden = tcfg.get("hidden_size")
    ckpt = Checkpoint(args.model_dir)
    layers = discover_units(ckpt, hidden, tcfg.get("num_hidden_layers", 0))
    if args.max_layers:
        layers = {l: layers[l] for l in sorted(layers)[: args.max_layers]}
    if not layers:
        sample = [n for n in ckpt.weight_map if ".layers." in n or n.startswith("layers.")][:60]
        sys.exit("no FFN units recognised; sample tensor names:\n" + "\n".join(sample))

    units = []
    for l in sorted(layers):
        for u in layers[l]:
            u["layer"], u["gid"] = l, len(units)
            units.append(u)
    layer_list = sorted(layers)
    L, nU = len(layer_list), len(units)
    lpos = {l: i for i, l in enumerate(layer_list)}
    d = load_unit(ckpt, units[0], "cpu", torch.float32)[0].shape[1]

    # whole units as queries, randomly sampled to fit the memory budget
    rng = np.random.default_rng(args.seed)
    budget = int(args.query_gb * 2**30 / (3 * d * torch.finfo(dt).bits // 8))
    q_units, total = [], 0
    for gid in rng.permutation(nU).tolist():
        if total + units[gid]["f"] > budget:
            continue
        q_units.append(gid)
        total += units[gid]["f"]
    q_units.sort()
    nQU, Q = len(q_units), total
    print(json.dumps({"model": name, "layers": L, "units": nU, "d_model": d, "neurons": sum(u["f"] for u in units),
                      "query_units": nQU, "query_neurons": Q}), flush=True)

    Gq = torch.empty(Q, d, dtype=dt, device=dev)
    Uq, Wq = torch.empty_like(Gq), torch.empty_like(Gq)
    q_unit = torch.empty(Q, dtype=torch.long)
    q_pos = torch.empty(Q, dtype=torch.long)
    o = 0
    for qi, gid in enumerate(q_units):
        g, u, w = load_unit(ckpt, units[gid], dev, dt)
        f = g.shape[0]
        Gq[o : o + f], Uq[o : o + f], Wq[o : o + f] = g, u, w
        q_unit[o : o + f], q_pos[o : o + f] = qi, torch.arange(f)
        o += f
    q_unit, q_pos = q_unit.to(dev), q_pos.to(dev)
    q_layer = torch.tensor([lpos[units[g]["layer"]] for g in q_units], device=dev)[q_unit]
    q_f = torch.tensor([units[g]["f"] for g in q_units], device=dev)[q_unit]
    q_gid = torch.tensor(q_units, device=dev)[q_unit]

    gen = torch.Generator().manual_seed(args.seed + 1)
    nq = min(args.null_queries, Q)
    sel = torch.randperm(Q, generator=gen)[:nq].to(dev)
    R = torch.linalg.qr(torch.randn(d, d, generator=gen))[0].to(dev)
    Gn, Un, Wn = ((X[sel].float() @ R).to(dt) for X in (Gq, Uq, Wq))

    best_by_layer = torch.full((Q, L), NEG, device=dev)
    best_own = torch.full((Q,), NEG, device=dev)
    null_best = torch.full((nq,), NEG, device=dev)
    pair_sum = torch.zeros(nQU, nU, device=dev)
    pair_cov = torch.zeros(nQU, nU, device=dev)
    same_sum = torch.zeros(nQU, nU, device=dev)

    for l in layer_list:
        li = lpos[l]
        lu = layers[l]
        C = sum(u["f"] for u in lu)
        Gc = torch.empty(C, d, dtype=dt, device=dev)
        Uc, Wc = torch.empty_like(Gc), torch.empty_like(Gc)
        start_of, groups, s0 = {}, [], 0  # groups: runs of consecutive units with the same f
        for u in lu:
            g, uu, w = load_unit(ckpt, u, dev, dt)
            f = g.shape[0]
            Gc[s0 : s0 + f], Uc[s0 : s0 + f], Wc[s0 : s0 + f] = g, uu, w
            start_of[u["gid"]] = s0
            if groups and groups[-1][0] == f:
                groups[-1][2].append(u["gid"])
            else:
                groups.append((f, s0, [u["gid"]]))
            s0 += f
        chunk = max(128, int(args.chunk_elems // C))

        for a in range(0, nq, chunk):
            b = min(nq, a + chunk)
            null_best[a:b] = torch.maximum(null_best[a:b], sim_block(Gn[a:b], Un[a:b], Wn[a:b], Gc, Uc, Wc).max(1).values.float())

        for a in range(0, Q, chunk):
            b = min(Q, a + chunk)
            S = sim_block(Gq[a:b], Uq[a:b], Wq[a:b], Gc, Uc, Wc)
            rows = (q_layer[a:b] == li).nonzero().flatten()
            if rows.numel():  # queries of this layer: score their own unit separately, then mask it
                st = torch.tensor([start_of[g] for g in q_gid[a:b][rows].tolist()], device=dev)
                fr = q_f[a:b][rows]
                ar_f = torch.arange(int(fr.max()), device=dev)
                valid = ar_f[None, :] < fr[:, None]
                cols = torch.where(valid, st[:, None] + ar_f[None, :], st[:, None])
                own = S[rows[:, None], cols].float().masked_fill(~valid, NEG)
                own[torch.arange(rows.numel(), device=dev), q_pos[a:b][rows]] = NEG  # self
                best_own[a + rows] = own.max(1).values
                S[rows[:, None], cols] = NEG
            best_by_layer[a:b, li] = S.max(1).values.float()
            qu = q_unit[a:b]
            for f, g0, gids in groups:
                n = len(gids)
                umax = S[:, g0 : g0 + n * f].view(b - a, n, f).max(-1).values.float()
                cu = torch.tensor(gids, device=dev)
                pair_sum[:, cu] += torch.zeros(nQU, n, device=dev).index_add_(0, qu, umax)
                pair_cov[:, cu] += torch.zeros(nQU, n, device=dev).index_add_(0, qu, (umax > args.cov_threshold).float())
                sf = (q_f[a:b] == f).nonzero().flatten()
                if sf.numel():
                    idx = g0 + torch.arange(n, device=dev)[None, :] * f + q_pos[a:b][sf][:, None]
                    vals = S[sf[:, None], idx].float()
                    same_sum[:, cu] += torch.zeros(nQU, n, device=dev).index_add_(0, qu[sf], vals)
            del S
        del Gc, Uc, Wc
        print(json.dumps({"layer": l, "candidates": C, "elapsed_s": round(time.time() - t0)}), flush=True)

    # ---------------- reduce ----------------
    nan = float("nan")
    qg = torch.tensor(q_units, device=dev)
    qf_u = torch.tensor([units[g]["f"] for g in q_units], device=dev, dtype=torch.float32)
    f_all = torch.tensor([u["f"] for u in units], device=dev, dtype=torch.float32)
    pair = pair_sum / qf_u[:, None]
    cov = pair_cov / qf_u[:, None]
    same = torch.where(qf_u[:, None] == f_all[None, :], same_sum / qf_u[:, None], torch.full_like(same_sum, nan))
    diag = (torch.arange(nQU, device=dev), qg)
    for M in (pair, cov, same):
        M[diag] = nan

    def nanmax(x, dim):
        return torch.nan_to_num(x, nan=-9.0).max(dim)

    def as_np(x):
        x = x.float().cpu().numpy()
        x[x <= -8] = np.nan
        return x

    bbl = best_by_layer.masked_fill(best_by_layer <= NEG + 1, nan)
    ar = torch.arange(Q, device=dev)
    same_layer = bbl[ar, q_layer]
    other = bbl.clone()
    other[ar, q_layer] = nan
    other_layers = nanmax(other, 1).values
    adj = torch.full((Q,), -9.0, device=dev)
    for off in (-1, 1):
        nb = q_layer + off
        ok = (nb >= 0) & (nb < L)
        adj[ok] = torch.maximum(adj[ok], torch.nan_to_num(bbl[ar[ok], nb[ok]], nan=-9.0))
    own = best_own.masked_fill(best_own <= NEG + 1, -9.0)
    anyb = torch.stack([torch.nan_to_num(same_layer, nan=-9.0), other_layers, own]).max(0).values

    neuron = {
        "own_unit": hist_summary(as_np(own)),
        "same_layer_other_units": hist_summary(as_np(torch.nan_to_num(same_layer, nan=-9.0))),
        "adjacent_layers": hist_summary(as_np(adj)),
        "other_layers": hist_summary(as_np(other_layers)),
        "any": hist_summary(as_np(anyb)),
        "null_rotated": hist_summary(as_np(null_best)),
    }
    kind_q = torch.tensor([KINDS.index(units[g]["kind"]) for g in q_units], device=dev)[q_unit]
    by_kind = {k: {"any": hist_summary(as_np(anyb[kind_q == i])),
                   "other_layers": hist_summary(as_np(other_layers[kind_q == i]))}
               for i, k in enumerate(KINDS) if bool((kind_q == i).any())}
    profile = {}
    for delta in range(-(L - 1), L):
        nb = q_layer + delta
        ok = (nb >= 0) & (nb < L)
        if delta and bool(ok.any()):
            profile[str(delta)] = round(float(torch.nanmean(bbl[ar[ok], nb[ok]])), 4)

    u_layer = torch.tensor([lpos[u["layer"]] for u in units], device=dev)
    same_l = u_layer[qg][:, None] == u_layer[None, :]

    def best_in(M, mask):
        return nanmax(torch.where(mask, M, torch.full_like(M, nan)), 1)

    unit_stats = {}
    for scope, mask in (("same_layer", same_l), ("other_layers", ~same_l)):
        unit_stats[scope] = {
            "soft_align_best": hist_summary(as_np(best_in(pair, mask).values)),
            "coverage_best": hist_summary(as_np(best_in(cov, mask).values)),
            "same_index_best": hist_summary(as_np(best_in(same, mask).values)),
        }
    bv, bi = nanmax(pair, 1)

    def uname(g):
        u = units[g]
        return f"L{u['layer']}.{u['kind']}" + (str(u["id"]) if u["kind"] == "routed" else "")

    top = []
    for i in torch.argsort(bv, descending=True)[:30].tolist():
        if float(bv[i]) <= -8:  # no other unit to compare with
            break
        j = int(bi[i])
        sv = float(same[i, j])
        top.append({"a": uname(q_units[i]), "b": uname(j), "soft_align": round(float(bv[i]), 4),
                    "coverage": round(float(cov[i, j]), 4), "same_index": None if math.isnan(sv) else round(sv, 4)})

    result = {
        "model": name, "model_dir": args.model_dir, "model_type": tcfg.get("model_type"), "d_model": d,
        "layers": L, "units": nU, "query_units": nQU, "query_neurons": Q,
        "unit_kinds": {k: sum(u["kind"] == k for u in units) for k in KINDS},
        "unit_f": sorted({u["f"] for u in units}), "cov_threshold": args.cov_threshold,
        "bins": [float(BINS[0]), float(BINS[-1]), len(BINS) - 1],
        "neuron": neuron, "neuron_by_kind": by_kind, "layer_distance_profile": profile,
        "unit": unit_stats, "top_pairs": top, "elapsed_s": round(time.time() - t0),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f)
    np.savez_compressed(os.path.join(args.out, "unit_pairs.npz"), query_units=np.array(q_units),
                        soft_align=pair.cpu().numpy().astype(np.float16),
                        coverage=cov.cpu().numpy().astype(np.float16),
                        same_index=same.cpu().numpy().astype(np.float16))
    slim = {k: v for k, v in result.items() if k not in ("neuron", "neuron_by_kind", "unit")}
    slim["neuron_frac_above"] = {k: v.get("frac_above") for k, v in neuron.items()}
    slim["neuron_quantiles"] = {k: v.get("quantiles") for k, v in neuron.items()}
    slim["unit_frac_above"] = {s: {m: v[m].get("frac_above") for m in v} for s, v in unit_stats.items()}
    slim["unit_quantiles"] = {s: {m: v[m].get("quantiles") for m in v} for s, v in unit_stats.items()}
    print("RESULT_JSON " + json.dumps(slim), flush=True)


if __name__ == "__main__":
    main()
