#!/usr/bin/env python
"""How much does the allocation of a neuron-deletion budget matter? OBS (Fisher metric) with
least-squares compensation under three allocations: global (cheapest units anywhere), the same
fraction in every layer, the same number in every expert; plus random deletion. For every
configuration: validation loss, hidden units a token actually passes through (routing is not
changed by deletion), and how many units each expert keeps.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functional_redundancy import (  # noqa: E402
    apply_neuron_deletion, autocast, collect, fisher_factor, layer_weights, load_model, neuron_rule,
    ridge_inverse, select_allocation, spearman, val_batches,
)


@torch.no_grad()
def evaluate_active(model, batches, survivors):
    """Mean validation loss and, per layer, mean surviving hidden units over a token's top-k experts."""
    cfg = model.cfg
    dev = batches[0][0].device
    acc = [torch.zeros((), dtype=torch.float64, device=dev) for _ in range(cfg.n_layers)]
    cnt = [0]

    def pre_hook(li):
        def hook(mod, args):
            x = args[0].reshape(-1, cfg.d_model)
            with torch.autocast(device_type=dev.type, enabled=False):
                top_r = mod.route_logits(x.float()).topk(min(cfg.top_k, len(mod._s["reps"])), dim=-1).indices
            acc[li] += survivors[li][mod._s["local_of_row"][top_r]].sum().double()
            if li == 0:
                cnt[0] += x.shape[0]
        return hook

    hs = [blk.moe.register_forward_pre_hook(pre_hook(li)) for li, blk in enumerate(model.blocks)]
    try:
        tot = torch.zeros((), dtype=torch.float64, device=dev)
        for x, y in batches:
            with autocast(dev):
                loss, _ = model(x, y)
            tot += loss.double()
    finally:
        for h in hs:
            h.remove()
    return float(tot) / len(batches), [float(a) / cnt[0] for a in acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fractions", default="0.5,0.75")
    ap.add_argument("--calib-batches", type=int, default=128)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    model = load_model(args.model, dev)
    cfg = model.cfg
    f, k, Lyr = cfg.d_ff, cfg.top_k, cfg.n_layers
    tokens = np.memmap(args.val, dtype=np.uint16, mode="r")
    eval_b, eval_end = val_batches(tokens, 0, args.eval_batches, args.eval_bs, cfg.seq_len, dev)
    span = args.calib_batches * args.calib_bs * (cfg.seq_len + 1)
    if len(tokens) - span < eval_end:
        raise ValueError("calibration and evaluation tokens overlap")
    calib_b, _ = val_batches(tokens, len(tokens) - span, args.calib_batches, args.calib_bs, cfg.seq_len, dev)
    orig = {kk: v.clone() for kk, v in model.state_dict().items()}
    full = [torch.full((len(m._s["reps"]),), float(f), device=dev) for m in model.moe_layers()]
    base, base_active = evaluate_active(model, eval_b, full)
    results = []

    def record(rec):
        rec["elapsed_s"] = round(time.time() - t0)
        results.append(rec)
        print("REC " + json.dumps(rec), flush=True)

    record({"config": "base", "val_loss": base, "active_per_layer": base_active, "active_max": k * f})
    st, n_tok, _ = collect(model, calib_b, 0)
    del calib_b
    lays = []
    for li in range(Lyr):
        W = layer_weights(model, li)
        s = st[li]
        lays.append({"W": W, "Wd": W[:, 2].double(), "P": s["P"], "Pinv": ridge_inverse(s["P"], args.ridge),
                     "G": s["G"], "R": fisher_factor(s["G"]), "mass": s["mass"]})
    items = {"obs_fisher": [neuron_rule("obs_fisher", L, gen) for L in lays],
             "random": [neuron_rule("random", L, gen) for L in lays]}
    record({"config": "calibrated", "calib_tokens": n_tok})

    configs = [("obs_fisher", m) for m in ("global", "layer", "expert")] + [("random", "global")]
    for q in [float(x) for x in args.fractions.split(",")]:
        for rule, mode in configs:
            model.load_state_dict(orig)
            masks = select_allocation(items[rule], q, mode)
            pred = apply_neuron_deletion(model, lays, masks, True, n_tok)
            model.rebuild()
            surv = [(f - m.sum(1)).float() for m in masks]
            v, active = evaluate_active(model, eval_b, surv)
            per_layer = []
            for li, sv in enumerate(surv):
                s_np = sv.cpu().numpy()
                per_layer.append({
                    "deleted_frac": round(float(masks[li].float().mean()), 4),
                    "survivors_quantiles": [int(np.quantile(s_np, p)) for p in (0, 0.1, 0.5, 0.9, 1.0)],
                    "experts_empty": int((s_np == 0).sum()),
                    "spearman_mass_vs_survivors": spearman(lays[li]["mass"].cpu().numpy(), s_np),
                    "active_per_token": round(active[li], 1),
                })
            record({"config": f"{rule}/{mode}", "frac": q, "val_loss": v, "delta": v - base, "pred_delta": pred,
                    "active_frac": round(sum(active) / (Lyr * k * f), 4),
                    "nonzero_expert_frac": round(1 - float(sum(m.sum() for m in masks)) / sum(m.numel() for m in masks), 4),
                    "per_layer": per_layer})
    with open(os.path.join(args.out, "results.json"), "w") as fo:
        json.dump(results, fo)
    print("REC " + json.dumps({"done": True, "elapsed_s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
