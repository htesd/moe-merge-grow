#!/usr/bin/env python
"""Functional redundancy of a HuggingFace sparse-MoE checkpoint (Qwen3-MoE style blocks with
`gate` router and `experts`): the neuron-deletion part of functional_redundancy.py on a large
pretrained model.

Every MoE block is replaced by an equivalent grouped-GEMM block (checked against the original on
random inputs) that can collect calibration statistics and whose expert weights can be edited.
Expert hidden units are deleted by a rule (LS-OBS in the Fisher metric under a global / per-layer /
per-expert budget, or random / weight magnitude / weight similarity), always with least-squares
compensation of the down projections, and every configuration is scored by the real change of
the held-out loss. Text: documents decoded from a GPT-2 token file (scripts/prepare_fineweb.py
val.bin) and re-tokenized with the model's tokenizer, or a token-id .npy file.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functional_redundancy import (  # noqa: E402
    fisher_factor, obs_eliminate, refit_after_removal, ridge_inverse, select_allocation,
)
from mmg.moe import _dispatch, expert_forward, expert_forward_grouped, grouped_mm_available  # noqa: E402

GPT2_EOT = 50256


def expert_weights(block, hidden):
    """Router [E, d] and experts [E, 3, f, d] (gate, up, down^T) of a HF sparse-MoE block."""
    router = block.gate.weight
    ex = block.experts
    if isinstance(ex, nn.ModuleList):
        W = torch.stack([torch.stack([m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight.t()]) for m in ex])
    else:  # fused expert tensors (newer transformers)
        gu, dn = ex.gate_up_proj, ex.down_proj
        if gu.shape[-1] != hidden:  # [E, d, 2f] -> [E, 2f, d]
            gu = gu.transpose(1, 2)
        if dn.shape[-1] != hidden:  # [E, d, f] -> [E, f, d]
            dn = dn.transpose(1, 2)
        f = gu.shape[1] // 2
        W = torch.stack([gu[:, :f], gu[:, f:], dn], dim=1)
    return router.detach(), W.detach().contiguous()


def gram_update(s, x, local, top_p, W):
    """Accumulate P_e = sum p^2 a a^T of every expert's hidden activations a (fp32)."""
    E = W.shape[0]
    order, tok, pos_sorted, _, _, rows = _dispatch(local, E, None)
    cap = rows // E
    xw = x.to(W.dtype)
    xp = xw.new_zeros(E * cap, x.shape[1]).index_copy(0, pos_sorted, xw.index_select(0, tok)).view(E, cap, -1)
    gp = torch.zeros(E * cap, device=x.device).index_copy(0, pos_sorted, top_p.reshape(-1)[order].float())
    h = (F.silu(torch.bmm(xp, W[:, 0].transpose(1, 2))) * torch.bmm(xp, W[:, 1].transpose(1, 2))).float()
    h = h * gp.view(E, cap, 1)
    s["P"] += torch.bmm(h.transpose(1, 2), h)
    flat = local.reshape(-1)
    s["mass"].index_add_(0, flat, top_p.reshape(-1).double())
    s["count"] += torch.bincount(flat, minlength=E).double()


class GroupedMoE(nn.Module):
    def __init__(self, router, W, top_k, norm_topk, ret_tuple):
        super().__init__()
        self.register_buffer("router", router)
        self.register_buffer("W", W)
        self.top_k, self.norm_topk, self.ret_tuple = top_k, norm_topk, ret_tuple
        self.stats = None

    def forward(self, h):
        shp = h.shape
        x = h.reshape(-1, shp[-1])
        logits = F.linear(x, self.router)
        p, i = logits.float().softmax(-1).topk(self.top_k, dim=-1)
        if self.norm_topk:
            p = p / p.sum(-1, keepdim=True)
        if self.stats is not None:
            with torch.no_grad():
                gram_update(self.stats, x.detach(), i, p, self.W)
        fwd = expert_forward_grouped if grouped_mm_available(x, self.W) else expert_forward
        y = fwd(x, i, p, self.W).to(h.dtype).view(shp)
        if self.stats is not None and y.requires_grad:
            st = self.stats

            def hook(g):
                g = g.detach().reshape(-1, shp[-1]).float()
                st["G"] += (g.T @ g).double()

            y.register_hook(hook)
        return (y, logits) if self.ret_tuple else y


def replace_blocks(model, check_tokens=32):
    """Swap every sparse-MoE block for a GroupedMoE; returns the new blocks and the largest
    relative output difference seen in the equivalence check."""
    cfg = model.config
    hidden = cfg.hidden_size
    blocks, worst = [], 0.0
    p0 = next(model.parameters())
    g = torch.Generator(device=p0.device).manual_seed(0)
    for layer in model.model.layers:
        blk = layer.mlp
        if not hasattr(blk, "experts"):
            continue
        router, W = expert_weights(blk, hidden)
        x = torch.randn(1, check_tokens, hidden, device=p0.device, generator=g).to(p0.dtype)
        with torch.no_grad():
            ref = blk(x)
            ret_tuple = isinstance(ref, tuple)
            ref = ref[0] if ret_tuple else ref
            new = GroupedMoE(router, W, cfg.num_experts_per_tok, bool(getattr(cfg, "norm_topk_prob", False)), ret_tuple)
            out = new(x)
            out = out[0] if ret_tuple else out
        worst = max(worst, float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)))
        layer.mlp = new
        blocks.append(new)
        del blk
    if p0.is_cuda:
        torch.cuda.empty_cache()
    return blocks, worst


def make_windows(args, tokenizer):
    """(eval windows, calibration windows), each a LongTensor [N, seq_len]."""
    L = args.seq_len
    n_eval, n_cal = args.eval_windows, args.calib_windows
    if args.token_file:
        ids = np.load(args.token_file).astype(np.int64)
        need = (n_eval + n_cal) * L
        if len(ids) < need:
            raise ValueError(f"token file: need {need} tokens, have {len(ids)}")
        w = torch.from_numpy(ids[:need].copy()).view(-1, L)
        return w[:n_eval], w[n_eval:]
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    toks = np.asarray(np.memmap(args.gpt2_val, dtype=np.uint16, mode="r"))
    cuts = np.flatnonzero(toks == GPT2_EOT)
    docs = [toks[a + 1 : b] for a, b in zip(cuts, list(cuts[1:]) + [len(toks)]) if b > a + 1]
    sep = tokenizer.convert_tokens_to_ids("<|endoftext|>")  # document separator of pretraining
    if sep is None or sep == tokenizer.unk_token_id:
        sep = tokenizer.eos_token_id

    def pack(doc_iter, n):
        buf, out = [], []
        for d in doc_iter:
            buf.extend(tokenizer(enc.decode(d.tolist()), add_special_tokens=False)["input_ids"])
            buf.append(sep)
            while len(buf) >= L and len(out) < n:
                out.append(buf[:L])
                buf = buf[L:]
            if len(out) == n:
                return torch.tensor(out)
        raise ValueError("not enough text for the requested windows")

    eval_w = pack(iter(docs), n_eval)  # from the start of the file
    cal_w = pack(reversed(docs), n_cal)  # from the end: disjoint documents
    return eval_w, cal_w


@torch.no_grad()
def evaluate(model, windows, bs, dev):
    tot = n = 0.0
    for i in range(0, len(windows), bs):
        w = windows[i : i + bs].to(dev)
        out = model(input_ids=w, labels=w)
        k = w.numel() - w.shape[0]
        tot += float(out.loss) * k
        n += k
    return tot / n


def calibrate(model, blocks, windows, bs, dev):
    E, _, f, d = blocks[0].W.shape
    for b in blocks:
        b.stats = {"P": torch.zeros(E, f, f, device=dev), "G": torch.zeros(d, d, dtype=torch.float64, device=dev),
                   "mass": torch.zeros(E, dtype=torch.float64, device=dev),
                   "count": torch.zeros(E, dtype=torch.float64, device=dev)}
    emb = model.get_input_embeddings()
    n_tok = 0
    try:
        for i in range(0, len(windows), bs):
            w = windows[i : i + bs].to(dev)
            x = emb(w).detach().requires_grad_(True)
            out = model(inputs_embeds=x, labels=w)
            k = w.numel() - w.shape[0]
            (out.loss.float() * k).backward()  # per-token loss sum
            n_tok += w.numel()
    finally:
        stats = [b.stats for b in blocks]
        for b in blocks:
            b.stats = None
    for s in stats:
        s["G"] /= n_tok
    return stats, n_tok


def weight_rules(W):
    """Data-free deletion scores [E, f] (lower = delete first)."""
    Wf = W.float()
    mag = Wf.norm(dim=-1).prod(1).double()
    g, u, w = (F.normalize(Wf[:, r], dim=-1) for r in range(3))
    s = (torch.bmm(g, g.transpose(1, 2)) + (torch.bmm(u, u.transpose(1, 2)) + torch.bmm(w, w.transpose(1, 2))).abs()) / 3
    s.diagonal(dim1=1, dim2=2).fill_(-9.0)
    return mag, -s.max(-1).values.double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpt2-val", default="", help="GPT-2 uint16 token file with <|endoftext|> separators")
    ap.add_argument("--token-file", default="", help="alternatively: .npy of token ids of this model")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=512)
    ap.add_argument("--calib-windows", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--calib-bs", type=int, default=1)
    ap.add_argument("--fractions", default="0.1,0.25,0.5")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    results = []

    def record(rec):
        rec["elapsed_s"] = round(time.time() - t0)
        results.append(rec)
        print("REC " + json.dumps(rec), flush=True)

    tok = None if args.token_file else AutoTokenizer.from_pretrained(args.model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype, device_map={"": dev})
    except (ImportError, ValueError):  # no accelerate
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype, low_cpu_mem_usage=True).to(dev)
    model.eval()
    model.requires_grad_(False)
    blocks, worst = replace_blocks(model)
    E, _, f, d = blocks[0].W.shape
    eval_w, cal_w = make_windows(args, tok)
    base = evaluate(model, eval_w, args.bs, dev)
    record({"part": 0, "model": os.path.basename(os.path.normpath(args.model_dir)), "moe_layers": len(blocks),
            "experts": E, "d_ff": f, "d_model": d, "top_k": blocks[0].top_k, "equivalence_rel_err": worst,
            "eval_tokens": int(eval_w.numel()), "calib_tokens": int(cal_w.numel()), "base_val_loss": base})

    stats, n_tok = calibrate(model, blocks, cal_w, args.calib_bs, dev)
    gen = torch.Generator().manual_seed(args.seed)
    items = {"obs_fisher": [], "random": [], "magnitude": [], "weight_sim": []}
    Pinv, Gs = [], []
    for b, s in zip(blocks, stats):
        Pi = ridge_inverse(s["P"].double(), args.ridge)
        R = fisher_factor(s["G"])
        order, marg, _ = obs_eliminate(Pi, b.W[:, 2].double() @ R, f)
        items["obs_fisher"].append(("prefix", order, marg))
        items["random"].append(("score", torch.rand(E, f, generator=gen, dtype=torch.float64).to(dev)))
        mag, wsim = weight_rules(b.W)
        items["magnitude"].append(("score", mag))
        items["weight_sim"].append(("score", wsim))
        Pinv.append(Pi)
        Gs.append(s["G"])
        del R, s["P"]
    counts = [s["count"] for s in stats]
    record({"part": 0, "calibrated_tokens": n_tok, "dead_experts": [int((c == 0).sum()) for c in counts]})

    orig = [b.W.detach().to("cpu", copy=True) for b in blocks]

    def restore():
        for b, w in zip(blocks, orig):
            b.W.copy_(w)

    configs = [("obs_fisher", m) for m in ("global", "layer", "expert")] + \
              [(r, "global") for r in ("random", "magnitude", "weight_sim")]
    for q in [float(x) for x in args.fractions.split(",")]:
        for rule, mode in configs:
            masks = select_allocation(items[rule], q, mode)
            pred = 0.0
            with torch.no_grad():
                for b, D, Pi, G in zip(blocks, masks, Pinv, Gs):
                    for e in torch.nonzero(D.any(1)).flatten().tolist():
                        Bn, c = refit_after_removal(Pi[e], b.W[e, 2].double(), D[e], G)
                        b.W[e, 0][D[e]] = 0
                        b.W[e, 1][D[e]] = 0
                        b.W[e, 2].copy_(Bn.to(b.W.dtype))
                        pred += c
            v = evaluate(model, eval_w, args.bs, dev)
            active = sum(float((c * (f - D.sum(1))).sum()) for c, D in zip(counts, masks)) / \
                sum(float(c.sum()) * f for c in counts)
            record({"part": 1, "rule": rule, "mode": mode, "frac": q, "val_loss": v, "delta": v - base,
                    "pred_delta": pred / (2 * n_tok), "active_frac": round(active, 4),
                    "frac_per_layer": [round(float(D.float().mean()), 4) for D in masks]})
            restore()
    again = evaluate(model, eval_w, args.bs, dev)
    record({"part": 0, "restored_val_loss": again, "restore_diff": again - base})
    with open(os.path.join(args.out, "results.json"), "w") as fo:
        json.dump(results, fo)
    print("REC " + json.dumps({"done": True, "elapsed_s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
