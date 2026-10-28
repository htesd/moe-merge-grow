#!/usr/bin/env python
"""Text samples, next-token predictions and validation loss of trained models (final_model.pt),
optionally after OBS neuron pruning (see functional_redundancy.py). GPT-2 BPE via tiktoken.

Variant spec: <name>=<final_model.pt>[@<prune fraction>], e.g. e256=/x/final_model.pt@0.5
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functional_redundancy import autocast, evaluate, load_model, obs_prune, val_batches  # noqa: E402

PROMPTS = [
    "The water cycle describes",
    "Photosynthesis is the process by which plants",
    "The French Revolution began in 1789 when",
    "To find the area of a circle, you",
    "Vaccines work by",
    "The Great Wall of China",
    "In computer science, an algorithm is",
    "Dinosaurs became extinct about 66 million years ago, most likely because",
]
NEXT_TOKEN = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water freezes at 0 degrees",
    "The chemical symbol for gold is",
    "Albert Einstein is famous for his theory of",
    "The first President of the United States was George",
]


@torch.no_grad()
def generate(model, ids, n_new, temperature, top_k, gen, n_samples):
    dev = model.embed.weight.device
    x = torch.tensor([ids] * n_samples, device=dev)
    for _ in range(n_new):
        with autocast(dev):
            logits, _ = model(x[:, -model.cfg.seq_len :])
        logits = logits[:, -1].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        x = torch.cat([x, nxt], 1)
    return x[:, len(ids) :].tolist()


@torch.no_grad()
def next_token(model, ids, k=5):
    dev = model.embed.weight.device
    with autocast(dev):
        logits, _ = model(torch.tensor([ids], device=dev))
    p = logits[0, -1].float().softmax(-1)
    v, i = p.topk(k)
    return list(zip(i.tolist(), v.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", help="name=path[@prune_frac]")
    ap.add_argument("--val", required=True)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--eval-bs", type=int, default=32)
    ap.add_argument("--calib-batches", type=int, default=128)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tokens = np.memmap(args.val, dtype=np.uint16, mode="r")
    for spec in args.variants:
        name, rest = spec.split("=", 1)
        path, _, frac = rest.partition("@")
        model = load_model(path, dev)
        seq = model.cfg.seq_len
        rec = {"variant": name, "model": path}
        if frac:
            span = args.calib_batches * args.calib_bs * (seq + 1)
            calib, _ = val_batches(tokens, len(tokens) - span, args.calib_batches, args.calib_bs, seq, dev)
            rec["pruned_frac"] = float(frac)
            rec["pruned_per_layer"] = obs_prune(model, calib, float(frac))
            del calib
        eval_b, _ = val_batches(tokens, 0, args.eval_batches, args.eval_bs, seq, dev)
        rec["val_loss"] = evaluate(model, eval_b)
        rec["perplexity"] = math.exp(rec["val_loss"])
        del eval_b
        print("INFO " + json.dumps(rec), flush=True)
        for p in NEXT_TOKEN:
            top = next_token(model, enc.encode(p))
            print("NEXT " + json.dumps({"variant": name, "prompt": p,
                                        "top": [[enc.decode([i]), round(v, 4)] for i, v in top]}), flush=True)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        for p in PROMPTS:
            ids = enc.encode(p)
            greedy = generate(model, ids, args.max_new, 0.0, 0, gen, 1)
            samples = generate(model, ids, args.max_new, args.temperature, args.top_k, gen, args.samples)
            for kind, outs in (("greedy", greedy), ("sample", samples)):
                for o in outs:
                    if enc.eot_token in o:
                        o = o[: o.index(enc.eot_token)]
                    print("GEN " + json.dumps({"variant": name, "prompt": p, "kind": kind,
                                               "text": enc.decode(o)}), flush=True)
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None


if __name__ == "__main__":
    main()
