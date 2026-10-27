#!/usr/bin/env python
"""Time forward / backward / optimizer of one micro-step and print the top CUDA ops."""

import argparse
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmg.data import SyntheticData  # noqa: E402
from mmg.model import ModelConfig, MoETransformer, configure_optimizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts-per-layer", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--rows", type=int, default=25)
    args = ap.parse_args()

    dev = torch.device("cuda", torch.cuda.current_device())
    cfg = ModelConfig(d_model=args.d_model, n_layers=args.n_layers, d_ff=args.d_ff,
                      experts_per_layer=args.experts_per_layer, top_k=args.top_k)
    model = MoETransformer(cfg).to(dev)
    model.compute_dtype = torch.bfloat16
    opt = configure_optimizer(model, 1e-3, 0.1)
    data = SyntheticData(cfg.vocab_size, cfg.seq_len, dev)

    def step(timed=False):
        x, y = data.train_batch(args.micro_batch)
        t = [time.time()]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, aux = model(x, y)
            total = loss + 0.01 * aux["lb"]
        torch.cuda.synchronize()
        t.append(time.time())
        total.backward()
        torch.cuda.synchronize()
        t.append(time.time())
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t.append(time.time())
        if timed:
            fwd, bwd, o = (t[1] - t[0]) * 1e3, (t[2] - t[1]) * 1e3, (t[3] - t[2]) * 1e3
            tok = args.micro_batch * cfg.seq_len
            print(f"fwd {fwd:.1f} ms  bwd {bwd:.1f} ms  opt {o:.1f} ms  "
                  f"-> {tok / (t[3] - t[0]):.0f} tok/s per micro-step", flush=True)

    for _ in range(3):
        step()
    torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(3):
        step(timed=True)
    print(f"peak allocated {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.rows), flush=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12), flush=True)


if __name__ == "__main__":
    main()
