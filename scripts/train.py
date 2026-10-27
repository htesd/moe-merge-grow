#!/usr/bin/env python
"""Train a slot-pool MoE LM under an optional GPU memory budget. Resumes from out_dir/ckpt.pt."""

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmg.data import SyntheticData, TokenData  # noqa: E402
from mmg.model import ModelConfig, MoETransformer, configure_optimizer  # noqa: E402
from mmg.probe import layer_usage  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    g = ap.add_argument_group("model")
    g.add_argument("--vocab-size", type=int, default=50304)
    g.add_argument("--seq-len", type=int, default=1024)
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--n-layers", type=int, default=8)
    g.add_argument("--n-heads", type=int, default=8)
    g.add_argument("--d-ff", type=int, default=256)
    g.add_argument("--experts-per-layer", type=int, default=64)
    g.add_argument("--top-k", type=int, default=8)
    g.add_argument("--pool-slots", type=int, default=0)
    g.add_argument("--max-rows", type=int, default=0)
    g = ap.add_argument_group("optimisation")
    g.add_argument("--micro-batch", type=int, default=8)
    g.add_argument("--grad-accum", type=int, default=8)
    g.add_argument("--total-tokens", type=float, default=2e9)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--min-lr-ratio", type=float, default=0.1)
    g.add_argument("--warmup-steps", type=int, default=300)
    g.add_argument("--decay-frac", type=float, default=0.2, help="WSD: final fraction of steps that decays")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--lb-coef", type=float, default=0.01)
    g.add_argument("--z-coef", type=float, default=0.001)
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--act-ckpt", action="store_true", help="recompute blocks in backward")
    g.add_argument("--ce-chunk", type=int, default=2048, help="tokens per cross-entropy chunk")
    g = ap.add_argument_group("run")
    g.add_argument("--data-dir", default="")
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--mem-budget-gb", type=float, default=0.0, help="cap via set_per_process_memory_fraction")
    g.add_argument("--max-steps", type=int, default=0, help="stop early (smoke / memory probe)")
    g.add_argument("--log-interval", type=int, default=20)
    g.add_argument("--eval-interval", type=int, default=500)
    g.add_argument("--eval-batches", type=int, default=40)
    g.add_argument("--final-eval-batches", type=int, default=200)
    g.add_argument("--ckpt-interval", type=int, default=1000)
    g.add_argument("--no-final-ckpt", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def lr_at(step, total, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    decay_start = int(total * (1 - args.decay_frac))
    if step < decay_start:
        return args.lr
    frac = (step - decay_start) / max(1, total - decay_start)
    return args.lr * (1 - (1 - args.min_lr_ratio) * frac)


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    total = 0.0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, y)
        total += loss.item()
    model.train()
    return total / len(batches)


def save_ckpt(path, model, opt, step, data, args):
    tmp = path + ".tmp"
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
         "data_state": data.state(), "args": vars(args)},
        tmp,
    )
    os.replace(tmp, path)


def gb(x):
    return round(x / 2**30, 3)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    done_path = os.path.join(args.out_dir, "summary.json")
    if os.path.exists(done_path) and not args.max_steps:
        with open(done_path) as f:
            prev = json.load(f)
        if prev.get("status") == "ok" and prev.get("steps_done", 0) >= prev.get("total_steps", 1):
            print(f"{args.out_dir} already finished, skipping", flush=True)
            return
    torch.manual_seed(args.seed)
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    total_mem = torch.cuda.get_device_properties(dev).total_memory
    if args.mem_budget_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_budget_gb * 2**30 / total_mem), dev)

    cfg = ModelConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, experts_per_layer=args.experts_per_layer, top_k=args.top_k,
        pool_slots=args.pool_slots, max_rows=args.max_rows,
    )
    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    total_steps = int(args.total_tokens) // tokens_per_step
    run_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps

    status, err = "ok", None
    summary = {"out_dir": args.out_dir, "experts_per_layer": args.experts_per_layer,
               "mem_budget_gb": args.mem_budget_gb, "tokens_per_step": tokens_per_step,
               "total_steps": total_steps}
    try:
        model = MoETransformer(cfg).to(dev)
        model.compute_dtype = torch.bfloat16
        model.act_ckpt = args.act_ckpt
        model.ce_chunk = args.ce_chunk
        opt = configure_optimizer(model, args.lr, args.weight_decay)
        if args.synthetic:
            data = SyntheticData(args.vocab_size, args.seq_len, dev, seed=args.seed)
        else:
            data = TokenData(args.data_dir, args.seq_len, dev, seed=args.seed)

        step = 0
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            data.load_state(ck["data_state"])
            step = ck["step"]
            del ck
            print(f"resumed from step {step}", flush=True)

        stats = model.structure_stats()
        summary["structure"] = stats
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({"args": vars(args), "model": dataclasses.asdict(cfg), "structure": stats,
                       "train_tokens_available": data.train_tokens}, f, indent=2)
        print(json.dumps({"structure": stats, "tokens_per_step": tokens_per_step,
                          "total_steps": total_steps}), flush=True)

        val_small = data.val_batches(args.micro_batch, args.eval_batches)
        log = open(os.path.join(args.out_dir, "log.jsonl"), "a")
        model.train()
        t0, steps_since = time.time(), 0
        t_start = time.time()
        while step < run_steps:
            lr = lr_at(step, total_steps, args)
            for grp in opt.param_groups:
                grp["lr"] = lr
            loss_acc = lb_acc = z_acc = 0.0
            for _ in range(args.grad_accum):
                x, y = data.train_batch(args.micro_batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, aux = model(x, y)
                    total = loss + args.lb_coef * aux["lb"] + args.z_coef * aux["z"]
                (total / args.grad_accum).backward()
                loss_acc += loss.detach()
                lb_acc += aux["lb"].detach()
                z_acc += aux["z"].detach()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            steps_since += 1

            if step % args.log_interval == 0 or step == run_steps:
                torch.cuda.synchronize()
                dt = time.time() - t0
                rec = {
                    "step": step, "tokens": step * tokens_per_step, "lr": lr,
                    "loss": float(loss_acc) / args.grad_accum, "lb": float(lb_acc) / args.grad_accum,
                    "z": float(z_acc) / args.grad_accum, "grad_norm": float(gnorm),
                    "tok_per_s": round(steps_since * tokens_per_step / dt),
                    "mem_gb": gb(torch.cuda.memory_allocated(dev)),
                    "peak_gb": gb(torch.cuda.max_memory_allocated(dev)),
                    "reserved_gb": gb(torch.cuda.max_memory_reserved(dev)),
                }
                if not math.isfinite(rec["loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                t0, steps_since = time.time(), 0
            if args.eval_interval and step % args.eval_interval == 0 and step < run_steps:
                rec = {"step": step, "tokens": step * tokens_per_step,
                       "val_loss": evaluate(model, val_small), "usage": layer_usage(model)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if args.ckpt_interval and step % args.ckpt_interval == 0 and step < run_steps:
                save_ckpt(ckpt_path, model, opt, step, data, args)

        summary["train_seconds_this_run"] = round(time.time() - t_start)
        del val_small
        val_full = data.val_batches(args.micro_batch, args.final_eval_batches)
        summary["final_val_loss"] = evaluate(model, val_full)
        summary["final_val_tokens"] = len(val_full) * args.micro_batch * args.seq_len
        summary["final_usage"] = layer_usage(model)
        summary["steps_done"] = step
        summary["tokens_done"] = step * tokens_per_step
        if step >= total_steps:
            # finished: keep weights + structure only, drop the (3x larger) resumable checkpoint
            if not args.no_final_ckpt:
                final_path = os.path.join(args.out_dir, "final_model.pt")
                params = {n for n, _ in model.named_parameters()}
                sd = {k: (v.to(torch.bfloat16) if k in params else v)
                      for k, v in model.state_dict().items()}  # bf16 weights, exact buffers
                torch.save({"model": sd, "config": dataclasses.asdict(cfg), "step": step},
                           final_path + ".tmp")
                os.replace(final_path + ".tmp", final_path)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        elif not args.no_final_ckpt:
            save_ckpt(ckpt_path, model, opt, step, data, args)
    except torch.cuda.OutOfMemoryError as e:
        status, err = "oom", str(e).splitlines()[0]
    summary["status"] = status
    summary["error"] = err
    summary["peak_gb"] = gb(torch.cuda.max_memory_allocated(dev))
    summary["peak_reserved_gb"] = gb(torch.cuda.max_memory_reserved(dev))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"summary": summary}), flush=True)
    if status != "ok":
        sys.exit(3)


if __name__ == "__main__":
    main()
