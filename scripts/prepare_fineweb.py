#!/usr/bin/env python
"""Download FineWeb-Edu parquet shards and tokenize them (GPT-2 BPE) into uint16 .bin files.

Output: <out>/val.bin (first --val-docs documents of shard 0), <out>/train_XXX.bin, <out>/meta.json.
Each document is prefixed with <|endoftext|>. Re-running skips finished shards.
"""

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np


def write_docs(fh, docs, eot):
    if not docs:
        return 0
    arr = np.concatenate([np.asarray([eot] + d, dtype=np.uint16) for d in docs])
    fh.write(arr.tobytes())
    return int(arr.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample/10BT")
    ap.add_argument("--num-files", type=int, default=4)
    ap.add_argument("--val-docs", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch-docs", type=int, default=2048)
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(args.out, exist_ok=True)
    prefix = args.subset.rstrip("/") + "/"
    files = sorted(
        f for f in HfApi().list_repo_files(args.repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not files:
        sys.exit(f"no parquet files under {args.repo}/{prefix}")
    files = files[: args.num_files]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    meta_path = os.path.join(args.out, "meta.json")
    meta = {"tokenizer": "gpt2", "repo": args.repo, "files": {}}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    for i, name in enumerate(files):
        dst = os.path.join(args.out, f"train_{i:03d}.bin")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        local = hf_hub_download(args.repo, name, repo_type="dataset",
                                local_dir=os.path.join(args.out, "_parquet"))
        pfile = pq.ParquetFile(local)
        val_left = args.val_docs if i == 0 else 0
        n_train = n_val = 0
        val_tmp = os.path.join(args.out, "val.bin.tmp")
        with open(dst + ".tmp", "wb") as ftr, (
            open(val_tmp, "wb") if i == 0 else contextlib.nullcontext()
        ) as fva:
            for batch in pfile.iter_batches(batch_size=args.batch_docs, columns=["text"]):
                toks = enc.encode_ordinary_batch(batch.column(0).to_pylist(), num_threads=args.threads)
                if val_left > 0:
                    take = min(val_left, len(toks))
                    n_val += write_docs(fva, toks[:take], eot)
                    toks = toks[take:]
                    val_left -= take
                n_train += write_docs(ftr, toks, eot)
        if i == 0:
            os.replace(val_tmp, os.path.join(args.out, "val.bin"))
        os.replace(dst + ".tmp", dst)
        if not args.keep_parquet:
            os.remove(local)
        meta["files"][name] = {"bin": os.path.basename(dst), "train_tokens": n_train, "val_tokens": n_val}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{name}: {n_train / 1e6:.1f}M train tokens, {n_val / 1e6:.2f}M val "
              f"({time.time() - t0:.0f}s)", flush=True)

    total = sum(v["train_tokens"] for v in meta["files"].values())
    print(f"done: {total / 1e9:.2f}B train tokens in {len(meta['files'])} shards", flush=True)


if __name__ == "__main__":
    main()
