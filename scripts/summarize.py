#!/usr/bin/env python
"""Print a markdown table of every <root>/*/summary.json."""

import glob
import json
import os
import sys


def main():
    root = sys.argv[1]
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        st = s.get("structure", {})
        rows.append((
            os.path.basename(os.path.dirname(path)),
            s.get("experts_per_layer"),
            st.get("total_params", 0) / 1e6,
            st.get("active_params_non_embed", 0) / 1e6,
            s.get("peak_reserved_gb"),
            s.get("tokens_done", 0) / 1e9,
            s.get("final_val_loss"),
            s.get("status"),
        ))
    rows.sort(key=lambda r: (r[1] or 0))
    print("| run | experts/layer | total params (M) | active non-emb (M) | peak reserved (GB) "
          "| tokens (B) | val loss | status |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        val = f"{r[6]:.4f}" if isinstance(r[6], float) else "-"
        print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]} | {r[5]:.2f} | {val} | {r[7]} |")


if __name__ == "__main__":
    main()
