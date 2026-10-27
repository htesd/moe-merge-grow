import math

import torch


@torch.no_grad()
def layer_usage(model):
    """Load statistics of the last forward pass, per MoE layer."""
    out = []
    for m in model.moe_layers():
        c = m.last_counts
        if c is None or c.numel() < 2:
            continue
        p = c / c.sum()
        ent = float(-(p * p.clamp_min(1e-12).log()).sum() / math.log(p.numel()))
        out.append(
            {
                "n": p.numel(),
                "entropy": round(ent, 4),
                "max_over_mean": round(float(p.max() * p.numel()), 3),
                "dead": int((c == 0).sum()),
            }
        )
    return out
