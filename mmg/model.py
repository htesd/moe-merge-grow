import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .moe import MoELayer
from .pool import ExpertPool


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 256  # hidden size of one expert
    experts_per_layer: int = 64  # initial logical experts per layer
    top_k: int = 8
    pool_slots: int = 0  # physical slots; 0 -> n_layers * experts_per_layer
    max_rows: int = 0  # router rows per layer; 0 -> 2 * experts_per_layer
    init_std: float = 0.02
    rope_theta: float = 10000.0
    qk_norm: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def rope_tables(seq_len, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv)
    return freqs.cos(), freqs.sin()


def _ce_sum(h, weight, targets):
    return F.cross_entropy(F.linear(h, weight).float(), targets, reduction="sum")


def chunked_cross_entropy(h, weight, targets, chunk):
    """Mean CE without materialising all [tokens, vocab] logits: each chunk is recomputed
    in backward, so peak logits memory is one chunk."""
    h = h.reshape(-1, h.size(-1))
    targets = targets.reshape(-1)
    total = 0.0
    for i in range(0, h.size(0), chunk):
        args = (h[i : i + chunk], weight, targets[i : i + chunk])
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_sum, *args, use_reentrant=False)
        else:
            total = total + _ce_sum(*args)
    return total / h.size(0)


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, cfg, max_rows):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoELayer(cfg.d_model, max_rows, cfg.top_k, router_std=cfg.init_std)


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        n_slots = cfg.pool_slots or cfg.n_layers * cfg.experts_per_layer
        max_rows = cfg.max_rows or 2 * cfg.experts_per_layer
        if n_slots < cfg.n_layers * cfg.experts_per_layer:
            raise ValueError("pool_slots must cover n_layers * experts_per_layer")
        if max_rows < cfg.experts_per_layer:
            raise ValueError("max_rows must be >= experts_per_layer")
        out_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, max_rows) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.pool = ExpertPool(n_slots, cfg.d_model, cfg.d_ff, cfg.init_std, down_std=out_std)
        cos, sin = rope_tables(cfg.seq_len, cfg.d_model // cfg.n_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.compute_dtype = None  # e.g. torch.bfloat16 on GPU; None = parameter dtype
        self.act_ckpt = False  # recompute each block in backward
        self.ce_chunk = 2048  # tokens per cross-entropy chunk

        nn.init.normal_(self.embed.weight, std=cfg.init_std)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.qkv.weight, std=cfg.init_std)
            nn.init.normal_(blk.attn.proj.weight, std=out_std)

        with torch.no_grad():
            for blk in self.blocks:
                m = blk.moe
                for r in range(cfg.experts_per_layer):
                    m.row_active[r] = True
                    m.rep_slot[r] = self.pool.alloc()
        self.rebuild()

    def moe_layers(self):
        return [blk.moe for blk in self.blocks]

    def rebuild(self):
        for m in self.moe_layers():
            m.rebuild()

    def load_state_dict(self, *args, **kwargs):
        out = super().load_state_dict(*args, **kwargs)
        self.rebuild()
        return out

    def slot_users(self, s):
        """[(layer index, representative row)] of logical experts served by slot s."""
        out = []
        for li, m in enumerate(self.moe_layers()):
            for r, slot in zip(m._s["reps"], m._s["slots"]):
                if slot == s:
                    out.append((li, r))
        return out

    @torch.no_grad()
    def redirect_slot(self, old, new):
        """Point every logical expert served by `old` at `new`. In a layer that already uses
        `new`, the two logical experts become one group (router rows are kept)."""
        for m in self.moe_layers():
            groups = m.groups()
            reps_old = [r for r in groups if int(m.rep_slot[r]) == old]
            if not reps_old:
                continue
            ro = reps_old[0]
            reps_new = [r for r in groups if int(m.rep_slot[r]) == new]
            if reps_new:
                rn = reps_new[0]
                for x in groups[ro]:
                    m.row_group[x] = rn
                m.rep_slot[ro] = -1
            else:
                m.rep_slot[ro] = new

    def _block(self, blk, x, cos, sin):
        x = x + blk.attn(blk.norm1(x), cos, sin)
        y, lb, z = blk.moe(blk.norm2(x), self.pool, self.compute_dtype)
        return x + y, lb, z

    def forward(self, idx, targets=None):
        B, L = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        lb = z = 0.0
        for blk in self.blocks:
            if self.act_ckpt and torch.is_grad_enabled():
                x, l, zz = checkpoint(self._block, blk, x, cos, sin, use_reentrant=False)
            else:
                x, l, zz = self._block(blk, x, cos, sin)
            lb = lb + l
            z = z + zz
        h = self.norm_f(x)
        aux = {"lb": lb / len(self.blocks), "z": z / len(self.blocks)}
        if targets is None:
            return F.linear(h, self.embed.weight), aux
        return chunked_cross_entropy(h, self.embed.weight, targets, self.ce_chunk), aux

    def structure_stats(self):
        cfg = self.cfg
        per_slot = self.pool.params_per_slot()
        physical = int(self.pool.in_use.sum())
        logical = [len(m._s["reps"]) for m in self.moe_layers()]
        non_pool = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("pool."))
        attn = sum(p.numel() for blk in self.blocks for p in blk.attn.parameters())
        routers = sum(m._s["n_rows"] * cfg.d_model for m in self.moe_layers())
        active = attn + sum(min(cfg.top_k, n) for n in logical) * per_slot + routers
        return {
            "physical_experts": physical,
            "logical_experts": sum(logical),
            "logical_per_layer": logical,
            "pool_slots": self.pool.num_slots,
            "free_slots": self.pool.num_free(),
            "params_per_expert": per_slot,
            "total_params": non_pool + physical * per_slot,
            "active_params_non_embed": active,
            "embed_params": self.embed.weight.numel(),
        }


def configure_optimizer(model, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, foreach=True)
