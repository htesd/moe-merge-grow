import torch
from torch import nn


class ExpertPool(nn.Module):
    """Fixed pool of physical SwiGLU expert slots shared by every MoE layer.

    Slot s is one Parameter of shape [3, d_ff, d_model]: [0] = W_gate, [1] = W_up, [2] = W_down^T.
    One Parameter per slot is what keeps structural edits cheap:
      * tying = several logical experts read the same Parameter, autograd sums their grads;
      * a free slot is never read, so its grad stays None and AdamW skips it;
      * optimizer state (exp_avg, exp_avg_sq, step) is per slot and can be copied or reset alone;
      * the parameter list never changes, so the optimizer and checkpoints need no re-plumbing.
    """

    def __init__(self, num_slots, d_model, d_ff, init_std=0.02, down_std=None):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.d_ff = d_ff
        self.init_std = init_std
        self.down_std = init_std if down_std is None else down_std
        self.slots = nn.ParameterList(
            [nn.Parameter(torch.empty(3, d_ff, d_model)) for _ in range(num_slots)]
        )
        self.register_buffer("in_use", torch.zeros(num_slots, dtype=torch.bool))
        for s in range(num_slots):
            self.init_slot(s)

    @torch.no_grad()
    def init_slot(self, s, zero_down=False):
        p = self.slots[s]
        p[0].normal_(0.0, self.init_std)
        p[1].normal_(0.0, self.init_std)
        if zero_down:
            p[2].zero_()
        else:
            p[2].normal_(0.0, self.down_std)

    def alloc(self):
        free = (~self.in_use).nonzero().flatten()
        if free.numel() == 0:
            raise RuntimeError("expert pool exhausted; raise pool_slots")
        s = int(free[0])
        self.in_use[s] = True
        return s

    def release(self, s):
        self.in_use[s] = False
        self.slots[s].grad = None

    def num_free(self):
        return int((~self.in_use).sum())

    def params_per_slot(self):
        return 3 * self.d_ff * self.d_model
