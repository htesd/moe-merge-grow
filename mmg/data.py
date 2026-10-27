import glob
import os

import numpy as np
import torch


class TokenData:
    """uint16 token shards: train_*.bin and val.bin (see scripts/prepare_fineweb.py)."""

    def __init__(self, data_dir, seq_len, device, seed=0):
        paths = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
        if not paths:
            raise FileNotFoundError(f"no train_*.bin under {data_dir}")
        self.train = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        lens = np.array([len(m) for m in self.train], dtype=np.float64)
        self.probs = lens / lens.sum()
        self.train_tokens = int(lens.sum())
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

    def _to_device(self, buf):
        t = torch.from_numpy(buf)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def train_batch(self, batch_size):
        L = self.seq_len
        files = self.rng.choice(len(self.train), size=batch_size, p=self.probs)
        buf = np.empty((batch_size, L + 1), dtype=np.int64)
        for i, f in enumerate(files):
            m = self.train[f]
            o = int(self.rng.integers(0, len(m) - L - 1))
            buf[i] = m[o : o + L + 1]
        return self._to_device(buf)

    def val_batches(self, batch_size, n_batches):
        """Fixed, non-overlapping windows from the start of val.bin."""
        span = batch_size * (self.seq_len + 1)
        n_batches = min(n_batches, len(self.val) // span)
        arr = np.asarray(self.val[: n_batches * span], dtype=np.int64)
        arr = arr.reshape(n_batches, batch_size, self.seq_len + 1)
        return [self._to_device(a.copy()) for a in arr]

    def state(self):
        return self.rng.bit_generator.state

    def load_state(self, state):
        self.rng.bit_generator.state = state


class SyntheticData:
    """Uniform random tokens, for smoke and memory tests only."""

    def __init__(self, vocab_size, seq_len, device, seed=0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.gen = torch.Generator().manual_seed(seed)
        self.train_tokens = 0

    def _batch(self, batch_size, gen):
        t = torch.randint(0, self.vocab_size, (batch_size, self.seq_len + 1), generator=gen)
        t = t.to(self.device)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def train_batch(self, batch_size):
        return self._batch(batch_size, self.gen)

    def val_batches(self, batch_size, n_batches):
        gen = torch.Generator().manual_seed(1234)
        return [self._batch(batch_size, gen) for _ in range(n_batches)]

    def state(self):
        return None

    def load_state(self, state):
        pass
