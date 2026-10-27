import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmg.moe import expert_forward, expert_forward_grouped, grouped_mm_available  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class GroupedMM(unittest.TestCase):
    def test_grouped_matches_padded_bmm(self):
        dev = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(0)
        T, d, dff, n, k = 3000, 128, 64, 24, 4
        w0 = (torch.randn(n, 3, dff, d, device=dev) * 0.05).to(torch.bfloat16)
        x0 = torch.randn(T, d, device=dev)
        local = torch.stack([torch.randperm(n - 4, device=dev)[:k] for _ in range(T)])  # 4 empty experts
        g0 = torch.rand(T, k, device=dev)
        if not grouped_mm_available(x0, w0):
            self.skipTest("torch._grouped_mm unavailable on this GPU / build")
        outs, grads = [], []
        for fn in (expert_forward, expert_forward_grouped):
            w = w0.clone().requires_grad_()
            x = x0.clone().requires_grad_()
            g = g0.clone().requires_grad_()
            y = fn(x, local, g, w)
            (y.square().sum()).backward()
            outs.append(y.detach())
            grads.append((x.grad, g.grad, w.grad.float()))
        torch.testing.assert_close(outs[1], outs[0], atol=2e-2, rtol=2e-2)
        for a, b in zip(grads[1], grads[0]):
            torch.testing.assert_close(a, b, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
