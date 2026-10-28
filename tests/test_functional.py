import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from functional_redundancy import (  # noqa: E402
    fisher_factor, functional_merge, obs_eliminate, refit_after_removal, ridge_inverse,
)
from mmg.model import ModelConfig, MoETransformer  # noqa: E402


def lstsq_residual(Phi, Y, keep, G):
    beta = torch.linalg.lstsq(Phi[:, keep], Y).solution
    r = Y - Phi[:, keep] @ beta
    return float((r @ G * r).sum()), beta


class LeastSquaresElimination(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.Phi = torch.randn(300, 12, generator=g, dtype=torch.float64)
        self.Phi[:, 11] = self.Phi[:, 0] - 2 * self.Phi[:, 3]  # an exactly redundant feature
        self.Phi[:, 11] += 1e-3 * torch.randn(300, generator=g, dtype=torch.float64)
        self.B = torch.randn(12, 5, generator=g, dtype=torch.float64)
        A = torch.randn(5, 5, generator=g, dtype=torch.float64)
        self.G = A @ A.T + 0.1 * torch.eye(5, dtype=torch.float64)
        self.Y = self.Phi @ self.B  # current fit is exact, as for an expert's own down rows

    def test_greedy_costs_match_bruteforce(self):
        P = self.Phi.T @ self.Phi
        Pinv = ridge_inverse(P, 0.0)
        R = fisher_factor(self.G)
        order, marg, alive = obs_eliminate(Pinv[None], (self.B @ R)[None], 6)
        keep = torch.ones(12, dtype=torch.bool)
        total = 0.0
        for s in range(6):
            keep[order[0, s]] = False
            total += float(marg[0, s])
            ref, _ = lstsq_residual(self.Phi, self.Y, keep, self.G)
            self.assertAlmostEqual(total, ref, delta=1e-6 * max(1.0, ref))
        self.assertTrue(torch.equal(alive[0], keep))
        self.assertIn(11, order[0, :2].tolist())  # the redundant feature goes (almost) first

    def test_refit_matches_lstsq(self):
        P = self.Phi.T @ self.Phi
        Pinv = ridge_inverse(P, 0.0)
        D = torch.zeros(12, dtype=torch.bool)
        D[[1, 4, 7, 9]] = True
        Bn, cost = refit_after_removal(Pinv, self.B, D, self.G)
        ref, beta = lstsq_residual(self.Phi, self.Y, ~D, self.G)
        self.assertAlmostEqual(cost, ref, delta=1e-6 * max(1.0, ref))
        torch.testing.assert_close(Bn[~D], beta, rtol=1e-6, atol=1e-8)
        self.assertEqual(float(Bn[D].abs().sum()), 0.0)


class FunctionalMerge(unittest.TestCase):
    def test_identical_experts_merge_without_loss(self):
        g = torch.Generator().manual_seed(1)
        E, f, d, T = 3, 6, 10, 400
        W = torch.randn(E, 3, f, d, generator=g) * 0.5
        W[1] = W[0][:, torch.randperm(f, generator=g)]  # same expert, hidden units permuted
        x = torch.randn(T, d, generator=g)
        idx = torch.randint(0, E, (T, 2), generator=g)
        idx[: T // 2, 0], idx[: T // 2, 1] = 0, 1
        gate = torch.rand(T, 2, generator=g) * 0.5
        s = {"x": x, "idx": idx, "gate": gate}
        R = fisher_factor(torch.eye(d, dtype=torch.float64))
        merged, pred = functional_merge(W, s, R, 0, 1, 1e-9, T)
        self.assertLess(pred, 1e-8)
        u = (idx == 0).any(1) | (idx == 1).any(1)
        pk = (gate * (idx == 0)).sum(1)[u]
        pr = (gate * (idx == 1)).sum(1)[u]
        X = x[u]

        def out(w):
            return (torch.nn.functional.silu(X @ w[0].T) * (X @ w[1].T)) @ w[2]

        target = pk[:, None] * out(W[0]) + pr[:, None] * out(W[1])
        torch.testing.assert_close((pk + pr)[:, None] * out(merged), target, rtol=1e-4, atol=1e-4)


class ScriptSmoke(unittest.TestCase):
    def test_runs_on_cpu(self):
        cfg = ModelConfig(vocab_size=64, seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=8,
                          experts_per_layer=8, top_k=2)
        torch.manual_seed(0)
        model = MoETransformer(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "final_model.pt")
            torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "step": 0}, path)
            val = os.path.join(tmp, "val.bin")
            np.random.default_rng(0).integers(0, 64, 20000, dtype=np.uint16).tofile(val)
            out = os.path.join(tmp, "out")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "functional_redundancy.py"),
                            "--model", path, "--val", val, "--out", out,
                            "--calib-batches", "6", "--calib-bs", "4", "--store-batches", "6",
                            "--eval-batches", "2", "--eval-bs", "4", "--single-eval-batches", "1",
                            "--fractions", "0.25,0.5", "--pairs", "1,2", "--single-layers", "0,1",
                            "--single-per-rule", "2", "--reroute-samples", "16"],
                           check=True, env=env, capture_output=True)
            with open(os.path.join(out, "results.json")) as f:
                r = json.load(f)
        res = r["results"]
        p1 = [x for x in res if x["part"] == 1]
        p2 = [x for x in res if x["part"] == 2 and "op" in x]
        p3 = [x for x in res if x["part"] == 3 and "layer" in x]
        self.assertEqual(len(p1), 7 * 2 + 2 * 2)
        self.assertGreaterEqual(len(p2), 4 * 2 * 3)
        self.assertGreater(len(p3), 0)
        for x in p1 + p2:
            self.assertTrue(np.isfinite(x["delta"]))
        # with compensation, OBS in the Fisher metric never predicts more than plain deletion
        for q in (0.25, 0.5):
            a = [x for x in p1 if x["rule"] == "obs_fisher" and x["frac"] == q]
            comp = [x["pred_delta"] for x in a if x["compensate"]][0]
            plain = [x["pred_delta"] for x in a if not x["compensate"]][0]
            self.assertLessEqual(comp, plain + 1e-12)
        self.assertIn("spearman_vs_delta", [x for x in res if x["part"] == 3][-1])


if __name__ == "__main__":
    unittest.main()
