import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    save_file = None


def make_checkpoint(path, d=32):
    """layer 0: dense MLP; layers 1-2: shared expert + 4 routed experts. Planted redundancy:
    L1.routed2 = exact copy of L1.routed1 (same index order),
    L2.routed3 = L1.routed0 with permuted hidden units and rescaled / sign-flipped (up, down),
    gate row 0 of L2.routed0 = 2 * gate row 3 of L1.routed1 (up / down unrelated)."""
    g = torch.Generator().manual_seed(0)

    def ffn(f):
        return [torch.randn(f, d, generator=g), torch.randn(f, d, generator=g), torch.randn(d, f, generator=g)]

    t = {}

    def put(prefix, w):
        t[prefix + "gate_proj.weight"], t[prefix + "up_proj.weight"], t[prefix + "down_proj.weight"] = w

    put("model.layers.0.mlp.", ffn(16))
    experts = {}
    for l in (1, 2):
        put(f"model.layers.{l}.mlp.shared_experts.", ffn(8))
        for e in range(4):
            experts[(l, e)] = ffn(8)
    experts[(1, 2)] = [x.clone() for x in experts[(1, 1)]]
    gate, up, down = experts[(1, 0)]
    perm = torch.randperm(8, generator=g)
    c = (torch.rand(8, generator=g) + 0.5) * torch.where(torch.rand(8, generator=g) < 0.5, -1.0, 1.0)
    experts[(2, 3)] = [gate[perm].clone(), up[perm] * c[:, None], down[:, perm] / c[None, :]]
    experts[(2, 0)][0][0] = 2.0 * experts[(1, 1)][0][3]  # gate row only: a single-vector match
    for (l, e), w in experts.items():
        put(f"model.layers.{l}.mlp.experts.{e}.", w)
    t["model.layers.1.mlp.gate.weight"] = torch.randn(4, d, generator=g)  # router, must be ignored
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": "toy", "hidden_size": d, "num_hidden_layers": 3}, f)


@unittest.skipIf(save_file is None, "needs safetensors")
class ExpertSimilarity(unittest.TestCase):
    def test_planted_duplicates_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck, out = os.path.join(tmp, "ck"), os.path.join(tmp, "out")
            os.makedirs(ck)
            make_checkpoint(ck)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "expert_similarity.py"), ck,
                            "--out", out, "--null-queries", "64", "--chunk-elems", "500"],
                           check=True, env=env, capture_output=True)
            with open(os.path.join(out, "result.json")) as f:
                r = json.load(f)
            t = dict(np.load(os.path.join(out, "layer_tables.npz")))
        roles = list(t["roles"])
        top = lambda h: int(h[..., -1].sum())  # best match in [0.99, 1.0]
        # query layer 2 matched in layer 1: 8 permuted neurons in every role, plus one gate row
        self.assertEqual(top(t["lm"][roles.index("triple"), 2, 1]), 8)
        self.assertEqual(top(t["lm"][roles.index("gate"), 2, 1]), 9)
        self.assertEqual(top(t["lm"][roles.index("up"), 2, 1]), 8)
        self.assertEqual(top(t["lm"][roles.index("down"), 2, 1]), 8)
        # the exact copy inside layer 1 shows up as same-layer, other-unit matches (16 neurons)
        self.assertEqual(top(t["lm"][roles.index("triple"), 1, 1]), 16)
        self.assertEqual(t["n_query"].tolist(), [16, 40, 40])
        self.assertEqual(t["n_strong"].tolist(), [16, 40, 40])
        self.assertLess(r["neuron_roles"]["gate"]["null_any"]["0.9"], 0.01)
        self.assertEqual(r["units"], 11)
        self.assertEqual(r["query_units"], 11)
        self.assertEqual(r["unit_kinds"], {"dense": 1, "shared": 2, "routed": 8})
        pairs = {(p["a"], p["b"]): p for p in r["top_pairs"]}
        perm_pair = pairs[("L2.routed3", "L1.routed0")]
        self.assertGreater(perm_pair["soft_align"], 0.999)
        self.assertEqual(perm_pair["coverage"], 1.0)
        self.assertLess(perm_pair["same_index"], 0.9)  # unaligned comparison misses it
        copy_pair = pairs[("L1.routed2", "L1.routed1")]
        self.assertGreater(copy_pair["same_index"], 0.999)
        # random neurons in d=32 rarely match well; planted ones always do
        self.assertLess(r["neuron"]["null_rotated"]["quantiles"]["0.5"], 0.7)
        self.assertGreater(r["neuron"]["other_layers"]["frac_above"]["0.9"], 0.0)


if __name__ == "__main__":
    unittest.main()
