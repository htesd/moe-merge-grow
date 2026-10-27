import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    save_file = None


def make_checkpoint(path, d=32):
    """layer 0: dense MLP; layers 1-2: shared expert + 4 routed experts. Planted redundancy:
    L1.routed2 = exact copy of L1.routed1 (same index order),
    L2.routed3 = L1.routed0 with permuted hidden units and rescaled / sign-flipped (up, down)."""
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
