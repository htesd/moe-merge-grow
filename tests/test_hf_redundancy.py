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
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
except ImportError:  # pragma: no cover
    Qwen3MoeConfig = None


@unittest.skipIf(Qwen3MoeConfig is None, "needs transformers with Qwen3-MoE")
class HFRedundancy(unittest.TestCase):
    def test_tiny_qwen3_moe(self):
        cfg = Qwen3MoeConfig(vocab_size=128, hidden_size=32, intermediate_size=64, moe_intermediate_size=8,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                             num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, max_position_embeddings=64,
                             decoder_sparse_step=1, mlp_only_layers=[])
        torch.manual_seed(0)
        model = Qwen3MoeForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            mdir = os.path.join(tmp, "m")
            model.save_pretrained(mdir)
            ids = np.random.default_rng(0).integers(0, 128, 40 * 32)
            np.save(os.path.join(tmp, "ids.npy"), ids)
            out = os.path.join(tmp, "out")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "hf_redundancy.py"), mdir, "--out", out,
                                "--token-file", os.path.join(tmp, "ids.npy"), "--seq-len", "32",
                                "--eval-windows", "8", "--calib-windows", "16", "--bs", "4", "--calib-bs", "2",
                                "--fractions", "0.25,0.5"], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            with open(os.path.join(out, "results.json")) as f:
                res = json.load(f)
        head = res[0]
        self.assertEqual((head["moe_layers"], head["experts"], head["d_ff"], head["d_model"]), (2, 8, 8, 32))
        self.assertLess(head["equivalence_rel_err"], 1e-4)
        p1 = [r for r in res if r["part"] == 1]
        self.assertEqual(len(p1), 2 * 6)
        for r in p1:
            self.assertTrue(np.isfinite(r["delta"]))
        exp = [r for r in p1 if r["mode"] == "expert"]
        self.assertTrue(all(r["frac_per_layer"] == [r["frac"]] * 2 for r in exp))
        restored = [r for r in res if "restore_diff" in r][0]
        self.assertLess(abs(restored["restore_diff"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
