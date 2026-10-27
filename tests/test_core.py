import io
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmg import ops  # noqa: E402
from mmg.model import ModelConfig, MoETransformer, configure_optimizer  # noqa: E402
from mmg.moe import expert_forward  # noqa: E402

VOCAB, SEQ = 64, 16


def tiny(seed=0, **kw):
    cfg = dict(
        vocab_size=VOCAB, seq_len=SEQ, d_model=32, n_layers=2, n_heads=4, d_ff=16,
        experts_per_layer=4, top_k=2, pool_slots=12, max_rows=8,
    )
    cfg.update(kw)
    torch.manual_seed(seed)
    return MoETransformer(ModelConfig(**cfg))


def batch(seed=0, b=3):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, VOCAB, (b, SEQ), generator=g)
    tgt = torch.randint(0, VOCAB, (b, SEQ), generator=g)
    return idx, tgt


def random_unit_transform(dff, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dff, generator=g)
    mag = torch.rand(dff, generator=g) + 0.5
    sign = torch.where(torch.rand(dff, generator=g) < 0.5, -1.0, 1.0)
    return perm, mag * sign


def expert_fn(p, x):
    return expert_forward(x, torch.zeros(x.shape[0], 1, dtype=torch.long),
                          torch.ones(x.shape[0], 1), [p])


def train_step(model, opt, seed=0):
    idx, tgt = batch(seed)
    loss, aux = model(idx, tgt)
    (loss + 0.01 * aux["lb"]).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


class ExpertForward(unittest.TestCase):
    def test_matches_naive_loop(self):
        torch.manual_seed(0)
        T, d, dff, n, k = 50, 8, 6, 5, 2
        x = torch.randn(T, d)
        ws = [torch.randn(3, dff, d) for _ in range(n)]
        local = torch.stack([torch.randperm(n)[:k] for _ in range(T)])
        gate = torch.rand(T, k)
        y = expert_forward(x, local, gate, ws)
        ref = torch.zeros(T, d)
        for t in range(T):
            for j in range(k):
                w = ws[int(local[t, j])]
                h = F.silu(w[0] @ x[t]) * (w[1] @ x[t])
                ref[t] += gate[t, j] * (h @ w[2])
        torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


class MemorySavers(unittest.TestCase):
    def test_chunked_ce_and_activation_checkpointing_match_plain(self):
        m = tiny()
        idx, tgt = batch()
        logits, _ = m(idx)
        ref = F.cross_entropy(logits.float().reshape(-1, VOCAB), tgt.reshape(-1))
        ref.backward()
        ref_grads = [p.grad.clone() if p.grad is not None else None for p in m.parameters()]
        m.zero_grad(set_to_none=True)

        m.ce_chunk = 7  # does not divide 3 * 16 tokens
        m.act_ckpt = True
        loss, _ = m(idx, tgt)
        loss.backward()
        self.assertAlmostEqual(loss.item(), ref.item(), places=5)
        for p, g in zip(m.parameters(), ref_grads):
            if g is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-5)


class Alignment(unittest.TestCase):
    def test_transform_is_function_preserving_and_alignment_recovers_it(self):
        torch.manual_seed(1)
        dff, d = 16, 8
        pa = torch.randn(3, dff, d)
        pb = pa.clone()
        ops.transform_units(pb, None, *random_unit_transform(dff, seed=3))
        x = torch.randn(20, d)
        torch.testing.assert_close(expert_fn(pb, x), expert_fn(pa, x), atol=1e-5, rtol=1e-5)
        self.assertGreater(float((pa - pb).norm() / pa.norm()), 0.5)
        perm, scale = ops.align_units(pa, pb)
        ops.transform_units(pb, None, perm, scale)
        torch.testing.assert_close(pb, pa, atol=1e-5, rtol=1e-5)


class StructuralOps(unittest.TestCase):
    def test_cross_layer_tie_is_exact_and_sums_grads(self):
        m = tiny()
        s0 = int(m.blocks[0].moe.rep_slot[0])
        s1 = int(m.blocks[1].moe.rep_slot[0])
        with torch.no_grad():
            m.pool.slots[s1].copy_(m.pool.slots[s0])
        idx, tgt = batch()
        loss, _ = m(idx, tgt)
        loss.backward()
        expected = m.pool.slots[s0].grad + m.pool.slots[s1].grad
        m.zero_grad(set_to_none=True)

        ops.merge_slots(m, None, keep=s0, drop=s1, align=False)
        loss2, _ = m(idx, tgt)
        loss2.backward()
        self.assertAlmostEqual(loss2.item(), loss.item(), places=5)
        torch.testing.assert_close(m.pool.slots[s0].grad, expected, atol=1e-6, rtol=1e-5)
        self.assertIsNone(m.pool.slots[s1].grad)
        self.assertFalse(bool(m.pool.in_use[s1]))
        self.assertEqual(m.slot_users(s0), [(0, 0), (1, 0)])

    def test_same_layer_merge_keeps_mixture_under_dense_routing(self):
        m = tiny(top_k=8)  # top_k >= experts: every expert is used, so the mixture is exact
        layer = m.blocks[0].moe
        sa, sb = int(layer.rep_slot[0]), int(layer.rep_slot[1])
        with torch.no_grad():
            m.pool.slots[sb].copy_(m.pool.slots[sa])
            ops.transform_units(m.pool.slots[sb], None, *random_unit_transform(16, seed=5))
        idx, _ = batch()
        out0, _ = m(idx)
        ops.merge_slots(m, None, keep=sa, drop=sb, align=True)
        out1, _ = m(idx)
        torch.testing.assert_close(out1, out0, atol=1e-5, rtol=1e-5)
        self.assertEqual(len(layer._s["reps"]), 3)
        self.assertEqual(layer.groups()[0], [0, 1])

    def test_split_and_unmerge_keep_mixture_under_dense_routing(self):
        m = tiny(top_k=8)
        layer = m.blocks[0].moe
        idx, _ = batch()
        out0, _ = m(idx)
        t = ops.split_expert(m, None, 0, rep=2)
        out1, _ = m(idx)
        torch.testing.assert_close(out1, out0, atol=1e-5, rtol=1e-5)
        self.assertEqual(len(layer._s["reps"]), 5)
        self.assertTrue(bool(m.pool.in_use[t]))

        # merge a copy back, then split the merged group (copy-on-write un-merge)
        sa, sb = int(layer.rep_slot[0]), int(layer.rep_slot[1])
        with torch.no_grad():
            m.pool.slots[sb].copy_(m.pool.slots[sa])
        out2, _ = m(idx)
        ops.merge_slots(m, None, keep=sa, drop=sb, align=False)
        ops.split_expert(m, None, 0, rep=0)
        out3, _ = m(idx)
        torch.testing.assert_close(out3, out2, atol=1e-5, rtol=1e-5)
        self.assertEqual(layer.groups()[0], [0])
        self.assertEqual(layer.groups()[1], [1])

    def test_untie_restores_private_slots(self):
        m = tiny()
        s0 = int(m.blocks[0].moe.rep_slot[0])
        s1 = int(m.blocks[1].moe.rep_slot[0])
        with torch.no_grad():
            m.pool.slots[s1].copy_(m.pool.slots[s0])
        idx, _ = batch()
        out0, _ = m(idx)
        ops.merge_slots(m, None, keep=s0, drop=s1, align=False)
        t = ops.untie_expert(m, None, 1, rep=0)
        out1, _ = m(idx)
        torch.testing.assert_close(out1, out0, atol=1e-5, rtol=1e-5)
        self.assertEqual(m.slot_users(s0), [(0, 0)])
        self.assertEqual(m.slot_users(t), [(1, 0)])

    def test_drop_expert_frees_unreferenced_slot(self):
        m = tiny()
        s = int(m.blocks[0].moe.rep_slot[3])
        ops.drop_expert(m, None, 0, rep=3)
        self.assertFalse(bool(m.pool.in_use[s]))
        self.assertEqual(len(m.blocks[0].moe._s["reps"]), 3)
        m(batch()[0])

    def test_soft_pull_shrinks_distance(self):
        m = tiny()
        a, b = int(m.blocks[0].moe.rep_slot[0]), int(m.blocks[1].moe.rep_slot[0])
        d0 = float((m.pool.slots[a] - m.pool.slots[b]).detach().norm())
        ops.soft_pull(m, [(a, b)], strength=0.25)
        d1 = float((m.pool.slots[a] - m.pool.slots[b]).detach().norm())
        self.assertAlmostEqual(d1, 0.75 * d0, places=4)


class OptimizerState(unittest.TestCase):
    def test_merge_free_and_reuse(self):
        m = tiny()
        opt = configure_optimizer(m, lr=1e-2, weight_decay=0.1)
        train_step(m, opt)
        sa = int(m.blocks[0].moe.rep_slot[0])
        sb = int(m.blocks[1].moe.rep_slot[0])
        pa, pb = m.pool.slots[sa], m.pool.slots[sb]
        ma, mb = opt.state[pa]["exp_avg"].clone(), opt.state[pb]["exp_avg"].clone()
        va, vb = opt.state[pa]["exp_avg_sq"].clone(), opt.state[pb]["exp_avg_sq"].clone()

        ops.merge_slots(m, opt, keep=sa, drop=sb, align=False)
        torch.testing.assert_close(opt.state[pa]["exp_avg"], ma + mb)
        torch.testing.assert_close(opt.state[pa]["exp_avg_sq"], (va.sqrt() + vb.sqrt()) ** 2)
        self.assertNotIn(pb, opt.state)

        frozen = pb.detach().clone()
        train_step(m, opt, seed=1)
        torch.testing.assert_close(pb.detach(), frozen, atol=0, rtol=0)  # free slot untouched
        self.assertNotIn(pb, opt.state)

        t = ops.add_fresh_expert(m, opt, 0, donor_rep=1)
        self.assertEqual(t, sb)  # the freed slot is reused
        train_step(m, opt, seed=2)
        self.assertEqual(float(opt.state[m.pool.slots[t]]["step"]), 1.0)  # fresh bias correction

    def test_alignment_transforms_adam_state(self):
        m = tiny()
        opt = configure_optimizer(m, lr=1e-2, weight_decay=0.0)
        train_step(m, opt)
        s = int(m.blocks[0].moe.rep_slot[0])
        p = m.pool.slots[s]
        st = opt.state[p]
        m0, v0 = st["exp_avg"].clone(), st["exp_avg_sq"].clone()
        perm, scale = random_unit_transform(16, seed=7)
        ops.transform_units(p, st, perm, scale)
        sc = scale[:, None]
        torch.testing.assert_close(st["exp_avg"][1], m0[1][perm] / sc)
        torch.testing.assert_close(st["exp_avg"][2], m0[2][perm] * sc)
        torch.testing.assert_close(st["exp_avg_sq"][2], v0[2][perm] * sc * sc)


class Checkpoint(unittest.TestCase):
    def test_state_dict_roundtrip_with_edited_structure(self):
        m = tiny(top_k=8)
        opt = configure_optimizer(m, lr=1e-2, weight_decay=0.1)
        train_step(m, opt)
        layer = m.blocks[0].moe
        ops.merge_slots(m, opt, keep=int(layer.rep_slot[0]), drop=int(layer.rep_slot[1]))
        ops.split_expert(m, opt, 1, rep=2, noise=0.01)
        idx, _ = batch()
        out0, _ = m(idx)

        buf = io.BytesIO()  # serialise like a real checkpoint; in-process state_dicts share tensors
        torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, buf)
        buf.seek(0)
        ck = torch.load(buf, weights_only=False)
        m2 = tiny(seed=123, top_k=8)
        m2.load_state_dict(ck["model"])
        opt2 = configure_optimizer(m2, lr=1e-2, weight_decay=0.1)
        opt2.load_state_dict(ck["opt"])
        out1, _ = m2(idx)
        torch.testing.assert_close(out1, out0, atol=1e-6, rtol=1e-6)
        self.assertEqual(m2.structure_stats(), m.structure_stats())
        train_step(m, opt, seed=3)
        train_step(m2, opt2, seed=3)
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
