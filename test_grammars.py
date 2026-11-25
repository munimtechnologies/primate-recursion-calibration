# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
import itertools
import math
import unittest

import numpy as np

from grammars import (HMM, KNGram, MLUnigram, PCFG, batch_ll, compress, fit_em, hmm_em_batch,
                      hmm_loglik_batch, hmm_sample, kn_sample, pcfg_em_batch, pcfg_from_hmm,
                      pcfg_loglik_batch, pcfg_sample)


def brute_hmm(h, s):
    """Sum over every hidden path."""
    K = h.K
    tot = 0.0
    for path in itertools.product(range(K), repeat=len(s)):
        p = h.pi[path[0]]
        for t, k in enumerate(path):
            p *= h.B[k, s[t]]
            p *= h.T[k, path[t + 1]] if t + 1 < len(s) else h.T[k, K]
        tot += p
    return tot


def brute_inside(g, s, a):
    """P(a =>* s) by recursion over the first split (a second, independent implementation)."""
    if len(s) == 1:
        return g.E[a, s[0]]
    tot = 0.0
    for k in range(1, len(s)):
        for b in range(g.N):
            for c in range(g.N):
                r = g.R[a, b, c]
                if r:
                    tot += r * brute_inside(g, s[:k], b) * brute_inside(g, s[k:], c)
    return tot


SEQS = [(0,), (1, 0), (0,), (1, 0), (2,), (0, 2, 1), (2, 2, 0, 1), (0,), (1, 1, 2)]


class TestUnigram(unittest.TestCase):
    def test_hand(self):
        m = MLUnigram(2).fit([(0,), (0, 1)])
        # counts: 0 -> 2, 1 -> 1, end -> 2; total 5
        np.testing.assert_allclose(m.p, [0.4, 0.2, 0.4])
        self.assertAlmostEqual(m.loglik((0, 1)), math.log(0.4) + math.log(0.2) + math.log(0.4))
        m2 = MLUnigram(3).fit([(0,), (0, 1)])
        self.assertEqual(m2.loglik((2,)), -math.inf)


class TestKN(unittest.TestCase):
    def test_hand_unigram(self):
        m = KNGram(1, 2).fit([(0,), (0, 1)])
        # counts {0: 2, 1: 1, EOS: 2}; n1 = 1, n2 = 2 -> Y = 0.2, D1 = 0.2, D2 = 1.95 (clipped)
        self.assertAlmostEqual(m.D[1][0], 0.2, places=12)
        self.assertAlmostEqual(m.D[1][1], 1.95, places=12)
        gamma = (0.2 * 1 + 1.95 * 2) / 5.0
        self.assertAlmostEqual(m.prob((), 1), 0.8 / 5 + gamma / 3, places=12)
        self.assertAlmostEqual(m.prob((), 0), 0.05 / 5 + gamma / 3, places=12)
        self.assertAlmostEqual(m.prob((), 2), 0.05 / 5 + gamma / 3, places=12)

    def test_sums_to_one(self):
        rng = np.random.default_rng(1)
        V = 3
        seqs = [tuple(int(x) for x in rng.integers(0, V, size=int(rng.integers(1, 7))))
                for _ in range(60)]
        for order in (1, 2, 3, 4):
            m = KNGram(order, V).fit(seqs)
            toks = [-1] + list(range(V))
            for ctx in itertools.product(toks, repeat=order - 1):
                tot = sum(m.prob(ctx, w) for w in range(V + 1))
                self.assertAlmostEqual(tot, 1.0, places=10)

    def test_loglik_is_sum_of_events(self):
        m = KNGram(2, 2).fit([(0, 1), (1, 1, 0)])
        s = (0, 1, 1)
        want = (math.log(m.prob((-1,), 0)) + math.log(m.prob((0,), 1))
                + math.log(m.prob((1,), 1)) + math.log(m.prob((1,), 2)))
        self.assertAlmostEqual(m.loglik(s), want, places=12)

    def test_sample(self):
        m = KNGram(3, 3).fit(SEQS)
        rng = np.random.default_rng(2)
        for _ in range(50):
            s = kn_sample(m, rng, max_len=5)
            self.assertTrue(1 <= len(s) <= 5)
            self.assertTrue(all(0 <= x < 3 for x in s))


class TestHMM(unittest.TestCase):
    def test_forward_matches_brute(self):
        h = HMM(2, 3, rng=np.random.default_rng(3))
        for s in [(0,), (2, 1), (0, 1, 1), (2, 2, 0, 1)]:
            self.assertAlmostEqual(h.loglik(s), math.log(brute_hmm(h, s)), places=10)

    def test_total_mass(self):
        h = HMM(2, 2, rng=np.random.default_rng(4))
        h.T = np.array([[0.2, 0.2, 0.6], [0.3, 0.1, 0.6]])
        mass = 0.0
        for L in range(1, 9):
            for s in itertools.product(range(2), repeat=L):
                mass += math.exp(h.loglik(s))
        self.assertLessEqual(mass, 1.0 + 1e-9)
        self.assertGreater(mass, 1.0 - 0.4 ** 8 - 1e-9)

    def test_em_monotone(self):
        rng = np.random.default_rng(5)
        seqs = [tuple(int(x) for x in rng.integers(0, 3, size=int(rng.integers(1, 6))))
                for _ in range(40)]
        h = HMM(3, 3, rng=np.random.default_rng(6), pseudo=0.0)
        lls = [h.em_step(seqs) for _ in range(12)]
        for a, b in zip(lls[:-1], lls[1:]):
            self.assertGreaterEqual(b, a - 1e-8)
        self.assertAlmostEqual(h.pi.sum(), 1.0, places=12)
        np.testing.assert_allclose(h.T.sum(axis=1), 1.0, atol=1e-12)

    def test_batch_loglik_matches(self):
        h = HMM(3, 3, rng=np.random.default_rng(16))
        got = batch_ll(hmm_loglik_batch, h, SEQS)
        want = np.array([h.loglik(s) for s in SEQS])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)

    def test_batch_em_matches(self):
        h1 = HMM(3, 3, rng=np.random.default_rng(17), pseudo=0.05)
        h2 = HMM(3, 3, rng=np.random.default_rng(17), pseudo=0.05)
        for _ in range(3):
            t1 = h1.em_step(SEQS)
            t2 = hmm_em_batch(h2, compress(SEQS))
            self.assertAlmostEqual(t1, t2, places=9)
        np.testing.assert_allclose(h1.pi, h2.pi, atol=1e-10)
        np.testing.assert_allclose(h1.B, h2.B, atol=1e-10)
        np.testing.assert_allclose(h1.T, h2.T, atol=1e-10)

    def test_sample_cap(self):
        h = HMM(2, 3, rng=np.random.default_rng(18))
        h.T = np.array([[0.45, 0.45, 0.1], [0.45, 0.45, 0.1]])
        rng = np.random.default_rng(19)
        for _ in range(30):
            self.assertLessEqual(len(hmm_sample(h, rng, max_len=4)), 4)


class TestPCFG(unittest.TestCase):
    def test_inside_matches_brute(self):
        g = PCFG(3, 2, rng=np.random.default_rng(7))
        for s in [(1,), (0, 1), (1, 1, 0), (0, 1, 0, 1)]:
            self.assertAlmostEqual(g.loglik(s), math.log(brute_inside(g, s, 0)), places=10)

    def test_rules_normalised(self):
        g = PCFG(4, 3, rng=np.random.default_rng(8))
        tot = g.R.reshape(4, -1).sum(axis=1) + g.E.sum(axis=1)
        np.testing.assert_allclose(tot, 1.0, atol=1e-12)

    def test_em_monotone(self):
        rng = np.random.default_rng(9)
        seqs = [tuple(int(x) for x in rng.integers(0, 2, size=int(rng.integers(1, 6))))
                for _ in range(30)]
        g = PCFG(3, 2, rng=np.random.default_rng(10), pseudo=0.0)
        lls = [g.em_step(seqs) for _ in range(10)]
        for a, b in zip(lls[:-1], lls[1:]):
            self.assertGreaterEqual(b, a - 1e-8)
        tot = g.R.reshape(3, -1).sum(axis=1) + g.E.sum(axis=1)
        np.testing.assert_allclose(tot, 1.0, atol=1e-12)

    def test_pcfg_nests_hmm(self):
        h = HMM(2, 3, rng=np.random.default_rng(11))
        g = pcfg_from_hmm(h)
        tot = g.R.reshape(g.N, -1).sum(axis=1) + g.E.sum(axis=1)
        np.testing.assert_allclose(tot, 1.0, atol=1e-12)
        for s in [(0,), (1, 2), (2, 0, 1), (0, 0, 1, 2)]:
            self.assertAlmostEqual(g.loglik(s), h.loglik(s), places=10)

    def test_batch_loglik_matches(self):
        g = PCFG(4, 3, rng=np.random.default_rng(20))
        got = batch_ll(pcfg_loglik_batch, g, SEQS)
        want = np.array([g.loglik(s) for s in SEQS])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)

    def test_batch_em_matches(self):
        g1 = PCFG(3, 3, rng=np.random.default_rng(21), pseudo=0.05)
        g2 = PCFG(3, 3, rng=np.random.default_rng(21), pseudo=0.05)
        for _ in range(3):
            t1 = g1.em_step(SEQS)
            t2 = pcfg_em_batch(g2, compress(SEQS))
            self.assertAlmostEqual(t1, t2, places=9)
        np.testing.assert_allclose(g1.R, g2.R, atol=1e-10)
        np.testing.assert_allclose(g1.E, g2.E, atol=1e-10)

    def test_sample_cap(self):
        g = PCFG(3, 2, rng=np.random.default_rng(22))
        rng = np.random.default_rng(23)
        for _ in range(30):
            s = pcfg_sample(g, rng, max_len=5)
            self.assertTrue(1 <= len(s) <= 5)


class TestBatchHelpers(unittest.TestCase):
    def test_compress_hand(self):
        groups = compress([(0,), (1, 0), (0,), (1, 0), (2,)])
        self.assertEqual(len(groups), 2)
        S1, w1 = groups[0]
        S2, w2 = groups[1]
        self.assertEqual(S1.tolist(), [[0], [2]])
        self.assertEqual(w1.tolist(), [2.0, 1.0])
        self.assertEqual(S2.tolist(), [[1, 0]])
        self.assertEqual(w2.tolist(), [2.0])

    def test_fit_em_stops(self):
        h = HMM(2, 3, rng=np.random.default_rng(24), pseudo=0.01)
        n = fit_em(h, hmm_em_batch, compress(SEQS), iters=500, tol=1e-6)
        self.assertTrue(2 <= n <= 500)
        self.assertTrue(np.isfinite(batch_ll(hmm_loglik_batch, h, SEQS)).all())


class TestEndToEnd(unittest.TestCase):
    def test_small_world(self):
        rng = np.random.default_rng(12)
        h = HMM(2, 3, rng=np.random.default_rng(13))
        h.T = np.array([[0.6, 0.1, 0.3], [0.1, 0.6, 0.3]])
        h.B = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        seqs = [hmm_sample(h, rng, max_len=8) for _ in range(80)]
        train, test = seqs[:60], seqs[60:]
        kn = KNGram(2, 3).fit(train)
        hm = HMM(2, 3, rng=np.random.default_rng(14), pseudo=0.01)
        fit_em(hm, hmm_em_batch, compress(train), iters=30)
        pg = PCFG(3, 3, rng=np.random.default_rng(15), pseudo=0.01)
        fit_em(pg, pcfg_em_batch, compress(train), iters=5)
        for ll in (np.array([kn.loglik(s) for s in test]),
                   batch_ll(hmm_loglik_batch, hm, test),
                   batch_ll(pcfg_loglik_batch, pg, test)):
            self.assertTrue(np.isfinite(ll).all())
            self.assertLess(float(ll.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
