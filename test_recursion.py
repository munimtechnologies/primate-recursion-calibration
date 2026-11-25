# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
import os
import tempfile
import unittest

import numpy as np

import recursion_test as rt
from grammars import HMM, hmm_sample


class TestData(unittest.TestCase):
    def test_load(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('block;sequence\n"s1";"phee, trill,chatter"\n"s2";"chatter"\n'
                         's3;tsk\nbad line\n')
            rows = rt.load(path)
        finally:
            os.remove(path)
        self.assertEqual(rows, [["phee", "trill"], ["tsk"]])

    def test_remove_singletons(self):
        rows = [["a", "b"], ["a"], ["c", "a"], ["b"]]
        kept, single, counts = rt.remove_singletons(rows)
        self.assertEqual(kept, [["a", "b"], ["a"], ["b"]])
        self.assertEqual(single, ["c"])
        self.assertEqual(counts, {"a": 3, "b": 2, "c": 1})
        units, seqs = rt.encode(kept)
        self.assertEqual(units, ["a", "b"])
        self.assertEqual(seqs, [(0, 1), (0,), (1,)])

    def test_fold_ids_balanced(self):
        f = rt.fold_ids(45, 20, 3)
        c = np.bincount(f, minlength=20)
        self.assertEqual(int(c.sum()), 45)
        self.assertTrue(c.max() - c.min() <= 1)


class TestStats(unittest.TestCase):
    def test_fold_stats_hand(self):
        s = rt.fold_stats([1.0, -1.0, 2.0, 0.0], [0, 0, 1, 1], folds=2, perms=20)
        self.assertEqual(s["fold_elld"], [0.0, 1.0])
        self.assertEqual(s["F"], 0.5)
        self.assertAlmostEqual(s["mean_elld"], 0.5)
        self.assertAlmostEqual(s["t_folds"], 1.0)
        self.assertAlmostEqual(s["Q"], 0.5)
        self.assertTrue(0.0 < s["p_Q"] < 1.0)
        self.assertTrue(0.0 <= s["F_noise"] <= 1.0)

    def test_chi2_wh(self):
        # chi2(19) median is about 18.34; upper 5% point 30.14
        self.assertAlmostEqual(rt.chi2_sf_wh(18.338, 19), 0.5, places=2)
        self.assertAlmostEqual(rt.chi2_sf_wh(30.144, 19), 0.05, places=2)

    def test_headline(self):
        self.assertEqual(rt.headline({}), "INSUFFICIENT_WORLDS")
        self.assertEqual(rt.headline({"B": [0.5] * 8, "H": [0.0] * 8}), "NOT_DIAGNOSTIC")
        self.assertEqual(rt.headline({"B": [0.0] * 8, "H": [0.1] * 8}), "DIAGNOSTIC")
        self.assertEqual(rt.headline({"B": [0.2] * 8}), "PARTIAL")
        self.assertEqual(rt.headline({"B": [0.5] * 7, "H": [0.0] * 8}), "PARTIAL")
        self.assertEqual(rt.calibration_flag({"G": [0.5] * 8}), "MODEL_UNCALIBRATED")
        self.assertEqual(rt.calibration_flag({"G": [0.0] * 8}), "CONTROL_OK")
        self.assertEqual(rt.calibration_flag({"G": [0.0] * 3}), "CONTROL_INSUFFICIENT")


class TestPipeline(unittest.TestCase):
    def test_tiny_cv(self):
        rng = np.random.default_rng(30)
        h = HMM(2, 3, rng=np.random.default_rng(31))
        h.T = np.array([[0.5, 0.1, 0.4], [0.1, 0.5, 0.4]])
        seqs = [hmm_sample(h, rng, max_len=6) for _ in range(30)]
        ll, fold = rt.cv_all(seqs, 3, ((2, 0.1),), 5, folds=3, iters=3, orders=(1, 2),
                             ks=(1, 2))
        self.assertEqual(sorted(ll), ["hmm1", "hmm2", "ng1", "ng2", "pcfg2_0.1"])
        for k in ("hmm1", "hmm2", "ng2", "pcfg2_0.1"):
            self.assertTrue(np.isfinite(ll[k]).all(), k)
        s = rt.summarise(ll, fold, ["pcfg2_0.1"], folds=3)
        self.assertEqual(len(s["vs_ngram"]["fold_elld"]), 3)
        self.assertTrue(0.0 <= s["vs_hmm"]["F"] <= 1.0)

    def test_bootstrap_world(self):
        real = [(0,), (1, 2), (2,)]
        w = rt.make_world("B", {}, real, 10, np.random.default_rng(1))
        self.assertEqual(len(w), 10)
        self.assertTrue(all(s in real for s in w))


if __name__ == "__main__":
    unittest.main()
