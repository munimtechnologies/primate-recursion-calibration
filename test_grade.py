# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Tests for grade.py (standard library only)."""
import json
import os
import tempfile
import unittest

import grade


def _row(fam, seed, f, f_ps=None, f_hmm=0.1):
    return {"family": fam, "seed": seed, "world": seed, "vs_ngram_F": f,
            "vs_ngram_perseq_F": f if f_ps is None else f_ps, "vs_hmm_F": f_hmm}


def _real():
    s = lambda F, m, p, q, pq: {"F": F, "F_noise": 0.5, "mean_elld": m,  # noqa: E731
                                "p_t_folds": p, "Q": q, "p_Q": pq}
    return {"T0": {"n_sequences": 5800, "c_types": 6},
            "primary": {"best_ngram": "ng2", "best_hmm": "hmm3", "best_pcfg": "pcfg6_0.01",
                        "vs_ngram": s(0.5, 0.01, 0.5, 20.0, 0.4),
                        "vs_hmm": s(0.3, -0.02, 0.1, 18.0, 0.5),
                        "hmm_vs_ngram": s(0.7, 0.03, 0.04, 15.0, 0.6),
                        "vs_ngram_perseq": s(0.6, 0.05, 0.2, 19.0, 0.45)}}


def _worlds():
    a = [_row("G", i, 0.0) for i in range(1, 9)]
    a += [_row("B", i, 0.4) for i in range(11, 19)]
    a += [_row("H", i, 0.5) for i in range(21, 25)]
    b = [_row("H", i, 0.5) for i in range(25, 29)]
    b.append(_row("G", 1, 0.9))  # duplicate seed: the first copy is kept
    return [{"worlds": a}, {"worlds": b}]


class TestQuantile(unittest.TestCase):
    def test_numpy_linear(self):
        self.assertAlmostEqual(grade.median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(grade.quantile([1, 2, 3, 4], 0.95), 3.85)
        self.assertAlmostEqual(grade.quantile([4, 1, 3, 2], 0.05), 1.15)
        self.assertAlmostEqual(grade.median([7]), 7.0)


class TestRules(unittest.TestCase):
    def test_pool_dedup(self):
        rows = grade.pool_worlds(_worlds())
        self.assertEqual(len(rows), 24)
        g1 = [r for r in rows if r["family"] == "G" and r["seed"] == 1]
        self.assertEqual(len(g1), 1)
        self.assertEqual(g1[0]["vs_ngram_F"], 0.0)

    def test_headline(self):
        self.assertEqual(grade.headline({"B": [0.3] * 8}), "NOT_DIAGNOSTIC")
        self.assertEqual(grade.headline({"B": [0.05] * 8, "H": [0.1] * 8}), "DIAGNOSTIC")
        self.assertEqual(grade.headline({"B": [0.1] * 8}), "PARTIAL")
        self.assertEqual(grade.headline({"B": [0.1] * 8, "H": [0.2] * 8}), "PARTIAL")
        self.assertEqual(grade.headline({"B": [0.9] * 7, "H": [0.9] * 7}),
                         "INSUFFICIENT_WORLDS")
        # H counted with median >= .25 even though B is short
        self.assertEqual(grade.headline({"B": [0.0] * 3, "H": [0.25] * 8}), "NOT_DIAGNOSTIC")

    def test_calibration(self):
        self.assertEqual(grade.calibration_flag({"G": [0.3] * 8}), "MODEL_UNCALIBRATED")
        self.assertEqual(grade.calibration_flag({"G": [0.25] * 8}), "CONTROL_OK")
        self.assertEqual(grade.calibration_flag({"G": [0.0] * 7}), "CONTROL_INSUFFICIENT")

    def test_t3(self):
        self.assertEqual(grade.t3_reading({"mean_elld": 0.0, "p_t_folds": 0.01}),
                         "HMM_MATCHES_OR_BEATS")
        self.assertEqual(grade.t3_reading({"mean_elld": 0.1, "p_t_folds": 0.01}),
                         "PCFG_BEYOND_HMM")
        self.assertEqual(grade.t3_reading({"mean_elld": 0.1, "p_t_folds": 0.2}), "UNDECIDED")


class TestEndToEnd(unittest.TestCase):
    def test_grade_synthetic(self):
        g = grade.grade(_real(), _worlds())
        self.assertEqual(g["headline"], "NOT_DIAGNOSTIC")
        self.assertEqual(g["calibration"], "CONTROL_OK")
        self.assertEqual(g["T3"], "HMM_MATCHES_OR_BEATS")
        self.assertEqual(g["n_worlds_pooled"], 24)
        out = {p["id"]: p["outcome"] for p in g["predictions"]}
        for pid in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10", "P11"):
            self.assertIs(out[pid], True, pid)
        self.assertIsNone(out["P9"])
        self.assertEqual(g["n_graded"], 10)
        self.assertAlmostEqual(g["brier_mean"], 0.16025)
        self.assertEqual(g["share_of_worlds_with_F_ge_real"], {"B": 0.0, "G": 0.0, "H": 1.0})

    def test_main_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "real.json")
            wa = os.path.join(d, "wa.json")
            wb = os.path.join(d, "wb.json")
            ws = _worlds()
            for path, obj in ((rp, _real()), (wa, ws[0]), (wb, ws[1])):
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(obj, fh)
            out = os.path.join(d, "res")
            self.assertEqual(grade.main(["--real", rp, "--worlds", wa, wb, "--out-dir", out]), 0)
            with open(os.path.join(out, "grade_results.json"), encoding="utf-8") as fh:
                g = json.load(fh)
            self.assertEqual(g["headline"], "NOT_DIAGNOSTIC")
            with open(os.path.join(out, "GRADE-RECURSION.md"), encoding="utf-8") as fh:
                md = fh.read()
            self.assertIn("Headline: **NOT_DIAGNOSTIC**", md)
            self.assertIn("| P9 |", md)
            self.assertIn("not graded", md)


if __name__ == "__main__":
    unittest.main()
