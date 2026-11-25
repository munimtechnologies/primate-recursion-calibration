# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""primate-recursion-calibration: grade the pre-registered test (PREREG.md) from the job outputs.

Written 2026-11-25, BEFORE any job output existed (commit history shows this). Self-contained,
standard library only. Pools the per-world rows of the worlds jobs (deduplicated by seed),
recomputes the headline and calibration flag exactly as PREREG.md fixes them, reads T0/T2/T3/T4
from the real-data job, and Brier-scores the eleven registered predictions.

  python grade.py --real real_results.json --worlds worlds_a.json worlds_b.json --out-dir results
"""
import argparse
import json
import math
import os
import sys

MIN_WORLDS = 8
PUBLISHED_N, PUBLISHED_C = 5800, 6
PRIORS = {"P1": 0.80, "P2": 0.45, "P3": 0.65, "P4": 0.50, "P5": 0.45, "P6": 0.55,
          "P7": 0.60, "P8": 0.85, "P9": 0.55, "P10": 0.80, "P11": 0.60}
TEXT = {
    "P1": "T0 coding gives n = 5800 and c = 6",
    "P2": "real F (vs n-gram, A) in [0.25, 0.75]",
    "P3": "control G: median F <= 0.15",
    "P4": "HMM family H: median F >= 0.25",
    "P5": "bootstrap family B: median F >= 0.25",
    "P6": "headline NOT_DIAGNOSTIC",
    "P7": "T3: HMM_MATCHES_OR_BEATS on real data",
    "P8": "T2: Cochran's Q p > 0.05",
    "P9": "power family P: median F >= 0.5",
    "P10": "real F under selection B >= real F under selection A",
    "P11": "real |mean ELLD| vs n-gram < 0.05 nats per sequence",
}


def quantile(values, q):
    """Linear interpolation between order statistics (numpy's default method)."""
    v = sorted(float(x) for x in values)
    if not v:
        raise ValueError("empty")
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def median(values):
    return quantile(values, 0.5)


def pool_worlds(world_docs):
    """All per-world rows of all worlds files, deduplicated by (family, seed)."""
    seen = {}
    for doc in world_docs:
        for r in doc.get("worlds", []):
            key = (r["family"], int(r["seed"]))
            seen.setdefault(key, r)
    return [seen[k] for k in sorted(seen)]


def families(rows, field="vs_ngram_F"):
    fam = {}
    for r in rows:
        x = r.get(field)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        fam.setdefault(r["family"], []).append(float(x))
    return fam


def headline(fam, min_worlds=MIN_WORLDS):
    regs = [k for k in ("B", "H") if len(fam.get(k, [])) >= min_worlds]
    if not regs:
        return "INSUFFICIENT_WORLDS"
    if any(median(fam[k]) >= 0.25 for k in regs):
        return "NOT_DIAGNOSTIC"
    if len(regs) == 2 and all(quantile(fam[k], 0.95) <= 0.15 for k in regs):
        return "DIAGNOSTIC"
    return "PARTIAL"


def calibration_flag(fam, min_worlds=MIN_WORLDS):
    g = fam.get("G", [])
    if len(g) < min_worlds:
        return "CONTROL_INSUFFICIENT"
    return "MODEL_UNCALIBRATED" if median(g) > 0.25 else "CONTROL_OK"


def t3_reading(vs_hmm):
    m = vs_hmm.get("mean_elld")
    p = vs_hmm.get("p_t_folds")
    if m is None:
        return "NOT_COMPUTED"
    if m <= 0:
        return "HMM_MATCHES_OR_BEATS"
    if p is not None and p < 0.05:
        return "PCFG_BEYOND_HMM"
    return "UNDECIDED"


def family_summary(fam):
    out = {}
    for k in sorted(fam):
        v = fam[k]
        out[k] = {"n_worlds": len(v), "median": median(v), "q05": quantile(v, 0.05),
                  "q95": quantile(v, 0.95),
                  "share_ge_0.25": sum(1 for x in v if x >= 0.25) / float(len(v))}
    return out


def grade(real, world_docs, min_worlds=MIN_WORLDS):
    rows = pool_worlds(world_docs)
    fam = families(rows, "vs_ngram_F")
    fam_b = families(rows, "vs_ngram_perseq_F")
    fam_h = families(rows, "vs_hmm_F")
    head = headline(fam, min_worlds)
    calib = calibration_flag(fam, min_worlds)
    prim = real["primary"]
    t0 = real["T0"]
    vs_ng = prim["vs_ngram"]
    vs_hmm = prim["vs_hmm"]
    perseq = prim["vs_ngram_perseq"]
    t3 = t3_reading(vs_hmm)

    def fam_ok(k):
        return len(fam.get(k, [])) >= min_worlds

    out = {}
    out["P1"] = t0["n_sequences"] == PUBLISHED_N and t0["c_types"] == PUBLISHED_C
    out["P2"] = 0.25 <= vs_ng["F"] <= 0.75
    out["P3"] = (median(fam["G"]) <= 0.15) if fam_ok("G") else None
    out["P4"] = (median(fam["H"]) >= 0.25) if fam_ok("H") else None
    out["P5"] = (median(fam["B"]) >= 0.25) if fam_ok("B") else None
    out["P6"] = (head == "NOT_DIAGNOSTIC") if head != "INSUFFICIENT_WORLDS" else None
    out["P7"] = (t3 == "HMM_MATCHES_OR_BEATS") if t3 != "NOT_COMPUTED" else None
    out["P8"] = (vs_ng["p_Q"] > 0.05) if vs_ng.get("p_Q") is not None else None
    out["P9"] = (median(fam["P"]) >= 0.5) if fam_ok("P") else None
    out["P10"] = perseq["F"] >= vs_ng["F"]
    out["P11"] = abs(vs_ng["mean_elld"]) < 0.05

    preds = []
    sq = []
    for pid in sorted(PRIORS, key=lambda s: int(s[1:])):
        o = out[pid]
        row = {"id": pid, "prediction": TEXT[pid], "p": PRIORS[pid],
               "outcome": None if o is None else bool(o)}
        if o is not None:
            row["brier"] = (PRIORS[pid] - (1.0 if o else 0.0)) ** 2
            sq.append(row["brier"])
        preds.append(row)
    brier = sum(sq) / len(sq) if sq else None

    real_f_in_null = {}
    for k, v in sorted(fam.items()):
        real_f_in_null[k] = sum(1 for x in v if x >= vs_ng["F"]) / float(len(v))

    return {
        "headline": head,
        "calibration": calib,
        "T3": t3,
        "n_worlds_pooled": len(rows),
        "families_vs_ngram_F": family_summary(fam),
        "families_vs_ngram_perseq_F": family_summary(fam_b),
        "families_vs_hmm_F": family_summary(fam_h),
        "share_of_worlds_with_F_ge_real": real_f_in_null,
        "real": {"n_sequences": t0["n_sequences"], "c_types": t0["c_types"],
                 "best_ngram": prim["best_ngram"], "best_hmm": prim["best_hmm"],
                 "best_pcfg": prim["best_pcfg"],
                 "vs_ngram": _pick(vs_ng), "vs_hmm": _pick(vs_hmm),
                 "hmm_vs_ngram": _pick(prim["hmm_vs_ngram"]),
                 "vs_ngram_perseq": _pick(perseq)},
        "predictions": preds,
        "brier_mean": brier,
        "n_graded": len(sq),
        "min_worlds": min_worlds,
    }


def _pick(s):
    return {k: s.get(k) for k in ("F", "F_noise", "mean_elld", "p_t_folds", "Q", "p_Q")}


def _f(x, fmt="%.4f"):
    return "-" if x is None else fmt % x


def to_md(g):
    L = ["# primate-recursion-calibration: grade (computed by grade.py)", "",
         "Headline: **%s**. Calibration: %s. T3 (pCFG vs HMM, real): %s. Worlds pooled: %d."
         % (g["headline"], g["calibration"], g["T3"], g["n_worlds_pooled"]), "",
         "## Simulated worlds: F = share of 20 folds where the pCFG beats the best n-gram", "",
         "| family | worlds | median F | 5% | 95% | share F >= 0.25 | share F >= real F |",
         "|---|---|---|---|---|---|---|"]
    for k, s in g["families_vs_ngram_F"].items():
        L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            k, s["n_worlds"], s["median"], s["q05"], s["q95"], s["share_ge_0.25"],
            g["share_of_worlds_with_F_ge_real"].get(k, float("nan"))))
    r = g["real"]
    L += ["", "## Real marmoset data", "",
          "n = %d sequences, c = %d call types. Best: n-gram %s, HMM %s, PCFG %s." % (
              r["n_sequences"], r["c_types"], r["best_ngram"], r["best_hmm"], r["best_pcfg"]),
          "", "| comparison | F | F_noise | mean ELLD (nats/seq) | p (fold t) | Q | p(Q) |",
          "|---|---|---|---|---|---|---|"]
    for lab in ("vs_ngram", "vs_hmm", "hmm_vs_ngram", "vs_ngram_perseq"):
        s = r[lab]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            lab, _f(s["F"], "%.2f"), _f(s["F_noise"], "%.3f"), _f(s["mean_elld"]),
            _f(s["p_t_folds"], "%.4g"), _f(s["Q"], "%.2f"), _f(s["p_Q"], "%.4g")))
    L += ["", "## Registered predictions", "", "| id | prediction | p | outcome | Brier |",
          "|---|---|---|---|---|"]
    for p in g["predictions"]:
        o = "not graded" if p["outcome"] is None else ("true" if p["outcome"] else "false")
        L.append("| %s | %s | %.2f | %s | %s |" % (p["id"], p["prediction"], p["p"], o,
                                                 _f(p.get("brier"), "%.4f")))
    L += ["", "Mean Brier over %d graded predictions: %s" % (g["n_graded"],
                                                             _f(g["brier_mean"], "%.4f"))]
    return "\n".join(L) + "\n"


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--worlds", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    g = grade(_load(a.real), [_load(p) for p in a.worlds])
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "grade_results.json"), "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=1)
    with open(os.path.join(a.out_dir, "GRADE-RECURSION.md"), "w", encoding="utf-8") as fh:
        fh.write(to_md(g))
    print(g["headline"], g["calibration"], g["T3"], "Brier", g["brier_mean"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
