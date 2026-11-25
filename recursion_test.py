# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""primate-recursion-calibration: does the held-out likelihood test of Nitschke et al. 2026
('Supra-regular recursion in primate vocal sequences', bioRxiv 10.64898/2026.09.19.752870)
fire on sources that are REGULAR by construction? Pre-registered in PREREG.md.

  python recursion_test.py real   --data-dir D --out-dir O [--workers 4]
  python recursion_test.py worlds --data-dir D --out-dir O [--workers 4] [--worlds 30]
                                  [--budget-minutes 320] [--seed-offset 0]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from grammars import (HMM, KNGram, MLUnigram, PCFG, batch_ll, compress, fit_em,  # noqa: E402
                      hmm_em_batch, hmm_loglik_batch, hmm_sample, kn_sample, n_params_hmm,
                      pcfg_em_batch, pcfg_loglik_batch, pcfg_sample)

SEED = 20261125
FOLDS = 20
NGRAM_ORDERS = (1, 2, 3, 4, 5, 6)
HMM_KS = (1, 2, 3, 4, 6, 8)
PSEUDO_HMM = 0.01
PCFG_GRID = ((6, 0.01), (12, 0.01))
PCFG_GRID_REAL_EXTRA = ((20, 0.01), (6, 0.1), (12, 0.1))
EM_ITERS = 300
TOL = 1e-6
MAX_LEN = 12
GEN_HMM_KS = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_KN_ORDER = 3
GEN_PCFG = (6, 0.01)
FAMILIES = ("G", "B", "H", "P")
MIN_WORLDS = 8
PUBLISHED_N, PUBLISHED_C = 5800, 6
DATA_FILE = "marmoset_sequencing_data.csv"


# ----------------------------------------------------------------------------- data

def load(path):
    """Bosshard et al.'s reading (as in marmoset-order-test): skip the header, split each line
    on ';', take field 1, remove spaces and quotes, split on ',', drop 'chatter'."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    rows = []
    for line in lines[1:]:
        f = line.split(";")
        if len(f) < 2:
            continue
        field = f[1].replace(" ", "").replace('"', "")
        calls = [x for x in field.split(",") if x and x != "chatter"]
        if calls:
            rows.append(calls)
    return rows


def remove_singletons(rows):
    """Nitschke et al.: 'Sequences that contained symbols which occurred only once in the
    dataset were removed' (one pass)."""
    counts = {}
    for r in rows:
        for c in r:
            counts[c] = counts.get(c, 0) + 1
    single = {c for c, n in counts.items() if n == 1}
    kept = [r for r in rows if not any(c in single for c in r)]
    return kept, sorted(single), counts


def encode(rows):
    units = sorted({c for r in rows for c in r})
    idx = {u: i for i, u in enumerate(units)}
    return units, [tuple(idx[c] for c in r) for r in rows]


# ----------------------------------------------------------------------------- CV pipeline

def fold_ids(n, folds, seed):
    rng = np.random.default_rng(seed)
    f = np.empty(n, dtype=int)
    f[rng.permutation(n)] = np.arange(n) % folds
    return f


def pcfg_name(N, p):
    return "pcfg%d_%g" % (N, p)


def fold_logliks(task):
    """Fit every model on the training folds; per-sequence held-out log-likelihoods."""
    seqs, V, train_idx, test_idx, grid, seed, iters, orders, ks = task
    train = [seqs[i] for i in train_idx]
    test = [seqs[i] for i in test_idx]
    groups = compress(train)
    res = {}
    for n in orders:
        m = MLUnigram(V).fit(train) if n == 1 else KNGram(n, V).fit(train)
        res["ng%d" % n] = np.array([m.loglik(s) for s in test])
    for K in ks:
        h = HMM(K, V, rng=np.random.default_rng(seed + 101 * K), pseudo=PSEUDO_HMM)
        fit_em(h, hmm_em_batch, groups, iters, TOL)
        res["hmm%d" % K] = batch_ll(hmm_loglik_batch, h, test)
    for N, p in grid:
        g = PCFG(N, V, rng=np.random.default_rng(seed + 1009 * N + int(round(p * 1e4))),
                 pseudo=p)
        fit_em(g, pcfg_em_batch, groups, iters, TOL)
        res[pcfg_name(N, p)] = batch_ll(pcfg_loglik_batch, g, test)
    return test_idx, res


def cv_all(seqs, V, grid, seed, folds=FOLDS, pool=None, iters=EM_ITERS,
           orders=NGRAM_ORDERS, ks=HMM_KS):
    fold = fold_ids(len(seqs), folds, seed)
    tasks = []
    for f in range(folds):
        tasks.append((seqs, V, np.flatnonzero(fold != f), np.flatnonzero(fold == f), grid,
                      seed + 31 * f, iters, orders, ks))
    results = pool.map(fold_logliks, tasks) if pool is not None else [fold_logliks(t)
                                                                      for t in tasks]
    n = len(seqs)
    ll = {}
    for test_idx, res in results:
        for k, v in res.items():
            ll.setdefault(k, np.full(n, np.nan))[test_idx] = v
    return ll, fold


# ----------------------------------------------------------------------------- statistics

def pick(ll, names):
    """Setting with the highest total held-out log-likelihood (their selection of n-gram order
    and number of non-terminals by held-out likelihood)."""
    tots = {k: float(np.sum(ll[k])) for k in names}
    return max(sorted(names), key=lambda k: tots[k]), tots


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_sf_wh(q, df):
    """Wilson-Hilferty approximation of the chi-square upper tail."""
    if df <= 0:
        return float("nan")
    z = ((q / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return _norm_sf(z)


def t_two_sided(t, df):
    try:
        from scipy import stats  # optional; the job installs scipy
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:  # noqa: BLE001
        return float(2.0 * _norm_sf(abs(t)))


def fold_stats(d, fold, folds=FOLDS, perms=200, seed=SEED):
    """d: per-sequence held-out difference (pCFG minus baseline), nats. Returns the fold means
    (= their per-fold ELLD), the share of folds with ELLD > 0 (F), the share expected if the
    folds were re-drawn at random from the same per-sequence differences (F_noise), a fold-level
    t test of mean ELLD = 0 and Cochran's Q for excess between-fold heterogeneity."""
    d = np.asarray(d, dtype=float)
    fold = np.asarray(fold)
    nf = np.bincount(fold, minlength=folds).astype(float)
    m = np.bincount(fold, weights=d, minlength=folds) / nf
    out = {"fold_elld": [float(x) for x in m], "F": float(np.mean(m > 0)),
           "mean_elld": float(d.mean()), "finite": bool(np.isfinite(d).all())}
    if not out["finite"]:
        return out
    sd = float(m.std(ddof=1)) if folds > 1 else 0.0
    t = float(m.mean() / (sd / math.sqrt(folds))) if sd > 0 else 0.0
    out["t_folds"] = t
    out["p_t_folds"] = t_two_sided(t, folds - 1)
    v = np.array([d[fold == f].var(ddof=1) / nf[f] if nf[f] > 1 else 0.0
                  for f in range(folds)])
    v = np.maximum(v, 1e-12)
    w = 1.0 / v
    mbar = float((w * m).sum() / w.sum())
    q = float((w * (m - mbar) ** 2).sum())
    out["Q"] = q
    out["p_Q"] = chi2_sf_wh(q, folds - 1)
    rng = np.random.default_rng(seed)
    sh = []
    for _ in range(perms):
        pf = fold[rng.permutation(len(fold))]
        mm = np.bincount(pf, weights=d, minlength=folds) / nf
        sh.append(float(np.mean(mm > 0)))
    out["F_noise"] = float(np.mean(sh))
    return out


def summarise(ll, fold, pcfg_names, folds=FOLDS):
    ng = sorted(k for k in ll if k.startswith("ng"))
    hm = sorted(k for k in ll if k.startswith("hmm"))
    kn, tn = pick(ll, ng)
    kh, th = pick(ll, hm)
    kp, tp = pick(ll, pcfg_names)
    res = {"best_ngram": kn, "best_hmm": kh, "best_pcfg": kp,
           "totals": {**tn, **th, **tp},
           "vs_ngram": fold_stats(ll[kp] - ll[kn], fold, folds),
           "vs_hmm": fold_stats(ll[kp] - ll[kh], fold, folds),
           "hmm_vs_ngram": fold_stats(ll[kh] - ll[kn], fold, folds)}
    mx = lambda names: np.max(np.vstack([ll[k] for k in names]), axis=0)  # noqa: E731
    res["vs_ngram_perseq"] = fold_stats(mx(pcfg_names) - mx(ng), fold, folds)
    res["vs_hmm_perseq"] = fold_stats(mx(pcfg_names) - mx(hm), fold, folds)
    return res


def headline(fam, min_worlds=MIN_WORLDS):
    """fam: family -> list of per-world F (pCFG vs best n-gram, selection A). Regular families:
    B (bootstrap of the real sequences) and H (HMM fitted to them)."""
    regs = [k for k in ("B", "H") if len(fam.get(k, [])) >= min_worlds]
    if not regs:
        return "INSUFFICIENT_WORLDS"
    if any(float(np.median(fam[k])) >= 0.25 for k in regs):
        return "NOT_DIAGNOSTIC"
    if len(regs) == 2 and all(float(np.quantile(fam[k], 0.95)) <= 0.15 for k in regs):
        return "DIAGNOSTIC"
    return "PARTIAL"


def calibration_flag(fam, min_worlds=MIN_WORLDS):
    g = fam.get("G", [])
    if len(g) < min_worlds:
        return "CONTROL_INSUFFICIENT"
    return "MODEL_UNCALIBRATED" if float(np.median(g)) > 0.25 else "CONTROL_OK"


# ----------------------------------------------------------------------------- real data

def read_real(data_dir):
    rows = load(os.path.join(data_dir, DATA_FILE))
    kept, single, counts = remove_singletons(rows)
    units, seqs = encode(kept)
    lengths = np.bincount([len(s) for s in seqs])
    info = {"rows_after_chatter_drop": len(rows), "singleton_types": single,
            "type_counts": {k: int(v) for k, v in sorted(counts.items())},
            "n_sequences": len(seqs), "c_types": len(units), "units": units,
            "n_calls": int(sum(len(s) for s in seqs)),
            "length_counts": {str(L): int(c) for L, c in enumerate(lengths) if c},
            "n_distinct": len(set(seqs)),
            "matches_published": len(seqs) == PUBLISHED_N and len(units) == PUBLISHED_C}
    return units, seqs, info


def cmd_real(args):
    t0 = time.time()
    units, seqs, info = read_real(args.data_dir)
    V = len(units)
    print("T0:", json.dumps({k: info[k] for k in ("n_sequences", "c_types", "n_calls",
                                                  "matches_published")}), flush=True)
    grid = PCFG_GRID + PCFG_GRID_REAL_EXTRA
    with mp.Pool(args.workers) as pool:
        ll, fold = cv_all(seqs, V, grid, SEED, pool=pool)
    prim = summarise(ll, fold, [pcfg_name(N, p) for N, p in PCFG_GRID])
    sec = summarise(ll, fold, [pcfg_name(N, p) for N, p in grid])
    res = {"T0": info, "primary": prim, "secondary_full_grid": sec,
           "settings": {"folds": FOLDS, "ngram_orders": NGRAM_ORDERS, "hmm_ks": HMM_KS,
                        "pcfg_grid": PCFG_GRID, "pcfg_grid_extra": PCFG_GRID_REAL_EXTRA,
                        "em_iters": EM_ITERS, "tol": TOL, "seed": SEED},
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "real_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(args.out_dir, "RESULTS-REAL.md"), "w", encoding="utf-8") as fh:
        fh.write(real_md(res))
    for lab in ("vs_ngram", "vs_hmm", "vs_ngram_perseq"):
        s = prim[lab]
        print(lab, "F %.2f F_noise %s mean %.4f" % (s["F"], s.get("F_noise"), s["mean_elld"]),
              flush=True)
    print("done in %.0f s" % (time.time() - t0), flush=True)
    return 0


def real_md(res):
    p = res["primary"]
    L = ["# primate-recursion-calibration: real marmoset data (job output)", "",
         "T0: %d sequences, %d call types (published 5800 / 6): %s" % (
             res["T0"]["n_sequences"], res["T0"]["c_types"], res["T0"]["matches_published"]),
         "", "Best settings (selection by total held-out log-likelihood): n-gram %s, HMM %s, "
         "PCFG %s" % (p["best_ngram"], p["best_hmm"], p["best_pcfg"]), "",
         "| comparison | F (share of folds pCFG better) | F_noise | mean ELLD (nats/seq) | "
         "p (fold t) | Q | p(Q) |", "|---|---|---|---|---|---|---|"]
    for lab in ("vs_ngram", "vs_hmm", "hmm_vs_ngram", "vs_ngram_perseq", "vs_hmm_perseq"):
        s = p[lab]
        L.append("| %s | %.2f | %s | %.4f | %s | %s | %s |" % (
            lab, s["F"], _f(s.get("F_noise")), s["mean_elld"], _f(s.get("p_t_folds")),
            _f(s.get("Q")), _f(s.get("p_Q"))))
    L.append("")
    L.append("Totals: " + ", ".join("%s %.1f" % (k, v) for k, v in sorted(p["totals"].items())))
    return "\n".join(L) + "\n"


def _f(x):
    return "-" if x is None else "%.4g" % x


# ----------------------------------------------------------------------------- worlds

def fit_generators(seqs, V, seed=SEED):
    groups = compress(seqs)
    n_seq = len(seqs)
    bic = {}
    best = None
    for K in GEN_HMM_KS:
        h = HMM(K, V, rng=np.random.default_rng(seed + 7 * K), pseudo=PSEUDO_HMM)
        fit_em(h, hmm_em_batch, groups, EM_ITERS, TOL)
        ll = float(sum(w @ hmm_loglik_batch(h, S) for S, w in groups))
        b = -2.0 * ll + n_params_hmm(K, V) * math.log(n_seq)
        bic[str(K)] = b
        if best is None or b < best[0]:
            best = (b, K, h)
    kn = KNGram(GEN_KN_ORDER, V).fit(seqs)
    N, p = GEN_PCFG
    g = PCFG(N, V, rng=np.random.default_rng(seed + 13), pseudo=p)
    fit_em(g, pcfg_em_batch, groups, EM_ITERS, TOL)
    info = {"hmm_bic": bic, "hmm_K": best[1], "kn_order": GEN_KN_ORDER, "pcfg": GEN_PCFG}
    return {"H": best[2], "G": kn, "P": g}, info


def make_world(family, gens, real, n, rng):
    if family == "B":
        idx = rng.integers(0, len(real), size=n)
        return [real[i] for i in idx]
    if family == "H":
        return [hmm_sample(gens["H"], rng, MAX_LEN) for _ in range(n)]
    if family == "G":
        return [kn_sample(gens["G"], rng, MAX_LEN) for _ in range(n)]
    if family == "P":
        return [pcfg_sample(gens["P"], rng, MAX_LEN) for _ in range(n)]
    raise ValueError(family)


def run_world(task):
    family, w, gens, real, V, n, seed = task
    t0 = time.time()
    rng = np.random.default_rng(seed)
    seqs = make_world(family, gens, real, n, rng)
    ll, fold = cv_all(seqs, V, PCFG_GRID, seed)
    s = summarise(ll, fold, [pcfg_name(N, p) for N, p in PCFG_GRID])
    lens = [len(x) for x in seqs]
    keep = ("F", "F_noise", "mean_elld", "p_t_folds", "p_Q")
    return {"family": family, "world": w, "seed": seed, "seconds": round(time.time() - t0, 1),
            "n_distinct": len(set(seqs)), "max_len": int(max(lens)),
            "share_len1": float(np.mean(np.asarray(lens) == 1)),
            "best_ngram": s["best_ngram"], "best_hmm": s["best_hmm"],
            "best_pcfg": s["best_pcfg"],
            **{"%s_%s" % (lab, k): s[lab].get(k) for lab in
               ("vs_ngram", "vs_hmm", "hmm_vs_ngram", "vs_ngram_perseq") for k in keep}}


def family_table(worlds):
    fam = {}
    for r in worlds:
        fam.setdefault(r["family"], []).append(r)
    out = {}
    for k, rs in sorted(fam.items()):
        row = {"n_worlds": len(rs)}
        for lab in ("vs_ngram_F", "vs_hmm_F", "vs_ngram_perseq_F", "vs_ngram_mean_elld",
                    "vs_hmm_mean_elld"):
            v = np.array([r[lab] for r in rs], dtype=float)
            row[lab] = {"median": float(np.median(v)), "q05": float(np.quantile(v, 0.05)),
                        "q95": float(np.quantile(v, 0.95)),
                        "share_ge_0.25": float(np.mean(v >= 0.25))} if "_F" in lab else {
                "median": float(np.median(v)), "mean": float(v.mean())}
        out[k] = row
    return out


def cmd_worlds(args):
    t0 = time.time()
    units, real, info = read_real(args.data_dir)
    V = len(units)
    n = len(real)
    gens, ginfo = fit_generators(real, V)
    print("generators:", json.dumps(ginfo), flush=True)
    off = {f: 1000 * i for i, f in enumerate(FAMILIES)}
    tasks = [(f, w, gens, real, V, n, SEED + args.seed_offset + off[f] + w)
             for w in range(args.worlds) for f in FAMILIES]
    os.makedirs(args.out_dir, exist_ok=True)
    worlds = []
    budget = args.budget_minutes * 60.0
    stopped = "all tasks done"
    pool = mp.Pool(args.workers)
    try:
        it = pool.imap_unordered(run_world, tasks)
        while True:
            remaining = budget - (time.time() - t0)
            if remaining <= 0:
                stopped = "time budget"
                break
            try:
                r = it.next(timeout=remaining)
            except StopIteration:
                break
            except mp.TimeoutError:
                stopped = "time budget"
                break
            worlds.append(r)
            print("%s world %d: F_ng %.2f F_hmm %.2f (%.0f s; %d done, %.0f min)" % (
                r["family"], r["world"], r["vs_ngram_F"], r["vs_hmm_F"], r["seconds"],
                len(worlds), (time.time() - t0) / 60), flush=True)
            write_worlds(args.out_dir, info, ginfo, worlds, stopped, args)
    finally:
        pool.terminate()
    write_worlds(args.out_dir, info, ginfo, worlds, stopped, args)
    print("stopped:", stopped, "worlds:", len(worlds), flush=True)
    return 0


def write_worlds(out_dir, info, ginfo, worlds, stopped, args):
    fam = {}
    for r in worlds:
        fam.setdefault(r["family"], []).append(r["vs_ngram_F"])
    res = {"T0": info, "generators": ginfo, "seed_offset": args.seed_offset,
           "stopped": stopped, "n_worlds": len(worlds), "families": family_table(worlds),
           "provisional_headline": headline(fam), "calibration": calibration_flag(fam),
           "worlds": sorted(worlds, key=lambda r: (r["family"], r["world"]))}
    tmp = os.path.join(out_dir, "worlds_results.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, "worlds_results.json"))


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("real", "worlds"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=30)
    ap.add_argument("--budget-minutes", type=float, default=320.0)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    return cmd_real(args) if args.mode == "real" else cmd_worlds(args)


if __name__ == "__main__":
    sys.exit(main())
