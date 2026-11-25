# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Probabilistic sequence models over a finite call vocabulary 0..V-1, each giving the
probability of a WHOLE sequence (ending included), so their held-out log-likelihoods compare:

  MLUnigram - maximum-likelihood unigram with an end token (order-1 'pREG' of Nitschke et al.)
  KNGram    - interpolated modified Kneser-Ney n-gram (the n-gram 'pREG' class of Nitschke et
              al. 2026), with n-1 start pads and an end token.
  HMM       - probabilistic finite automaton with K hidden states and an end transition: the
              full class of probabilistic REGULAR grammars (right-linear rules A -> a B | a),
              which strictly contains every n-gram model.
  PCFG      - Chomsky-normal-form grammar with N non-terminals (A -> B C | a), start symbol 0,
              fitted by inside-outside EM with Dirichlet pseudo-counts (pseudo > 0 keeps every
              rule non-zero, i.e. the grammar fully connected).

The *_batch functions do the same computations for many distinct sequences of one length at a
time, weighted by how often each occurs (compress()); the unit tests check them against the
plain per-sequence versions and against brute-force sums. numpy only. Sequences are tuples of
ints; empty sequences are not allowed.
"""
import math

import numpy as np


# ============================================================================ unigram (ML)

class MLUnigram:
    """P(s) = prod_t p(s_t) * p(end), maximum likelihood (Nitschke et al.: no smoothing at n=1)."""

    def __init__(self, V):
        self.V = int(V)

    def fit(self, seqs):
        c = np.zeros(self.V + 1)
        for s in seqs:
            for x in s:
                c[x] += 1
            c[self.V] += 1
        self.p = c / c.sum()
        return self

    def loglik(self, s):
        ps = [self.p[x] for x in s] + [self.p[self.V]]
        if min(ps) <= 0:
            return -math.inf
        return float(sum(math.log(p) for p in ps))


# ============================================================================ n-gram (KN)

class KNGram:
    """Interpolated modified Kneser-Ney (Chen & Goodman 1999). Highest order uses raw counts;
    lower orders use continuation counts (number of distinct left extensions). Discounts D1, D2,
    D3+ per order from counts-of-counts, clipped into (0, k). Base: uniform over V + 1 outcomes
    (V calls + end). P sums to 1 over the V + 1 outcomes for every context."""

    def __init__(self, order, V):
        if order < 1:
            raise ValueError("order >= 1")
        self.n = int(order)
        self.V = int(V)
        self.BOS = -1
        self.EOS = self.V

    def _events(self, s):
        pad = [self.BOS] * (self.n - 1)
        toks = pad + list(s) + [self.EOS]
        for t in range(self.n - 1, len(toks)):
            yield tuple(toks[t - self.n + 1:t]), toks[t]

    def fit(self, seqs):
        n = self.n
        # counts[m][ctx][w], ctx of length m - 1, m = 1..n
        counts = [None] + [dict() for _ in range(n)]
        for s in seqs:
            for ctx, w in self._events(s):
                d = counts[n].setdefault(ctx, {})
                d[w] = d.get(w, 0) + 1
        # continuation counts for lower orders: N1+(. ctx' w) over distinct first tokens
        for m in range(n - 1, 0, -1):
            seen = {}
            for ctx, row in counts[m + 1].items():
                for w in row:
                    key = (ctx[1:], w)
                    seen.setdefault(key, set()).add(ctx[0])
            for (ctx, w), firsts in seen.items():
                counts[m].setdefault(ctx, {})[w] = len(firsts)
        self.counts = counts
        self.D = [None]
        for m in range(1, n + 1):
            cc = [0, 0, 0, 0, 0]
            for row in counts[m].values():
                for c in row.values():
                    if c <= 4:
                        cc[c] += 1
            self.D.append(_kn_discounts(cc[1], cc[2], cc[3], cc[4]))
        self.tot = [None] + [{ctx: sum(row.values()) for ctx, row in counts[m].items()}
                             for m in range(1, n + 1)]
        self.gam = [None]
        for m in range(1, n + 1):
            D1, D2, D3 = self.D[m]
            g = {}
            for ctx, row in counts[m].items():
                k1 = sum(1 for c in row.values() if c == 1)
                k2 = sum(1 for c in row.values() if c == 2)
                k3 = sum(1 for c in row.values() if c >= 3)
                g[ctx] = (D1 * k1 + D2 * k2 + D3 * k3) / self.tot[m][ctx]
            self.gam.append(g)
        return self

    def _disc(self, m, c):
        if c <= 0:
            return 0.0
        D1, D2, D3 = self.D[m]
        return D1 if c == 1 else (D2 if c == 2 else D3)

    def prob(self, ctx, w):
        """P(w | ctx), ctx a tuple of the n - 1 previous tokens (BOS = -1 for pads)."""
        p = 1.0 / (self.V + 1)
        for m in range(1, self.n + 1):
            c_ctx = tuple(ctx[len(ctx) - (m - 1):]) if m > 1 else ()
            tot = self.tot[m].get(c_ctx, 0)
            if tot == 0:
                continue
            c = self.counts[m][c_ctx].get(w, 0)
            p = max(c - self._disc(m, c), 0.0) / tot + self.gam[m][c_ctx] * p
        return p

    def loglik(self, s):
        return float(sum(math.log(self.prob(ctx, w)) for ctx, w in self._events(s)))


def _kn_discounts(n1, n2, n3, n4):
    if n1 > 0 and n2 > 0:
        Y = n1 / (n1 + 2.0 * n2)
        D1 = 1.0 - 2.0 * Y * n2 / n1
        D2 = 2.0 - 3.0 * Y * n3 / n2
        D3 = 3.0 - 4.0 * Y * n4 / n3 if n3 > 0 else 1.5
    else:
        D1, D2, D3 = 0.5, 1.0, 1.5
    return (min(max(D1, 0.05), 0.95), min(max(D2, 0.05), 1.95), min(max(D3, 0.05), 2.95))


def kn_sample(m, rng, max_len=12):
    """One sequence from a fitted KNGram; empty or longer-than-max_len draws are rejected."""
    while True:
        ctx = [m.BOS] * (m.n - 1)
        out = []
        while True:
            probs = np.array([m.prob(tuple(ctx), w) for w in range(m.V + 1)])
            w = _draw(probs, rng)
            if w == m.EOS:
                break
            out.append(w)
            if len(out) > max_len:
                break
            if m.n > 1:
                ctx = ctx[1:] + [w]
        if out and len(out) <= max_len:
            return tuple(out)


# ============================================================================ HMM (PFA)

class HMM:
    """K hidden states. pi[k] start; B[k, v] emission; T[k, j] next state, T[k, K] = end.
    Each emission is followed by either a move or the end, so every sequence has length >= 1."""

    def __init__(self, K, V, rng=None, pseudo=0.1):
        self.K, self.V, self.pseudo = int(K), int(V), float(pseudo)
        rng = rng if rng is not None else np.random.default_rng(0)
        self.pi = _rownorm(rng.random(self.K) + 0.5)
        self.B = _rownorm(rng.random((self.K, self.V)) + 0.5)
        self.T = _rownorm(rng.random((self.K, self.K + 1)) + 0.5)

    def _forward(self, s):
        K = self.K
        L = len(s)
        a = np.zeros((L, K))
        logc = 0.0
        v = self.pi * self.B[:, s[0]]
        for t in range(L):
            if t > 0:
                v = (a[t - 1] @ self.T[:, :K]) * self.B[:, s[t]]
            z = v.sum()
            a[t] = v / z
            logc += math.log(z)
        end = float(a[L - 1] @ self.T[:, K])
        return a, logc, end

    def loglik(self, s):
        _, logc, end = self._forward(s)
        return logc + math.log(end)

    def em_step(self, seqs):
        """One Baum-Welch step with additive pseudo-counts. Returns the training log-lik under
        the parameters BEFORE the step."""
        K = self.K
        cpi = np.zeros(K)
        cB = np.zeros((K, self.V))
        cT = np.zeros((K, K + 1))
        total = 0.0
        TK = self.T[:, :K]
        for s in seqs:
            L = len(s)
            a, logc, end = self._forward(s)
            total += logc + math.log(end)
            # scaled backward: b[t] proportional to P(rest | state t)
            b = np.zeros((L, K))
            b[L - 1] = self.T[:, K] / end
            for t in range(L - 2, -1, -1):
                v = TK @ (self.B[:, s[t + 1]] * b[t + 1])
                b[t] = v / v.sum()
            # posteriors from normalised products (scales cancel within each t)
            g = a * b
            g = g / g.sum(axis=1, keepdims=True)
            cpi += g[0]
            for t in range(L):
                cB[:, s[t]] += g[t]
            for t in range(L - 1):
                xi = a[t][:, None] * TK * (self.B[:, s[t + 1]] * b[t + 1])[None, :]
                cT[:, :K] += xi / xi.sum()
            cT[:, K] += g[L - 1]
        p = self.pseudo
        self.pi = _rownorm(cpi + p)
        self.B = _rownorm(cB + p)
        self.T = _rownorm(cT + p)
        return total

    def fit(self, seqs, iters=200, tol=1e-6):
        last = -np.inf
        for _ in range(iters):
            ll = self.em_step(seqs)
            if abs(ll - last) < tol * max(1.0, abs(ll)):
                break
            last = ll
        return self

    def sample(self, rng, max_len=200):
        K = self.K
        k = _draw(self.pi, rng)
        out = []
        while True:
            out.append(_draw(self.B[k], rng))
            nxt = _draw(self.T[k], rng)
            if nxt == K or len(out) >= max_len:
                return tuple(out)
            k = nxt


def hmm_sample(h, rng, max_len=12):
    """One sequence from an HMM; draws longer than max_len are rejected (not truncated)."""
    while True:
        s = h.sample(rng, max_len=max_len + 1)
        if len(s) <= max_len:
            return s


def n_params_hmm(K, V):
    return (K - 1) + K * (V - 1) + K * K


# ============================================================================ PCFG (CNF)

class PCFG:
    """N non-terminals, start = 0. R[a, b, c] = P(a -> b c), E[a, v] = P(a -> v); for every a,
    R[a].sum() + E[a].sum() = 1. Inside-outside EM with additive pseudo-count `pseudo` on every
    rule (pseudo > 0 keeps the grammar fully connected)."""

    def __init__(self, N, V, rng=None, pseudo=0.1, p_term=0.5):
        self.N, self.V, self.pseudo = int(N), int(V), float(pseudo)
        rng = rng if rng is not None else np.random.default_rng(0)
        R = rng.random((self.N, self.N, self.N)) + 0.5
        E = rng.random((self.N, self.V)) + 0.5
        R = R / R.reshape(self.N, -1).sum(axis=1)[:, None, None] * (1.0 - p_term)
        E = E / E.sum(axis=1, keepdims=True) * p_term
        self.R, self.E = R, E

    def inside(self, s):
        N, L = self.N, len(s)
        ins = np.zeros((L, L + 1, N))
        R2 = self.R.reshape(N, N * N)
        for i in range(L):
            ins[i, i + 1] = self.E[:, s[i]]
        for span in range(2, L + 1):
            for i in range(0, L - span + 1):
                j = i + span
                acc = np.zeros((N, N))
                for k in range(i + 1, j):
                    acc += np.outer(ins[i, k], ins[k, j])
                ins[i, j] = R2 @ acc.reshape(-1)
        return ins

    def loglik(self, s):
        z = self.inside(s)[0, len(s), 0]
        return math.log(z) if z > 0 else -np.inf

    def em_step(self, seqs):
        N = self.N
        cR = np.zeros_like(self.R)
        cE = np.zeros_like(self.E)
        total = 0.0
        for s in seqs:
            L = len(s)
            ins = self.inside(s)
            Z = ins[0, L, 0]
            total += math.log(Z)
            out = np.zeros_like(ins)
            out[0, L, 0] = 1.0
            for span in range(L, 1, -1):
                for i in range(0, L - span + 1):
                    j = i + span
                    o = out[i, j]
                    if not o.any():
                        continue
                    M = np.tensordot(o, self.R, axes=(0, 0))  # (N, N): sum_a o_a R[a,b,c]
                    acc = np.zeros((N, N))
                    for k in range(i + 1, j):
                        out[i, k] += M @ ins[k, j]
                        out[k, j] += ins[i, k] @ M
                        acc += np.outer(ins[i, k], ins[k, j])
                    cR += o[:, None, None] * self.R * acc[None, :, :] / Z
            for i in range(L):
                cE[:, s[i]] += out[i, i + 1] * self.E[:, s[i]] / Z
        p = self.pseudo
        cR += p
        cE += p
        tot = cR.reshape(N, -1).sum(axis=1) + cE.sum(axis=1)
        self.R = cR / tot[:, None, None]
        self.E = cE / tot[:, None]
        return total

    def fit(self, seqs, iters=300, tol=1e-6):
        last = -np.inf
        for _ in range(iters):
            ll = self.em_step(seqs)
            if abs(ll - last) < tol * max(1.0, abs(ll)):
                break
            last = ll
        return self


def pcfg_sample(g, rng, max_len=12):
    """One sequence from a PCFG by top-down expansion (left child first). Every pending
    non-terminal yields at least one call, so a draw is rejected as soon as pending + emitted
    exceeds max_len."""
    flatR = g.R.reshape(g.N, -1)
    while True:
        out = []
        stack = [0]
        ok = True
        while stack:
            a = stack.pop()
            if rng.random() < g.E[a].sum():
                out.append(_draw(g.E[a], rng))
            else:
                b, c = divmod(_draw(flatR[a], rng), g.N)
                stack.append(c)
                stack.append(b)
            if len(stack) + len(out) > max_len:
                ok = False
                break
        if ok:
            return tuple(out)


def pcfg_from_hmm(hmm):
    """The CNF PCFG that generates exactly the HMM's distribution (right-branching only, so it
    is regular). Non-terminals: 0 = start S, 1..K = A_k ('emit from state k, then continue'),
    K+1..K+V = U_v (-> v with probability 1).
      A_k -> v        : B[k, v] * T[k, end]       A_k -> U_v A_j : B[k, v] * T[k, j]
      S   -> v        : sum_k pi_k B[k, v] T[k, end]
      S   -> U_v A_j  : sum_k pi_k B[k, v] T[k, j]"""
    K, V = hmm.K, hmm.V
    N = 1 + K + V
    g = PCFG(N, V)
    R = np.zeros((N, N, N))
    E = np.zeros((N, V))
    for k in range(K):
        for v in range(V):
            E[1 + k, v] = hmm.B[k, v] * hmm.T[k, K]
            for j in range(K):
                R[1 + k, 1 + K + v, 1 + j] = hmm.B[k, v] * hmm.T[k, j]
    E[0] = hmm.pi @ E[1:1 + K]
    R[0] = np.tensordot(hmm.pi, R[1:1 + K], axes=(0, 0))
    for v in range(V):
        E[1 + K + v, v] = 1.0
    g.R, g.E = R, E
    return g


# ============================================================================ batched (by length)

def compress(seqs):
    """Distinct sequences grouped by length: list of (S, w), S an int array (B, L), w the
    multiplicities. Groups sorted by L and rows sorted, so the result is deterministic."""
    counts = {}
    for s in seqs:
        t = tuple(int(x) for x in s)
        if not t:
            raise ValueError("empty sequence")
        counts[t] = counts.get(t, 0) + 1
    by_len = {}
    for t in sorted(counts):
        by_len.setdefault(len(t), []).append(t)
    groups = []
    for L in sorted(by_len):
        rows = by_len[L]
        S = np.array(rows, dtype=np.int64).reshape(len(rows), L)
        w = np.array([counts[t] for t in rows], dtype=float)
        groups.append((S, w))
    return groups


def pcfg_inside_batch(g, S):
    B, L = S.shape
    N = g.N
    R2T = g.R.reshape(N, N * N).T
    ins = np.zeros((B, L, L + 1, N))
    for i in range(L):
        ins[:, i, i + 1, :] = g.E[:, S[:, i]].T
    for span in range(2, L + 1):
        for i in range(L - span + 1):
            j = i + span
            acc = np.zeros((B, N, N))
            for k in range(i + 1, j):
                acc += ins[:, i, k, :, None] * ins[:, k, j, None, :]
            ins[:, i, j, :] = acc.reshape(B, N * N) @ R2T
    return ins


def pcfg_loglik_batch(g, S):
    Z = pcfg_inside_batch(g, S)[:, 0, S.shape[1], 0]
    with np.errstate(divide="ignore"):
        return np.log(Z)


def pcfg_em_batch(g, groups):
    """Same update as PCFG.em_step on the expanded list of sequences."""
    N, V = g.N, g.V
    R2 = g.R.reshape(N, N * N)
    raw = np.zeros((N, N * N))
    cET = np.zeros((V, N))
    total = 0.0
    for S, w in groups:
        B, L = S.shape
        ins = pcfg_inside_batch(g, S)
        Z = ins[:, 0, L, 0]
        total += float(w @ np.log(Z))
        f = w / Z
        out = np.zeros_like(ins)
        out[:, 0, L, 0] = 1.0
        for span in range(L, 1, -1):
            for i in range(L - span + 1):
                j = i + span
                o = out[:, i, j, :]
                M = (o @ R2).reshape(B, N, N)
                acc = np.zeros((B, N, N))
                for k in range(i + 1, j):
                    out[:, i, k, :] += np.einsum("nbc,nc->nb", M, ins[:, k, j, :])
                    out[:, k, j, :] += np.einsum("nb,nbc->nc", ins[:, i, k, :], M)
                    acc += ins[:, i, k, :, None] * ins[:, k, j, None, :]
                raw += (o * f[:, None]).T @ acc.reshape(B, N * N)
        for i in range(L):
            contrib = out[:, i, i + 1, :] * g.E[:, S[:, i]].T * f[:, None]
            np.add.at(cET, S[:, i], contrib)
    p = g.pseudo
    cR = raw * R2 + p
    cE = cET.T + p
    tot = cR.sum(axis=1) + cE.sum(axis=1)
    g.R = (cR / tot[:, None]).reshape(N, N, N)
    g.E = cE / tot[:, None]
    return total


def hmm_forward_batch(h, S):
    """Unscaled forward (fine for the short sequences used here: max_len 12)."""
    B, L = S.shape
    K = h.K
    TK = h.T[:, :K]
    a = np.zeros((B, L, K))
    a[:, 0, :] = h.pi[None, :] * h.B[:, S[:, 0]].T
    for t in range(1, L):
        a[:, t, :] = (a[:, t - 1, :] @ TK) * h.B[:, S[:, t]].T
    Z = a[:, L - 1, :] @ h.T[:, K]
    return a, Z


def hmm_loglik_batch(h, S):
    _, Z = hmm_forward_batch(h, S)
    with np.errstate(divide="ignore"):
        return np.log(Z)


def hmm_em_batch(h, groups):
    """Same update as HMM.em_step on the expanded list of sequences."""
    K, V = h.K, h.V
    TK = h.T[:, :K]
    cpi = np.zeros(K)
    cBT = np.zeros((V, K))
    cT = np.zeros((K, K + 1))
    total = 0.0
    for S, w in groups:
        B, L = S.shape
        a, Z = hmm_forward_batch(h, S)
        total += float(w @ np.log(Z))
        f = w / Z
        b = np.zeros((B, L, K))
        b[:, L - 1, :] = h.T[:, K][None, :]
        for t in range(L - 2, -1, -1):
            b[:, t, :] = (h.B[:, S[:, t + 1]].T * b[:, t + 1, :]) @ TK.T
        g = a * b * f[:, None, None]
        cpi += g[:, 0, :].sum(axis=0)
        for t in range(L):
            np.add.at(cBT, S[:, t], g[:, t, :])
        for t in range(L - 1):
            y = h.B[:, S[:, t + 1]].T * b[:, t + 1, :] * f[:, None]
            cT[:, :K] += (a[:, t, :].T @ y) * TK
        cT[:, K] += g[:, L - 1, :].sum(axis=0)
    p = h.pseudo
    h.pi = _rownorm(cpi + p)
    h.B = _rownorm(cBT.T + p)
    h.T = _rownorm(cT + p)
    return total


def fit_em(model, step, groups, iters=300, tol=1e-6):
    """Run `step` until the relative change of the training log-lik is below tol. Returns the
    number of steps taken."""
    last = None
    n = 0
    for n in range(1, iters + 1):
        ll = step(model, groups)
        if last is not None and abs(ll - last) < tol * max(1.0, abs(ll)):
            break
        last = ll
    return n


def batch_ll(fn, model, seqs):
    """Per-sequence log-likelihoods of `seqs` (list order) from a batched function."""
    table = {}
    for S, _ in compress(seqs):
        ll = fn(model, S)
        for row, v in zip(S, ll):
            table[tuple(int(x) for x in row)] = float(v)
    return np.array([table[tuple(int(x) for x in s)] for s in seqs])


# ============================================================================ helpers

def _rownorm(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x / x.sum()
    return x / x.sum(axis=1, keepdims=True)


def _draw(p, rng):
    c = np.cumsum(p)
    return int(min(np.searchsorted(c, rng.random() * c[-1], side="right"), len(p) - 1))
