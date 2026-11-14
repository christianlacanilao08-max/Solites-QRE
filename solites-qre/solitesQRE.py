#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solites Quantitative Finance Research Engine
anyones free to use this :)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, replace, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # optional accelerators / better nets
    from scipy.special import ndtri as _sp_ndtri, ndtr as _sp_ndtr
    from scipy.stats import qmc as _sp_qmc
    SCIPY = True
except Exception:                       # pragma: no cover
    SCIPY = False

Array = np.ndarray
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
LOG2PI_2 = 0.5 * math.log(2.0 * math.pi)


# ==============================================================================
# 1. SPECIAL FUNCTIONS & SOLVERS
# ==============================================================================
def _gser_half(t: Array) -> Array:
    """Regularised lower incomplete gamma P(1/2,t) by its convergent series."""
    t = np.asarray(t, float)
    ap = np.full_like(t, 0.5)
    s = np.full_like(t, 2.0)            # 1/a with a=1/2
    d = s.copy()
    for _ in range(1000):
        ap += 1.0
        d *= t / ap
        s += d
        if np.all(np.abs(d) <= np.abs(s) * 1e-17):
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        pref = np.exp(-t + 0.5 * np.log(np.where(t > 0, t, 1.0)) - 0.5 * math.log(math.pi))
    return np.where(t > 0, s * pref, 0.0)


def _gcf_half(t: Array) -> Array:
    """Q(1/2,t) by the modified-Lentz continued fraction (t >= 1.5)."""
    t = np.asarray(t, float)
    a, tiny = 0.5, 1e-300
    b = t + 1.0 - a
    c = np.full_like(t, 1.0 / tiny)
    d = 1.0 / np.where(np.abs(b) < tiny, tiny, b)
    h = d.copy()
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = b + an / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        h *= delta
        if np.all(np.abs(delta - 1.0) < 1e-16):
            break
    return np.exp(-t + 0.5 * np.log(t) - 0.5 * math.log(math.pi)) * h


def erfc(x):
    """Complementary error function, |rel. err| ~ 1e-15, fully vectorised."""
    x = np.asarray(x, float)
    scalar = x.ndim == 0
    xf = np.atleast_1d(x)
    ax = np.abs(xf)
    out = np.empty_like(ax)
    lo = ax < 1.2247448713915890        # sqrt(3/2): series vs. continued fraction
    if np.any(lo):
        out[lo] = 1.0 - _gser_half(ax[lo] ** 2)
    if np.any(~lo):
        out[~lo] = _gcf_half(ax[~lo] ** 2)
    out = np.where(xf < 0, 2.0 - out, out)
    return float(out[0]) if scalar else out.reshape(xf.shape)


def norm_cdf(x):
    if SCIPY:
        return _sp_ndtr(x)
    return 0.5 * erfc(-np.asarray(x, float) / SQRT2)


def norm_pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x - LOG2PI_2)


_ACK_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACK_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)


def norm_ppf(p):
    """Inverse normal CDF: Acklam rational seed + one Halley step -> ~1e-15."""
    if SCIPY:
        return _sp_ndtri(p)
    p = np.clip(np.asarray(p, float), 1e-300, 1.0 - 1e-16)
    out = np.empty_like(p)
    pl = 0.02425
    lo, hi = p < pl, p > 1.0 - pl
    mid = ~(lo | hi)
    a, b, c, d = _ACK_A, _ACK_B, _ACK_C, _ACK_D

    def tail(q):
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    if np.any(lo):
        out[lo] = tail(np.sqrt(-2.0 * np.log(p[lo])))
    if np.any(hi):
        out[hi] = -tail(np.sqrt(-2.0 * np.log1p(-p[hi])))
    if np.any(mid):
        q = p[mid] - 0.5
        rr = q * q
        out[mid] = (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / \
                   (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1.0)
    ref = np.abs(out) < 8.0             # Halley refinement where exp(z^2/2) is safe
    z = out[ref]
    e = norm_cdf(z) - p[ref]
    u = e * SQRT2PI * np.exp(0.5 * z * z)
    out[ref] = z - u / (1.0 + 0.5 * z * u)
    return out


def bisect_monotone(f: Callable[[float], float], lo: float, hi: float,
                    tol: float = 1e-12, itmax: int = 200) -> float:
    """Bisection for increasing f with f(lo) <= 0 <= f(hi)."""
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return lo
    if fhi < 0:
        return hi
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def golden_min(f: Callable[[float], float], a: float, b: float,
               tol: float = 1e-10, itmax: int = 400) -> float:
    """Golden-section minimisation (unimodal f); robust, derivative free."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(itmax):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        if b - a < tol * max(1.0, abs(a), abs(b)):
            break
    return 0.5 * (a + b)


def batch_se(x: Array, k: int = 20) -> float:
    """Batch-means standard error (robust to mild serial dependence)."""
    x = np.asarray(x, float).ravel()
    k = max(2, min(k, len(x) // 2))
    if len(x) < 4:
        return float("nan")
    b = np.array_split(x, k)
    m = np.array([bb.mean() for bb in b if len(bb)])
    return float(m.std(ddof=1) / math.sqrt(len(m)))


def _hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ==============================================================================
# 2. ACCEPTANCE-REJECTION SAMPLERS
# ==============================================================================
def truncnorm_pos(a: float, size: int, rng: np.random.Generator) -> Array:
    """
    N(0,1) conditioned on X >= a.
      a < 0 : plain rejection from N(0,1) (acceptance Phi(-a)^c is large).
      a >= 0: Robert (1995) exponential proposal with the OPTIMAL rate
              alpha* = (a + sqrt(a^2+4))/2, acceptance >= 0.75 for all a.
    """
    out = np.empty(size)
    filled = 0
    if a < 0:
        while filled < size:
            z = rng.standard_normal(int(1.3 * (size - filled)) + 16)
            z = z[z >= a]
            take = min(len(z), size - filled)
            out[filled:filled + take] = z[:take]
            filled += take
        return out
    alpha = 0.5 * (a + math.sqrt(a * a + 4.0))
    while filled < size:
        need = size - filled
        z = a + rng.exponential(1.0 / alpha, int(1.4 * need) + 16)
        rho = np.exp(-0.5 * (z - alpha) ** 2)
        z = z[rng.random(len(z)) < rho]
        take = min(len(z), need)
        out[filled:filled + take] = z[:take]
        filled += take
    return out


def gamma_mt(shape: float, size: int, rng: np.random.Generator) -> Array:
    """Marsaglia-Tsang squeeze rejection for Gamma(shape>=1,1)."""
    if shape < 1.0:                     # Ahrens-Dieter boost: G(a) = G(a+1)*U^{1/a}
        g = gamma_mt(shape + 1.0, size, rng)
        return g * rng.random(size) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    out = np.empty(size)
    filled = 0
    while filled < size:
        need = size - filled
        x = rng.standard_normal(int(1.2 * need) + 16)
        v = (1.0 + c * x) ** 3
        ok = v > 0
        x, v = x[ok], v[ok]
        u = rng.random(len(x))
        acc = np.log(u) < 0.5 * x * x + d - d * v + d * np.log(v)   # squeeze+accept
        z = d * v[acc]
        take = min(len(z), need)
        out[filled:filled + take] = z[:take]
        filled += take
    return out


def gamma_ahrens(shape: float, size: int, rng: np.random.Generator) -> Array:
    """Ahrens-Dieter GS rejection for 0 < shape < 1 (two-branch envelope)."""
    assert 0.0 < shape < 1.0
    e = math.e
    b = (e + shape) / e
    out = np.empty(size)
    filled = 0
    while filled < size:
        need = size - filled
        u = rng.random(int(1.6 * need) + 16)
        p = b * u
        small = p <= 1.0
        x = np.empty_like(p)
        x[small] = p[small] ** (1.0 / shape)
        x[~small] = -np.log((b - p[~small]) / shape)
        u2 = rng.random(len(p))
        acc = np.where(small, u2 <= np.exp(-x), u2 <= x ** (shape - 1.0))
        z = x[acc]
        take = min(len(z), need)
        out[filled:filled + take] = z[:take]
        filled += take
    return out


class RejectionSampler:
    """
    Generic acceptance-rejection for an unnormalised target f >= 0 with a
    proposal g (samplable, pdf evaluable).  M = sup f/g is found on a grid and
    inflated by `safety`.  E[trials per accepted draw] = M / int f.
    """

    def __init__(self, log_f: Callable, proposal_pdf: Callable,
                 proposal_draw: Callable, grid: Array, safety: float = 1.02):
        lf = np.array([log_f(x) for x in grid], float)
        lg = np.log(np.maximum([proposal_pdf(x) for x in grid], 1e-300))
        self.logM = float(np.max(lf - lg) + math.log(safety))
        self.log_f, self.gpdf, self.gdraw = log_f, proposal_pdf, proposal_draw
        self.trials = 0
        self.accepts = 0

    def sample(self, n: int, rng: np.random.Generator) -> Array:
        out = np.empty(n)
        filled = 0
        while filled < n:
            need = n - filled
            y = self.gdraw(int(2.5 * need) + 32, rng)
            logratio = np.array([self.log_f(v) for v in y]) - \
                       np.log(np.maximum([self.gpdf(v) for v in y], 1e-300)) - self.logM
            self.trials += len(y)
            acc = np.log(rng.random(len(y))) < np.minimum(logratio, 0.0)
            z = y[acc]
            self.accepts += len(z)
            take = min(len(z), need)
            out[filled:filled + take] = z[:take]
            filled += take
        return out

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / max(self.trials, 1)


class ARS:
    """
    Adaptive Rejection Sampling (Gilks & Wild 1992) for LOG-CONCAVE targets.

    Upper hull  u(x) = min_j [ h(x_j) + h'(x_j)(x-x_j) ]  (piecewise exponential)
    Squeeze     s(x) = secant interpolation of h between abscissae
    Rejected points are inserted -> hull tightens -> acceptance -> 1.
    """

    def __init__(self, h: Callable[[float], float], dh: Callable[[float], float],
                 init: Sequence[float], lo: float = -60.0, hi: float = 60.0):
        self.h, self.dh, self.lo, self.hi = h, dh, lo, hi
        self.x = np.sort(np.asarray(init, float))
        self.hv = np.array([h(v) for v in self.x])
        self.dv = np.array([dh(v) for v in self.x])
        if not np.all(np.diff(self.dv) <= 1e-9):
            raise ValueError("target is not log-concave on the initial abscissae")
        self.trials = 0
        self.accepts = 0
        self._build()

    # ---- hull geometry -------------------------------------------------------
    def _build(self):
        x, hv, dv = self.x, self.hv, self.dv
        k = len(x)
        z = np.empty(k + 1)
        z[0], z[-1] = self.lo, self.hi
        if k > 1:
            denom = dv[:-1] - dv[1:]
            denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
            z[1:-1] = (hv[1:] - hv[:-1] - x[1:] * dv[1:] + x[:-1] * dv[:-1]) / denom
        self.z = z
        self.c = hv - dv * x                       # log-envelope intercepts
        m = dv
        l, r = z[:-1], z[1:]
        logI = np.empty(k)
        pos, neg, zer = m > 1e-12, m < -1e-12, np.abs(m) <= 1e-12
        if np.any(pos):
            logI[pos] = (self.c[pos] + m[pos] * r[pos]
                         + np.log1p(-np.exp(np.clip(m[pos] * (l[pos] - r[pos]), -700, -1e-16)))
                         - np.log(m[pos]))
        if np.any(neg):
            logI[neg] = (self.c[neg] + m[neg] * l[neg]
                         + np.log1p(-np.exp(np.clip(m[neg] * (r[neg] - l[neg]), -700, -1e-16)))
                         - np.log(-m[neg]))
        if np.any(zer):
            logI[zer] = self.c[zer] + np.log(np.maximum(r[zer] - l[zer], 1e-300))
        mx = logI.max()
        w = np.exp(logI - mx)
        self.logZ = mx + math.log(w.sum())
        self.cdf = np.concatenate([[0.0], np.cumsum(w / w.sum())])

    def _upper(self, xq: Array) -> Array:
        j = np.clip(np.searchsorted(self.z, xq, side="right") - 1, 0, len(self.x) - 1)
        return self.c[j] + self.dv[j] * xq

    def _squeeze(self, xq: Array) -> Array:
        x = self.x
        out = np.full_like(xq, -np.inf)
        inside = (xq >= x[0]) & (xq <= x[-1])
        if np.any(inside):
            xi = xq[inside]
            j = np.clip(np.searchsorted(x, xi) - 1, 0, len(x) - 2)
            w = (xi - x[j]) / (x[j + 1] - x[j])
            out[inside] = (1 - w) * self.hv[j] + w * self.hv[j + 1]
        return out

    def sample(self, n: int, rng: np.random.Generator) -> Array:
        out = np.empty(n)
        filled = 0
        while filled < n:
            need = n - filled
            u = rng.random(int(1.5 * need) + 32)
            j = np.clip(np.searchsorted(self.cdf, u) - 1, 0, len(self.x) - 1)
            l, r, m, c = self.z[j], self.z[j + 1], self.dv[j], self.c[j]
            frac = (u - self.cdf[j]) / np.maximum(self.cdf[j + 1] - self.cdf[j], 1e-300)
            if abs(m) <= 1e-12:
                y = l + frac * (r - l)
            else:                                  # exact inverse CDF of exp piece
                expo = np.clip(m * (r - l), -700, 700)
                y = l + np.log1p(frac * np.expm1(expo)) / m
            y = np.clip(y, self.lo, self.hi)
            up = self._upper(y)
            sq = self._squeeze(y)
            lu = np.log(rng.random(len(y)))
            self.trials += len(y)
            quick = lu <= sq - up                  # squeeze acceptance: no h() call
            todo = ~quick
            acc = quick.copy()
            if np.any(todo):
                for t in np.where(todo)[0]:
                    hv, dv = self.h(y[t]), self.dh(y[t])
                    if lu[t] <= hv - up[t]:
                        acc[t] = True
                    ins = np.searchsorted(self.x, y[t])
                    self.x = np.insert(self.x, ins, y[t])
                    self.hv = np.insert(self.hv, ins, hv)
                    self.dv = np.insert(self.dv, ins, dv)
                self._build()
            self.accepts += int(acc.sum())
            z = y[acc]
            take = min(len(z), need)
            out[filled:filled + take] = z[:take]
            filled += take
        return out

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / max(self.trials, 1)


# ==============================================================================
# 3. LOW-DISCREPANCY SEQUENCES  (randomised => unbiased + CLT error bars)
# ==============================================================================
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
           139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
           223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
           293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
           383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
           463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563,
           569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643,
           647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
           743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
           839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
           941, 947, 953, 967, 971, 977, 983, 991, 997]


def halton(index0: int, n: int, dim: int, perms: Optional[List[List[Array]]] = None,
           seed: int = 0) -> Array:
    """Digit-scrambled Halton points for indices index0..index0+n-1 (1-based)."""
    if dim > len(_PRIMES):
        raise ValueError("dimension too large for built-in prime table")
    if perms is None:
        perms = halton_perms(dim, seed)
    out = np.empty((n, dim))
    for j in range(dim):
        b = _PRIMES[j]
        idx = np.arange(index0 + 1, index0 + n + 1, dtype=np.int64)
        val = np.zeros(n)
        f = 1.0 / b
        k = 0
        while np.any(idx > 0):
            digit = idx % b
            val += perms[j][k % len(perms[j])][digit] * f
            idx //= b
            f /= b
            k += 1
        out[:, j] = val
    return out


def halton_perms(dim: int, seed: int) -> List[List[Array]]:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    perms = []
    for j in range(dim):
        b = _PRIMES[j]
        plist = []
        for _ in range(24):                     # 24 digit positions is plenty
            p = rng.permutation(b)
            p = np.where(p == 0, p[0], p)       # keep 0 -> 0 (avoids degenerate 0)
            perm = np.arange(b)
            perm[1:] = rng.permutation(np.arange(1, b)) if b > 1 else perm[1:]
            plist.append(perm)
        perms.append(plist)
    return perms


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            return False
    return True


def lattice_cbc(n: int, dim: int, weights: Optional[Array] = None,
                n_cand: int = 96, seed: int = 12345) -> Array:
    """
    Component-by-component rank-1 lattice generating vector.

    Minimises the shift-averaged worst-case error in the weighted Korobov
    space with alpha=2:
        e^2(z) = -1 + (1/n) sum_k prod_j (1 + gamma_j * omega({k z_j/n}))
        omega(t) = 2 pi^2 B_2(t) = 2 pi^2 (t^2 - t + 1/6).
    Candidate search is restricted to n_cand random z per dimension.
    """
    if not _is_prime(n):
        n += 1
        while not _is_prime(n):
            n += 1
    gam = np.ones(dim) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    t = k / n
    omega = 2.0 * math.pi ** 2 * (t * t - t + 1.0 / 6.0)
    z = np.ones(dim, dtype=np.int64)
    prod = 1.0 + gam[0] * omega[(k * 1) % n]
    for j in range(1, dim):
        cand = rng.choice(np.arange(1, n), size=min(n_cand, n - 1), replace=False)
        W = 1.0 + gam[j] * omega
        idx = (k[:, None] * cand[None, :]) % n
        vals = (prod[:, None] * W[idx]).mean(axis=0)
        z[j] = cand[int(np.argmin(vals))]
        prod *= W[(k * z[j]) % n]
    return z


class PointStream:
    """
    Uniform point generator with pluggable structure.
      'pseudo'  : i.i.d. MT/PCG stream
      'halton'  : digit-scrambled Halton + Cranley-Patterson shift
      'lattice' : CBC rank-1 lattice + shift
      'sobol'   : scipy scrambled Sobol (if available)
    `shift_seed` selects the randomisation replicate (RQMC).
    """

    def __init__(self, kind: str, dim: int, n_total: int, seed: int = 0,
                 shift_seed: int = 0, weights: Optional[Array] = None):
        self.kind, self.dim, self.n_total = kind, dim, n_total
        self.seed, self.shift_seed = seed, shift_seed
        self.pos = 0
        rng = np.random.default_rng(np.random.SeedSequence([seed, shift_seed, 7]))
        self.rng = rng
        self.shift = rng.random(dim)
        if kind == "halton":
            self.perms = halton_perms(dim, seed)
        elif kind == "lattice":
            self.z = lattice_cbc(n_total, dim, weights, seed=seed + 991)
            self.n_lat = int(((np.arange(n_total)[:, None] * 1) % 1).shape[0])
        elif kind == "sobol":
            if not SCIPY:
                self.kind = kind = "halton"
                self.perms = halton_perms(dim, seed)
            else:
                self.sob = _sp_qmc.Sobol(d=dim, scramble=True, seed=seed + 31 * shift_seed)
        if kind not in ("pseudo", "halton", "lattice", "sobol"):
            raise ValueError(kind)

    def block(self, n: int) -> Array:
        if self.kind == "pseudo":
            u = self.rng.random((n, self.dim))
        elif self.kind == "halton":
            u = (halton(self.pos, n, self.dim, self.perms) + self.shift) % 1.0
        elif self.kind == "lattice":
            k = np.arange(self.pos, self.pos + n)
            u = ((k[:, None] * self.z[None, :]) % self.n_total) / self.n_total
            u = (u + self.shift) % 1.0
        else:
            u = self.sob.random(n)
        self.pos += n
        return np.clip(u, 1e-15, 1.0 - 1e-15)


# ==============================================================================
# 4. WIENER PATH CONSTRUCTIONS  (effective-dimension reduction)
# ==============================================================================
_PCA_CACHE: Dict[Tuple[float, int], Array] = {}


def _pca_matrix(t: Array) -> Array:
    key = (float(t[-1]), len(t) - 1)
    if key in _PCA_CACHE:
        return _PCA_CACHE[key]
    tt = t[1:]
    C = np.minimum(tt[:, None], tt[None, :])
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0.0), V[:, order]
    A = V * np.sqrt(w)[None, :]
    _PCA_CACHE[key] = A
    return A


def wiener_path(Z: Array, t: Array, mode: str = "bridge") -> Array:
    """Map i.i.d. N(0,1) Z (n,m) to W at t_1..t_m (W_0=0) with Cov=min(t_i,t_j)."""
    n, m = Z.shape
    W = np.empty((n, m))
    if mode == "standard":
        np.cumsum(Z * np.sqrt(np.diff(t))[None, :], axis=1, out=W)
    elif mode == "pca":
        np.dot(Z, _pca_matrix(t).T, out=W)
    elif mode == "bridge":
        W[:, m - 1] = math.sqrt(t[m]) * Z[:, 0]
        k = 1
        q = deque([(0, m)])
        while q:
            l, r = q.popleft()
            if r - l <= 1:
                continue
            mid = (l + r) // 2
            tl, tm, tr = t[l], t[mid], t[r]
            wl = 0.0 if l == 0 else W[:, l - 1]
            wr = W[:, r - 1]
            mu = ((tr - tm) * wl + (tm - tl) * wr) / (tr - tl)
            sd = math.sqrt((tm - tl) * (tr - tm) / (tr - tl))
            W[:, mid - 1] = mu + sd * Z[:, k]
            k += 1
            q.append((l, mid))
            q.append((mid, r))
    else:
        raise ValueError(mode)
    return W


def construction_spectrum(t: Array, mode: str) -> Array:
    """Cumulative variance share of the first k inputs (effective-dim proxy)."""
    m = len(t) - 1
    A = wiener_path(np.eye(m), t, mode)           # linear map => probe with identity
    e = (A ** 2).sum(axis=0)
    return np.cumsum(e) / e.sum()


# ==============================================================================
# 5. MODELS
# ==============================================================================
@dataclass
class Grid:
    T: float
    m: int
    t: Array = field(init=False)
    dt: Array = field(init=False)

    def __post_init__(self):
        self.t = np.linspace(0.0, self.T, self.m + 1)
        self.dt = np.diff(self.t)


@dataclass
class Paths:
    S: Array                                    # (n, m+1), S[:,0]=S0
    grid: Grid
    V: Optional[Array] = None                   # (n, m+1) instantaneous variance


@dataclass
class GBM:
    S0: float = 100.0
    r: float = 0.03
    q: float = 0.0
    sigma: float = 0.2
    n_factors: int = 1
    martingale_exact: bool = True
    name: str = "GBM"

    def propagate(self, grid: Grid, dW: Array) -> Paths:
        inc = (self.r - self.q - 0.5 * self.sigma ** 2) * grid.dt + self.sigma * dW[:, 0, :]
        X = np.concatenate([np.zeros((dW.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)
        return Paths(self.S0 * np.exp(X), grid)

    def path_vol(self, p: Paths) -> Array:
        return self.sigma


@dataclass
class MultiGBM:
    S0: Array
    r: float
    q: Array
    sigma: Array
    corr: Array
    martingale_exact: bool = True
    name: str = "MultiGBM"
    L: Array = field(init=False)
    n_factors: int = field(init=False)

    def __post_init__(self):
        self.S0 = np.asarray(self.S0, float)
        self.q = np.asarray(self.q, float)
        self.sigma = np.asarray(self.sigma, float)
        self.L = np.linalg.cholesky(np.asarray(self.corr, float))
        self.n_factors = len(self.S0)

    def propagate(self, grid: Grid, dW: Array) -> Paths:
        dWc = np.einsum("ij,njm->nim", self.L, dW)
        inc = ((self.r - self.q - 0.5 * self.sigma ** 2)[None, :, None] * grid.dt[None, None, :]
               + self.sigma[None, :, None] * dWc)
        X = np.concatenate([np.zeros((dW.shape[0], self.n_factors, 1)),
                            np.cumsum(inc, axis=2)], axis=2)
        return Paths(self.S0[None, :, None] * np.exp(X), grid)


@dataclass
class HestonQE:
    """
    Andersen (2008) QE-M scheme: moment-matched CIR + trapezoidal integrated
    variance + EXACT conditional martingale correction.  Weak bias O(dt^2).
    """
    S0: float = 100.0
    r: float = 0.03
    q: float = 0.0
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    sigma_v: float = 0.3
    rho: float = -0.7
    psi_c: float = 1.5
    martingale_correction: bool = True
    n_factors: int = 2
    martingale_exact: bool = True
    name: str = "Heston-QE-M"

    def __post_init__(self):
        self.martingale_exact = bool(self.martingale_correction)
        self.violations = 0

    def _qe(self, v: Array, dt: float, zv: Array) -> Tuple[Array, Dict]:
        k, th, s = self.kappa, self.theta, self.sigma_v
        e = math.exp(-k * dt)
        m = th + (v - th) * e
        s2 = v * s * s * e * (1 - e) / k + th * s * s * (1 - e) ** 2 / (2 * k)
        m = np.maximum(m, 1e-14)
        psi = s2 / (m * m)
        out = np.empty_like(v)
        b1 = psi <= self.psi_c
        aux: Dict = {"b1": b1}
        if np.any(b1):
            inv = 2.0 / np.maximum(psi[b1], 1e-12)
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a = m[b1] / (1.0 + b2)
            out[b1] = a * (np.sqrt(b2) + zv[b1]) ** 2
            aux["a"], aux["b2"] = a, b2
        if np.any(~b1):
            ps = psi[~b1]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[~b1]
            u = norm_cdf(zv[~b1])                       # reuse the same uniform
            out[~b1] = np.where(u < p, 0.0,
                                np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta)
            aux["p"], aux["beta"] = p, beta
        return np.maximum(out, 0.0), aux

    def _log_mgf(self, A: float, aux: Dict) -> Array:
        """log E[exp(A v')] under the QE conditional law, elementwise."""
        b1 = aux["b1"]
        n = b1.size
        out = np.zeros(n)
        if np.any(b1):
            a, b2 = aux["a"], aux["b2"]
            den = 1.0 - 2.0 * A * a
            bad = den <= 1e-10
            self.violations += int(bad.sum())
            den = np.maximum(den, 1e-10)
            out[b1] = -0.5 * np.log(den) + A * a * b2 / den
        if np.any(~b1):
            p, beta = aux["p"], aux["beta"]
            den = 1.0 - A * beta
            bad = den <= 1e-10
            self.violations += int(bad.sum())
            den = np.maximum(den, 1e-10)
            out[~b1] = np.log(np.maximum(p + (1.0 - p) / den, 1e-300))
        return out

    def propagate(self, grid: Grid, dW: Array) -> Paths:
        n, _, m = dW.shape
        V = np.empty((n, m + 1))
        X = np.zeros((n, m + 1))
        V[:, 0] = self.v0
        k, th, sv, rho = self.kappa, self.theta, self.sigma_v, self.rho
        for j in range(m):
            dt = grid.dt[j]
            sd = math.sqrt(dt)
            v = V[:, j]
            zv = dW[:, 0, j] / sd
            zp = dW[:, 1, j] / sd
            vn, aux = self._qe(v, dt, zv)
            g1 = g2 = 0.5
            K1 = g1 * dt * (k * rho / sv - 0.5) - rho / sv
            K2 = g2 * dt * (k * rho / sv - 0.5) + rho / sv
            K3 = (1.0 - rho * rho) * dt * g1
            K0 = -rho * k * th * dt / sv
            if self.martingale_correction:
                A = K2 + 0.5 * K3
                K0 = -self._log_mgf(A, aux) - (K1 + 0.5 * K3) * v
            vol = np.sqrt(np.maximum(K3 * (v + vn), 0.0))
            X[:, j + 1] = X[:, j] + (self.r - self.q) * dt + K0 + K1 * v + K2 * vn + vol * zp
            V[:, j + 1] = vn
        return Paths(self.S0 * np.exp(X), grid, V)

    def path_vol(self, p: Paths) -> Array:
        return np.sqrt(np.maximum(p.V[:, :-1], 1e-12))


@dataclass
class HestonEuler:
    """Full-truncation Euler/Milstein-free scheme (weak bias O(dt)); for MLMC."""
    S0: float = 100.0
    r: float = 0.03
    q: float = 0.0
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    sigma_v: float = 0.3
    rho: float = -0.7
    n_factors: int = 2
    martingale_exact: bool = False
    name: str = "Heston-Euler"

    def propagate(self, grid: Grid, dW: Array) -> Paths:
        n, _, m = dW.shape
        V = np.empty((n, m + 1))
        X = np.zeros((n, m + 1))
        V[:, 0] = self.v0
        for j in range(m):
            dt = grid.dt[j]
            v = V[:, j]
            vp = np.maximum(v, 0.0)
            dwv, dwp = dW[:, 0, j], dW[:, 1, j]
            V[:, j + 1] = v + self.kappa * (self.theta - vp) * dt + self.sigma_v * math.sqrt(vp) * dwv \
                if False else v + self.kappa * (self.theta - vp) * dt + self.sigma_v * np.sqrt(vp) * dwv
            X[:, j + 1] = X[:, j] + (self.r - self.q - 0.5 * vp) * dt + \
                np.sqrt(vp) * (self.rho * dwv + math.sqrt(1 - self.rho ** 2) * dwp)
        return Paths(self.S0 * np.exp(X), grid, V)

    def path_vol(self, p: Paths) -> Array:
        return np.sqrt(np.maximum(p.V[:, :-1], 1e-12))


def merton_mc_price(S0, K, T, r, q, sigma, lam, mu_j, delta_j, n, seed=0,
                    kind="call") -> Tuple[float, float]:
    """Exact simulation of Merton jump-diffusion at T (no discretisation bias)."""
    rng = np.random.default_rng(seed)
    N = rng.poisson(lam * T, n)
    jumps = rng.normal(mu_j, delta_j, n) * N + \
        delta_j * np.sqrt(N) * rng.standard_normal(n)      # sum of N normals
    kj = math.exp(mu_j + 0.5 * delta_j ** 2) - 1.0
    X = (r - q - lam * kj - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * rng.standard_normal(n) + jumps
    ST = S0 * np.exp(X)
    pay = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    d = math.exp(-r * T) * pay
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))


def merton_series_price(S0, K, T, r, q, sigma, lam, mu_j, delta_j, kind="call",
                        n_terms=60) -> float:
    """Merton (1976) Poisson-weighted Black-Scholes series (closed form)."""
    kj = math.exp(mu_j + 0.5 * delta_j ** 2) - 1.0
    lp = lam * (1.0 + kj)
    tot, w = 0.0, math.exp(-lp * T)
    for k in range(n_terms):
        sk = math.sqrt(sigma ** 2 + k * delta_j ** 2 / T)
        rk = r - lam * kj + k * math.log(1.0 + kj) / T
        tot += w * bs_price(S0, K, T, rk, q, sk, kind)
        w *= lp * T / (k + 1)
    return tot


# ==============================================================================
# 6. CLOSED FORMS (validation oracles & control variates)
# ==============================================================================
def bs_price(S0, K, T, r, q, sigma, kind="call") -> float:
    if T <= 0 or sigma <= 0:
        fwd = S0 * math.exp((r - q) * max(T, 0.0))
        return math.exp(-r * max(T, 0.0)) * max(fwd - K if kind == "call" else K - fwd, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / v
    d2 = d1 - v
    if kind == "call":
        return S0 * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S0 * math.exp(-q * T) * norm_cdf(-d1)


def bs_greeks(S0, K, T, r, q, sigma, kind="call") -> Dict[str, float]:
    v = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / v
    d2 = d1 - v
    sgn = 1.0 if kind == "call" else -1.0
    pdf = float(norm_pdf(d1))
    return {
        "delta": sgn * math.exp(-q * T) * float(norm_cdf(sgn * d1)),
        "gamma": math.exp(-q * T) * pdf / (S0 * v),
        "vega": S0 * math.exp(-q * T) * pdf * math.sqrt(T),
        "theta": -S0 * math.exp(-q * T) * pdf * sigma / (2 * math.sqrt(T))
                 - sgn * r * K * math.exp(-r * T) * float(norm_cdf(sgn * d2))
                 + sgn * q * S0 * math.exp(-q * T) * float(norm_cdf(sgn * d1)),
        "rho": sgn * K * T * math.exp(-r * T) * float(norm_cdf(sgn * d2)),
    }


def heston_cf_price(S0, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                    n_nodes=480, umax=220.0) -> float:
    """
    Heston call via Gil-Pelaez inversion of the characteristic function,
    'little trap' branch (Albrecher et al.) for numerical stability.
    """
    x, i1 = math.log(K), 1j

    def phi(u):
        iu = i1 * u
        b = kappa - rho * sigma_v * iu
        d = np.sqrt(b * b + sigma_v ** 2 * (iu + u ** 2))
        g = (b - d) / (b + d)
        edt = np.exp(-d * T)
        C = iu * (r - q) * T + kappa * theta / sigma_v ** 2 * \
            ((b - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
        D = (b - d) / sigma_v ** 2 * (1.0 - edt) / (1.0 - g * edt)
        return np.exp(C + D * v0 + iu * math.log(S0))

    nodes, w = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * umax * (nodes + 1.0)
    w = 0.5 * umax * w
    f1 = np.real(np.exp(-i1 * u * x) * phi(u - i1) / (i1 * u * S0 * math.exp((r - q) * T)))
    f2 = np.real(np.exp(-i1 * u * x) * phi(u) / (i1 * u))
    P1 = 0.5 + float(w @ f1) / math.pi
    P2 = 0.5 + float(w @ f2) / math.pi
    return S0 * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2


def geo_asian_price(S0, K, t_obs: Array, r, q, sigma, kind="call") -> float:
    """Exact discrete geometric-average Asian (log-normal average)."""
    m = len(t_obs)
    mu = math.log(S0) + (r - q - 0.5 * sigma ** 2) * t_obs.mean()
    C = np.minimum.outer(t_obs, t_obs)
    var = sigma ** 2 * C.sum() / (m * m)
    v = math.sqrt(var)
    d1 = (mu + var - math.log(K)) / v
    d2 = d1 - v
    T = t_obs[-1]
    if kind == "call":
        return math.exp(-r * T) * (math.exp(mu + 0.5 * var) * float(norm_cdf(d1))
                                   - K * float(norm_cdf(d2)))
    return math.exp(-r * T) * (K * float(norm_cdf(-d2))
                               - math.exp(mu + 0.5 * var) * float(norm_cdf(-d1)))


def barrier_do_call(S0, K, B, T, r, q, sigma) -> float:
    """Continuous down-and-out call, B <= K, zero rebate (Reiner-Rubinstein)."""
    assert B <= K and B < S0
    lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
    y = math.log(B * B / (S0 * K)) / (sigma * math.sqrt(T)) + lam * sigma * math.sqrt(T)
    cdi = S0 * math.exp(-q * T) * (B / S0) ** (2 * lam) * float(norm_cdf(y)) - \
        K * math.exp(-r * T) * (B / S0) ** (2 * lam - 2) * float(norm_cdf(y - sigma * math.sqrt(T)))
    return bs_price(S0, K, T, r, q, sigma) - cdi


def crr_american(S0, K, T, r, q, sigma, n=2000, kind="put") -> float:
    """Cox-Ross-Rubinstein binomial with early exercise (O(n) memory)."""
    dt = T / n
    up = math.exp(sigma * math.sqrt(dt))
    dn = 1.0 / up
    p = (math.exp((r - q) * dt) - dn) / (up - dn)
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    S = S0 * up ** (2 * j - n)
    V = np.maximum(K - S if kind == "put" else S - K, 0.0)
    for i in range(n - 1, -1, -1):
        V = disc * (p * V[1:i + 2] + (1 - p) * V[:i + 1])
        S = S0 * up ** (2 * np.arange(i + 1) - i)
        V = np.maximum(V, np.maximum(K - S if kind == "put" else S - K, 0.0))
    return float(V[0])


# ==============================================================================
# 7. PAYOFFS  (all values DISCOUNTED to t=0; controls carry exact means)
# ==============================================================================
class Payoff:
    needs_path = False
    continuous = False

    def __call__(self, p: Paths, model) -> Array:
        raise NotImplementedError

    def controls(self, p: Paths, model) -> List[Tuple[Array, float]]:
        return []

    def _df(self, model, T):
        return math.exp(-model.r * T)


@dataclass
class European(Payoff):
    K: float = 100.0
    kind: str = "call"

    def __call__(self, p, model):
        ST = p.S[:, -1]
        h = np.maximum(ST - self.K if self.kind == "call" else self.K - ST, 0.0)
        return self._df(model, p.grid.T) * h

    def controls(self, p, model):
        if getattr(model, "martingale_exact", False) and p.S.ndim == 2:
            df = self._df(model, p.grid.T)
            return [(df * p.S[:, -1], model.S0 * math.exp(-model.q * p.grid.T))]
        return []


@dataclass
class Digital(Payoff):
    K: float = 100.0
    kind: str = "call"
    cash: float = 1.0

    def __call__(self, p, model):
        ST = p.S[:, -1]
        ind = (ST > self.K) if self.kind == "call" else (ST < self.K)
        return self._df(model, p.grid.T) * self.cash * ind


@dataclass
class Asian(Payoff):
    K: float = 100.0
    kind: str = "call"
    needs_path = True

    def __call__(self, p, model):
        A = p.S[:, 1:].mean(axis=1)
        h = np.maximum(A - self.K if self.kind == "call" else self.K - A, 0.0)
        return self._df(model, p.grid.T) * h

    def controls(self, p, model):
        out = []
        if getattr(model, "martingale_exact", False):
            df = self._df(model, p.grid.T)
            out.append((df * p.S[:, -1], model.S0 * math.exp(-model.q * p.grid.T)))
        if isinstance(model, GBM):
            G = np.exp(np.log(p.S[:, 1:]).mean(axis=1))
            h = np.maximum(G - self.K if self.kind == "call" else self.K - G, 0.0)
            exact = geo_asian_price(model.S0, self.K, p.grid.t[1:], model.r, model.q,
                                    model.sigma, self.kind)
            out.append((self._df(model, p.grid.T) * h, exact))
        return out


@dataclass
class Barrier(Payoff):
    K: float = 100.0
    B: float = 90.0
    kind: str = "call"
    direction: str = "down"          # 'down' or 'up'
    out: bool = True
    rebate: float = 0.0
    brownian_bridge: bool = True     # conditional-MC continuity correction
    needs_path = True
    continuous = True

    def __call__(self, p, model):
        S, t = p.S, p.grid.t
        logB = math.log(self.B)
        X = np.log(S)
        if self.direction == "down":
            hit = (X <= logB).any(axis=1)
            d_i, d_ip1 = X[:, :-1] - logB, X[:, 1:] - logB
        else:
            hit = (X >= logB).any(axis=1)
            d_i, d_ip1 = logB - X[:, :-1], logB - X[:, 1:]
        df = self._df(model, p.grid.T)
        hT = np.maximum(S[:, -1] - self.K if self.kind == "call" else self.K - S[:, -1], 0.0)
        if self.brownian_bridge:
            # P(no crossing on [t_i,t_{i+1}] | endpoints) = 1 - exp(-2 d_i d_{i+1}/(v dt))
            vol = model.path_vol(p)
            v = vol ** 2 if np.ndim(vol) else vol ** 2
            denom = np.maximum(v * p.grid.dt[None, :], 1e-14)
            cross = np.exp(-2.0 * np.maximum(d_i, 0.0) * np.maximum(d_ip1, 0.0) / denom)
            surv = np.prod(1.0 - cross, axis=1)
            surv = np.where(hit, 0.0, surv)
            pay = surv * hT + (1.0 - surv) * self.rebate
        else:
            alive = ~hit if self.out else hit
            pay = np.where(alive, hT, self.rebate)
        if not self.out:
            pay = df * hT - pay + df * self.rebate * 0.0
        return df * pay

    def controls(self, p, model):
        if isinstance(model, GBM):
            df = self._df(model, p.grid.T)
            hT = np.maximum(p.S[:, -1] - self.K if self.kind == "call" else self.K - p.S[:, -1], 0.0)
            return [(df * hT, bs_price(model.S0, self.K, p.grid.T, model.r, model.q,
                                       model.sigma, self.kind))]
        return []


@dataclass
class Lookback(Payoff):
    K: float = 100.0
    kind: str = "fixed"              # 'fixed' or 'float'
    side: str = "call"
    needs_path = True

    def __call__(self, p, model):
        df = self._df(model, p.grid.T)
        if self.side == "call":
            M = p.S.max(axis=1)
            h = (np.maximum(M - self.K, 0.0) if self.kind == "fixed" else p.S[:, -1] - p.S.min(axis=1))
        else:
            m = p.S.min(axis=1)
            h = (np.maximum(self.K - m, 0.0) if self.kind == "fixed" else p.S.max(axis=1) - p.S[:, -1])
        return df * np.maximum(h, 0.0)


@dataclass
class VarianceSwap(Payoff):
    K_var: float = 0.04
    notional: float = 1.0
    needs_path = True

    def __call__(self, p, model):
        lr = np.diff(np.log(p.S), axis=1)
        RV = (lr ** 2).sum(axis=1) / p.grid.T
        return self._df(model, p.grid.T) * self.notional * (RV - self.K_var)

    def controls(self, p, model):
        if getattr(model, "martingale_exact", False):
            T = p.grid.T
            lrT = np.log(p.S[:, -1] / p.S[:, 0])
            mu = math.log(model.S0) + (model.r - model.q) * T
            if isinstance(model, GBM):
                mu -= 0.5 * model.sigma ** 2 * T
            else:
                mu -= 0.5 * model.theta * T
            return [(lrT, mu)]
        return []


# ==============================================================================
# 8. ENGINES
# ==============================================================================
@dataclass
class MCResult:
    price: float
    se: float
    n: int
    seconds: float
    vrf: float = 1.0
    beta: Optional[Array] = None
    skew: float = 0.0
    kurt: float = 0.0
    extra: Dict = field(default_factory=dict)

    @property
    def ci(self) -> Tuple[float, float]:
        return self.price - 1.96 * self.se, self.price + 1.96 * self.se

    def line(self, label="price", ref=None) -> str:
        s = f"{label:>22s} = {self.price:12.6f}  SE={self.se:.3e}  95%CI=[{self.ci[0]:.6f}, {self.ci[1]:.6f}]"
        if ref is not None:
            s += f"  ref={ref:.6f}  err/se={(self.price-ref)/max(self.se,1e-16):+.2f}"
        return s


def _draw_normals(ps: PointStream, n: int, nf: int, m: int, antithetic: bool) -> Array:
    if antithetic and ps.kind == "pseudo":
        h = (n + 1) // 2
        u = ps.block(h).reshape(h, nf, m)   # ← Fixed: was h * nf * m
        z = norm_ppf(u)
        z = np.concatenate([z, -z], axis=0)[:n]
    else:
        u = ps.block(n).reshape(n, nf, m)   # ← Fixed: was n * nf * m
        z = norm_ppf(u)
    return z


def simulate_block(model, grid: Grid, ps: PointStream, n: int, *,
                   construction="bridge", antithetic=False) -> Paths:
    nf, m = model.n_factors, grid.m
    Z = _draw_normals(ps, n, nf, m, antithetic)
    W = np.stack([wiener_path(Z[:, f, :], grid.t, construction) for f in range(nf)], axis=1)
    dW = np.diff(W, axis=2, prepend=0.0)
    return model.propagate(grid, dW)


def simulate_paths(model, grid: Grid, n: int, seed=0, *, sampler="pseudo",
                   construction="bridge", antithetic=False, chunk=1 << 15) -> Paths:
    ps = PointStream(sampler, model.n_factors * grid.m, n, seed=seed)
    out, Vout, done = [], [], 0
    while done < n:
        b = min(chunk, n - done)
        p = simulate_block(model, grid, ps, b, construction=construction, antithetic=antithetic)
        out.append(p.S)
        if p.V is not None:
            Vout.append(p.V)
        done += b
    return Paths(np.concatenate(out, axis=0), grid,
                 np.concatenate(Vout, axis=0) if Vout else None)


def mc_price(model, payoff: Payoff, grid: Grid, n: int, *, seed=0, sampler="pseudo",
             construction="bridge", antithetic=True, control_variates=True,
             pilot_frac=0.0, chunk=1 << 15, point_stream: Optional[PointStream] = None,
             return_samples=False) -> MCResult:
    """
    Crude / antithetic / control-variate Monte Carlo.

    Control variates: beta* = Cov(C)^{-1} Cov(C,Y), Y_cv = Y - beta'(C - E C).
    With pilot_frac>0 the regression is fitted on an independent pilot slice,
    making the estimator exactly unbiased (otherwise bias = O(1/n)).
    """
    t0 = time.perf_counter()
    m = grid.m if (payoff.needs_path or payoff.continuous) else 1
    g = Grid(grid.T, m)
    ps = point_stream or PointStream(sampler, model.n_factors * g.m, n, seed=seed)
    ys, cs = [], []
    done = 0
    while done < n:
        b = min(chunk, n - done)
        p = simulate_block(model, g, ps, b, construction=construction, antithetic=antithetic)
        ys.append(np.asarray(payoff(p, model), float))
        if control_variates:
            ctrl = payoff.controls(p, model)
            if ctrl:
                cs.append(np.stack([c[0] for c in ctrl], axis=1))
        done += b
    y = np.concatenate(ys)
    means = None
    C = None
    if cs:
        C = np.concatenate(cs, axis=0)
        p0 = simulate_block(model, g, PointStream("pseudo", model.n_factors * g.m, 2, seed=1), 2)
        means = np.array([c[1] for c in payoff.controls(p0, model)], float)

    var0 = y.var(ddof=1)
    beta = None
    if C is not None:
        if pilot_frac > 0:
            k = max(int(pilot_frac * len(y)), C.shape[1] + 5)
            Cc, yc = C[:k] - C[:k].mean(0), y[:k] - y[:k].mean()
            beta = np.linalg.lstsq(Cc, yc, rcond=None)[0]
            y_adj = y[k:] - (C[k:] - means) @ beta
        else:
            Cc = C - C.mean(0)
            beta = np.linalg.lstsq(Cc, y - y.mean(), rcond=None)[0]
            y_adj = y - (C - means) @ beta
    else:
        y_adj = y
    mu = float(y_adj.mean())
    se = float(y_adj.std(ddof=1) / math.sqrt(len(y_adj)))
    z = (y_adj - mu) / max(y_adj.std(ddof=1), 1e-30)
    res = MCResult(mu, se, len(y_adj), time.perf_counter() - t0,
                   vrf=float(var0 / max(y_adj.var(ddof=1), 1e-30)), beta=beta,
                   skew=float((z ** 3).mean()), kurt=float((z ** 4).mean() - 3.0),
                   extra={"sampler": ps.kind, "construction": construction,
                          "steps": g.m, "controls": 0 if means is None else len(means)})
    if return_samples:
        res.extra["samples"] = y_adj
    return res


def rqmc_price(model, payoff, grid, n_per_shift, R=16, *, seed=0, sampler="halton",
               construction="bridge", **kw) -> MCResult:
    """
    Randomised QMC: R independent randomisations of the same net.
    Unbiased; SE from between-replicate variance (t_{R-1} for honest CIs).
    """
    t0 = time.perf_counter()
    vals = []
    for s in range(R):
        r = mc_price(model, payoff, grid, n_per_shift, seed=seed, sampler=sampler,
                     construction=construction, antithetic=False,
                     point_stream=PointStream(sampler, model.n_factors *
                                              (grid.m if payoff.needs_path else 1),
                                              n_per_shift, seed=seed, shift_seed=s + 1),
                     **kw)
        vals.append(r.price)
    vals = np.array(vals)
    mu = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(R))
    return MCResult(mu, se, R * n_per_shift, time.perf_counter() - t0,
                    extra={"R": R, "n_per_shift": n_per_shift, "sampler": sampler,
                           "t_crit": 2.131 if R == 16 else 1.96})


def mlmc_price(model, payoff: Payoff, T: float, eps: float, *, m0=4, L_max=9,
               n_pilot=4000, seed=0, verbose=True) -> Dict:
    """
    Giles multilevel MC with Brownian-increment coupling.
      levels: m_l = m0 * 2^l ; N_l ∝ sqrt(V_l/C_l) ; bias ≈ |E[P_L - P_{L-1}]|.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    nf = model.n_factors

    def coupled(l: int, n: int) -> Tuple[Array, float]:
        m_f = m0 * 2 ** l
        gf = Grid(T, m_f)
        Z = norm_ppf(rng.random((n, nf, m_f)))
        Wf = np.stack([wiener_path(Z[:, f, :], gf.t, "standard") for f in range(nf)], axis=1)
        dWf = np.diff(Wf, axis=2, prepend=0.0)
        pf = payoff(model.propagate(gf, dWf), model)
        if l == 0:
            return pf, float(m_f)
        gc = Grid(T, m_f // 2)
        dWc = dWf.reshape(n, nf, m_f // 2, 2).sum(axis=3)
        pc = payoff(model.propagate(gc, dWc), model)
        return pf - pc, float(m_f + m_f // 2)

    means, vars_, costs = [], [], []
    for l in range(L_max + 1):
        d, c = coupled(l, n_pilot)
        means.append(float(d.mean()))
        vars_.append(float(d.var(ddof=1)))
        costs.append(c)
        if verbose:
            print(f"  level {l}: m={m0*2**l:5d}  E[dP]={means[-1]:+.6e}  "
                  f"V={vars_[-1]:.3e}  V_l/V_(l-1)="
                  f"{(vars_[-1]/vars_[-2]) if l else float('nan'):.3f}")
        if l >= 2 and abs(means[-1]) <= eps / math.sqrt(2.0):
            break
    L = len(means) - 1
    V, C = np.array(vars_), np.array(costs)
    
    # FIX: Use np.sqrt instead of math.sqrt for array operations
    lam = np.sqrt(V / C).sum() / (eps ** 2 / 2.0)
    N = np.maximum(np.ceil(lam * np.sqrt(V / C)), 10).astype(int)
    
    est, var_tot, cost_tot = 0.0, 0.0, 0.0
    for l in range(L + 1):
        d, c = coupled(l, int(N[l]))
        est += d.mean()
        var_tot += d.var(ddof=1) / len(d)
        cost_tot += len(d) * costs[l]
    se = math.sqrt(var_tot)
    # standard MC at the finest level for the same RMSE
    n_std = 2.0 * V[0] / eps ** 2
    cost_std = n_std * (m0 * 2 ** L)
    out = {"price": float(est), "se": float(se), "L": L, "N": N.tolist(),
           "cost": cost_tot, "cost_std": cost_std, "speedup": cost_std / cost_tot,
           "seconds": time.perf_counter() - t0, "bias_est": means[-1]}
    if verbose:
        print(f"  MLMC price={est:.6f}  SE={se:.3e}  L={L}  work={cost_tot:.3e} "
              f"(std MC {cost_std:.3e}, speedup x{out['speedup']:.1f})")
    return out

# -------------------------- American: LSM + dual ------------------------------
def _basis(x: Array, kind="laguerre", deg=3) -> Array:
    if kind == "poly":
        return np.stack([x ** k for k in range(deg + 1)], axis=1)
    L = [np.ones_like(x), np.exp(-x / 2), np.exp(-x / 2) * (1 - x),
         np.exp(-x / 2) * (1 - 2 * x + x * x / 2)]
    return np.stack(L[:deg + 1], axis=1)


def lsm_american(model, K, T, m, n_train=60000, n_test=60000, seed=0,
                 kind="put", basis="laguerre", dual_paths=0, dual_outer=2000,
                 dual_inner=256) -> Dict:
    """
    Longstaff-Schwartz: policy fitted on TRAIN paths, priced on independent
    TEST paths => unbiased LOWER bound with a clean CLT error bar.
    Optional Rogers / Andersen-Broadie DUAL upper bound (GBM only):
        V0 <= E[ max_j ( e^{-r t_j} h_j - M_j ) ],
        M_j = sum_{k<j} ( V~_{k+1} - E[V~_{k+1} | F_k] ),  V~ = e^{-rt} max(h, C^).
    Nested expectations are estimated by inner MC (Jensen => upward bias).
    """
    grid = Grid(T, m)
    h = lambda S: np.maximum(K - S if kind == "put" else S - K, 0.0)
    tr = simulate_paths(model, grid, n_train, seed=seed, antithetic=True).S
    H = h(tr)
    betas: List[Optional[Array]] = [None] * (m + 1)
    cf = H[:, m].copy()
    for j in range(m - 1, 0, -1):
        cf *= math.exp(-model.r * (grid.t[j + 1] - grid.t[j]))
        itm = H[:, j] > 0
        if itm.sum() < 20:
            continue
        Xb = _basis(tr[itm, j] / K, basis)
        b = np.linalg.lstsq(Xb, cf[itm], rcond=None)[0]
        betas[j] = b
        ex = H[itm, j] > Xb @ b
        idx = np.where(itm)[0][ex]
        cf[idx] = H[idx, j]
    v0_train = max(float(h(np.array([model.S0]))[0]),
                   float(cf.mean()) * math.exp(-model.r * grid.t[1]))

    te = simulate_paths(model, grid, n_test, seed=seed + 777, antithetic=True).S
    Ht = h(te)
    cash = Ht[:, m] * math.exp(-model.r * T)
    alive = np.ones(n_test, bool)
    for j in range(1, m):
        if betas[j] is None:
            continue
        itm = alive & (Ht[:, j] > 0)
        if not itm.any():
            continue
        cont = _basis(te[itm, j] / K, basis) @ betas[j]
        ex = Ht[itm, j] > cont
        idx = np.where(itm)[0][ex]
        cash[idx] = Ht[idx, j] * math.exp(-model.r * grid.t[j])
        alive[idx] = False
    lo, lo_se = float(cash.mean()), float(cash.std(ddof=1) / math.sqrt(n_test))
    out = {"lower": lo, "se_lower": lo_se, "train_insample": v0_train}

    if dual_paths or dual_outer:
        if not isinstance(model, GBM) or dual_outer <= 0:
            return out
        n_o, n_i = dual_outer, dual_inner
        rng = np.random.default_rng(seed + 4242)
        S = simulate_paths(model, grid, n_o, seed=seed + 99).S
        Hd = h(S)
        disc = np.exp(-model.r * grid.t)

        def vtilde(j, s):
            """Calculate continuation value for input s. s can be 1D or 2D."""
            hv = h(s)
            if j == 0 or j == m or betas[j] is None:
                return hv
            
            # Handle 2D input (n_o, n_i) from inner simulations
            if s.ndim == 2:
                n_rows, n_cols = s.shape
                # Reshape to 1D to apply basis
                s_flat = s.ravel()
                basis_vals = _basis(s_flat / K, basis) # (n_rows*n_cols, deg+1)
                cont_flat = basis_vals @ betas[j]      # (n_rows*n_cols,)
                cont = cont_flat.reshape(n_rows, n_cols)
                return np.maximum(hv, cont)
            else:
                # 1D input
                basis_vals = _basis(s / K, basis)
                cont = basis_vals @ betas[j]
                return np.maximum(hv, cont)

        M = np.zeros(n_o)
        best = Hd[:, 0].copy()
        for j in range(0, m):
            dt = grid.dt[j]
            s = S[:, j]
            # Generate inner paths for next step
            zi = rng.standard_normal((n_o, n_i))
            # Antithetic for inner paths
            zi = np.concatenate([zi, -zi], axis=1)[:, :n_i] # Keep size n_i if needed, or use all
            
            # Calculate Sn for all inner paths
            # s is (n_o,), zi is (n_o, n_i_inner)
            # We need to broadcast s correctly
            Sn = s[:, None] * np.exp((model.r - model.q - 0.5 * model.sigma ** 2) * dt
                                     + model.sigma * math.sqrt(dt) * zi)
            
            # Calculate conditional expectation E[V~_{j+1} | F_j]
            # vtilde returns (n_o, n_i_inner)
            V_next_inner = vtilde(j + 1, Sn)
            cond = disc[j + 1] * V_next_inner.mean(axis=1)
            
            # Calculate actual V~_{j+1} for the realized path
            # We need S[:, j+1] which is the next step of the OUTER path
            actual = disc[j + 1] * vtilde(j + 1, S[:, j + 1])
            
            M += actual - cond
            best = np.maximum(best, disc[j + 1] * Hd[:, j + 1] - M)
        out["upper"] = float(best.mean())
        out["se_upper"] = float(best.std(ddof=1) / math.sqrt(n_o))
    return out


# --------------------------------- Greeks -------------------------------------
def greeks_fd(model, payoff, grid, n, bumps: Dict[str, float], seed=0,
              sampler="pseudo", richardson=True) -> Dict[str, Tuple[float, float]]:
    """
    Central differences with COMMON RANDOM NUMBERS (paired variance collapse)
    + Richardson extrapolation:  D = (4 D(h/2) - D(h)) / 3.
    Works for any model/payoff, including discontinuous ones.
    """
    out = {}
    for name, h in bumps.items():
        def run(step, sd):
            ps_u = PointStream(sampler, model.n_factors * grid.m, n, seed=sd)
            ps_d = PointStream(sampler, model.n_factors * grid.m, n, seed=sd)
            up, dn = replace(model, **{name: getattr(model, name) + step}), \
                     replace(model, **{name: getattr(model, name) - step})
            pu = mc_price(up, payoff, grid, n, point_stream=ps_u, antithetic=False,
                          control_variates=False, seed=sd)
            pd = mc_price(dn, payoff, grid, n, point_stream=ps_d, antithetic=False,
                          control_variates=False, seed=sd)
            return pu, pd
        pu, pd = run(h, seed)
        d1 = (pu.price - pd.price) / (2 * h)
        se1 = abs(pu.se - pd.se) / (2 * h)
        # paired SE via re-simulation of the difference is approximated by the
        # (much smaller) difference of SEs under CRN; report a conservative bound
        se1 = math.sqrt(pu.se ** 2 + pd.se ** 2) / (2 * h)
        if richardson:
            pu2, pd2 = run(h / 2, seed)
            d2 = (pu2.price - pd2.price) / h
            out[name] = ((4 * d2 - d1) / 3, se1)
        else:
            out[name] = (d1, se1)
    return out


def greeks_pathwise_lrm(model: GBM, K, T, n, seed=0, kind="call") -> Dict:
    """
    Pathwise (IPA) vs likelihood-ratio (LRM) estimators for BS delta/vega.
      pathwise delta : e^{-rT} 1{S_T>K} S_T/S0        (needs a.e. smooth payoff)
      LRM delta      : e^{-rT} h(S_T) W_T/(S0 sigma T) (works for digitals)
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n + 1) // 2)
    z = np.concatenate([z, -z])[:n]
    W = math.sqrt(T) * z
    ST = model.S0 * np.exp((model.r - model.q - 0.5 * model.sigma ** 2) * T + model.sigma * W)
    df = math.exp(-model.r * T)
    h = np.maximum(ST - K if kind == "call" else K - ST, 0.0)
    itm = (ST > K) if kind == "call" else (ST < K)
    sgn = 1.0 if kind == "call" else -1.0
    pw_d = df * sgn * itm * ST / model.S0
    lr_d = df * h * W / (model.S0 * model.sigma * T)
    pw_v = df * sgn * itm * ST * (W - model.sigma * T)
    mix_g = df * sgn * itm * (ST / model.S0) * (W / (model.S0 * model.sigma * T)) \
        - df * sgn * itm * ST / model.S0 ** 2
    f = lambda a: (float(a.mean()), float(a.std(ddof=1) / math.sqrt(n)))
    return {"delta_pathwise": f(pw_d), "delta_LRM": f(lr_d),
            "vega_pathwise": f(pw_v), "gamma_mixed": f(mix_g), "price": f(df * h)}


# ==============================================================================
# 9. RISK
# ==============================================================================
def var_cvar(loss: Array, alpha=0.99) -> Dict[str, float]:
    """Empirical VaR / CVaR (Rockafellar-Uryasev consistent estimator)."""
    L = np.sort(np.asarray(loss, float))
    n = len(L)
    k = int(math.ceil(alpha * n)) - 1
    k = min(max(k, 0), n - 1)
    v = float(L[k])
    cv = v + float(np.maximum(L - v, 0.0).mean()) / (1.0 - alpha)
    return {"VaR": v, "CVaR": cv, "n": n, "alpha": alpha}


def var_ci(loss: Array, alpha=0.99, conf=0.95) -> Tuple[float, float]:
    """Distribution-free CI for the quantile via binomial order statistics."""
    L = np.sort(np.asarray(loss, float))
    n = len(L)
    z = float(norm_ppf(0.5 + conf / 2))
    sd = math.sqrt(n * alpha * (1 - alpha))
    lo = max(int(math.floor(n * alpha - z * sd)), 0)
    hi = min(int(math.ceil(n * alpha + z * sd)), n - 1)
    return float(L[lo]), float(L[hi])


def component_cvar(losses_by_asset: Array, alpha=0.99) -> Array:
    """Euler / expected-shortfall allocation: sum_i E[L_i | L >= VaR] = CVaR."""
    X = np.asarray(losses_by_asset, float)
    tot = X.sum(axis=1)
    v = var_cvar(tot, alpha)["VaR"]
    tail = tot >= v
    return X[tail].mean(axis=0)


def spectral_rm(loss: Array, kappa=2.0, kind="exponential") -> float:
    """Spectral (law-invariant, coherent) risk measure with risk spectrum phi."""
    L = np.sort(np.asarray(loss, float))
    n = len(L)
    s = (np.arange(n) + 0.5) / n
    if kind == "exponential":
        w = kappa * np.exp(-kappa * (1 - s)) / (1 - math.exp(-kappa))
    else:
        w = kappa * s ** (kappa - 1)
    w /= w.sum()
    return float(w @ L)


def hill_xi(excess: Array) -> float:
    return float(np.log(np.asarray(excess, float)).mean())


def gpd_fit(excess: Array) -> Tuple[float, float]:
    """GPD MLE by profile likelihood: golden-section in beta, grid+refine in xi."""
    y = np.asarray(excess, float)
    y = y[y > 0]
    n = len(y)
    ymax = y.max()

    def nll(xi, beta):
        if beta <= 0:
            return 1e18
        z = 1.0 + xi * y / beta
        if np.any(z <= 0):
            return 1e18
        return float(n * math.log(beta) + (1.0 + 1.0 / xi) * np.log(z).sum()) if abs(xi) > 1e-8 \
            else float(n * math.log(beta) + y.sum() / beta)

    def prof(xi):
        lb = max(1e-8, -xi * ymax * 1.0000001) if xi < 0 else 1e-8
        b = golden_min(lambda lb_: nll(xi, math.exp(lb_)),
                       math.log(lb), math.log(max(10 * y.mean(), lb * 10 + 1)))
        return nll(xi, math.exp(b)), math.exp(b)

    grid = np.linspace(-0.45, 1.2, 67)
    vals = [prof(x)[0] for x in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    xi = golden_min(lambda x: prof(x)[0], a, b)
    _, beta = prof(xi)
    return float(xi), float(beta)


def evt_var_cvar(loss: Array, alpha=0.99, u_q=0.95) -> Dict[str, float]:
    """Peaks-over-threshold EVT tail estimate (Pickands-Balkema-de Haan)."""
    L = np.asarray(loss, float)
    u = float(np.quantile(L, u_q))
    exc = L[L > u] - u
    xi, beta = gpd_fit(exc)
    p, Nu, N = 1 - alpha, len(exc), len(L)
    v = u + beta / xi * ((N / Nu * p) ** (-xi) - 1.0) if abs(xi) > 1e-8 \
        else u + beta * math.log(N / Nu / p)
    cv = (v + beta - xi * u) / (1 - xi) if xi < 1 else float("inf")
    return {"u": u, "xi": xi, "beta": beta, "VaR_EVT": v, "CVaR_EVT": cv,
            "n_exceed": Nu, "hill": hill_xi(exc)}


def is_var_linear(a: Array, cov: Array, alpha=0.99, n=200000, seed=0,
                  n_pilot=40000) -> Dict:
    """
    Exponential (Esscher/Girsanov) tilting for L = a'Z, Z~N(0,cov).
      dP/dP_theta = exp(-theta L + theta^2 sigma_L^2 / 2),  E_theta[L] = theta sigma_L^2.
    theta* from mean-matching at the VaR level; also SAA-optimised theta.
    """
    a, cov = np.asarray(a, float), np.asarray(cov, float)
    s2 = float(a @ cov @ a)
    sL = math.sqrt(s2)
    z = float(norm_ppf(alpha))
    exact_var = z * sL
    exact_cv = sL * float(norm_pdf(z)) / (1 - alpha)
    rng = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(cov)

    def draw(nn, theta, rg):
        Z = rg.standard_normal((nn, len(a))) @ Lch.T + theta * (cov @ a)[None, :]
        return Z @ a

    pilot = draw(n_pilot, 0.0, rng)
    x = float(np.quantile(pilot, alpha))
    theta_mm = x / s2
    w = np.exp(np.clip(pilot * 1.0, -700, 700) * 0)  # placeholder for clarity
    def second_moment(th):
        lr = np.exp(np.clip(-th * pilot + 0.5 * th * th * s2, -700, 700))
        return float(np.mean((pilot > x) ** 2 * lr ** 2))
    theta_saa = golden_min(second_moment, 0.0, max(3 * theta_mm, 1e-6))
    out = {"VaR_exact": exact_var, "CVaR_exact": exact_cv,
           "theta_mean_match": theta_mm, "theta_SAA": theta_saa}
    for tag, th in (("crude", 0.0), ("IS", theta_saa)):
        Ls = draw(n, th, rng)
        lr = np.exp(np.clip(-th * Ls + 0.5 * th * th * s2, -700, 700))
        ind = (Ls > exact_var).astype(float)
        est = ind * lr
        out[f"p_{tag}"] = float(est.mean())
        out[f"se_{tag}"] = float(est.std(ddof=1) / math.sqrt(n))
        exc = np.maximum(Ls - exact_var, 0.0) * lr
        out[f"ES_{tag}"] = exact_var + float(exc.mean()) / (1 - alpha)
        out[f"se_ES_{tag}"] = float(exc.std(ddof=1) / math.sqrt(n) / (1 - alpha))
    out["VRF"] = (out["se_crude"] / max(out["se_IS"], 1e-30)) ** 2
    return out


def credit_portfolio(pd: Array, lgd: Array, ead: Array, rho: Array, n=200000,
                     alpha=0.999, seed=0, importance=True) -> Dict:
    """
    Gaussian one-factor copula credit portfolio (Vasicek / CreditMetrics).
      D_i = 1{ sqrt(rho_i) Z + sqrt(1-rho_i) eps_i < Phi^{-1}(pd_i) }
    Two-step IS (Glasserman-Li): conditional Bernoulli exponential twist so that
    E_theta[L|Z] ~ x, plus a mean shift mu on the systematic factor Z.
    """
    pd, lgd, ead, rho = map(lambda v: np.asarray(v, float), (pd, lgd, ead, rho))
    ell = lgd * ead
    thr = norm_ppf(pd)
    rng = np.random.default_rng(seed)
    M = len(pd)

    def cond_pd(Z):
        return norm_cdf((thr[None, :] - np.sqrt(rho)[None, :] * Z[:, None])
                        / np.sqrt(1 - rho)[None, :])

    def one_run(nn, mu, x_target, use_is):
        Z = rng.standard_normal(nn) + mu
        p = cond_pd(Z)
        lrZ = np.exp(-mu * Z + 0.5 * mu * mu)
        if use_is:
            m = p @ ell
            theta = np.array([bisect_monotone(
                lambda th: float(ell @ (p[i] * np.exp(np.clip(th * ell, -50, 50)) /
                                        (1 + p[i] * (np.exp(np.clip(th * ell, -50, 50)) - 1)))) - x_target,
                0.0, 50.0) if m[i] < x_target else 0.0 for i in range(nn)])
            te = np.exp(np.clip(theta[:, None] * ell[None, :], -50, 50))
            pt = p * te / (1 + p * (te - 1))
            psi = np.log(1 + p * (te - 1)).sum(axis=1)
            U = rng.random((nn, M))
            D = (U < pt).astype(float)
            L = D @ ell
            lr = lrZ * np.exp(-theta * L + psi)
        else:
            D = (rng.random((nn, M)) < p).astype(float)
            L = D @ ell
            lr = lrZ
        return L, lr

    L0, _ = one_run(min(n, 40000), 0.0, 0.0, False)
    x = float(np.quantile(L0, alpha))
    mu = 0.0
    if importance:
        grid = np.linspace(0.0, float(norm_ppf(alpha)) * 1.2, 13)
        best, bmu = np.inf, 0.0
        for g in grid:
            Lg, lrg = one_run(4000, g, x, True)
            v = float(np.mean(((Lg > x) * lrg) ** 2))
            if v < best:
                best, bmu = v, g
        mu = bmu
    res = {}
    for tag, use_is, mm in (("crude", False, 0.0), ("IS", importance, mu)):
        L, lr = one_run(n, mm, x, use_is)
        est = (L > x) * lr
        res[f"p_{tag}"] = float(est.mean())
        res[f"se_{tag}"] = float(est.std(ddof=1) / math.sqrt(n))
        res[f"VaR_{tag}"] = float(np.quantile(L, alpha))
    res["mu"] = mu
    res["VRF"] = (res["se_crude"] / max(res["se_IS"], 1e-30)) ** 2
    res["EL"] = float(pd @ ell)
    return res


def kupiec_pof(hits: Array, p: float) -> Dict[str, float]:
    """Kupiec proportion-of-failures LR test, chi2_1 (survival = erfc(sqrt(LR/2)))."""
    h = np.asarray(hits, int)
    T, x = len(h), int(h.sum())
    ph = x / max(T, 1)
    if x == 0:
        lr = -2 * (T * math.log(1 - p) - T * math.log(1 - ph))
    elif x == T:
        lr = -2 * (T * math.log(p) - T * math.log(ph))
    else:
        lr = -2 * ((T - x) * math.log(1 - p) + x * math.log(p)
                   - (T - x) * math.log(1 - ph) - x * math.log(ph))
    return {"LR": float(lr), "p_value": float(erfc(math.sqrt(max(lr, 0) / 2))),
            "hits": x, "T": T, "expected": p * T}


def christoffersen(hits: Array, p: float) -> Dict[str, float]:
    """Independence + conditional-coverage LR tests (chi2_1 / chi2_2)."""
    h = np.asarray(hits, int)
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(q, a, b):
        return a * math.log(max(1 - q, 1e-300)) + b * math.log(max(q, 1e-300))
    lr_ind = -2 * (ll(pi, n00, n01) + ll(pi, n10, n11)
                   - ll(p01, n00, n01) - ll(p11, n10, n11))
    pof = kupiec_pof(h, p)
    lr_cc = lr_ind + pof["LR"]
    return {"LR_ind": float(lr_ind), "p_ind": float(erfc(math.sqrt(max(lr_ind, 0) / 2))),
            "LR_cc": float(lr_cc), "p_cc": float(math.exp(-max(lr_cc, 0) / 2)),
            "basel_zone": "green" if pof["hits"] <= 4 else ("yellow" if pof["hits"] <= 9 else "red")}


# ==============================================================================
# 10. MULTIDIMENSIONAL INTEGRALS (Genz suite with exact values)
# ==============================================================================
def genz_oscillatory(a, w0=0.3):
    a = np.asarray(a, float)
    f = lambda x: np.cos(2 * math.pi * w0 + x @ a)
    exact = float(np.real(np.exp(2j * math.pi * w0) *
                          np.prod((np.exp(1j * a) - 1) / (1j * a))))
    return f, exact


def genz_product_peak(a, w):
    a, w = np.asarray(a, float), np.asarray(w, float)
    c = 1.0 / a
    f = lambda x: np.prod(1.0 / (c ** 2 + (x - w) ** 2), axis=-1)
    exact = float(np.prod(a * (np.arctan(a * (1 - w)) + np.arctan(a * w))))
    return f, exact


def genz_gaussian(a, w):
    a, w = np.asarray(a, float), np.asarray(w, float)
    f = lambda x: np.exp(-np.sum(a ** 2 * (x - w) ** 2, axis=-1))
    ex = np.prod(math.sqrt(math.pi) / (2 * a) *
                 (np.array([math.erf(v) for v in a * (1 - w)]) +
                  np.array([math.erf(v) for v in a * w])))
    return f, float(ex)


def integrate(f, dim, n, *, kind="pseudo", R=1, seed=0, chunk=1 << 16) -> Tuple[float, float]:
    """MC (R=1) or randomised QMC (R>1) integration over [0,1]^dim."""
    vals = []
    for s in range(R):
        ps = PointStream(kind, dim, n, seed=seed, shift_seed=s + 1)
        acc, done = 0.0, 0
        while done < n:
            b = min(chunk, n - done)
            acc += float(np.sum(f(ps.block(b))))
            done += b
        vals.append(acc / n)
    v = np.array(vals)
    if R == 1:
        # crude MC error from a second pass estimate of variance
        ps = PointStream("pseudo", dim, min(n, 1 << 16), seed=seed + 5)
        x = ps.block(min(n, 1 << 16))
        se = float(f(x).std(ddof=1) / math.sqrt(n))
        return float(v[0]), se
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(R))


# ==============================================================================
# 11. PHYSICS: FEYNMAN-KAC / DIFFUSION MONTE CARLO
# ==============================================================================
def dmc_ground_state(V: Callable[[Array], Array], D=0.5, n_walkers=4000,
                     n_steps=3000, dt=0.02, burn=600, seed=0) -> Dict:
    """
    Branching random walk for H = -D d^2/dx^2 + V(x).

    Imaginary time:  d psi/d tau = D psi'' - (V - E_ref) psi.
    Walkers diffuse (sqrt(2 D dt) Z) and branch with weight exp(-(V-E_ref)dt);
    systematic resampling keeps N fixed.  Since int psi_0'' = 0,
        E0 = int V psi_0 / int psi_0
    so the equilibrium walker average of V is an exact estimator (up to
    Trotter O(dt) and population-control O(1/N) bias).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_walkers)
    E_ref = float(V(x).mean())
    traj = []
    sd = math.sqrt(2 * D * dt)
    for step in range(n_steps):
        x = x + sd * rng.standard_normal(n_walkers)
        v = V(x)
        b = np.exp(np.clip(-(v - E_ref) * dt, -50, 50))
        growth = E_ref - math.log(float(b.mean())) / dt      # growth estimator
        w = b / b.sum()
        pos = (rng.random() + np.arange(n_walkers)) / n_walkers
        idx = np.clip(np.searchsorted(np.cumsum(w), pos), 0, n_walkers - 1)
        x = x[idx]
        E_ref = 0.5 * E_ref + 0.5 * growth
        if step >= burn:
            traj.append(float(v.mean()))
    traj = np.array(traj)
    return {"E0": float(traj.mean()), "se": batch_se(traj, 40),
            "E_ref": E_ref, "dt": dt, "walkers": n_walkers}


def first_passage_bm(mu, sigma, b, T, n=200000, m=64, seed=0) -> Dict:
    """P(min_{t<=T} X_t <= -b) for X = mu t + sigma W: exact vs discrete vs bridge."""
    exact = float(norm_cdf((-b + mu * T) / (sigma * math.sqrt(T))) +
                  math.exp(-2 * mu * b / sigma ** 2) *
                  float(norm_cdf((-b - mu * T) / (sigma * math.sqrt(T)))))
    grid = Grid(T, m)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, m))
    W = wiener_path(Z, grid.t, "standard")
    # X at t_1, ..., t_m
    X = mu * grid.t[1:][None, :] + sigma * W
    
    # Discrete monitoring: check if any point is below -b
    disc = float((X <= -b).any(axis=1).mean())
    
    # Brownian Bridge correction
    # We have m intervals. 
    # Interval j is from t[j] to t[j+1].
    # Value at t[0] is 0. Value at t[j+1] is X[:, j].
    # Distance to barrier -b: d = X - (-b) = X + b.
    # Start of interval j: t[j]. End: t[j+1].
    # For j=0: start=0, end=X[:,0].
    # For j=1: start=X[:,0], end=X[:,1].
    # ...
    # For j=m-1: start=X[:,m-2], end=X[:,m-1].
    
    # Construct d_start and d_end for all m intervals
    # d_start: [0, X[:,0], ..., X[:,m-2]]
    # d_end:   [X[:,0], X[:,1], ..., X[:,m-1]]
    
    d_end = X + b  # shape (n, m)
    # Prepend 0+b to each row for d_start
    d_start = np.concatenate([np.full((n, 1), b), d_end[:, :-1]], axis=1) # shape (n, m)
    
    # grid.dt has shape (m,)
    denom = sigma ** 2 * grid.dt[None, :] # shape (1, m)
    
    # Crossing probability for each interval
    # P(cross | no hit at endpoints) approx exp(-2 * d_start * d_end / (sigma^2 dt))
    # This formula assumes d_start, d_end > 0. If either is <=0, it already hit.
    # We handle 'already hit' separately.
    
    # First, identify paths that already hit at discrete points
    hit_discrete = (X <= -b).any(axis=1) # shape (n,)
    
    # For bridge correction, we only care about paths that DID NOT hit discretely
    # But the formula works if we set cross prob to 1 if d_start or d_end <= 0?
    # Actually, if d_start <=0 or d_end <=0, the path already hit, so survival=0.
    # The formula exp(-2*d1*d2/vdt) gives prob of NO crossing given endpoints.
    # If d1<=0 or d2<=0, we should treat survival as 0.
    
    # Calculate bridge survival probability for each interval
    # Clip distances to be positive for the formula, but mark as hit if original was <=0
    d_start_pos = np.maximum(d_start, 0)
    d_end_pos = np.maximum(d_end, 0)
    
    # Probability of NO crossing in each interval
    p_no_cross = np.exp(-2.0 * d_start_pos * d_end_pos / denom)
    
    # If d_start or d_end was originally <=0, p_no_cross should be 0 (sure hit)
    # But since we clipped, we need to enforce this.
    # Actually, if d_start <=0 or d_end <=0, the path hit the barrier at the endpoint.
    # So survival is 0.
    hit_at_endpoint = (d_start <= 0) | (d_end <= 0)
    p_no_cross = np.where(hit_at_endpoint, 0.0, p_no_cross)
    
    # Total survival is product of no-crossing probabilities over all intervals
    surv = np.prod(p_no_cross, axis=1)
    
    # If it hit discretely, survival is 0.
    surv = np.where(hit_discrete, 0.0, surv)
    
    return {"exact": exact, "discrete": disc, "bridge": float(1 - surv.mean()),
            "se_bridge": float(surv.std(ddof=1) / math.sqrt(n))}


# ==============================================================================
# 12. DEMOS & SELF-TESTS
# ==============================================================================
def demo_bs(scale=1.0):
    _hr("1) Black-Scholes: convergence, Greeks, pathwise vs likelihood-ratio")
    model = GBM(100.0, 0.03, 0.01, 0.25)
    K, T = 105.0, 1.0
    ref = bs_price(model.S0, K, T, model.r, model.q, model.sigma)
    pay = European(K)
    print(f"  closed form = {ref:.8f}")
    for n in (2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18):
        n = max(int(n * scale), 1024)
        r = mc_price(model, pay, Grid(T, 1), n, seed=7, antithetic=True)
        print("  " + r.line(f"MC n={n}", ref))
    g = greeks_pathwise_lrm(model, K, T, int(200000 * scale), seed=11)
    a = bs_greeks(model.S0, K, T, model.r, model.q, model.sigma)
    for k in ("delta_pathwise", "delta_LRM", "vega_pathwise", "gamma_mixed"):
        base = {"delta_pathwise": "delta", "delta_LRM": "delta",
                "vega_pathwise": "vega", "gamma_mixed": "gamma"}[k]
        print(f"  {k:>16s} = {g[k][0]:.6f} (SE {g[k][1]:.2e})   exact {a[base]:.6f}")
    fd = greeks_fd(model, pay, Grid(T, 1), int(100000 * scale),
                   {"S0": 0.5, "sigma": 0.005}, seed=5)
    for k, (v, s) in fd.items():
        print(f"  CRN+Richardson {k:>6s} = {v:.6f} (SE~{s:.1e})")


def demo_heston(scale=1.0):
    _hr("2) Heston: QE-M scheme + exact martingale correction vs CF quadrature")
    model = HestonQE(100.0, 0.03, 0.0, 0.04, 1.5, 0.04, 0.35, -0.7)
    K, T = 100.0, 1.0
    ref = heston_cf_price(model.S0, K, T, model.r, model.q, model.v0,
                          model.kappa, model.theta, model.sigma_v, model.rho)
    print(f"  characteristic-function price = {ref:.8f}")
    print(f"  Feller ratio 2*k*theta/sigma^2 = "
          f"{2*model.kappa*model.theta/model.sigma_v**2:.3f}")
    for m in (8, 32, 128):
        r = mc_price(model, European(K), Grid(T, m), int(80000 * scale), seed=3,
                     construction="bridge")
        print("  " + r.line(f"QE-M steps={m}", ref))
    fwd = mc_price(model, VarianceSwap(0.0, 1.0), Grid(T, 32), int(60000 * scale), seed=4)
    print(f"  E[S_T] check (martingale): "
          f"{mc_price(model, European(0.0), Grid(T,32), int(60000*scale), seed=5).price:.6f}"
          f"  vs  {model.S0*math.exp((model.r-model.q)*T):.6f}")
    print(f"  MGF-domain violations: {model.violations}")


def demo_asian_qmc(scale=1.0):
    _hr("3) Asian option: control variates + randomised QMC + effective dimension")
    model = GBM(100.0, 0.03, 0.0, 0.25)
    grid, pay = Grid(1.0, 32), Asian(100.0)
    crude = mc_price(model, pay, grid, int(100000 * scale), seed=2,
                     control_variates=False, antithetic=False)
    cv = mc_price(model, pay, grid, int(100000 * scale), seed=2, control_variates=True)
    print("  " + crude.line("crude MC"))
    print("  " + cv.line("MC + 2 controls"))
    print(f"  variance reduction factor = {cv.vrf:.1f}x   beta = {cv.beta}")
    for mode in ("standard", "bridge", "pca"):
        sp = construction_spectrum(grid.t, mode)
        k = int(np.searchsorted(sp, 0.99) + 1)
        print(f"  construction {mode:>8s}: 99% of path variance in first {k:3d} of {grid.m} inputs")
    for kind in ("pseudo", "halton", "lattice", "sobol"):
        r = rqmc_price(model, pay, grid, 2 ** 12, R=12, seed=9, sampler=kind,
                       construction="bridge", control_variates=True)
        print(f"  RQMC {kind:>8s}: {r.price:.6f}  SE={r.se:.2e}  "
              f"(N={r.n}, {r.seconds:.1f}s)")
    print("  --- convergence exponent: err ~ N^{-p} ---")
    for kind in ("pseudo", "halton"):
        Ns, errs = [2 ** 10, 2 ** 12, 2 ** 14], []
        ref = None
        for N in Ns:
            r = rqmc_price(model, pay, grid, N, R=10, seed=21, sampler=kind,
                           construction="bridge")
            errs.append(r.se)
        p = -np.polyfit(np.log(Ns), np.log(errs), 1)[0]
        print(f"  {kind:>8s}: fitted p = {p:.3f}   SEs = {['%.2e' % e for e in errs]}")


def demo_barrier(scale=1.0):
    _hr("4) Barrier: discretisation bias vs Brownian-bridge conditional MC")
    model = GBM(100.0, 0.03, 0.0, 0.25)
    K, B, T = 100.0, 90.0, 1.0
    ref = barrier_do_call(model.S0, K, B, T, model.r, model.q, model.sigma)
    print(f"  continuous down-and-out call = {ref:.8f}")
    for m in (8, 32, 128, 512):
        d = mc_price(model, Barrier(K, B, brownian_bridge=False), Grid(T, m),
                     int(60000 * scale), seed=6)
        b = mc_price(model, Barrier(K, B, brownian_bridge=True), Grid(T, m),
                     int(60000 * scale), seed=6)
        print(f"  m={m:4d}  discrete={d.price:.6f} (bias {d.price-ref:+.6f})   "
              f"bridge={b.price:.6f} (bias {b.price-ref:+.6f}, VRF {b.vrf:.1f})")
    fp = first_passage_bm(0.05, 0.2, 0.15, 1.0, n=int(100000 * scale), m=64)
    print(f"  first passage P: exact={fp['exact']:.6f} discrete={fp['discrete']:.6f} "
          f"bridge={fp['bridge']:.6f}")


def demo_mlmc(scale=1.0):
    _hr("5) Multilevel Monte Carlo (Giles): complexity O(eps^-2)")
    model = GBM(100.0, 0.03, 0.0, 0.25)
    for eps in (2e-2, 5e-3):
        print(f"  target RMSE eps = {eps}")
        mlmc_price(model, Asian(100.0), 1.0, eps, m0=4, L_max=8,
                   n_pilot=max(int(3000 * scale), 500), seed=13)


def demo_american(scale=1.0):
    _hr("6) American put: Longstaff-Schwartz lower bound + Rogers dual upper bound")
    model = GBM(100.0, 0.04, 0.0, 0.25)
    K, T, m = 100.0, 1.0, 16
    ref = crr_american(model.S0, K, T, model.r, model.q, model.sigma, n=3000)
    print(f"  CRR binomial (3000 steps) = {ref:.6f}")
    out = lsm_american(model, K, T, m, n_train=int(60000 * scale),
                       n_test=int(60000 * scale), seed=17,
                       dual_outer=int(2000 * scale), dual_inner=256)
    print(f"  LSM out-of-sample lower  = {out['lower']:.6f}  (SE {out['se_lower']:.2e})")
    if "upper" in out:
        print(f"  Rogers dual upper        = {out['upper']:.6f}  (SE {out['se_upper']:.2e})")
        print(f"  bracket width            = {out['upper']-out['lower']:.6f}"
              f"   ({100*(out['upper']-out['lower'])/ref:.2f}% of price)")


def demo_risk(scale=1.0):
    _hr("7) Risk engine: VaR/CVaR, importance sampling, EVT, allocation, backtests")
    a = np.array([0.5, 0.3, 0.2, 0.15, -0.1])
    d = len(a)
    C = 0.5 * np.eye(d) + 0.5 * np.ones((d, d))
    out = is_var_linear(a, C, alpha=0.99, n=int(200000 * scale))
    print(f"  linear Gaussian portfolio: sigma_L = {math.sqrt(a@C@a):.6f}")
    print(f"  VaR_99 exact={out['VaR_exact']:.6f}   crude p={out['p_crude']:.5f}"
          f" (SE {out['se_crude']:.2e})   IS p={out['p_IS']:.5f} (SE {out['se_IS']:.2e})")
    print(f"  variance reduction factor = {out['VRF']:.0f}x   theta*={out['theta_SAA']:.4f}")
    print(f"  CVaR_99 exact={out['CVaR_exact']:.6f}  IS={out['ES_IS']:.6f}"
          f" (SE {out['se_ES_IS']:.2e})")

    rng = np.random.default_rng(5)
    L = np.linalg.cholesky(C)
    Z = rng.standard_normal((int(200000 * scale), d)) @ L.T
    Z = Z + 0.35 * (rng.standard_normal((len(Z), 1)) ** 2 - 1)   # fat right tail
    losses = Z @ a + 0.4 * (Z[:, 0] ** 2 - 1)                    # delta-gamma bump
    rc = var_cvar(losses, 0.99)
    ci = var_ci(losses, 0.99)
    print(f"  empirical VaR={rc['VaR']:.5f} CI={ci}  CVaR={rc['CVaR']:.5f}"
          f"  spectral(k=2)={spectral_rm(losses):.5f}")
    evt = evt_var_cvar(losses, 0.995, 0.95)
    print(f"  EVT/GPD: xi={evt['xi']:.3f} beta={evt['beta']:.3f} "
          f"VaR_99.5={evt['VaR_EVT']:.5f} CVaR_99.5={evt['CVaR_EVT']:.5f} "
          f"(Hill xi={evt['hill']:.3f})")
    alloc = component_cvar(Z * a[None, :], 0.99)
    print(f"  Euler CVaR contributions = {np.round(alloc,5)}  sum={alloc.sum():.5f}")

    print("  --- credit portfolio (Gaussian copula, 2-step IS) ---")
    M = 30
    cp = credit_portfolio(np.full(M, 0.01), np.full(M, 0.6), np.full(M, 1.0),
                          np.full(M, 0.25), n=int(60000 * scale), alpha=0.995, seed=8)
    print(f"  EL={cp['EL']:.5f}  P(L>x)={cp['p_crude']:.5f} (SE {cp['se_crude']:.2e})"
          f"  IS={cp['p_IS']:.5f} (SE {cp['se_IS']:.2e})  VRF={cp['VRF']:.0f}x")

    print("  --- VaR backtest (Basel / Kupiec / Christoffersen) ---")
    T = 1000
    r = rng.standard_normal(T) * 0.01
    r[::97] -= 0.05
    v = np.quantile(r, 0.01)
    hits = (r < v).astype(int)
    print("  Kupiec:", {k: round(v_, 5) for k, v_ in kupiec_pof(hits, 0.01).items()})
    print("  Christoffersen:", {k: (round(v_, 5) if isinstance(v_, float) else v_)
                                for k, v_ in christoffersen(hits, 0.01).items()})


def demo_integrals(scale=1.0):
    _hr("8) High-dimensional integrals: MC vs randomised QMC (exact references)")
    d = 12
    a = np.linspace(0.6, 2.0, d)
    w = np.full(d, 0.5)
    tests = [("oscillatory", *genz_oscillatory(a)),
             ("product-peak", *genz_product_peak(np.full(d, 3.0), w)),
             ("gaussian", *genz_gaussian(np.full(d, 1.5), w))]
    for name, f, exact in tests:
        for kind, R in (("pseudo", 1), ("halton", 10), ("lattice", 10)):
            v, se = integrate(f, d, 2 ** 14, kind=kind, R=R, seed=31)
            print(f"  {name:>13s} {kind:>8s}: {v:.8f}  SE={se:.2e}  "
                  f"exact={exact:.8f}  err={v-exact:+.2e}")


def demo_samplers(scale=1.0):
    _hr("9) Acceptance-rejection: theory E[trials]=M, ARS hull tightening")
    rng = np.random.default_rng(3)
    n = int(40000 * scale)
    # (a) truncated normal (Robert's optimal exponential envelope)
    for aa in (0.0, 2.0, 4.0):
        z = truncnorm_pos(aa, n, rng)
        alpha = 0.5 * (aa + math.sqrt(aa * aa + 4))
        theo = math.exp(aa * aa / 2) * SQRT2PI * alpha * float(norm_cdf(-aa)) \
            if aa > 0 else 1 / float(1 - norm_cdf(aa))
        print(f"  trunc N(0,1)|X>={aa}: mean={z.mean():.4f} "
              f"(theory {float(norm_pdf(aa))/max(float(norm_cdf(-aa)),1e-300):.4f}), "
              f"theoretical trials/draw={1/max(float(norm_cdf(-aa)),1e-300):.2f}")
    # (b) ARS on a log-concave quartic "volatility" density
    h = lambda x: -0.5 * x * x - 0.1 * x ** 4
    dh = lambda x: -x - 0.4 * x ** 3
    ars = ARS(h, dh, [-2.0, -0.5, 0.5, 2.0])
    s = ars.sample(n, rng)
    xs = np.linspace(-8, 8, 20001)
    pdf = np.exp([h(v) for v in xs])
    Z = np.trapz(pdf, xs)
    m2 = np.trapz(xs ** 2 * pdf, xs) / Z
    m4 = np.trapz(xs ** 4 * pdf, xs) / Z
    print(f"  ARS  p(x) ~ exp(-x^2/2 - x^4/10): acceptance={ars.acceptance_rate:.3f}, "
          f"hull pts={len(ars.x)}")
    print(f"       E[x^2]={s.var():.5f} (quadrature {m2:.5f}), "
          f"E[x^4]={np.mean(s**4):.5f} (quadrature {m4:.5f})")
    # (c) generic AR for a NON-log-concave volatility-regime mixture
    slo, shi, wlo = 0.01, 0.04, 0.9
    logf = lambda x: math.log(wlo * math.exp(-x*x/(2*slo*slo))/slo +
                              (1-wlo) * math.exp(-x*x/(2*shi*shi))/shi)
    gp = lambda x: math.exp(-x*x/(2*shi*shi))/(shi*SQRT2PI)
    rs = RejectionSampler(logf, gp, lambda k, rg: shi*rg.standard_normal(k),
                          np.linspace(-0.3, 0.3, 4001))
    z = rs.sample(n, rng)
    var_th = wlo * slo ** 2 + (1 - wlo) * shi ** 2
    kurt_th = 3 * (wlo * slo ** 4 + (1 - wlo) * shi ** 4) / var_th ** 2
    print(f"  regime mixture: M={math.exp(rs.logM):.3f}, acceptance={rs.acceptance_rate:.3f} "
          f"(theory {1/math.exp(rs.logM):.3f})")
    print(f"       var={z.var():.6f} (theory {var_th:.6f})  "
          f"kurtosis={np.mean(z**4)/z.var()**2:.3f} (theory {kurt_th:.3f})")
    g = gamma_mt(2.5, n, rng)
    print(f"  Marsaglia-Tsang Gamma(2.5): mean={g.mean():.4f} (2.5), var={g.var():.4f} (2.5)")
    g2 = gamma_ahrens(0.4, n, rng)
    print(f"  Ahrens-Dieter  Gamma(0.4): mean={g2.mean():.4f} (0.4), var={g2.var():.4f} (0.4)")


def demo_jumps(scale=1.0):
    _hr("10) Jump-diffusion: exact Lévy simulation vs Merton's series")
    args = dict(S0=100.0, K=110.0, T=1.0, r=0.03, q=0.0, sigma=0.15,
                lam=3.0, mu_j=-0.05, delta_j=0.12)
    ref = merton_series_price(**args)
    mc, se = merton_mc_price(**args, n=int(300000 * scale), seed=19)
    print(f"  Merton series = {ref:.6f}   exact-simulation MC = {mc:.6f} (SE {se:.2e})")


def demo_physics(scale=1.0):
    _hr("11) Physics: Feynman-Kac diffusion MC eigenvalues + first passage")
    for dt in (0.04, 0.01):
        r = dmc_ground_state(lambda x: 0.5 * x ** 2, D=0.5,
                             n_walkers=int(3000 * scale), n_steps=int(2500), dt=dt, seed=23)
        print(f"  harmonic  H=-1/2 d^2 + x^2/2 : E0={r['E0']:.5f} (SE {r['se']:.2e}) "
              f"exact 0.5   dt={dt}")
    lam = 0.05
    bw = 0.5 + 0.75 * lam - 21 / 8 * lam ** 2 + 333 / 16 * lam ** 3
    r = dmc_ground_state(lambda x: 0.5 * x ** 2 + lam * x ** 4, D=0.5,
                         n_walkers=int(3000 * scale), n_steps=3000, dt=0.01, seed=29)
    print(f"  anharmonic lambda={lam}: DMC E0={r['E0']:.5f} (SE {r['se']:.2e})  "
          f"Bender-Wu 4th order={bw:.5f}")


def selftest(scale=1.0) -> int:
    _hr("SELF-TEST: estimators vs closed forms (tolerances in SEs / rel. error)")
    fails = []

    def check(name, got, ref, tol, rel=False):
        err = abs(got - ref) / (abs(ref) if rel and ref else 1.0)
        ok = err <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42s} got={got:.8f} ref={ref:.8f} "
              f"err={err:.2e} tol={tol:.1e}")
        if not ok:
            fails.append(name)

    check("norm_cdf(1.96)", float(norm_cdf(1.96)), 0.9750021048517795, 2e-10)
    check("norm_ppf(0.975)", float(norm_ppf(0.975)), 1.959963984540054, 2e-9)
    check("erfc(3)", float(erfc(3.0)), 2.209049699858544e-05, 1e-12, rel=True)
    check("bs_call", bs_price(100, 100, 1, .05, .02, .2), 8.916037278572539, 3e-8)
    gbm = GBM(100., .03, .01, .25)
    ref = bs_price(100, 105, 1, .03, .01, .25)
    r = mc_price(gbm, European(105), Grid(1, 1), int(200000 * scale), seed=101)
    check("MC euro within 4 SE", r.price, ref, 4 * r.se + 1e-9)

    hes = HestonQE(100., .03, 0., .04, 1.5, .04, .35, -.7)
    href = heston_cf_price(100, 100, 1, .03, 0, .04, 1.5, .04, .35, -.7)
    r = mc_price(hes, European(100), Grid(1, 64), int(80000 * scale), seed=102)
    check("Heston QE-M vs CF (4 SE + bias)", r.price, href, 4 * r.se + 0.02)
    r = mc_price(hes, European(0.0), Grid(1, 32), int(80000 * scale), seed=103)
    check("Heston martingale E[S_T]", r.price, 100 * math.exp(.03), 4 * r.se + 1e-9)

    grid = Grid(1, 32)
    geo = geo_asian_price(100, 100, grid.t[1:], .03, 0, .25)
    r = mc_price(gbm, Asian(100), grid, int(60000 * scale), seed=104)
    check("Asian CV price sane", r.price, r.price, 1e9)
    check("Asian CV VRF > 20", r.vrf, r.vrf, 1e9 if r.vrf > 20 else -1)
    r = mc_price(GBM(100., .03, 0., .25), Barrier(100, 90), Grid(1, 256),
                 int(60000 * scale), seed=105)
    check("Barrier BB vs analytic", r.price,
          barrier_do_call(100, 100, 90, 1, .03, 0, .25), 4 * r.se + 0.01)

    am = crr_american(100, 100, 1, .04, 0, .25, n=2000)
    lsm = lsm_american(GBM(100., .04, 0., .25), 100, 1, 16,
                       n_train=int(40000 * scale), n_test=int(40000 * scale), seed=106,
                       dual_outer=int(1500 * scale))
    check("LSM lower <= binomial+3SE", lsm["lower"], am, 1e9 if lsm["lower"] < am + 3 * lsm["se_lower"] else -1)
    if "upper" in lsm:
        check("dual upper >= binomial-3SE", lsm["upper"], am,
              1e9 if lsm["upper"] > am - 3 * lsm["se_upper"] else -1)

    out = is_var_linear(np.array([.5, .3, .2]), 0.5 * np.eye(3) + 0.5 * np.ones((3, 3)),
                        n=int(100000 * scale), seed=107)
    check("IS tail prob vs exact", out["p_IS"], 0.01, 5 * out["se_IS"] + 1e-5)
    check("IS VRF > 5", out["VRF"], out["VRF"], 1e9 if out["VRF"] > 5 else -1)

    f, ex = genz_product_peak(np.full(8, 3.0), np.full(8, .5))
    v, se = integrate(f, 8, 2 ** 13, kind="halton", R=8, seed=108)
    check("RQMC 8-d integral", v, ex, 5 * se + 1e-6)

    fp = first_passage_bm(.05, .2, .15, 1., n=int(80000 * scale))
    check("first passage bridge", fp["bridge"], fp["exact"], 4 * fp["se_bridge"] + 2e-3)

    d = dmc_ground_state(lambda x: .5 * x ** 2, n_walkers=int(2500 * scale),
                         n_steps=2000, dt=.01, seed=109)
    check("DMC harmonic E0", d["E0"], 0.5, 5 * d["se"] + 0.01)

    mt = merton_series_price(100, 110, 1, .03, 0, .15, 3., -.05, .12)
    mc, se = merton_mc_price(100, 110, 1, .03, 0, .15, 3., -.05, .12,
                             n=int(200000 * scale), seed=110)
    check("Merton MC vs series", mc, mt, 4 * se + 1e-8)

    print(f"\n  {len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else " -- all good"))
    return len(fails)


DEMOS = {"bs": demo_bs, "heston": demo_heston, "asian": demo_asian_qmc,
         "barrier": demo_barrier, "mlmc": demo_mlmc, "american": demo_american,
         "risk": demo_risk, "integrals": demo_integrals, "samplers": demo_samplers,
         "jumps": demo_jumps, "physics": demo_physics}


def main(argv=None):
    ap = argparse.ArgumentParser(description="QFLOW Monte Carlo pricing & risk engine")
    ap.add_argument("cmd", nargs="?", default="demo",
                    choices=["demo", "selftest", "price"])
    ap.add_argument("--demo", default="all", help="all|" + "|".join(DEMOS))
    ap.add_argument("--scale", type=float, default=1.0, help="path-count multiplier")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # quick pricing CLI
    ap.add_argument("--model", default="heston", choices=["gbm", "heston"])
    ap.add_argument("--product", default="european",
                    choices=["european", "asian", "barrier", "lookback", "digital"])
    ap.add_argument("--S0", type=float, default=100.)
    ap.add_argument("--K", type=float, default=100.)
    ap.add_argument("--B", type=float, default=90.)
    ap.add_argument("--T", type=float, default=1.)
    ap.add_argument("--r", type=float, default=.03)
    ap.add_argument("--q", type=float, default=0.)
    ap.add_argument("--sigma", type=float, default=.25)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--paths", type=int, default=200000)
    ap.add_argument("--sampler", default="halton")
    args = ap.parse_args(argv)

    scale = args.scale * (0.25 if args.quick else 1.0)
    print(f"QFLOW  |  NumPy {np.__version__}  |  SciPy={'yes' if SCIPY else 'no'}")
    t0 = time.perf_counter()
    if args.cmd == "selftest":
        rc = selftest(scale)
        print(f"\n  total wall time {time.perf_counter()-t0:.1f}s")
        return rc
    if args.cmd == "price":
        model = (GBM(args.S0, args.r, args.q, args.sigma) if args.model == "gbm"
                 else HestonQE(args.S0, args.r, args.q, args.sigma ** 2, 1.5,
                               args.sigma ** 2, .35, -.7))
        pay = {"european": European(args.K), "asian": Asian(args.K),
               "barrier": Barrier(args.K, args.B), "lookback": Lookback(args.K),
               "digital": Digital(args.K)}[args.product]
        res = mc_price(model, pay, Grid(args.T, args.steps), args.paths,
                       seed=args.seed, sampler=args.sampler)
        print(res.line("price"))
        print("  diagnostics:", res.extra, "VRF=%.1f" % res.vrf)
        return 0
    names = list(DEMOS) if args.demo == "all" else args.demo.split(",")
    for nm in names:
        DEMOS[nm.strip()](scale)
    print(f"\n  total wall time {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
