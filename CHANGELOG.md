# Changelog

All notable changes to Solites QRE are documented in this file.

## [1.0.0-beta] - 2026-11-15

### 🐛 Bug Fixes

#### ARS Sampler (`ARS.sample`)
- **Fixed:** `ValueError: truth value of an array is ambiguous` on line 395.
  - **Cause:** `if abs(m) <= 1e-12:` failed because `m = self.dv[j]` is a vector (one slope per sampled point), not a scalar.
  - **Fix:** Replaced scalar `if/else` with vectorized `np.where` using per-element mask `flat = np.abs(m) <= 1e-12`.
- **Fixed:** `RuntimeWarning: overflow encountered in expm1` on line 398.
  - **Cause:** `expo = m * (r - l)` could reach +700 on wide outer hull segments, causing `expm1(700)` to overflow when multiplied by `frac`.
  - **Fix:** Implemented numerically stable reflection identity — for positive slopes, integrate from the right endpoint (`y_hi`) so the exponent argument is always ≤ 0.

#### NumPy 2.x Compatibility (`demo_samplers`)
- **Fixed:** `AttributeError: module 'numpy' has no attribute 'trapz'` on line 2089.
  - **Cause:** `np.trapz` was removed in NumPy 2.0+; `getattr(np, "trapezoid", np.trapz)` eagerly evaluates the default argument, crashing before `getattr` can return.
  - **Fix:** Changed to two-step safe fallback: `_trap = getattr(np, "trapezoid", None)` followed by explicit `if _trap is None: _trap = np.trapz`.

#### Merton Jump-Diffusion (`merton_mc_price`)
- **Fixed:** MC price ~20.07 vs series reference ~7.54 (err/se > 30σ).
  - **Cause:** Jump generation used `rng.normal(mu_j, delta_j, n) * N + delta_j * sqrt(N) * randn(n)`, adding the jump mean **twice** (once via `normal(mu_j,...)*N`, once implicitly through the second normal draw).
  - **Fix:** Corrected to single normal draw: `jumps = mu_j * N + delta_j * sqrt(N) * rng.standard_normal(n)`.

#### First Passage Bridge (`first_passage_bm`)
- **Fixed:** Bridge estimate = 1.000000 vs exact = 0.370710.
  - **Cause 1:** Survival probability computed as `np.prod(cross)` instead of `np.prod(1 - cross)`, making survival ≈ 1e-23 → hit probability ≈ 1.0.
  - **Cause 2:** Reflection principle formula had Φ terms in wrong order.
  - **Fix:** Corrected to `surv = np.prod(1.0 - cross, axis=1)` and swapped Φ arguments to `Φ((-b-μT)/σ√T) + exp(...)·Φ((-b+μT)/σ√T)`.

#### Heston Martingale Self-Test
- **Fixed:** Test reported E[S_T] = 100.000000 vs expected 103.045453.
  - **Cause:** `European(K=0)` payoff equals its own control variate (`df * S_T`), causing β = 1 and `y_adj ≡ mean = S₀` exactly. The model drift was correct; the test design annihilated the signal.
  - **Fix:** Added `control_variates=False` to the martingale check and changed reference to discounted expectation `S₀·exp(-qT)`.

#### Black-Scholes Self-Test Reference
- **Fixed:** `bs_call` test failed with err = 3.11e-01.
  - **Cause:** Reference constant `8.916037278572539` corresponded to `(r=0.02, q=0.00)`, but test called `bs_price(100, 100, 1, 0.05, 0.02, 0.2)`.
  - **Fix:** Updated reference to correct value `9.227005508154036`.

#### LSM Dual Upper Bound (`lsm_american`)
- **Fixed:** `ValueError: matmul input operand dimension mismatch` in `vtilde()`.
  - **Cause:** Inner simulation paths `Sn` had shape `(n_o, n_i)`, but `_basis()` produced incompatible dimensions for matrix multiplication with `betas[j]`.
  - **Fix:** Added 2D reshape logic: flatten → apply basis → matmul → reshape back to `(n_rows, n_cols)`.

#### MLMC Array Operations (`mlmc_price`)
- **Fixed:** `TypeError` from `math.sqrt` on array.
  - **Cause:** `math.sqrt(V / C)` where `V` and `C` are arrays.
  - **Fix:** Changed to `np.sqrt(V / C)`.

### ✅ Validation
- All 18 self-test checks now pass with **0 failures**.
- All 11 demos produce statistically correct results within expected confidence intervals.
- Verified against closed-form solutions: Black-Scholes, Heston CF, Merton series, Reiner-Rubinstein barrier, reflection principle, geometric Asian, CRR binomial, DMC harmonic oscillator.
