# Solites Quantitative Finance Research Engine (`QRE`) 

A self-contained, NumPy-based quantitative finance, numerical integration, and statistical physics research engine. Built from the ground up with minimal dependencies (optional `scipy` acceleration), this engine implements 44 foundational stochastic models, variance reduction techniques, quasi-Monte Carlo generators, risk management frameworks, and quantum mechanical solvers—all cross-validated against exact closed-form analytical benchmarks.

---

## ⚠️ Release Note: Beta Version


This engine implements cutting-edge numerical methods (MLMC, ARS, Heston QE-M) that are typically found only in proprietary research libraries. While the mathematical logic has been rigorously validated against closed-form solutions and academic benchmarks, this is a **brand-new release**.

### 🐛 Known Limitations & Reporting
*   **Numerical Edge Cases:** Extreme parameter values (e.g., very high volatility or near-zero time to maturity) may require manual tuning of integration nodes or step sizes.
*   **Performance:** On mobile devices (Android/iOS), heavy modules like *Credit Portfolio Importance Sampling* or *American Option Dual Bounds* may take several minutes to complete due to nested optimization loops.
*   **Bug Reports:** If you encounter a crash, a numerical instability, or a logic error, please report it! 
    *   Include the **command** you ran.
    *   Include the **error traceback**.
    *   Include the **parameters** used.

> *"New code always comes with a few kinks. Your feedback helps make Solites the most robust open-source QRE available."*

---

## Quick Start & CLI Usage

### Requirements
* Python 3.8+
* `numpy` (Required)
* `scipy` (Optional, used for hardware-accelerated special functions and scrambled Sobol' nets)

### Command-Line Interface
Run the full benchmark suite, execute self-tests against analytical closed forms, or price specific derivatives directly from the terminal:

```bash
# Run all 11 demonstration modules
python solites.py demo

# Run a specific demo module (e.g., heston, asian, risk, mlmc, physics)
python solites.py demo --demo heston

# Run a fast smoke test (25% path count)
python solites.py demo --quick

# Run automated unit tests verifying estimators against closed-form formulas
python solites.py selftest

# Price a custom option via CLI
python solites.py price --model heston --product barrier --S0 100 --K 100 --B 90 --T 1.0 --paths 200000
```

---

# Packaging Solites: Creating `__init__.py`

A step-by-step guide to turning `solites.py` into an importable Python package —
on **Android (Pydroid 3)** and on **Windows / macOS / Linux**.

---

## 1. Why you need `__init__.py`

Without it, Python 3 still lets you `import solites` (implicit *namespace
packages*), but the folder exposes **nothing**:

```python
import solites
solites.GBM        # AttributeError — the folder imported, the code didn't
```

Verified behaviour:

| Folder contents | `import solites` | `solites.GBM` |
|---|---|---|
| `solites/solites.py` only | succeeds | ❌ missing |
| `solites/solites.py` + `__init__.py` | succeeds | ✅ works |

`__init__.py` is the file that says *"this folder is a package, and here is its
public API."*

---

## 2. Target structure

```text
Solites-QRE/                 ← project root (git repo)
├── solites/                 ← the package
│   ├── __init__.py          ← public API + version
│   └── solites.py           ← the engine (~2,300 lines)
├── docs/
│   └── INSTALL.md           ← this file
├── examples/
│   └── plot_convergence.py
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

> The folder is `solites/` and the file inside is `solites.py`. That repetition
> is normal (`numpy/numpy/…` does the same). It means users write
> `from solites import GBM`, not `from solites.solites import GBM`.

---

## 3. Create the file

### 3a. Android — Pydroid 3 (file manager method)

The terminal heredoc approach often fails on Android due to scoped storage.
Use a file manager instead:

1. Open **Files / ZArchiver / Solid Explorer**.
2. Navigate to `Internal Storage → Solites-QRE → solites`.
3. Tap **New → File**, name it exactly `__init__.py`.
4. Open it with a text editor (**QuickEdit**, **Acode**, or Pydroid's editor).
5. Paste the code from §4, save as **UTF-8**, **LF** line endings.

**Filename traps — all four of these silently fail:**

| Wrong name | Result |
|---|---|
| `init.py` | ❌ no exports |
| `_init_.py` (single underscores) | ❌ no exports |
| `__init__.PY` (capital extension) | ❌ no exports |
| `__init__.py.txt` (editor added `.txt`) | ❌ no exports |

It must be **two underscores**, `init`, **two underscores**, `.py` — all lowercase.

### 3b. Android — Pydroid 3 (terminal method)

```bash
cd /storage/emulated/0/Solites-QRE
mkdir -p solites
cp solites.py solites/solites.py     # if the engine is still at root
```

Then create `solites/__init__.py` with the editor (§3a) — heredocs are unreliable
on Android.

### 3c. Desktop — Windows

```powershell
cd C:\Users\<you>\Solites-QRE
mkdir solites
move solites.py solites\solites.py
notepad solites\__init__.py
```

Paste §4, then **File → Save As → Encoding: UTF-8**.

### 3d. Desktop — macOS / Linux

```bash
cd ~/Solites-QRE
mkdir -p solites
mv solites.py solites/solites.py
cat > solites/__init__.py << 'EOF'
# paste the code from §4 here
EOF
```

---

## 4. The `__init__.py` code

Every name below exists in `solites.py`. Verified: **48 functions/classes**,
**50 entries** in `__all__` including the two metadata strings.

```python
"""
Solites Quantitative Finance Research Engine
============================================
Monte Carlo pricing, risk analytics, and computational-statistics routines.

Author : solite
Version: 1.0.0-beta
"""

__version__ = "1.0.0-beta"
__author__ = "solite"

from .solites import (
    # Models
    GBM, MultiGBM, HestonQE, HestonEuler,
    # Engines
    mc_price, rqmc_price, mlmc_price, simulate_paths,
    # Payoffs
    European, Digital, Asian, Barrier, Lookback, VarianceSwap,
    # Risk
    var_cvar, var_ci, component_cvar, spectral_rm, evt_var_cvar,
    is_var_linear, credit_portfolio, kupiec_pof, christoffersen,
    # Greeks
    greeks_fd, greeks_pathwise_lrm,
    # Special functions & solvers
    norm_cdf, norm_ppf, norm_pdf, erfc,
    bisect_monotone, golden_min, batch_se,
    # Low-discrepancy sequences
    halton, lattice_cbc, PointStream,
    # Wiener path constructions
    wiener_path, construction_spectrum,
    # Closed forms
    bs_price, bs_greeks, heston_cf_price,
    geo_asian_price, barrier_do_call, crr_american,
    # Jump diffusion
    merton_mc_price, merton_series_price,
    # Physics / Feynman-Kac
    dmc_ground_state, first_passage_bm,
    # Utils
    Grid,
)

__all__ = [
    "__version__", "__author__",
    "GBM", "MultiGBM", "HestonQE", "HestonEuler",
    "mc_price", "rqmc_price", "mlmc_price", "simulate_paths",
    "European", "Digital", "Asian", "Barrier", "Lookback", "VarianceSwap",
    "var_cvar", "var_ci", "component_cvar", "spectral_rm", "evt_var_cvar",
    "is_var_linear", "credit_portfolio", "kupiec_pof", "christoffersen",
    "greeks_fd", "greeks_pathwise_lrm",
    "norm_cdf", "norm_ppf", "norm_pdf", "erfc",
    "bisect_monotone", "golden_min", "batch_se",
    "halton", "lattice_cbc", "PointStream",
    "wiener_path", "construction_spectrum",
    "bs_price", "bs_greeks", "heston_cf_price",
    "geo_asian_price", "barrier_do_call", "crr_american",
    "merton_mc_price", "merton_series_price",
    "dmc_ground_state", "first_passage_bm",
    "Grid",
]
```

### Two rules that matter

**Rule 1 — use the relative import `.solites`, not `solites`.**

```python
from .solites import GBM      # ✅ correct
from solites  import GBM      # ❌ ImportError: partially initialized module
                              #    (most likely due to a circular import)
```

The absolute form makes the package import *itself* mid-initialisation.

**Rule 2 — never wrap the import in `try / except ImportError: pass`.**

This pattern is common in tutorials and it is harmful. Demonstrated:

```python
try:
    from .solites import (
        GBM,
        MertonJumpDiffusion,   # ← typo, this name does not exist
        mc_price,
    )
except ImportError:
    pass
```

Result: **no error at all**, but `mc_price` is silently missing — because one
bad name aborts the *entire* import statement, and the `except` swallows it.
You then debug a "missing function" for an hour.

Remove the `try/except`. The same typo now fails loudly and instantly:

```text
ImportError: cannot import name 'MertonJumpDiffusion' from 'solites.solites'
```

That error message is the fix. Let it happen.

---

## 5. Verify the install

Save this as `check_install.py` in the **project root**:

```python
import solites as s

print("version :", getattr(s, "__version__", "unknown"))
names = getattr(s, "__all__", None)
if names is None:
    print("WARNING: __all__ missing -> __init__.py probably not loaded")
else:
    print("exports :", len(names))
    print("missing :", [n for n in names if not hasattr(s, n)] or "none")
print("location:", s.__file__)
```

Run it:

```bash
cd /storage/emulated/0/Solites-QRE     # Android
cd ~/Solites-QRE                       # macOS / Linux
python check_install.py
```

Expected:

```text
version : 1.0.0-beta
exports : 50
missing : none
location: /storage/emulated/0/Solites-QRE/solites/__init__.py
```

If you see `location: None` and the `__all__` warning, Python found a
*namespace* package — your `__init__.py` is misnamed or in the wrong folder.

Quick one-liner alternative:

```bash
python -c "import solites; print(solites.__version__, len(solites.__all__))"
```

---

## 6. Using it from your own scripts

### Option A — run from the project root (simplest)

Python puts the current directory on `sys.path`, so this just works:

```bash
cd /storage/emulated/0/Solites-QRE
python my_script.py
```

```python
from solites import GBM, European, mc_price, Grid

model  = GBM(S0=100, r=0.03, q=0.01, sigma=0.25)
payoff = European(K=105, kind="call")
grid   = Grid(T=1.0, m=1)

r = mc_price(model, payoff, grid, n=100_000, seed=42)
print(f"{r.price:.6f} ± {r.se:.6f}")
```

### Option B — run from anywhere (`sys.path` bootstrap)

Put this at the **top** of your script, before importing solites:

```python
import sys
from pathlib import Path

ROOT = Path("/storage/emulated/0/Solites-QRE")   # Android
# ROOT = Path.home() / "Solites-QRE"             # macOS / Linux
# ROOT = Path(r"C:\Users\you\Solites-QRE")       # Windows

sys.path.insert(0, str(ROOT))

from solites import GBM, mc_price
```

> **Use `insert(0, ...)`, not `append(...)`.** `append` puts your project *last*,
> so any stray `solites.py` sitting next to your script wins instead. Confirmed:
> with `append`, Python imported the wrong file; with `insert(0)`, the real
> package loaded. `Path` keeps it portable across Android/Windows separators.

### Option C — permanent registration via `.pth` (no bootstrap needed)

A `.pth` file containing one directory path adds that directory to `sys.path`
at interpreter startup. Find your site-packages:

```bash
python -c "import site; print(site.getsitepackages()[0])"
```

Create `solites.pth` there containing a single line:

```text
/storage/emulated/0/Solites-QRE
```

Pydroid 3 typical location:

```bash
echo "/storage/emulated/0/Solites-QRE" > \
  /data/user/0/ru.iiec.pydroid3/files/aarch64-linux-android/lib/python3.13/site-packages/solites.pth
```

Restart Pydroid. Now `from solites import GBM` works in **any** script.

> The `.pth` file must live in a real **site-packages** directory — Python only
> processes `.pth` files there, not in arbitrary folders on `PYTHONPATH`.

### Option D — proper install (desktop, recommended)

With `pyproject.toml` present:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```

`-e` (editable) means edits to `solites.py` take effect immediately, no
reinstall. Verified working: after `pip install -e .`, `import solites` succeeds
from any directory.

Minimal `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "solites"
version = "1.0.0-beta"
requires-python = ">=3.9"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
fast = ["scipy>=1.10"]
plot = ["matplotlib>=3.5"]

[tool.setuptools]
packages = ["solites"]
```

Then `pip install -e ".[fast,plot]"` pulls the optional extras.

---

## 7. Optional: `python -m solites`

Add `solites/__main__.py` to make the package runnable as a module:

```python
from .solites import main
import sys

if __name__ == "__main__":
    sys.exit(main())
```

Now both of these work and are equivalent:

```bash
python solites/solites.py selftest
python -m solites selftest
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'solites'` | Not running from root, and no bootstrap/install | §6 Option A, B, C, or D |
| Imports fine, but `solites.GBM` missing | `__init__.py` misnamed or absent → namespace package | Check exact filename (§3a table) |
| `solites.__file__` is `None` | Same as above | Same |
| Some functions present, others missing | `try/except ImportError: pass` swallowed a typo | Delete the `try/except` (§4 Rule 2) |
| `ImportError: partially initialized module` | Used `from solites import …` inside `__init__.py` | Change to `from .solites import …` |
| `ImportError: cannot import name 'X'` | `X` isn't defined in `solites.py` | Use the real name (e.g. `merton_mc_price`, not `MertonJumpDiffusion`) |
| `TabError: inconsistent use of tabs and spaces` | Editor mixed tabs and spaces | Set editor to **4 spaces**, no tabs |
| `IndentationError: unindent does not match…` | First line of a pasted block lost its indent | Re-indent the whole block consistently |
| Weird characters like `â€”` or `LÃ©vy` | File saved/read as Latin-1 instead of UTF-8 | Re-save as UTF-8; keep `# -*- coding: utf-8 -*-` |
| Edits to `solites.py` seem ignored | Stale bytecode | Delete `solites/__pycache__/`, or run `python -B` |
| Wrong `solites` imported | A stray `solites.py` shadows the package | Use `insert(0, …)`; print `solites.__file__` to confirm |
| `ModuleNotFoundError: numpy` | Dependency missing | `pip install numpy scipy` |

**Universal first diagnostic** — always print where the import came from:

```python
import solites
print(solites.__file__)
```

If that path isn't your `Solites-QRE/solites/__init__.py`, you're importing
something else entirely.

---

## 9. Checklist

- [ ] Folder is `solites/`, engine is `solites/solites.py`
- [ ] File is exactly `__init__.py` (2 underscores each side, lowercase `.py`)
- [ ] Saved as UTF-8, LF endings, 4-space indentation, no tabs
- [ ] Imports use `from .solites import …` (leading dot)
- [ ] No `try / except ImportError: pass` anywhere
- [ ] Every name in `__all__` exists in `solites.py`
- [ ] `python check_install.py` prints `exports: 50`, `missing: none`
- [ ] `solites.__file__` points at your project

---

## 10. Note on `__all__`

`__all__` controls `from solites import *` and documents your public surface.
It does **not** hide other attributes — `solites.some_private_helper` still
works if it was imported. Treat it as documentation plus star-import control,
not access control.

If you'd rather not hand-maintain the list, you can generate it:

```python
from . import solites as _core
from .solites import *          # noqa: F401,F403
__all__ = [n for n in dir(_core) if not n.startswith("_")]
```

This works, but it also re-exports third-party names like `np` and `math`.
The explicit list in §4 is cleaner for a library — keep it.

---

## 🛠️ Installation
```bash
    git clone https://github.com/christianlacanilao08-max/Solites-QRE.git
    cd Solites-QRE
    pip install -r requirements.txt
```

---

## Complete Compendium of Implemented Theories

The engine encompasses **44 distinct theories, models, and mathematical frameworks** organized across seven core domains:

### 1. Stochastic Processes & Asset Pricing Models
1. **Geometric Brownian Motion (GBM) & Multi-Asset GBM:** Continuous-time log-normal diffusion processes, utilizing Cholesky decomposition to correlate multi-factor Brownian motions.
2. **Black-Scholes-Merton (BSM) Theory:** Closed-form analytical pricing and exact sensitivities (Delta, Gamma, Vega, Theta, Rho) for European options.
3. **Heston Stochastic Volatility Model:** Mean-reverting Cox-Ingersoll-Ross (CIR) variance process coupled with correlated asset price returns.
4. **Andersen’s (2008) Quadratic-Exponential (QE-M) Scheme:** Moment-matched discretization of the CIR variance process paired with an exact conditional martingale correction to achieve $\mathcal{O}(\Delta t^2)$ weak bias.
5. **Full-Truncation Euler Discretization:** Absorbing/truncating negative variance excursions in stochastic volatility paths, utilized as a baseline for multilevel coupling.
6. **Gil-Pelaez Fourier Inversion & Albrecher’s "Little Trap":** Characteristic function integration via Gauss-Legendre quadrature using the numerically stable branch cut for semi-analytical Heston pricing.
7. **Merton Jump-Diffusion Model (1976):** Compound Poisson jump processes combined with continuous Brownian diffusion, solved via both exact Lévy simulation and Merton's Poisson-weighted Black-Scholes infinite series.
8. **Cox-Ross-Rubinstein (CRR) Binomial Tree:** Memory-efficient $\mathcal{O}(n)$ backward-induction lattice for pricing American options with early exercise.
9. **Exotic & Path-Dependent Derivative Pricing:** Formulations for European, Digital (Binary), Arithmetic & Geometric Asian, Continuous Barrier (Reiner-Rubinstein analytical formulas), Lookback (fixed/floating), and Variance Swaps.

### 2. Monte Carlo & Variance Reduction Theory
10. **Antithetic Variates:** Pairing standard normal draws $Z$ with $-Z$ to induce negative correlation and cancel odd-order error terms.
11. **Control Variates:** Optimal linear regression adjustment ($Y_{cv} = Y - \beta'(C - \mathbb{E}[C])$) using known martingale expectations and closed-form prices (e.g., geometric Asian options) with pilot-slicing to eliminate regression bias.
12. **Giles Multilevel Monte Carlo (MLMC):** Coupling coarse and fine Brownian increments across hierarchical time grids ($m_l = m_0 2^l$) to reduce computational complexity from $\mathcal{O}(\epsilon^{-3})$ to $\mathcal{O}(\epsilon^{-2})$.
13. **Brownian Bridge Conditional Monte Carlo:** Exact analytical survival probabilities ($1 - \exp(-2 d_i d_{i+1} / v \Delta t)$) between discrete grid points to eliminate continuous barrier-crossing discretization bias.
14. **Standard Random Walk Path Construction:** Sequential forward accumulation of independent Brownian increments.
15. **Brownian Bridge Path Construction:** Hierarchical time-interval bisection to concentrate variance into the initial uniform draws.
16. **Principal Component Analysis (PCA) / Karhunen-Loève Expansion:** Eigendecomposition of the Brownian covariance matrix $\text{Cov}(W_{t_i}, W_{t_j}) = \min(t_i, t_j)$ to maximize effective dimension reduction.
17. **Longstaff-Schwartz Least-Squares Monte Carlo (LSM):** Cross-sectional polynomial regression (Laguerre and power bases) on in-the-money paths to estimate continuation values for optimal stopping (primal lower bound).
18. **Rogers / Andersen-Broadie Dual Method:** Martingale duality utilizing nested Monte Carlo simulations to construct a strict upper bound for American option prices.
19. **Common Random Numbers (CRN) & Richardson Extrapolation:** Synchronized random streams across bumped parameters paired with $\frac{4D(h/2) - D(h)}{3}$ extrapolation to reduce finite-difference Greek bias and variance.
20. **Infinitesimal Perturbation Analysis (IPA / Pathwise Derivative Method):** Differentiating discounted payoffs directly along simulated trajectories for smooth payoffs.
21. **Likelihood Ratio Method (LRM / Score Function Method):** Differentiating the underlying probability density rather than the payoff to compute Greeks for discontinuous contracts (e.g., digitals) and mixed pathwise-LRM Gamma.

### 3. Quasi-Monte Carlo (QMC) & Discrepancy Theory
22. **Randomized Quasi-Monte Carlo (RQMC / Cranley-Patterson Shifts):** Applying modulo-1 random shifts across independent replicates of deterministic low-discrepancy nets to restore unbiasedness and Central Limit Theorem confidence intervals.
23. **Halton Sequences:** Digit-scrambled Van der Corput sequences constructed in prime bases.
24. **Rank-1 Lattice Rules (CBC Construction):** Component-by-Component construction minimizing shift-averaged worst-case error in weighted Korobov spaces via second-degree Bernoulli polynomials ($B_2$).
25. **Sobol' Sequences:** Scrambled $(t,m,s)$-nets in base 2.
26. **Genz Test Integrand Suite:** High-dimensional benchmark integrands (Oscillatory, Product-Peak, and Gaussian) with known analytical solutions.

### 4. Acceptance-Rejection & Exact Sampling Theory
27. **Adaptive Rejection Sampling (ARS - Gilks & Wild 1992):** Dynamically tightening piecewise exponential upper hulls and secant lower squeezes for arbitrary log-concave target densities.
28. **Robert’s (1995) Optimal Truncated Normal Sampler:** Translated exponential proposal distribution with optimal rate parameter $\alpha^* = \frac{a + \sqrt{a^2+4}}{2}$ guaranteeing $\ge 75\%$ acceptance across all tail thresholds.
29. **Marsaglia-Tsang Squeeze Rejection:** Fast transformation-rejection algorithm for $\text{Gamma}(\alpha \ge 1, 1)$ variates.
30. **Ahrens-Dieter GS Rejection & Boosting:** Two-branch envelope sampling and shape-boosting ($G(\alpha) = G(\alpha+1)U^{1/\alpha}$) for $\text{Gamma}(0 < \alpha < 1, 1)$.

### 5. Quantitative Risk Management & Extreme Value Theory (EVT)
31. **Coherent & Spectral Risk Measures:** Empirical Value-at-Risk (VaR), Conditional Value-at-Risk / Expected Shortfall (Rockafellar-Uryasev consistent formulation), Euler capital allocation (Component CVaR), and exponential/power-law Spectral Risk Measures.
32. **Order-Statistic Confidence Intervals:** Distribution-free non-parametric quantile confidence intervals derived from binomial order statistics.
33. **Pickands–Balkema–de Haan Theorem (EVT / POT):** Peaks-over-Threshold tail modeling via the Generalized Pareto Distribution (GPD), fitted using profile maximum likelihood.
34. **Hill Tail-Index Estimator:** Maximum likelihood tail-index ($\xi$) estimation for heavy-tailed loss distributions.
35. **Esscher / Girsanov Exponential Tilting:** Change-of-measure importance sampling for linear Gaussian portfolios using both mean-matching and Sample Average Approximation (SAA) variance minimization.
36. **Gaussian One-Factor Copula & Glasserman-Li Two-Step IS:** Vasicek / CreditMetrics structural credit portfolio model accelerated via systematic factor shifting and conditional Bernoulli exponential twisting.
37. **Kupiec Proportion-of-Failures (POF) Test:** Unconditional coverage likelihood-ratio test ($\chi^2_1$) for VaR exception frequency.
38. **Christoffersen Markov Chain Tests:** First-order Markov independence and joint conditional coverage likelihood-ratio tests ($\chi^2_1$ / $\chi^2_2$) alongside Basel traffic-light zone classification.

### 6. Statistical Physics & Quantum Mechanics
39. **Feynman-Kac Formula & Imaginary-Time Schrödinger Equation:** Mapping quantum ground-state eigenvalue problems ($H = -D\nabla^2 + V(x)$) to branching diffusion processes ($\partial_\tau \psi = D\psi'' - (V - E_{\text{ref}})\psi$).
40. **Diffusion Monte Carlo (DMC):** Population-controlled branching random walks utilizing systematic resampling and growth estimators to solve for quantum harmonic and anharmonic ground states.
41. **Bender-Wu Perturbation Theory:** Fourth-order asymptotic perturbation series for the quantum quartic anharmonic oscillator ground-state energy.
42. **First-Passage Time Theory (Reflection Principle):** Exact analytical hitting probabilities for drifted Brownian motion ($\mathbb{P}(\min_{t \le T} X_t \le -b)$) benchmarked against discrete and Brownian-bridge simulations.

### 7. Numerical Analysis & Special Function Approximations
43. **Acklam’s Rational Approximation + Halley’s Method:** High-precision inverse standard normal CDF ($\Phi^{-1}$) achieving $\sim 10^{-15}$ relative accuracy without external libraries.
44. **Special Function Series, Solvers & Batch Means:**
    * **Incomplete Gamma & $\text{erfc}$:** Convergent power series for small arguments paired with Modified Lentz continued fractions for large arguments.
    * **Derivative-Free Optimization & Root-Finding:** Golden-section unimodal minimization and monotone bisection solvers.
    * **Batch-Means Standard Error:** Non-overlapping batch estimation to account for serial autocorrelation in Markov chain and walker trajectories.

---

## License
Open-source / Public Domain (`anyones free to use this :)`).
