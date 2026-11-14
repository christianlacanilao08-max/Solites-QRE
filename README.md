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
