import sys
sys.path.insert(0, "/storage/emulated/0/Solites-QRE")

import numpy as np
import matplotlib.pyplot as plt
from solites import GBM, European, mc_price, Grid, bs_price

model = GBM(S0=100, r=0.03, q=0.01, sigma=0.25)
payoff = European(K=105, kind="call")
grid = Grid(T=1.0, m=1)

# Exact reference for comparison
exact_price = bs_price(model.S0, payoff.K, grid.T, model.r, model.q, model.sigma)

# Run MC at exponentially increasing path counts
path_counts = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
prices, ses, cis_lo, cis_hi = [], [], [], []

for n in path_counts:
    result = mc_price(model, payoff, grid, n=n, seed=42)
    prices.append(result.price)
    ses.append(result.se)
    cis_lo.append(result.ci[0])
    cis_hi.append(result.ci[1])

fig, ax = plt.subplots(figsize=(10, 6))

# Plot MC estimates with error bars
ax.errorbar(
    path_counts, prices, yerr=ses, fmt='-o', color='#2196F3',
    ecolor='#BBDEFB', elinewidth=2, capsize=4, markersize=6,
    label='MC Price ± SE'
)

# Fill 95% CI region
ax.fill_between(path_counts, cis_lo, cis_hi, alpha=0.15, color='#2196F3')

# Plot exact reference line
ax.axhline(y=exact_price, color='#E53935', linestyle='--', linewidth=2,
           label=f'Exact BS Price = {exact_price:.6f}')

# Formatting
ax.set_xscale('log')
ax.set_xlabel('Number of Paths (N)', fontsize=13)
ax.set_ylabel('Option Price', fontsize=13)
ax.set_title(f'Solites QRE — Black-Scholes MC Convergence\n'
             f'S={model.S0}, K={payoff.K}, T={grid.T}, σ={model.sigma}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, which='both', alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('/storage/emulated/0/Solites-QRE/mc_convergence.png', dpi=150)
plt.show()

print(f"\n✅ Chart saved to /storage/emulated/0/Solites-QRE/mc_convergence.png")
print(f"   Exact Price: {exact_price:.6f}")
print(f"   MC @ N=100k: {prices[-1]:.6f} ± {ses[-1]:.6f}")
