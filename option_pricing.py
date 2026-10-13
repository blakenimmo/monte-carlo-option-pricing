"""
Monte Carlo pricing of a European call option, validated against Black-Scholes.

Blake Nimmo, October 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------- parameters
S0 = 100.0     # initial stock price (£)
K = 105.0      # strike price (£)
r = 0.05       # risk-free rate (per year)
sigma = 0.20   # volatility (per year)
T = 1.0        # time to expiry (years)


# ---------------------------------------------------------------- models
def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng):
    """Simulate risk-neutral geometric Brownian motion paths (exact log-normal steps)."""
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S


def mc_call(S0, K, r, sigma, T, n, rng):
    """Monte Carlo call price and standard error, sampling terminal prices directly."""
    Z = rng.standard_normal(n)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0)
    disc = np.exp(-r * T)
    return disc * payoffs.mean(), disc * payoffs.std(ddof=1) / np.sqrt(n)


def black_scholes_call(S0, K, r, sigma, T):
    """Closed-form Black-Scholes price of a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1. Sample paths and sanity check E[S_T] = S0 e^{rT}
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=10_000, rng=rng)
    print(f"Mean terminal price: {S[:, -1].mean():.3f}  (theory {S0 * np.exp(r * T):.3f})")

    plt.figure(figsize=(6, 4))
    plt.plot(np.linspace(0, T, 253), S[:50].T, lw=0.7)
    plt.xlabel("Time (years)"); plt.ylabel("Stock price (£)")
    plt.title("Simulated GBM paths (risk-neutral)")
    plt.tight_layout(); plt.savefig("gbm_paths.png", dpi=150)

    # 2-4. Convergence study: price and standard error vs number of paths
    bs = black_scholes_call(S0, K, r, sigma, T)
    rng = np.random.default_rng(0)
    Ns = np.logspace(2, 6, 13).astype(int)          # 100 to 1,000,000 paths
    results = np.array([mc_call(S0, K, r, sigma, T, n, rng) for n in Ns])
    prices, errs = results[:, 0], results[:, 1]

    print(f"Black-Scholes:            £{bs:.4f}")
    print(f"Monte Carlo (N = 10^6):   £{prices[-1]:.4f} ± {errs[-1]:.4f}  "
          f"({(prices[-1] - bs) / errs[-1]:+.2f} std errs)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.errorbar(Ns, prices, yerr=errs, fmt="o", capsize=3, label="Monte Carlo")
    ax1.axhline(bs, color="k", ls="--", label=f"Black-Scholes (£{bs:.3f})")
    ax1.set_xscale("log"); ax1.set_xlabel("Number of paths N"); ax1.set_ylabel("Call price (£)")
    ax1.set_title("Monte Carlo price converges to Black-Scholes"); ax1.legend()

    ax2.loglog(Ns, errs, "o", label="Standard error")
    ax2.loglog(Ns, errs[0] * np.sqrt(Ns[0] / Ns), "k--", label="∝ N^(-1/2)")
    ax2.set_xlabel("Number of paths N"); ax2.set_ylabel("Standard error (£)")
    ax2.set_title("Error falls as 1/√N"); ax2.legend()
    plt.tight_layout(); plt.savefig("convergence.png", dpi=150)

    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    print(f"Convergence slope: {slope:.3f}  (theory -0.5)")
    plt.show()
