#!/usr/bin/env python3
"""
NumPy serial baseline for Step 8 HPC benchmark.

- 1,000,000 simulation paths
- Horizons: 12, 36, 60, 120 months
- Total updates = 1,000,000 * (12 + 36 + 60 + 120) = 228,000,000

This file is a completely independent demo for presentation in PPT:
- What is "global workload = 228M updates";
- What the NumPy serial baseline double for-loop looks like;
- Final value distribution (mean + quantiles) for each horizon.

# Example: python -m finmc_tech.simulation.numpy_mc_demo
"""

import time
import numpy as np

# 1. Global configuration
N_SIMS = 1_000_000
HORIZONS = [12, 36, 60, 120]  # 4 horizons: 12, 36, 60, 120 months

S0 = 100.0
MU_ANNUAL = 0.10      # 10% annual expected return
SIGMA_ANNUAL = 0.40   # 40% annual volatility
STEPS_PER_YEAR = 12   # Monthly steps

MU_STEP = MU_ANNUAL / STEPS_PER_YEAR
SIGMA_STEP = SIGMA_ANNUAL / np.sqrt(STEPS_PER_YEAR)

RANDOM_SEED = 42


def run_numpy_baseline() -> None:
    """
    Pure NumPy serial Monte Carlo baseline:
    - Outer loop: over simulations
    - Inner loop: over months
    - Accurately count total_updates to prove 228M updates
    - Print runtime and final value distribution for each horizon (mean, P5, median, P95)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    total_updates = 0
    summaries = {}

    t0 = time.perf_counter()

    for horizon in HORIZONS: # Outer loop: over horizon simulations
        # (1) Generate all shocks for current horizon, shape = (N_SIMS, horizon)
        shocks = rng.normal(0.0, 1.0, size=(N_SIMS, horizon))

        # (2) Run N_SIMS paths sequentially
        terminal = np.empty(N_SIMS, dtype=np.float64)

        for i in range(N_SIMS):          # Inner loop: i =1m
            price = S0
            for t in range(horizon):     # Inner loop: t = (1,2,...120) months in each horizon
                price *= (1.0 + MU_STEP + SIGMA_STEP * shocks[i, t])
                total_updates += 1       # Exact update count

            terminal[i] = price

        # (3) Summarize for each horizon
        mean_price = float(terminal.mean())
        p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
        summaries[horizon] = (mean_price, p5, p50, p95)

    t1 = time.perf_counter()
    runtime = t1 - t0

    # sanity check: ensure equal to 228,000,000
    expected_updates = N_SIMS * sum(HORIZONS)
    assert total_updates == expected_updates, (
        f"total_updates={total_updates} != expected={expected_updates}"
    )

    # (4) Print results
    print(f"\n[NumPy serial baseline]")
    print(f"  Paths        : {N_SIMS:,d}")
    print(f"  Horizons     : {HORIZONS} months")
    print(f"  Total updates: {total_updates:,d}")
    print(f"  Runtime      : {runtime:.3f} s\n")

    print("Horizon (months) |  Mean Price  |   P5   |  Median  |   P95")
    print("-" * 62)
    for h in HORIZONS:
        mean_price, p5, p50, p95 = summaries[h]
        print(
            f"{h:16d} | {mean_price:11.2f} | "
            f"{p5:6.2f} | {p50:8.2f} | {p95:7.2f}"
        )


if __name__ == "__main__":
    run_numpy_baseline()

