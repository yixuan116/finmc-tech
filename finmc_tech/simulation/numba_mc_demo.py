#!/usr/bin/env python3
"""
Numba-parallel Monte Carlo baseline for Step 8 HPC benchmark.

- 1,000,000 simulation paths
- Horizons: 12, 36, 60, 120 months
- Total updates = 228,000,000

This file mirrors numpy_mc_demo.py but replaces the
serial double for-loop with a Numba-parallel kernel
over simulation paths (prange).
# Example: python -m finmc_tech.simulation.numba_mc_demo
"""

import time
import numpy as np
from numba import njit, prange

# === 1. Global config (keep consistent with numpy_mc_demo.py) ===
N_SIMS = 1_000_000
HORIZONS = [12, 36, 60, 120]  # months

S0 = 100.0
MU_ANNUAL = 0.10
SIGMA_ANNUAL = 0.40
STEPS_PER_YEAR = 12

MU_STEP = MU_ANNUAL / STEPS_PER_YEAR
SIGMA_STEP = SIGMA_ANNUAL / np.sqrt(STEPS_PER_YEAR)

RANDOM_SEED = 42

# Runtime measured from numpy_mc_demo.py
NUMPY_BASELINE_RUNTIME = 37.0  # seconds (approximate baseline)


# === 2. Numba-parallel Monte Carlo kernel (path-level parallelism) ===
@njit(parallel=True) # JIT compiles the Python loops into machine code
def mc_numba_paths(
    S0: float,
    mu_step: float,
    sigma_step: float,
    n_sims: int,
    horizon_steps: int,
) -> np.ndarray:
    """
    Numba-parallel Monte Carlo kernel:
    - Parallel over simulation paths (prange over n_sims)
    - Each thread walks one full path over `horizon_steps` months
    - Same arithmetic return model as numpy_mc_demo.py:
      S_{t+1} = S_t * (1 + mu_step + sigma_step * eps)
    """
    # Note: Using simple array return logic here. 
    # For memory efficiency with 1M paths * 120 steps, usually one would only return terminal.
    # However, to mirror the structure and for potential visualization, keeping path storage logic
    # similar to typical MC implementations, but optimized.
    # Actually, the requirement asks for terminal stats.
    # Storing all steps for 1M paths * 120 steps = 120M floats ~ 960MB, which is fine.
    
    paths = np.zeros((n_sims, horizon_steps + 1))
    paths[:, 0] = S0

    for i in prange(n_sims):  # data-parallel over paths
        price = S0
        for t in range(horizon_steps):
            eps = np.random.normal()
            r_t = mu_step + sigma_step * eps
            # Optional: simplistic crash protection mirroring some logic, 
            # though standard geometric brownian motion usually does log returns.
            # The prompt provided specific arithmetic logic: price *= (1.0 + r_t)
            # We will stick to the exact arithmetic logic provided in the prompt skeleton.
            if r_t < -0.99:
                r_t = -0.99
            price *= (1.0 + r_t)
            paths[i, t + 1] = price

    return paths


def run_numba_baseline() -> None:
    """
    Run Numba Monte Carlo for all horizons and print:

    - Total updates (should be 228,000,000)
    - Runtime
    - Speedup vs NumPy serial baseline
    - Per-horizon terminal distribution (mean, P5, median, P95)
    """
    # Warm-up to exclude JIT compilation time from benchmark
    print("Warming up Numba JIT...")
    _ = mc_numba_paths(S0, MU_STEP, SIGMA_STEP, 10_000, 12)

    total_updates = 0
    summaries = {}

    t0 = time.perf_counter()

    for horizon in HORIZONS:
        # Run simulation
        paths = mc_numba_paths(S0, MU_STEP, SIGMA_STEP, N_SIMS, horizon)
        terminals = paths[:, -1]

        total_updates += N_SIMS * horizon

        mean_price = float(terminals.mean())
        p5, p50, p95 = np.percentile(terminals, [5, 50, 95])
        summaries[horizon] = (mean_price, p5, p50, p95)

    t1 = time.perf_counter()
    runtime = t1 - t0

    expected_updates = N_SIMS * sum(HORIZONS)
    assert total_updates == expected_updates, (
        f"total_updates={total_updates} != expected={expected_updates}"
    )

    speedup = NUMPY_BASELINE_RUNTIME / runtime if runtime > 0 else np.nan

    print(f"\n[Numba-parallel baseline]")
    print(f"  Paths        : {N_SIMS:,d}")
    print(f"  Horizons     : {HORIZONS} months")
    print(f"  Total updates: {total_updates:,d}")
    print(f"  Runtime      : {runtime:.3f} s")
    print(f"  Speedup vs NumPy ({NUMPY_BASELINE_RUNTIME:.3f} s): {speedup:.2f}x\n")

    print("Horizon (months) |  Mean Price  |   P5   |  Median  |   P95")
    print("-" * 62)
    for h in HORIZONS:
        mean_price, p5, p50, p95 = summaries[h]
        print(
            f"{h:16d} | {mean_price:11.2f} | "
            f"{p5:6.2f} | {p50:8.2f} | {p95:7.2f}"
        )


if __name__ == "__main__":
    run_numba_baseline()

