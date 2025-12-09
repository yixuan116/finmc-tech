"""
MPI Monte Carlo benchmark demo (HPC, Step 8).

Usage example:
    # 12 months (1Y) - default
    mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo # 4 core CPUs
    
    # 36 months (3Y)
    mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --steps 36
    
    # 60 months (5Y)
    mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --steps 60
    
    # 120 months (10Y)
    mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --steps 120

    (Default simulations: 1,000,000)

This will append a row to results/step8/hpc_benchmark_mpi.csv
with the measured parallel runtime.

This is an optional MPI demo only, not used in the main pipeline.
It shows how the same Monte Carlo workload can be split across MPI ranks.
It is conceptually aligned with your benchmark_scenario_concurrency and path-level Numba kernel,
but implemented via multi-process message passing.
"""

from mpi4py import MPI
import numpy as np
from pathlib import Path
import csv
import sys
import argparse

# Default parameters
TOTAL_SIMS = 1000000  # 1M sims for benchmark comparability
DEFAULT_N_STEPS = 12  # 12 months (1Y)
MU = 0.01
SIGMA = 0.40
S0 = 100.0
STEPS_PER_YEAR = 12  # Monthly steps

# Multi-horizon configuration
HORIZONS = {
    "1Y": 12,
    "3Y": 36,
    "5Y": 60,
    "10Y": 120,
}

def run_mc_paths(n_sims, n_steps, mu, sigma, s0, steps_per_year=12):
    """Simple sequential Monte Carlo kernel for a single rank."""
    rng = np.random.default_rng()  # Seed can be rank-dependent if needed
    sigma_step = sigma / np.sqrt(steps_per_year)
    
    # Pre-allocate
    terminals = np.zeros(n_sims)
    
    # Vectorized implementation for efficiency within rank
    # (or could be looped to match C demo structure exactly)
    paths = np.zeros((n_sims, n_steps + 1)) #the terminal prices will be stored in the last column
    paths[:, 0] = s0 #initial price
    
    Z = rng.standard_normal((n_sims, n_steps)) #the random increments will be stored in the last column
    
    # Calculate all steps
    rets = mu + sigma_step * Z
    rets = np.clip(rets, -0.99, None) #clip the returns to avoid extreme values
    
    # Cumulative product
    paths[:, 1:] = s0 * np.cumprod(1.0 + rets, axis=1)
    
    return paths[:, -1]

def run_single_mpi_benchmark(n_steps: int, total_sims: int, output_dir: Path) -> float:
    """
    Run the MPI Monte Carlo benchmark for a given number of steps.

    This enhanced version does three things:
    1. Each MPI rank prints a concise "work card" describing:
       - Which global path indices it is responsible for
       - How many simulations and MC updates it computed
       - Its local runtime
       - Its local terminal price S_T distribution (mean/std)
    2. Rank 0 aggregates all local terminal prices into global statistics:
       - mean(S_T), std(S_T), min(S_T), max(S_T)
       - global Monte Carlo workload (total_sims × n_steps updates)
    3. Rank 0 prints the global financial result and the effective parallel runtime,
       then appends the runtime result to results/step8/hpc_benchmark_mpi.csv
       using the same schema as before.

    Parameters
    ----------
    n_steps : int
        Number of time steps (months).
    total_sims : int
        Total number of Monte Carlo simulations across all ranks.
    output_dir : Path
        Output directory for the CSV benchmark file.

    Returns
    -------
    float
        Maximum parallel runtime across all ranks (seconds), returned on rank 0.
        Non-root ranks return 0.0.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Compute per-rank workload.
    #
    # We split `total_sims` as evenly as possible across ranks:
    # - base_chunk = total_sims // size
    # - remainder  = total_sims % size (assigned to the last rank)
    base_chunk = total_sims // size
    remainder = total_sims % size

    local_n_sims = base_chunk
    if rank == size - 1:
        local_n_sims += remainder

    # 2) Synchronize before timing, then run the Monte Carlo kernel on each rank.
    #
    # Only the actual MC kernel runtime is measured here; MPI reductions and printing
    # are NOT included in the timing.
    comm.Barrier()
    t0 = MPI.Wtime()
    
    # [HPC-MPI] Each rank simulates `local_n_sims` terminal prices for this horizon.
    local_terminals = run_mc_paths(
        local_n_sims, n_steps, MU, SIGMA, S0, STEPS_PER_YEAR
    )

    comm.Barrier()
    t1 = MPI.Wtime()
    local_time = t1 - t0  # Local runtime for this rank

    # 3) For interpretability: compute a conceptual global path index range
    #    for each rank and print a concise "work card".
    #
    # Conceptually:
    # - Rank 0 handles paths [0, base_chunk-1]
    # - Rank 1 handles the next chunk, and so on.
    # - The last rank additionally covers the remainder.
    if rank < size - 1:
        global_start = rank * base_chunk
        global_end = global_start + local_n_sims
    else:
        global_start = rank * base_chunk
        global_end = global_start + local_n_sims

    local_mean = float(local_terminals.mean())
    local_std = float(local_terminals.std())
    local_updates = local_n_sims * n_steps

    # Optional barrier for slightly more orderly printing (not required for correctness).
    comm.Barrier()
    print(
        f"[MPI RANK {rank}] "
        f"paths=[{global_start:,d}, {global_end - 1:,d}], "
        f"local_n_sims={local_n_sims:,d}, "
        f"mc_updates={local_updates:,d}, "
        f"local_time={local_time:.4f}s, "
        f"mean(S_T)={local_mean:.4f}, std(S_T)={local_std:.4f}"
    )

    # 4) Prepare local financial statistics for global aggregation on rank 0.
    #
    # We aggregate:
    # - Count:       n
    # - Sum:         sum(S_T)
    # - Sum of sq.:  sum(S_T^2)
    # - Min / Max:   min(S_T), max(S_T)
    local_n = np.array([local_terminals.size], dtype="int64")
    local_sum = np.array([local_terminals.sum()], dtype="float64")
    local_sum_sq = np.array([(local_terminals ** 2).sum()], dtype="float64")
    local_min = np.array([local_terminals.min()], dtype="float64")
    local_max = np.array([local_terminals.max()], dtype="float64")

    global_n = np.zeros_like(local_n)
    global_sum = np.zeros_like(local_sum)
    global_sum_sq = np.zeros_like(local_sum_sq)
    global_min = np.zeros_like(local_min)
    global_max = np.zeros_like(local_max)

    # [HPC-MPI] Reduce local statistics to rank 0.
    comm.Reduce(local_n, global_n, op=MPI.SUM, root=0)
    comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_sum_sq, global_sum_sq, op=MPI.SUM, root=0)
    comm.Reduce(local_min, global_min, op=MPI.MIN, root=0)
    comm.Reduce(local_max, global_max, op=MPI.MAX, root=0)

    # 5) Aggregate runtime: the effective parallel runtime is the maximum
    #    local_time across all ranks.
    max_time = comm.reduce(local_time, op=MPI.MAX, root=0)

    # 6) On rank 0, compute the global financial distribution of S_T
    #    and write the runtime benchmark to CSV as before.
    if rank == 0:
        # ---- 6.1 Global financial statistics for S_T ----
        n = int(global_n[0])
        mean_st = global_sum[0] / n
        var_st = global_sum_sq[0] / n - mean_st ** 2
        var_st = max(var_st, 0.0)  # Numerical guard against tiny negative values
        std_st = float(np.sqrt(var_st))
        min_st = float(global_min[0])
        max_st = float(global_max[0])

        total_updates = total_sims * n_steps  # Total MC updates across all ranks

        print("\n[MPI MC FINANCIAL RESULT]")
        print(f"  Horizon: {n_steps} months")
        print(f"  Paths (total_sims): {total_sims:,d}")
        print(
            f"  Global workload: {total_sims:,d} paths × {n_steps} steps "
            f"= {total_updates:,d} Monte Carlo updates"
        )
        print("  Model: S_{t+1} = S_t * (1 + r_t),  r_t = μ + σ_step * ε_t")
        print(
            f"         μ = {MU:.4f} per step, "
            f"σ = {SIGMA:.4f} per year, "
            f"steps_per_year = {STEPS_PER_YEAR}"
        )
        print("  Terminal price S_T distribution (across all ranks):")
        print(f"    mean(S_T) = {mean_st:.4f}")
        print(f"    std(S_T)  = {std_st:.4f}")
        print(f"    min(S_T)  = {min_st:.4f}")
        print(f"    max(S_T)  = {max_st:.4f}")
        print("")

        # ---- 6.2 Runtime benchmark result + CSV output ----
        csv_path = output_dir / "hpc_benchmark_mpi.csv"
        file_exists = csv_path.exists()

        # This preserves the original CSV schema:
        # backend, mode, n_sims, n_steps, n_ranks, time_sec
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["backend", "mode", "n_sims", "n_steps", "n_ranks", "time_sec"]
                )

            writer.writerow(
                ["mpi_python", "mpi_parallel", total_sims, n_steps, size, max_time]
            )

        print("[MPI MC RUNTIME]")
        print(
            f"  Effective parallel runtime (max rank time): {max_time:.4f} seconds"
        )
        print(f"  Results appended to: {csv_path}\n")

    # Non-root ranks return 0.0 to keep the signature consistent.
    return max_time if rank == 0 else 0.0

def print_total_hpc_comparison(mpi_total_time: float, n_ranks: int) -> None:
    """
    Print a total-runtime comparison table for all HPC backends.

    The non-MPI numbers are hard-coded from the existing slide:
    - NumPy total (4 horizons):          36.644 s
    - OpenMP-C total (4 horizons):        6.088 s
    - OpenMP-Horizon total (4 horizons):  1.189 s
    - Numba total: approximately 1.2–1.3 s

    Parameters
    ----------
    mpi_total_time : float
        Sum of MPI runtimes across all horizons in this multi-horizon run.
    n_ranks : int
        Size of MPI_COMM_WORLD (number of ranks used in this MPI run).
    """
    NUMPY_TOTAL = 36.644
    OMP_C_TOTAL = 6.088
    OMP_H_TOTAL = 1.189
    NUMBA_LOW = 1.2
    NUMBA_HIGH = 1.3

    print("\n[HPC TOTAL RUNTIME COMPARISON]")
    print("  Total over horizons: 12m, 36m, 60m, 120m\n")
    print(f"  NumPy (serial baseline):        {NUMPY_TOTAL:6.3f} s")
    print(f"  OpenMP-C (path-parallel C):     {OMP_C_TOTAL:6.3f} s")
    print(f"  OpenMP-Horizon (C):             {OMP_H_TOTAL:6.3f} s")
    print(f"  Numba parallel (estimate):      ≈{NUMBA_LOW:.1f}–{NUMBA_HIGH:.1f} s")
    print(f"  MPI Python (this run, {n_ranks} ranks): {mpi_total_time:6.3f} s")

    if mpi_total_time > 0:
        speedup_vs_numpy = NUMPY_TOTAL / mpi_total_time
        print(f"\n  Speedup vs NumPy baseline: {speedup_vs_numpy:.1f}× faster")

    print("")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MPI Monte Carlo benchmark demo for Step 8 HPC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_N_STEPS,
        help="Number of time steps (months). Common values: 12 (1Y), 36 (3Y), 60 (5Y), 120 (10Y)"
    )
    parser.add_argument(
        "--sims", type=int, default=TOTAL_SIMS,
        help="Total number of Monte Carlo simulations"
    )
    parser.add_argument(
        "--multi-horizon", action="store_true",
        help="Run benchmarks for all horizons (1Y, 3Y, 5Y, 10Y)"
    )
    
    # Only rank 0 parses arguments, then broadcasts to others
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0: #only the master rank parses the arguments
        args = parser.parse_args()
        n_steps = args.steps
        total_sims = args.sims
        multi_horizon = args.multi_horizon
    else:
        n_steps = None
        total_sims = None
        multi_horizon = None
    
    # Broadcast parameters to all ranks
    n_steps = comm.bcast(n_steps, root=0) #root process: rank 0 broadcasts the parameters to all ranks
    total_sims = comm.bcast(total_sims, root=0)
    multi_horizon = comm.bcast(multi_horizon, root=0)
    
    output_dir = Path("results/step8")
    output_dir.mkdir(parents=True, exist_ok=True) #create the output directory if it does not exist
    
    if multi_horizon:
        # Run benchmarks for all horizons
        if rank == 0:
            print(f"Starting Multi-Horizon MPI Monte Carlo Demo with {size} ranks...")
            print(f"Total Sims: {total_sims}")
            print(f"Horizons: {list(HORIZONS.keys())}\n")
        
        mpi_total_time = 0.0

        for horizon_label, horizon_steps in HORIZONS.items(): #loop through all horizons
            if rank == 0: #start,only the master rank prints the progress
                print(f"--- Running {horizon_label} ({horizon_steps} steps) ---")
            
            max_time = run_single_mpi_benchmark(horizon_steps, total_sims, output_dir) #run the benchmark for the current horizon
            
            if rank == 0: #end, only the master rank prints the result
                mpi_total_time += max_time
                print(f"✓ {horizon_label} finished in {max_time:.4f} seconds\n")
        
        if rank == 0:
            csv_path = output_dir / "hpc_benchmark_mpi.csv"
            print(f"All results appended to: {csv_path}")
            # New: print total HPC comparison using slide numbers + MPI total
            print_total_hpc_comparison(mpi_total_time, size)
    else:
        # Original single-horizon behavior
        if rank == 0:
            horizon_label = f"{n_steps}M"
            if n_steps == 12:
                horizon_label = "1Y"
            elif n_steps == 36:
                horizon_label = "3Y"
            elif n_steps == 60:
                horizon_label = "5Y"
            elif n_steps == 120:
                horizon_label = "10Y"
            
            print(f"Starting MPI Monte Carlo Demo with {size} ranks...")
            print(f"Total Sims: {total_sims}, Steps: {n_steps} ({horizon_label})")
        
        max_time = run_single_mpi_benchmark(n_steps, total_sims, output_dir) #run the benchmark for the current horizon
        
        if rank == 0:
            print(f"✓ MPI Parallel finished in {max_time:.4f} seconds")
            csv_path = output_dir / "hpc_benchmark_mpi.csv"
            print(f"Results appended to: {csv_path}")

if __name__ == "__main__":
    main()

