"""
MPI Monte Carlo benchmark demo (HPC extension, Step 8).

Usage example:
    mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo

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

# Parameters
TOTAL_SIMS = 1000000  # 1e6
N_STEPS = 12
MU = 0.01
SIGMA = 0.40
S0 = 100.0

def run_mc_paths(n_sims, n_steps, mu, sigma, s0):
    """Simple sequential Monte Carlo kernel for a single rank."""
    rng = np.random.default_rng()  # Seed can be rank-dependent if needed
    sigma_step = sigma / np.sqrt(12)
    
    # Pre-allocate
    terminals = np.zeros(n_sims)
    
    # Vectorized implementation for efficiency within rank
    # (or could be looped to match C demo structure exactly)
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = s0
    
    Z = rng.standard_normal((n_sims, n_steps))
    
    # Calculate all steps
    rets = mu + sigma_step * Z
    rets = np.clip(rets, -0.99, None)
    
    # Cumulative product
    paths[:, 1:] = s0 * np.cumprod(1.0 + rets, axis=1)
    
    return paths[:, -1]

def main():
    # [HPC-MPI] Rank-level decomposition: split total Monte Carlo paths across MPI ranks.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Starting MPI Monte Carlo Demo with {size} ranks...")
        print(f"Total Sims: {TOTAL_SIMS}, Steps: {N_STEPS}")

    # Calculate local workload
    local_n_sims = TOTAL_SIMS // size
    # Add remainder to last rank
    if rank == size - 1:
        local_n_sims += TOTAL_SIMS % size

    # Sync before timing
    comm.Barrier()
    t0 = MPI.Wtime()

    # [HPC-MPI] Parallel Monte Carlo execution: each rank simulates `local_n_sims` paths.
    local_terminals = run_mc_paths(local_n_sims, N_STEPS, MU, SIGMA, S0)
    
    # Wait for all to finish
    comm.Barrier()
    t1 = MPI.Wtime()
    
    local_time = t1 - t0
    
    # Gather max time (the effective parallel runtime)
    max_time = comm.reduce(local_time, op=MPI.MAX, root=0)

    if rank == 0:
        print(f"✓ MPI Parallel finished in {max_time:.4f} seconds")
        
        output_dir = Path("results/step8")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "hpc_benchmark_mpi.csv"
        
        file_exists = csv_path.exists()
        
        # [HPC-MPI] Benchmark result: written once on rank 0 to
        # results/step8/hpc_benchmark_mpi.csv
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["backend", "mode", "n_sims", "n_steps", "n_ranks", "time_sec"])
            
            writer.writerow(["mpi_python", "mpi_parallel", TOTAL_SIMS, N_STEPS, size, max_time])
            
        print(f"Results appended to: {csv_path}")

if __name__ == "__main__":
    main()

