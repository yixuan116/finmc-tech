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
TOTAL_SIMS = 1000000  # 1e6
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
    Run the MPI Monte Carlo benchmark for a given number of steps,
    return the parallel runtime (max rank time).
    Writes a row to hpc_benchmark_mpi.csv as before.
    
    Parameters
    ----------
    n_steps : int
        Number of time steps (months)
    total_sims : int
        Total number of Monte Carlo simulations
    output_dir : Path
        Output directory for CSV file
    
    Returns
    -------
    float
        Maximum parallel runtime across all ranks (seconds)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() #path simulation rank
    size = comm.Get_size() #number of ranks
    
    # Calculate local workload
    local_n_sims = total_sims // size #distribute the simulations evenly across all ranks
    # Add remainder to last rank, if cannot be evenly distributed
    if rank == size - 1:
        local_n_sims += total_sims % size
    
    # Sync before timing
    comm.Barrier() #wait for all ranks to reach this point
    t0 = MPI.Wtime() #start timing
    
    # [HPC-MPI] Parallel Monte Carlo execution: each rank simulates `local_n_sims` paths.
    local_terminals = run_mc_paths(local_n_sims, n_steps, MU, SIGMA, S0, STEPS_PER_YEAR)
    
    # Wait for all to finish
    comm.Barrier()
    t1 = MPI.Wtime()
    
    local_time = t1 - t0 #local time is the time taken by the current rank
    
    # Gather max time (the effective parallel runtime), return to rank 0
    max_time = comm.reduce(local_time, op=MPI.MAX, root=0)
    
    if rank == 0: #the master rank check if the file exists and append the result to the file
        csv_path = output_dir / "hpc_benchmark_mpi.csv"
        file_exists = csv_path.exists()
        
        # [HPC-MPI] Benchmark result: written once on rank 0 to
        # results/step8/hpc_benchmark_mpi.csv
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["backend", "mode", "n_sims", "n_steps", "n_ranks", "time_sec"]) #write the header if the file does not exist
            
            writer.writerow(["mpi_python", "mpi_parallel", total_sims, n_steps, size, max_time]) #write the result to the file
    
    return max_time if rank == 0 else 0.0 #return the max time to the master rank

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
        
        for horizon_label, horizon_steps in HORIZONS.items(): #loop through all horizons
            if rank == 0: #start,only the master rank prints the progress
                print(f"--- Running {horizon_label} ({horizon_steps} steps) ---")
            
            max_time = run_single_mpi_benchmark(horizon_steps, total_sims, output_dir) #run the benchmark for the current horizon
            
            if rank == 0: #end, only the master rank prints the result
                print(f"✓ {horizon_label} finished in {max_time:.4f} seconds\n")
        
        if rank == 0:
            csv_path = output_dir / "hpc_benchmark_mpi.csv"
            print(f"All results appended to: {csv_path}")
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

