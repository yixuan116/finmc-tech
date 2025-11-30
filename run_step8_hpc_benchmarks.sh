#!/bin/bash
# Step 8 HPC Multi-Horizon Benchmarks Runner
# This script runs all HPC benchmarks for 1Y/3Y/5Y horizons

set -e  # Exit on error

echo "=========================================="
echo "Step 8 HPC Multi-Horizon Benchmarks"
echo "=========================================="
echo ""

# Create output directory
mkdir -p results/step8

# Step 1: Python NumPy vs Numba benchmark (1Y/3Y/5Y)
echo "Step 1/3: Running Python NumPy vs Numba benchmark..."
echo "This will benchmark 1Y, 3Y, and 5Y horizons"
python -m finmc_tech.simulation.scenario_mc --hpc-multi-horizon --n 50000
echo "✓ Python benchmark complete"
echo ""

# Step 2: MPI benchmark (if MPI is available)
if command -v mpirun &> /dev/null; then
    echo "Step 2/3: Running MPI benchmark..."
    echo "This will benchmark 1Y, 3Y, and 5Y horizons with MPI"
    mpirun -n 4 python -m finmc_tech.hpc_demos.mpi_mc_demo --multi-horizon
    echo "✓ MPI benchmark complete"
else
    echo "Step 2/3: Skipping MPI benchmark (mpirun not found)"
fi
echo ""

# Step 3: OpenMP C benchmark (if compiled)
if [ -f "./openmp_mc_demo" ]; then
    echo "Step 3/3: Running OpenMP C benchmark..."
    echo "This will benchmark 1Y, 3Y, and 5Y horizons with OpenMP"
    ./openmp_mc_demo
    echo "✓ OpenMP benchmark complete"
else
    echo "Step 3/3: Skipping OpenMP benchmark (openmp_mc_demo not found)"
    echo "  To compile: gcc -O3 -fopenmp finmc_tech/hpc_demos/openmp_mc_demo.c -o openmp_mc_demo"
fi
echo ""

# Step 4: Generate summary
echo "Generating summary table..."
python -m finmc_tech.simulation.scenario_mc --hpc-summary
echo "✓ Summary generated"
echo ""

echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results saved in results/step8/:"
echo "  - hpc_benchmark_paths_1y_3y_5y.csv (Python NumPy vs Numba)"
echo "  - hpc_benchmark_mpi.csv (MPI results)"
echo "  - hpc_benchmark_openmp.csv (OpenMP results)"
echo "  - hpc_benchmark_summary_1y_3y_5y.csv (Summary table)"
echo ""

