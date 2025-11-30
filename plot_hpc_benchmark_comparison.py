#!/usr/bin/env python3
"""
Plot HPC benchmark comparison across all backends and horizons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_benchmark_data(output_dir="results/step8"):
    """Load all benchmark CSV files."""
    out_dir = Path(output_dir)
    
    # Load Python NumPy vs Numba
    paths_df = pd.read_csv(out_dir / "hpc_benchmark_paths_1y_3y_5y.csv")
    
    # Load MPI
    mpi_df = pd.read_csv(out_dir / "hpc_benchmark_mpi.csv")
    
    # Load OpenMP
    openmp_df = pd.read_csv(out_dir / "hpc_benchmark_openmp.csv")
    
    return paths_df, mpi_df, openmp_df

def prepare_comparison_data(paths_df, mpi_df, openmp_df):
    """Prepare data for comparison plot."""
    horizons = ["1Y", "3Y", "5Y"]
    n_steps_map = {"1Y": 12, "3Y": 36, "5Y": 60}
    
    # Initialize data structure
    data = {
        "horizon": [],
        "n_steps": [],
        "backend": [],
        "time_sec": [],
        "n_sims": []
    }
    
    # Extract Python NumPy baseline
    for horizon in horizons:
        n_steps = n_steps_map[horizon]
        baseline_row = paths_df[
            (paths_df["horizon"] == horizon) & 
            (paths_df["backend"] == "baseline_numpy")
        ]
        if not baseline_row.empty:
            data["horizon"].append(horizon)
            data["n_steps"].append(n_steps)
            data["backend"].append("NumPy Baseline")
            data["time_sec"].append(baseline_row["time_sec"].iloc[0])
            data["n_sims"].append(50000)  # Python uses 50K sims
    
    # Extract Numba parallel
    for horizon in horizons:
        n_steps = n_steps_map[horizon]
        numba_row = paths_df[
            (paths_df["horizon"] == horizon) & 
            (paths_df["backend"] == "numba_parallel")
        ]
        if not numba_row.empty:
            data["horizon"].append(horizon)
            data["n_steps"].append(n_steps)
            data["backend"].append("Numba Parallel")
            data["time_sec"].append(numba_row["time_sec"].iloc[0])
            data["n_sims"].append(50000)
    
    # Extract MPI (use most recent run for each horizon)
    for horizon in horizons:
        n_steps = n_steps_map[horizon]
        mpi_rows = mpi_df[
            (mpi_df["n_steps"] == n_steps) &
            (mpi_df["backend"] == "mpi_python") &
            (mpi_df["mode"] == "mpi_parallel")
        ]
        if not mpi_rows.empty:
            # Use the most recent run
            mpi_time = mpi_rows["time_sec"].iloc[-1]
            data["horizon"].append(horizon)
            data["n_steps"].append(n_steps)
            data["backend"].append("MPI (4 ranks)")
            data["time_sec"].append(mpi_time)
            data["n_sims"].append(1000000)  # MPI uses 1M sims
    
    # Extract OpenMP parallel (use most recent run for each horizon)
    for horizon in horizons:
        n_steps = n_steps_map[horizon]
        openmp_rows = openmp_df[
            (openmp_df["n_steps"] == n_steps) &
            (openmp_df["backend"] == "openmp_c") &
            (openmp_df["mode"] == "openmp_parallel")
        ]
        if not openmp_rows.empty:
            # Use the most recent run
            openmp_time = openmp_rows["time_sec"].iloc[-1]
            data["horizon"].append(horizon)
            data["n_steps"].append(n_steps)
            data["backend"].append("OpenMP C")
            data["time_sec"].append(openmp_time)
            data["n_sims"].append(1000000)  # OpenMP uses 1M sims
    
    return pd.DataFrame(data)

def plot_benchmark_comparison(df, output_path="results/step8/hpc_benchmark_comparison.png"):
    """Create comparison plot."""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    horizons = ["1Y", "3Y", "5Y"]
    backends = ["NumPy Baseline", "Numba Parallel", "MPI (4 ranks)", "OpenMP C"]
    colors = {
        "NumPy Baseline": "#2E86AB",
        "Numba Parallel": "#A23B72",
        "MPI (4 ranks)": "#F18F01",
        "OpenMP C": "#C73E1D"
    }
    
    # Plot 1: Bar chart comparing all backends
    ax1 = axes[0]
    x = np.arange(len(horizons))
    width = 0.2
    
    for i, backend in enumerate(backends):
        backend_data = df[df["backend"] == backend]
        times = []
        for horizon in horizons:
            horizon_data = backend_data[backend_data["horizon"] == horizon]
            if not horizon_data.empty:
                times.append(horizon_data["time_sec"].iloc[0])
            else:
                times.append(0)
        
        offset = (i - len(backends) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, times, width, label=backend, 
                      color=colors.get(backend, "#808080"), alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s',
                        ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Time Horizon', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('HPC Benchmark Comparison: Runtime by Backend and Horizon', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')  # Use log scale for better visualization
    
    # Plot 2: Speedup comparison (relative to NumPy baseline)
    ax2 = axes[1]
    
    # Calculate speedups
    baseline_times = {}
    for horizon in horizons:
        baseline_data = df[
            (df["horizon"] == horizon) & 
            (df["backend"] == "NumPy Baseline")
        ]
        if not baseline_data.empty:
            baseline_times[horizon] = baseline_data["time_sec"].iloc[0]
    
    speedup_data = []
    for backend in ["Numba Parallel", "MPI (4 ranks)", "OpenMP C"]:
        backend_speedups = []
        for horizon in horizons:
            backend_data = df[
                (df["horizon"] == horizon) & 
                (df["backend"] == backend)
            ]
            if not backend_data.empty and horizon in baseline_times:
                speedup = baseline_times[horizon] / backend_data["time_sec"].iloc[0]
                backend_speedups.append(speedup)
            else:
                backend_speedups.append(0)
        speedup_data.append(backend_speedups)
    
    x2 = np.arange(len(horizons))
    width2 = 0.25
    
    for i, (backend, speedups) in enumerate(zip(["Numba Parallel", "MPI (4 ranks)", "OpenMP C"], speedup_data)):
        offset = (i - 1) * width2
        bars = ax2.bar(x2 + offset, speedups, width2, 
                      label=backend,
                      color=colors.get(backend, "#808080"), alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}x',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Time Horizon', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (vs NumPy Baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup Comparison: Parallel Backends vs NumPy Baseline', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(horizons)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add note about simulation counts
    note_text = ("Note: NumPy/Numba use 50K simulations per horizon.\n"
                "MPI/OpenMP use 1M simulations per horizon.\n"
                "Times are not directly comparable due to different simulation counts.")
    fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, 
             style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    
    plt.close()

def main():
    """Main function."""
    print("Loading benchmark data...")
    paths_df, mpi_df, openmp_df = load_benchmark_data()
    
    print("Preparing comparison data...")
    comparison_df = prepare_comparison_data(paths_df, mpi_df, openmp_df)
    
    print("Creating comparison plot...")
    plot_benchmark_comparison(comparison_df)
    
    print("\n" + "="*60)
    print("Benchmark comparison plot generated successfully!")
    print("="*60)
    
    # Print summary table
    print("\nSummary Table:")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()

