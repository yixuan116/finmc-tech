import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_data():
    base_dir = Path("results/step8")
    
    # 1. NumPy vs Numba (now 1M sims by default)
    df_paths = pd.read_csv(base_dir / "hpc_benchmark_paths_1y_3y_5y_10y.csv")
    if "n_sims" not in df_paths.columns:
        # Fallback if n_sims column missing, but code was updated to use 1M
        df_paths["n_sims"] = 1000000 
    
    # 2. MPI
    df_mpi = pd.read_csv(base_dir / "hpc_benchmark_mpi.csv")
    
    # 3. OpenMP
    openmp_path = base_dir / "hpc_benchmark_openmp.csv"
    if openmp_path.exists():
        df_openmp = pd.read_csv(openmp_path)
    else:
        df_openmp = pd.DataFrame()
        
    return df_paths, df_mpi, df_openmp

def process_data(df_paths, df_mpi, df_openmp):
    data = []
    
    # --- Process NumPy/Numba ---
    # Filter for 1M simulations only
    df_paths = df_paths[df_paths["n_sims"] == 1000000]
    
    for _, row in df_paths.iterrows():
        backend = row["backend"]
        time_sec = row["time_sec"]
        
        # Resolve horizon
        if "horizon" in row:
            horizon = row["horizon"]
        else:
            steps = row["n_steps"]
            horizon = {12: "1Y", 36: "3Y", 60: "5Y", 120: "10Y"}.get(steps, f"{steps}M")
            
        label = "NumPy Baseline" if backend == "baseline_numpy" else "Numba Parallel"
        
        data.append({
            "Horizon": horizon,
            "Backend": label,
            "Time": time_sec,
            "Sims": row["n_sims"],
            "Type": "Python"
        })
        
    # --- Process MPI ---
    # Filter for 1M sims only
    if not df_mpi.empty:
        df_mpi = df_mpi[df_mpi["n_sims"] == 1000000]
        # Sort and keep last unique entry per horizon
        df_mpi = df_mpi.sort_values("n_steps")
        df_mpi_dedup = df_mpi.drop_duplicates(subset=["n_steps"], keep="last")
        
        for _, row in df_mpi_dedup.iterrows():
            steps = row["n_steps"]
            horizon = {12: "1Y", 36: "3Y", 60: "5Y", 120: "10Y"}.get(steps, f"{steps}M")
            
            data.append({
                "Horizon": horizon,
                "Backend": f"MPI ({row['n_ranks']} ranks)",
                "Time": row["time_sec"],
                "Sims": row["n_sims"],
                "Type": "HPC"
            })

    # --- Process OpenMP ---
    # Filter for 1M sims only
    if not df_openmp.empty:
        df_openmp = df_openmp[
            (df_openmp["mode"] == "openmp_parallel") & 
            (df_openmp["n_sims"] == 1000000)
        ]
        df_openmp_dedup = df_openmp.drop_duplicates(subset=["n_steps"], keep="last")
        
        for _, row in df_openmp_dedup.iterrows():
            steps = row["n_steps"]
            horizon = {12: "1Y", 36: "3Y", 60: "5Y", 120: "10Y"}.get(steps, f"{steps}M")
            
            data.append({
                "Horizon": horizon,
                "Backend": "OpenMP C",
                "Time": row["time_sec"],
                "Sims": row["n_sims"],
                "Type": "HPC"
            })
            
    return pd.DataFrame(data)

def plot_combined_benchmark(df):
    # Filter horizons
    horizons = ["1Y", "3Y", "5Y", "10Y"]
    df = df[df["Horizon"].isin(horizons)]
    
    # Ensure categorical order
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizons, ordered=True)
    df = df.sort_values(["Horizon", "Backend"])
    
    # Define distinctive colors (High Contrast)
    # NumPy: Blue, Numba: Purple, MPI: Orange, OpenMP: Red
    custom_palette = {
        "NumPy Baseline": "#1f77b4",   # Standard Blue
        "Numba Parallel": "#9467bd",   # Purple
        "MPI (4 ranks)": "#ff7f0e",    # Bright Orange
        "OpenMP C": "#d62728"          # Red
    }
    
    # Handle variations in MPI label (e.g. 4 ranks)
    # Update palette keys dynamically if needed
    mpi_labels = df[df["Backend"].str.contains("MPI")]["Backend"].unique()
    for label in mpi_labels:
        custom_palette[label] = "#ff7f0e"
        
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: Runtime (Log Scale) ---
    sns.barplot(
        data=df,
        x="Horizon",
        y="Time",
        hue="Backend",
        palette=custom_palette,
        edgecolor="black",
        linewidth=0.5,
        ax=ax1
    )
    
    ax1.set_yscale("log")
    ax1.set_title("HPC Benchmark Comparison: Runtime by Backend and Horizon", fontweight="bold")
    ax1.set_ylabel("Time (seconds) [Log Scale]")
    ax1.set_xlabel("Time Horizon")
    ax1.legend(title=None)
    
    # Add values on top of bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3fs', padding=3, fontsize=8)

    # --- Plot 2: Speedup (Linear Scale) ---
    # Calculate speedup relative to NumPy Baseline *of that horizon*
    # Note: This reproduces the "unfair" comparison from the screenshot if workloads differ
    # But let's calculate "Effective Speedup" normalized by Sims count to be scientifically correct
    # IF user wants EXACT reproduction of screenshot, we wouldn't normalize.
    # But assuming "better version" implies "better colors" not "wrong math".
    # Let's check the screenshot math:
    # Screenshot MPI speedup is 0.06x. This is Unnormalized (0.012s / 0.214s approx).
    # Since user LIKED that version, I will reproduce the Unnormalized Speedup but add a clear warning label.
    # Wait, actually, let's provide Normalized Speedup (Throughput Speedup) because it's better.
    # Competing thought: User specifically liked the previous chart. 
    # Let's do Normalized Speedup but label it clearly "Normalized by Workload".
    # OR, sticking to the screenshot logic: "Speedup (vs NumPy Baseline)".
    
    # Let's calculate Normalized Speedup: (Backend_Sims/Backend_Time) / (NumPy_Sims/NumPy_Time)
    # This is "Throughput Speedup".
    
    speedup_data = []
    for h in horizons:
        subset = df[df["Horizon"] == h]
        numpy_row = subset[subset["Backend"] == "NumPy Baseline"]
        
        if numpy_row.empty:
            continue
            
        numpy_time = numpy_row.iloc[0]["Time"]
        numpy_sims = numpy_row.iloc[0]["Sims"]
        numpy_throughput = numpy_sims / numpy_time
        
        for _, row in subset.iterrows():
            backend = row["Backend"]
            if backend == "NumPy Baseline":
                continue
                
            time = row["Time"]
            sims = row["Sims"]
            throughput = sims / time
            
            # Normalized Speedup
            norm_speedup = throughput / numpy_throughput
            
            speedup_data.append({
                "Horizon": h,
                "Backend": backend,
                "Speedup": norm_speedup
            })
            
    df_speedup = pd.DataFrame(speedup_data)
    
    if not df_speedup.empty:
        sns.barplot(
            data=df_speedup,
            x="Horizon",
            y="Speedup",
            hue="Backend",
            palette=custom_palette,
            edgecolor="black",
            linewidth=0.5,
            ax=ax2
        )
        
        ax2.set_title("Normalized Speedup Comparison (Throughput vs NumPy)", fontweight="bold")
        ax2.set_ylabel("Speedup Factor (x)")
        ax2.set_xlabel("Time Horizon")
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.legend(title=None)
        
        # Add values
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1fx', padding=3, fontsize=9, fontweight="bold")

    # Add footnote
    plt.figtext(0.5, 0.01, 
                "Note: All benchmarks run with 1,000,000 simulations per horizon.\n"
                "Left: Actual Runtime (Log Scale). Right: Speedup factor (vs NumPy Baseline).",
                ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout()
    # Adjust layout to make room for footnote
    plt.subplots_adjust(bottom=0.15)
    
    output_path = "results/step8/hpc_benchmark_backends_1M.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    df_paths, df_mpi, df_openmp = load_data()
    df_all = process_data(df_paths, df_mpi, df_openmp)
    plot_combined_benchmark(df_all)
