import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np

# Use 'Agg' backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

# --- 1. Data Preparation ---

mpi_csv = Path("results/step8/hpc_benchmark_mpi.csv")
omp_csv = Path("results/step8/hpc_benchmark_openmp.csv")

dfs = []

# MPI Data
if mpi_csv.exists():
    df_mpi = pd.read_csv(mpi_csv)
    df_mpi = df_mpi[(df_mpi["backend"] == "mpi_python") & (df_mpi["mode"] == "mpi_parallel")]
    df_mpi = df_mpi[df_mpi["n_sims"] == 1000000]
    if not df_mpi.empty:
        df_mpi = df_mpi.groupby("n_steps", as_index=False)["time_sec"].mean()
        df_mpi["Backend"] = "MPI (4 ranks)"
        dfs.append(df_mpi)

# OpenMP Data
if omp_csv.exists():
    df_omp = pd.read_csv(omp_csv)
    df_omp = df_omp[(df_omp["backend"] == "openmp_c") & (df_omp["mode"] == "openmp_parallel")]
    df_omp = df_omp[df_omp["n_sims"] == 1000000]
    if not df_omp.empty:
        df_omp = df_omp.groupby("n_steps", as_index=False)["time_sec"].mean()
        df_omp["Backend"] = "OpenMP C"
        dfs.append(df_omp)

# Baseline Data (NumPy & Numba)
numpy_times = [1.93, 5.79, 9.64, 19.28] 
numba_times = [0.066, 0.197, 0.329, 0.658]
n_steps_list = [12, 36, 60, 120]

df_numpy = pd.DataFrame({"n_steps": n_steps_list, "time_sec": numpy_times, "Backend": "NumPy Baseline"})
df_numba = pd.DataFrame({"n_steps": n_steps_list, "time_sec": numba_times, "Backend": "Numba Parallel"})

dfs.append(df_numpy)
dfs.append(df_numba)

if not dfs:
    print("No data available.")
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)
horizons_map = {12: "1Y", 36: "3Y", 60: "5Y", 120: "10Y"}
df_all = df_all[df_all["n_steps"].isin([12, 36, 60, 120])]
df_all["Horizon"] = df_all["n_steps"].map(horizons_map)

# Calculate Speedup
speedup_records = []
baseline_map = df_all[df_all["Backend"] == "NumPy Baseline"].set_index("n_steps")["time_sec"].to_dict()

for idx, row in df_all.iterrows():
    step = row["n_steps"]
    backend = row["Backend"]
    time = row["time_sec"]
    base = baseline_map.get(step)
    
    if backend == "NumPy Baseline":
        continue 
        
    if base and time > 0:
        speedup = base / time
        speedup_records.append({
            "Horizon": horizons_map[step],
            "Backend": backend,
            "Speedup": speedup,
            "n_steps": step
        })

df_speedup = pd.DataFrame(speedup_records)

# --- 2. Plotting Style & Layout ---

sns.set_theme(style="darkgrid")
plt.rcParams["font.family"] = "sans-serif"

custom_palette = {
    "NumPy Baseline": "#5B8FA8", 
    "Numba Parallel": "#A55194", 
    "MPI (4 ranks)":  "#F1A340", 
    "OpenMP C":       "#D6604D"  
}

# Create Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

# --- Plot 1: Runtime Comparison ---
hue_order = ["NumPy Baseline", "Numba Parallel", "MPI (4 ranks)", "OpenMP C"]

sns.barplot(
    data=df_all, 
    x="Horizon", 
    y="time_sec", 
    hue="Backend", 
    palette=custom_palette,
    hue_order=hue_order,
    ax=ax1,
    edgecolor="white",
    linewidth=0.5
)

ax1.set_yscale("log") 
ax1.set_title("HPC Benchmark Comparison: Runtime by Backend and Horizon\n(Lower is Better)", fontsize=14, fontweight="bold", pad=15)
ax1.set_xlabel("Time Horizon", fontsize=12, fontweight="bold")
ax1.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
# Move legend outside to prevent overlap
ax1.legend(loc="upper left", bbox_to_anchor=(0, 1), frameon=True, fontsize=10, title=None)

# Label with increased padding
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.3fs', padding=4, fontsize=9)

# Increase y-limit to make room for labels
# For log scale, getting the current limits and multiplying the top one is safer
y_bottom, y_top = ax1.get_ylim()
ax1.set_ylim(y_bottom, y_top * 3) 


# --- Plot 2: Speedup Comparison ---
speedup_palette = {k: v for k, v in custom_palette.items() if k != "NumPy Baseline"}
hue_order_speedup = ["Numba Parallel", "MPI (4 ranks)", "OpenMP C"]

sns.barplot(
    data=df_speedup, 
    x="Horizon", 
    y="Speedup", 
    hue="Backend", 
    palette=speedup_palette,
    hue_order=hue_order_speedup,
    ax=ax2,
    edgecolor="white",
    linewidth=0.5
)

ax2.set_title("Speedup Comparison: Parallel Backends vs NumPy Baseline\n(Higher is Better)", fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Time Horizon", fontsize=12, fontweight="bold")
ax2.set_ylabel("Speedup (vs NumPy Baseline)", fontsize=12, fontweight="bold")
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

# Move legend outside or to a better spot. 
# Since bars are high, upper left inside might still overlap. Let's try upper right or outside.
ax2.legend(loc="upper left", bbox_to_anchor=(0, 1), frameon=True, fontsize=10, title=None)

# Label with padding
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.1fx', padding=4, fontsize=9, fontweight="bold")

# Increase y-limit to make room for labels
y_bottom, y_top = ax2.get_ylim()
ax2.set_ylim(y_bottom, y_top * 1.15) 

plt.tight_layout()

output_path = "results/step8/hpc_benchmark_updated_1M_styled.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved Styled Plot: {output_path}")
