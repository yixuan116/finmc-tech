import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load latest data from CSV instead of hardcoding
csv_path = Path("results/step8/hpc_benchmark_paths_1y_3y_5y_10y.csv")
if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Filter for n_sims=100000 if possible, otherwise take all (assuming one run per horizon)
if "n_sims" in df.columns:
    df = df[df["n_sims"] == 100000]

# Extract horizons
# Mapping n_steps to label if horizon column is missing or needs sorting
step_map = {12: "1Y", 36: "3Y", 60: "5Y", 120: "10Y"}
horizons = ["1Y", "3Y", "5Y", "10Y"]

baseline_time = []
best_parallel_time = []
speedup = []

for h in horizons:
    # Find steps for this horizon
    steps = [k for k, v in step_map.items() if v == h][0]
    
    # Get baseline (NumPy)
    row_base = df[(df["n_steps"] == steps) & (df["backend"] == "baseline_numpy")]
    t_base = row_base["time_sec"].iloc[0] if not row_base.empty else 0
    baseline_time.append(t_base)
    
    # Get parallel (Numba)
    row_numba = df[(df["n_steps"] == steps) & (df["backend"] == "numba_parallel")]
    t_numba = row_numba["time_sec"].iloc[0] if not row_numba.empty else 0
    best_parallel_time.append(t_numba)
    
    # Calculate speedup
    sp = t_base / t_numba if t_numba > 0 else 0
    speedup.append(sp)

fig, ax1 = plt.subplots(figsize=(7, 4))
x = range(len(horizons))

# Runtime curves
ax1.plot(x, baseline_time, marker="o", label="Baseline (NumPy)")
ax1.plot(x, best_parallel_time, marker="o", label="Numba parallel")
ax1.set_xlabel("Horizon")
ax1.set_ylabel("Runtime (seconds)")
ax1.set_xticks(x)
ax1.set_xticklabels(horizons)
ax1.set_title("HPC Benchmark: NumPy vs Numba Across Horizons (100k Sims)")
ax1.legend(loc="upper left")
ax1.grid(True, which="both", axis="y", alpha=0.3)

# Speedup curve on secondary axis
ax2 = ax1.twinx()
ax2.plot(x, speedup, marker="s", linestyle="--", color="green", label="Speedup (baseline / numba)")
ax2.set_ylabel("Speedup (×)")
ax2.legend(loc="upper right")

# Add labels above speedup points
for i, sp in enumerate(speedup):
    ax2.text(i, sp + 0.1, f"{sp:.2f}x", ha="center", va="bottom", fontsize=8)

fig.tight_layout()

output_path = "results/step8/hpc_benchmark_plot.png"
plt.savefig(output_path, dpi=200)
print(f"Saved: {output_path}")


