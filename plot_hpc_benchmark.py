import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load latest data from CSV instead of hardcoding
csv_path = Path("results/step8/hpc_benchmark_paths_1y_3y_5y_10y.csv")
if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Filter for n_sims=1000000 if possible, otherwise take all (assuming one run per horizon)
if "n_sims" in df.columns:
    df = df[df["n_sims"] == 1000000]

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
ln1 = ax1.plot(x, baseline_time, marker="o", label="Baseline (NumPy)")
ln2 = ax1.plot(x, best_parallel_time, marker="o", label="Numba parallel")
ax1.set_xlabel("Horizon")
ax1.set_ylabel("Runtime (seconds)")
ax1.set_xticks(x)
ax1.set_xticklabels(horizons)
ax1.set_title("HPC Benchmark: NumPy vs Numba Across Horizons (1M Sims)")
ax1.grid(True, which="both", axis="y", alpha=0.3)

# Speedup curve on secondary axis
ax2 = ax1.twinx()
ln3 = ax2.plot(x, speedup, marker="s", linestyle="--", color="green", label="Speedup (baseline / numba)")
ax2.set_ylabel("Speedup (×)")

# Combine legends into one box
lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="upper left")

# Add labels for runtime values (Baseline)
for i, val in enumerate(baseline_time):
    # Default: left side
    offset_x = -0.15
    offset_y = 0
    ha = "right"
    va = "bottom"
    
    # Special case for 1Y (i=0) to put it "middle" vertically (above orange, below green)
    if i == 0:
        offset_x = 0
        offset_y = 0.9 # Move UP significantly
        ha = "center"
        va = "bottom"
        
    # Special case for 10Y (i=3)
    if i == 3:
        offset_y = -0.5 
    
    # Special case for 5Y (i=2) - move to right of point
    if i == 2:
        offset_x = 0.15
        offset_y = 0.0
        ha = "left"
        va = "center"
        
    ax1.text(i + offset_x, val + offset_y, f"{val:.3f}s", ha=ha, va=va, fontsize=8, color="blue")

# Add labels for runtime values (Parallel)
for i, val in enumerate(best_parallel_time):
    # Default: below
    offset_y = -0.2
    ha = "center"
    va_default = "top"
    
    # Special case for 1Y (i=0) - avoid hitting X axis, move UP
    if i == 0:
        offset_y = 0.3 # Move UP above the point
        va_default = "bottom"
        
    # Special case for 3Y (i=1) - move to middle between orange and blue points
    if i == 1:
        offset_y = 0.35
        va_default = "bottom"
        
    ax1.text(i, val + offset_y, f"{val:.3f}s", ha=ha, va=va_default, fontsize=8, color="orange")

# Add labels above speedup points
for i, sp in enumerate(speedup):
    # Default offsets
    offset_y = 0.6
    offset_x = 0
    
    # Special case for 1Y (i=0) - Highest
    if i == 0:
        offset_y = 2.8 # Move UP significantly
        
    # Special case for the last point (10Y, i=3) to avoid overlap
    if i == 3:
        offset_y = 0.0 
        offset_x = -0.2
        
    ax2.text(i + offset_x, sp + offset_y, f"{sp:.2f}x", ha="center", va="bottom", fontsize=9, color="green")

fig.tight_layout()

output_path = "results/step8/hpc_benchmark_plot_1M.png"
plt.savefig(output_path, dpi=200)
print(f"Saved: {output_path}")


