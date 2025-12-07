# %% [markdown]
# # Run All Cells
# Click "Run Cell" on this cell to execute all cells in sequence, or use Cmd+Shift+P → "Run All Cells"

# %%
# This cell will be executed first, but you can also run individual cells below
print("=" * 80)
print("Starting HPC MC Demo - All Parts")
print("=" * 80)

# %% [markdown]
# # Part 0 – Environment Setup
# Load packages and reload scenario_mc so that changes to mc.py
# are immediately reflected when running this notebook.

# %%
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt

# Add project root to Python path
# Try multiple methods to find project root
if '__file__' in globals():
    # Running as script
    project_root = Path(__file__).parent.parent
else:
    # Running in notebook/interactive mode
    # Try to find project root by looking for finmc_tech directory
    current = Path.cwd()
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "finmc_tech").exists() and (parent / "finmc_tech" / "simulation").exists():
            project_root = parent
            break
    if project_root is None:
        # Fallback: assume we're in notebooks/ directory
        project_root = Path.cwd().parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path includes: {str(project_root) in sys.path}")

# Define display function for non-notebook environments
try:
    from IPython.display import display, Image
    HAS_IPYTHON = True
except ImportError:
    def display(obj):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string())
        else:
            print(obj)
    HAS_IPYTHON = False

from finmc_tech.simulation import scenario_mc
reload(scenario_mc)

print("scenario_mc loaded.")


# %% [markdown]
# # Part 1 – Historical Drift (Baseline)
# 
# This reproduces the original method used in scenario_mc.py:
# - Load NVDA historical data
# - Compute monthly mean return
# - Convert to annualized drift
# - Linearly scale to 1Y / 3Y / 5Y / 10Y mu_horizon
#
# This baseline will later be compared to ML-based horizon drift.

# %%
# Locate data file exactly as scenario_mc does
# Use absolute paths relative to project root
DEFAULT_DATA_PATHS = [
    str(project_root / path) if not Path(path).is_absolute() else path
    for path in scenario_mc.DEFAULT_DATA_PATHS
]

data_path = None
for p in DEFAULT_DATA_PATHS:
    if Path(p).exists():
        data_path = Path(p)
        break
assert data_path is not None, f"No data file found. Tried: {DEFAULT_DATA_PATHS}"

df = pd.read_csv(data_path)

# Auto-detect date column
date_col = None
for col in ["px_date", "date", "Date", "period_end", "timestamp"]:
    if col in df.columns:
        date_col = col
        break

assert date_col is not None, "No date column found"

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).set_index(date_col)

assert "adj_close" in df.columns, "adj_close not found in data"
returns = df["adj_close"].pct_change().dropna()

steps_per_year = 12
mu_monthly = returns.mean()
mu_annual = mu_monthly * steps_per_year

print(f"Monthly mean return: {mu_monthly:.4%}")
print(f"Annualized drift (historical mean): {mu_annual:.4%}")

# Assemble baseline horizon drifts
HORIZON_STEPS = scenario_mc.HORIZON_STEPS

records = []
for h_name, n_steps in HORIZON_STEPS.items():
    horizon_months = n_steps
    mu_horizon_hist = mu_annual * (horizon_months / 12.0)
    records.append({
        "horizon": h_name,
        "n_steps": n_steps,
        "months": horizon_months,
        "mu_horizon_hist": mu_horizon_hist,
    })

mu_hist_df = pd.DataFrame(records)
print("\nHistorical drift (baseline):")
display(mu_hist_df)


# %% [markdown]
# # Part 2 – ML-Based Horizon Drift
#
# If the user has horizon-specific models (1Y / 3Y / 5Y / 10Y),
# then we compute mu_horizon using those models.
#
# This corresponds to the literature-consistent approach:
#   E[R_{t+h}] = f_h(x_t)
#
# If a given model is missing, we record "None" for that horizon.

# %%
from finmc_tech.simulation.scenario_mc import (
    HORIZON_MODEL_PATHS,
    predict_horizon_return,
    load_latest_features,
)
import joblib

# Ensure we're using absolute paths for data files
# Temporarily modify DEFAULT_DATA_PATHS to use absolute paths
original_paths = scenario_mc.DEFAULT_DATA_PATHS.copy()
scenario_mc.DEFAULT_DATA_PATHS = [
    str(project_root / path) if not Path(path).is_absolute() else path
    for path in original_paths
]

# Also update model paths to use absolute paths
HORIZON_MODEL_PATHS_ABS = {}
for h_name, paths in HORIZON_MODEL_PATHS.items():
    HORIZON_MODEL_PATHS_ABS[h_name] = {
        "model": str(project_root / paths["model"]) if not Path(paths["model"]).is_absolute() else paths["model"],
        "scaler": str(project_root / paths["scaler"]) if not Path(paths["scaler"]).is_absolute() else paths["scaler"],
    }

# Load latest feature row
X_last, S0, dates, freq_info = load_latest_features("NVDA")
display(X_last.tail(1))

ml_records = []

for h_name, paths in HORIZON_MODEL_PATHS_ABS.items():
    m_path = Path(paths["model"])
    s_path = Path(paths["scaler"])
    if m_path.exists() and s_path.exists():
        model_h = joblib.load(m_path)
        scaler_h = joblib.load(s_path)
        mu_horizon_ml = predict_horizon_return(model_h, scaler_h, X_last)
        ml_records.append({
            "horizon": h_name,
            "mu_horizon_ml": mu_horizon_ml,
            "model_path": str(m_path),
        })
    else:
        ml_records.append({
            "horizon": h_name,
            "mu_horizon_ml": None,
            "model_path": f"NOT FOUND ({m_path})",
        })

mu_ml_df = pd.DataFrame(ml_records)
print("\nML-based horizon drift:")
display(mu_ml_df)

# Merge for comparison
print("\nComparison (Historical vs ML):")
mu_compare = mu_hist_df.merge(mu_ml_df, on="horizon", how="left")
display(mu_compare)


# %% [markdown]
# # Part 3 – Configure a Small Monte Carlo Experiment
#
# Now that we have horizon drifts (historical vs ML),
# we set up a *small* Monte Carlo experiment to:
# - Fix one horizon (e.g. 1Y = 12 steps)
# - Choose a reasonable volatility
# - Choose a small number of paths (e.g. 2,000)
# - Compute the "workload" = n_paths × n_steps
#
# This lets us reason about computational cost and where HPC helps.

# %%
from finmc_tech.simulation.scenario_mc import (
    run_driver_aware_mc_fast,
    summarize_paths,
)

# Basic config for the toy MC experiment
S0_demo = float(S0)  # reuse last price from Part 2
horizon_label_demo = "1Y"
horizon_steps_demo = scenario_mc.HORIZON_STEPS[horizon_label_demo]

# Volatility: estimate from history as in mc.py, or default to 40%
if "adj_close" in df.columns:
    returns_hist = df["adj_close"].pct_change().dropna()
    sigma_annual_demo = returns_hist.std() * np.sqrt(12)
else:
    sigma_annual_demo = 0.40

# Number of simulations for this small demo
n_sims_demo = 2000
steps_per_year_demo = 12

print(f"Horizon label   : {horizon_label_demo}")
print(f"Horizon steps   : {horizon_steps_demo}")
print(f"S0 (demo)       : {S0_demo:.2f}")
print(f"Annual volatility (demo): {sigma_annual_demo:.2%}")
print(f"n_sims (demo)   : {n_sims_demo:,}")

# Compute a simple "workload" metric: how many path-steps
workload = n_sims_demo * horizon_steps_demo
print(f"Approximate workload (path-steps) = n_sims × n_steps = {workload:,}")


# %% [markdown]
# # Part 4 – Run Baseline NumPy Monte Carlo (All Horizons)
#
# In this part we:
# - Loop through all horizons (1Y, 3Y, 5Y, 10Y)
# - For each horizon, choose a drift:
#     - either historical-based mu_horizon_hist
#     - or ML-based mu_horizon_ml, if available
# - Convert mu_horizon into a per-step drift sequence (mu_seq)
# - Call `run_driver_aware_mc_fast` to simulate all paths
# - Store results for visualization in Part 4B
#
# This gives a concrete view of the Monte Carlo engine BEFORE
# we add HPC accelerations like Numba / MPI / OpenMP.

# %%
# Store results for all horizons
all_horizon_results = {}

# Loop through all horizons
for horizon_name in ["1Y", "3Y", "5Y", "10Y"]:
    print("\n" + "=" * 80)
    print(f"Processing {horizon_name} horizon...")
    print("=" * 80)
    
    # Get horizon steps
    horizon_steps = scenario_mc.HORIZON_STEPS[horizon_name]
    
    # Pick the horizon row from the comparison table built in Part 2
    row = mu_compare.loc[mu_compare["horizon"] == horizon_name].iloc[0]
    
    mu_horizon_hist = row["mu_horizon_hist"]
    mu_horizon_ml = row.get("mu_horizon_ml", None)
    
    print(f"{horizon_name} horizon drift candidates:")
    print(f"  Historical-based mu_horizon: {mu_horizon_hist:.2%}")
    if pd.notnull(mu_horizon_ml):
        print(f"  ML-based mu_horizon        : {mu_horizon_ml:.2%}")
    else:
        print("  ML-based mu_horizon        : None (model not available)")
    
    # Choose which drift to use for this demo
    if pd.notnull(mu_horizon_ml):
        mu_horizon = float(mu_horizon_ml)
        drift_source = "ML"
    else:
        mu_horizon = float(mu_horizon_hist)
        drift_source = "historical"
    
    print(f"\nUsing {drift_source} mu_horizon = {mu_horizon:.2%} for {horizon_name} MC demo.")
    
    # Build a per-step drift sequence (monthly steps)
    # For simplicity: equal drift each step, so sum over steps ≈ mu_horizon
    mu_step = mu_horizon / horizon_steps
    mu_seq = np.full(horizon_steps, mu_step, dtype=float)
    
    print(f"Per-step drift (mu_step): {mu_step:.4%}")
    print(f"Length of mu_seq        : {len(mu_seq)}")
    print(f"Horizon steps           : {horizon_steps}")
    
    # Run the baseline vectorized NumPy Monte Carlo
    paths, terminals = run_driver_aware_mc_fast(
        S0=S0_demo,
        mu_seq=mu_seq,
        sigma_annual=sigma_annual_demo,
        n_sims=n_sims_demo,
        horizon_steps=horizon_steps,
        seed=42,
        steps_per_year=steps_per_year_demo,
    )
    
    print(f"\nMC demo finished for {horizon_name}.")
    print(f"paths shape     : {paths.shape}  (n_sims, n_steps+1)")
    print(f"terminals shape : {terminals.shape}")
    
    # Summarize terminal distribution
    summary = summarize_paths(terminals, S0_demo)
    print(f"\nTerminal distribution summary ({horizon_name}):")
    for k, v in summary.items():
        if isinstance(v, float):
            if k in ["exp_return", "up_prob"]:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Store results
    all_horizon_results[horizon_name] = {
        "paths": paths,
        "terminals": terminals,
        "mu_horizon": mu_horizon,
        "mu_step": mu_step,
        "horizon_steps": horizon_steps,
        "drift_source": drift_source,
        "summary": summary,
    }

print("\n" + "=" * 80)
print("All horizons processed successfully!")
print("=" * 80)


# %% [markdown]
# # Part 3B – One-Step Monte Carlo Math Check (1Y Example)
#
# This section fully expands the one-step Monte Carlo update formula using a 1Y example:
#
# - Annualized volatility: `sigma_annual_demo`
# - Steps per year: `steps_per_year_demo = 12`
# - Per-step volatility:
#   σ_step = σ_annual / sqrt(steps_per_year)
#
# - Per-step return:
#   r_t = μ_step + σ_step * eps_t,   eps_t ~ N(0, 1)
#
# - Price update:
#   S_{t+1} = S_t * (1 + r_t)
#
# We take only 3 paths over the full 1Y horizon (12 steps), and print
# eps, returns, and prices for each step to facilitate explanation in
# presentations/demos.

# %%
# Use 1Y horizon for the math demonstration
horizon_name_math = "1Y"
result_1y = all_horizon_results[horizon_name_math]
mu_step_1y = result_1y["mu_step"]
horizon_steps_1y = result_1y["horizon_steps"]

# Tiny example: 3 paths, full 1Y horizon
n_paths_tiny = 3
steps_tiny = horizon_steps_1y  # 12 for 1Y
sigma_step_demo = sigma_annual_demo / np.sqrt(steps_per_year_demo)

print(f"Per-step volatility (sigma_step_demo): {sigma_step_demo:.4%}")
print(f"Per-step drift (mu_step_1y): {mu_step_1y:.4%}")
print(f"Horizon steps (1Y): {steps_tiny}")

rng_tiny = np.random.default_rng(123)

# Standard normal eps for each path and step
eps_tiny = rng_tiny.normal(0.0, 1.0, size=(n_paths_tiny, steps_tiny))

# Per-step returns: r_t = mu_step_1y + sigma_step_demo * eps_t
rets_tiny = mu_step_1y + sigma_step_demo * eps_tiny

# Clip extreme negative returns to avoid negative prices
rets_tiny = np.clip(rets_tiny, -0.99, None)

print("\nEpsilon (eps_tiny) for 3 paths, first 5 steps:")
df_eps = pd.DataFrame(
    eps_tiny[:, :5].T,
    columns=[f"path{i}" for i in range(n_paths_tiny)]
)
df_eps.index.name = "step"
display(df_eps)

print("\nPer-step returns (rets_tiny) for 3 paths, first 5 steps:")
df_rets = pd.DataFrame(
    rets_tiny[:, :5].T,
    columns=[f"path{i}" for i in range(n_paths_tiny)]
)
df_rets.index.name = "step"
display(df_rets)

# Explicit price recursion for the tiny example
prices_tiny = np.zeros((n_paths_tiny, steps_tiny + 1))
prices_tiny[:, 0] = S0_demo
for t in range(steps_tiny):
    prices_tiny[:, t + 1] = prices_tiny[:, t] * (1.0 + rets_tiny[:, t])

print("\nTiny example prices for 3 paths (first 10 steps):")
prices_tiny_df = pd.DataFrame(
    prices_tiny.T,  # shape: (step, path)
    columns=[f"path{i}" for i in range(n_paths_tiny)]
)
prices_tiny_df.index.name = "step"
display(prices_tiny_df.head(10))


# %% [markdown]
# # Part 4B – Visualize MC Demo Paths and Terminal Distribution (All Horizons)
#
# This section visualizes the Monte Carlo demo from Part 4 for all horizons:
# - Plot a subset of paths on one figure to show price trajectories over time
# - Create a histogram of terminal returns to visualize the distribution shape
#
# These plots can be directly used in HPC / Monte Carlo section presentations.

# %%
# Generate plots for each horizon
for horizon_name in ["1Y", "3Y", "5Y", "10Y"]:
    print(f"\n{'='*80}")
    print(f"Generating plots for {horizon_name}...")
    print(f"{'='*80}")
    
    result = all_horizon_results[horizon_name]
    paths = result["paths"]
    terminals = result["terminals"]
    horizon_steps = result["horizon_steps"]
    
    # Extract parameters for display
    mu_horizon = result["mu_horizon"]
    mu_step = result["mu_step"]
    drift_source = result["drift_source"]
    horizon_steps = result["horizon_steps"]
    sigma_step = sigma_annual_demo / np.sqrt(steps_per_year_demo)
    
    # Print mathematical formulation and parameters
    print("\n" + "="*80)
    print(f"MONTE CARLO SIMULATION - MATHEMATICAL FORMULATION ({horizon_name})")
    print("="*80)
    print("\nUpdate Formula:")
    print("  r_t = μ_step + σ_step × ε_t")
    print("  S_{t+1} = S_t × (1 + r_t)")
    print("\nwhere:")
    print(f"  • r_t: per-step return at time t")
    print(f"  • μ_step: per-step drift = {mu_step:.6f} ({mu_step:.4%})")
    print(f"  • σ_step: per-step volatility = {sigma_step:.6f} ({sigma_step:.4%})")
    print(f"  • ε_t: random shock ~ N(0, 1)")
    print(f"  • S_t: price at time t")
    print("\nParameters:")
    print(f"  • S_0 (Initial Price) = ${S0_demo:.2f}")
    print(f"  • μ_horizon (Total Drift) = {mu_horizon:.6f} ({mu_horizon:.2%}) [{drift_source}]")
    print(f"  • μ_step (Per-Step Drift) = {mu_step:.6f} ({mu_step:.4%})")
    print(f"  • σ_annual (Annual Volatility) = {sigma_annual_demo:.6f} ({sigma_annual_demo:.2%})")
    print(f"  • σ_step (Per-Step Volatility) = {sigma_step:.6f} ({sigma_step:.4%})")
    print(f"  • Steps = {horizon_steps} (months)")
    print(f"  • Steps per Year = {steps_per_year_demo}")
    print(f"  • Horizon = {horizon_name}")
    print(f"  • Number of Paths = {n_sims_demo:,}")
    print("\nCalculation:")
    print(f"  σ_step = σ_annual / √steps_per_year")
    print(f"         = {sigma_annual_demo:.6f} / √{steps_per_year_demo}")
    print(f"         = {sigma_annual_demo:.6f} / {np.sqrt(steps_per_year_demo):.6f}")
    print(f"         = {sigma_step:.6f}")
    print(f"\n  μ_step = μ_horizon / horizon_steps")
    print(f"         = {mu_horizon:.6f} / {horizon_steps}")
    print(f"         = {mu_step:.6f}")
    print("="*80 + "\n")
    
    # Plot a subset of simulated price paths (clean chart, no text overlay)
    n_plot_paths = min(20, paths.shape[0])
    time_axis = np.arange(paths.shape[1])  # 0..horizon_steps
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_plot_paths):
        ax.plot(time_axis, paths[i, :], alpha=0.3, linewidth=0.8)
    ax.axhline(S0_demo, linestyle="--", linewidth=2, color="red", label=f"S0 = ${S0_demo:.2f}")
    ax.set_xlabel("Step (month)", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title(f"Sample Simulated Price Paths ({horizon_name} Horizon)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path_1 = project_root / "notebooks" / f"mc_demo_price_paths_{horizon_name.lower()}.png"
    plt.savefig(plot_path_1, dpi=200, bbox_inches="tight")
    print(f"✓ Saved price paths plot: {plot_path_1}")
    
    # Display the image directly in notebook
    if HAS_IPYTHON:
        display(Image(str(plot_path_1)))
    else:
        plt.show()
    plt.close()
    
    # Terminal return distribution: (S_T / S0 - 1)
    terminal_returns = terminals / S0_demo - 1.0
    summary = result["summary"]
    
    # Calculate statistics from terminal returns
    p5_val = np.percentile(terminal_returns, 5)
    p25_val = np.percentile(terminal_returns, 25)
    p50_val = np.percentile(terminal_returns, 50)
    p75_val = np.percentile(terminal_returns, 75)
    p95_val = np.percentile(terminal_returns, 95)
    exp_return = np.mean(terminal_returns)
    up_prob = np.mean(terminal_returns > 0)
    
    # VaR and CVaR (in return terms, not price terms)
    var_5_return = p5_val
    cvar_5_return = np.mean(terminal_returns[terminal_returns <= p5_val])
    
    # Print terminal distribution statistics
    print("\n" + "="*80)
    print(f"TERMINAL RETURN DISTRIBUTION STATISTICS ({horizon_name})")
    print("="*80)
    print("\nPercentiles:")
    print(f"  • P5  = {p5_val:.6f} ({p5_val:.2%})")
    print(f"  • P25 = {p25_val:.6f} ({p25_val:.2%})")
    print(f"  • P50 = {p50_val:.6f} ({p50_val:.2%}) [Median]")
    print(f"  • P75 = {p75_val:.6f} ({p75_val:.2%})")
    print(f"  • P95 = {p95_val:.6f} ({p95_val:.2%})")
    print("\nExpected Value:")
    print(f"  • E[R] (Expected Return) = {exp_return:.6f} ({exp_return:.2%})")
    print("\nRisk Metrics:")
    print(f"  • Up Probability = {up_prob:.6f} ({up_prob:.2%})")
    print(f"  • VaR_5 (5th Percentile Return) = {var_5_return:.6f} ({var_5_return:.2%})")
    print(f"  • CVaR_5 (Conditional VaR) = {cvar_5_return:.6f} ({cvar_5_return:.2%})")
    print("\nTerminal Price Statistics:")
    terminal_p5 = np.percentile(terminals, 5)
    terminal_p50 = np.percentile(terminals, 50)
    terminal_p95 = np.percentile(terminals, 95)
    terminal_mean = np.mean(terminals)
    print(f"  • P5 Price  = ${terminal_p5:.2f}")
    print(f"  • P50 Price = ${terminal_p50:.2f} [Median]")
    print(f"  • P95 Price = ${terminal_p95:.2f}")
    print(f"  • Mean Price = ${terminal_mean:.2f}")
    print("="*80 + "\n")
    
    # Plot terminal distribution (clean chart, no text overlay)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(terminal_returns, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Terminal Return", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Terminal Return Distribution ({horizon_name} Horizon)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add vertical lines for key statistics (minimal labels)
    ax.axvline(p5_val, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"P5: {p5_val:.2%}")
    ax.axvline(p50_val, color="green", linestyle="--", linewidth=2, alpha=0.7, label=f"P50: {p50_val:.2%}")
    ax.axvline(p95_val, color="blue", linestyle="--", linewidth=2, alpha=0.7, label=f"P95: {p95_val:.2%}")
    ax.axvline(exp_return, color="orange", linestyle=":", linewidth=2, alpha=0.7, label=f"E[R]: {exp_return:.2%}")
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    
    # Save the plot
    plot_path_2 = project_root / "notebooks" / f"mc_demo_terminal_distribution_{horizon_name.lower()}.png"
    plt.savefig(plot_path_2, dpi=200, bbox_inches="tight")
    print(f"✓ Saved terminal distribution plot: {plot_path_2}")
    
    # Display the image directly in notebook
    if HAS_IPYTHON:
        display(Image(str(plot_path_2)))
    else:
        plt.show()
    plt.close()

print("\n" + "="*80)
print("All plots generated successfully!")
print("="*80)
