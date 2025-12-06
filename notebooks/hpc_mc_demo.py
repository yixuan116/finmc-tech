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
    from IPython.display import display
except ImportError:
    def display(obj):
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string())
        else:
            print(obj)

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
# # Part 4 – Run Baseline NumPy Monte Carlo
#
# In this part we:
# - Choose a drift for the 1Y horizon:
#     - either historical-based mu_horizon_hist
#     - or ML-based mu_horizon_ml, if available
# - Convert mu_horizon into a per-step drift sequence (mu_seq)
# - Call `run_driver_aware_mc_fast` to simulate all paths
# - Inspect:
#     - shape of the paths
#     - a few sample paths
#     - terminal distribution summary
#
# This gives a concrete view of the Monte Carlo engine BEFORE
# we add HPC accelerations like Numba / MPI / OpenMP.

# %%
# Pick the horizon row from the comparison table built in Part 2
row_1y = mu_compare.loc[mu_compare["horizon"] == "1Y"].iloc[0]

mu_horizon_hist_1y = row_1y["mu_horizon_hist"]
mu_horizon_ml_1y = row_1y.get("mu_horizon_ml", None)

print("1Y horizon drift candidates:")
print(f"  Historical-based mu_horizon: {mu_horizon_hist_1y:.2%}")
if pd.notnull(mu_horizon_ml_1y):
    print(f"  ML-based mu_horizon        : {mu_horizon_ml_1y:.2%}")
else:
    print("  ML-based mu_horizon        : None (model not available)")

# Choose which drift to use for this demo
if pd.notnull(mu_horizon_ml_1y):
    mu_horizon_demo = float(mu_horizon_ml_1y)
    drift_source = "ML"
else:
    mu_horizon_demo = float(mu_horizon_hist_1y)
    drift_source = "historical"

print(f"\nUsing {drift_source} mu_horizon_demo = {mu_horizon_demo:.2%} for 1Y MC demo.")

# Build a per-step drift sequence (monthly steps)
# For simplicity: equal drift each step, so sum over 12 steps ≈ mu_horizon_demo
mu_step_demo = mu_horizon_demo / horizon_steps_demo
mu_seq_demo = np.full(horizon_steps_demo, mu_step_demo, dtype=float)

print(f"Per-step drift (mu_step_demo): {mu_step_demo:.4%}")
print(f"Length of mu_seq_demo        : {len(mu_seq_demo)}")

# Run the baseline vectorized NumPy Monte Carlo
paths_demo, terminals_demo = run_driver_aware_mc_fast(
    S0=S0_demo,
    mu_seq=mu_seq_demo,
    sigma_annual=sigma_annual_demo,
    n_sims=n_sims_demo,
    horizon_steps=horizon_steps_demo,
    seed=42,
    steps_per_year=steps_per_year_demo,
)

print("\nMC demo finished.")
print(f"paths_demo shape     : {paths_demo.shape}  (n_sims, n_steps+1)")
print(f"terminals_demo shape : {terminals_demo.shape}")

# Show first few simulated paths
demo_df = pd.DataFrame(paths_demo[:5, :].T)
demo_df.index.name = "step"
print("\nFirst 5 simulated paths (prices):")
display(demo_df)

# Summarize terminal distribution
summary_demo = summarize_paths(terminals_demo, S0_demo)
print("\nTerminal distribution summary (1Y demo):")
for k, v in summary_demo.items():
    if isinstance(v, float):
        if k in ["exp_return", "up_prob"]:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

