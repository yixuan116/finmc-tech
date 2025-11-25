#!/usr/bin/env python3
"""
Step 7: Scenario-Based Monte Carlo Forecasting with Driver-Aware Shocks.

This module implements driver-aware Monte Carlo simulation by:
1. Building macro scenarios aligned with Step 5 drivers (TNX/VIX + interactions)
2. Generating conditional drift from champion RF model
3. Running Monte Carlo paths for 12-month horizon
4. Producing forecast tables, fan charts, and distribution plots
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Default paths
DEFAULT_DATA_PATHS = [
    "data/processed/nvda_features_extended.csv",
    "data/processed/NVDA_revenue_features.csv",
]
DEFAULT_MODEL_PATH = "models/champion_model.pkl"
DEFAULT_SCALER_PATH = "models/feature_scaler.pkl"
DEFAULT_N_SIMS = 500 #test for simulation
DEFAULT_HORIZON_MONTHS = 12
RANDOM_STATE = 42


def load_latest_features(ticker: str = "NVDA") -> Tuple[pd.DataFrame, float, pd.DatetimeIndex, Dict]:
    """
    Load latest features from processed data files.
    
    Parameters
    ----------
    ticker : str
        Stock ticker (default: "NVDA")
    
    Returns
    -------
    X_last : pd.DataFrame
        Latest feature vector (single row)
    S0 : float
        Current stock price (adj_close)
    dates : pd.DatetimeIndex
        Date index from data
    freq_info : dict
        Frequency information (freq, is_quarterly, steps_per_year)
    """
    # Try to find data file
    data_path = None
    for path in DEFAULT_DATA_PATHS:
        if Path(path).exists():
            data_path = Path(path)
            break
    
    if data_path is None:
        raise FileNotFoundError(f"No data file found. Tried: {DEFAULT_DATA_PATHS}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Auto-detect date column
    date_col = None
    for col in ["px_date", "date", "Date", "period_end", "timestamp"]:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("No date column found")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    
    # Get latest row
    X_last = df.iloc[[-1]].copy()
    
    # Get current price
    if "adj_close" in X_last.columns:
        S0 = float(X_last["adj_close"].iloc[0])
    elif "close" in X_last.columns:
        S0 = float(X_last["close"].iloc[0])
    else:
        raise ValueError("No price column found (adj_close or close)")
    
    # Detect frequency
    dates = df.index
    if len(dates) > 1:
        freq_days = (dates[-1] - dates[-2]).days
        if freq_days <= 35:  # Monthly or daily
            freq = "monthly"
            is_quarterly = False
            steps_per_year = 12
        else:  # Quarterly
            freq = "quarterly"
            is_quarterly = True
            steps_per_year = 4
    else:
        freq = "monthly"
        is_quarterly = False
        steps_per_year = 12
    
    freq_info = {
        "freq": freq,
        "is_quarterly": is_quarterly,
        "steps_per_year": steps_per_year,
    }
    
    print(f"✓ Loaded latest features: {len(X_last.columns)} features")
    print(f"  Current price (S0): ${S0:.2f}")
    print(f"  Date: {X_last.index[0]}")
    print(f"  Frequency: {freq} ({steps_per_year} steps/year)")
    
    return X_last, S0, dates, freq_info


def build_scenarios(X_last: pd.DataFrame, history: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build scenario shock specifications.
    
    Parameters
    ----------
    X_last : pd.DataFrame
        Latest feature vector
    history : pd.DataFrame
        Historical data for percentile calculations
    
    Returns
    -------
    scenarios : dict
        Dictionary mapping scenario names to shock specifications
    """
    scenarios = {}
    
    # Get historical percentiles for VIX
    if "vix_level" in history.columns:
        vix_history = history["vix_level"].dropna()
        vix_p12 = vix_history.quantile(0.12)
        vix_p90 = vix_history.quantile(0.90)
    else:
        vix_p12 = 12.0  # Default low VIX
        vix_p90 = 30.0  # Default high VIX
    
    # Baseline: no shock
    scenarios["baseline"] = {}
    
    # Rate cut: TNX down 50bp
    scenarios["rate_cut"] = {
        "tnx_yield": -0.50,  # Absolute change in percentage points
        "tnx_change_3m": None,  # Will be computed
    }
    
    # Rate spike: TNX up 100bp
    scenarios["rate_spike"] = {
        "tnx_yield": +1.00,
        "tnx_change_3m": None,
    }
    
    # VIX crash: down to 12th percentile
    current_vix = X_last["vix_level"].iloc[0] if "vix_level" in X_last.columns else 20.0
    scenarios["vix_crash"] = {
        "vix_level": vix_p12 - current_vix,  # Absolute change
        "vix_change_3m": None,  # Will be computed
    }
    
    # VIX spike: up to 90th percentile
    scenarios["vix_spike"] = {
        "vix_level": vix_p90 - current_vix,
        "vix_change_3m": None,
    }
    
    print(f"\nBuilt {len(scenarios)} scenarios:")
    for name, spec in scenarios.items():
        print(f"  - {name}: {spec}")
    
    return scenarios


def recompute_interaction_features(
    X: pd.DataFrame,
    base_features: Dict[str, float],
) -> pd.DataFrame:
    """
    Recompute interaction features after shocking base macro variables.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature vector
    base_features : dict
        Updated base feature values
    
    Returns
    -------
    X_new : pd.DataFrame
        Feature vector with recomputed interactions
    """
    X_new = X.copy()
    
    # Update base features
    for feat, value in base_features.items():
        if feat in X_new.columns:
            X_new[feat] = value
    
    # Find interaction features and recompute
    interaction_cols = [col for col in X_new.columns if col.startswith("ix_")]
    
    for ix_col in interaction_cols:
        # Parse interaction: ix_macro__micro
        parts = ix_col.replace("ix_", "").split("__")
        if len(parts) == 2:
            macro_feat, micro_feat = parts
            
            # Get macro value (from base_features or X_new)
            if macro_feat in base_features:
                macro_val = base_features[macro_feat]
            elif macro_feat in X_new.columns:
                macro_val = X_new[macro_feat].iloc[0]
            else:
                continue
            
            # Get micro value (from X_new)
            if micro_feat in X_new.columns:
                micro_val = X_new[micro_feat].iloc[0]
            else:
                continue
            
            # Recompute interaction
            X_new[ix_col] = macro_val * micro_val
    
    return X_new


def apply_shock(
    X: pd.DataFrame,
    shock_spec: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply shock to feature vector.
    
    Parameters
    ----------
    X : pd.DataFrame
        Original feature vector
    shock_spec : dict
        Shock specification (e.g., {"tnx_yield": -0.50})
    
    Returns
    -------
    X_shocked : pd.DataFrame
        Shocked feature vector
    """
    X_shocked = X.copy()
    base_features = {}
    
    # Apply direct shocks
    for feat, shock_value in shock_spec.items():
        if feat in X_shocked.columns:
            if shock_value is None:
                continue
            
            current_val = X_shocked[feat].iloc[0]
            
            # If shock is absolute change, add it
            if isinstance(shock_value, (int, float)):
                new_val = current_val + shock_value
                X_shocked[feat] = new_val
                base_features[feat] = new_val
            else:
                base_features[feat] = current_val
    
    # Compute derived changes (e.g., tnx_change_3m from tnx_yield change)
    if "tnx_yield" in shock_spec and "tnx_change_3m" in X_shocked.columns:
        # Approximate: if TNX changes by X, change_3m should reflect this
        tnx_change = shock_spec["tnx_yield"]
        if tnx_change is not None:
            # Rough approximation: change_3m ≈ change / 3 (for 3-month change)
            current_change = X_shocked["tnx_change_3m"].iloc[0]
            new_change = current_change + (tnx_change / 3.0)
            X_shocked["tnx_change_3m"] = new_change
            base_features["tnx_change_3m"] = new_change
    
    if "vix_level" in shock_spec and "vix_change_3m" in X_shocked.columns:
        # Similar for VIX
        vix_change = shock_spec["vix_level"]
        if vix_change is not None:
            current_change = X_shocked["vix_change_3m"].iloc[0]
            new_change = current_change + (vix_change / 3.0)
            X_shocked["vix_change_3m"] = new_change
            base_features["vix_change_3m"] = new_change
    
    # Recompute interaction features
    X_shocked = recompute_interaction_features(X_shocked, base_features)
    
    return X_shocked


def predict_conditional_drift(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    X_seq: pd.DataFrame,
    horizon_steps: int,
) -> np.ndarray:
    """
    Predict conditional drift for multi-step horizon.
    
    For short-horizon (12M), we assume macro regime persists.
    Predict μ once from shocked X_last, then repeat for all steps.
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained champion model
    scaler : StandardScaler
        Feature scaler
    X_seq : pd.DataFrame
        Initial feature vector
    horizon_steps : int
        Number of steps ahead
    
    Returns
    -------
    mu_seq : np.ndarray
        Conditional drift sequence (shape: [horizon_steps])
    """
    # Strict training-order alignment
    feat_order = list(scaler.feature_names_in_)
    
    # Build X_current with exactly these columns in this order
    X_current = pd.DataFrame(index=X_seq.index, columns=feat_order, dtype=float)
    
    missing = []
    for f in feat_order:
        if f in X_seq.columns:
            # Convert to float, handling any non-numeric values
            try:
                X_current[f] = pd.to_numeric(X_seq[f], errors='coerce').fillna(0.0).astype(float)
            except:
                X_current[f] = 0.0
        else:
            X_current[f] = 0.0
            missing.append(f)
    
    if missing:
        print(f"⚠ Missing {len(missing)} training features; filled with 0.0")
    
    # Assert shape matches
    assert X_current.shape[1] == len(feat_order), \
        f"Feature count mismatch: {X_current.shape[1]} != {len(feat_order)}"
    
    # Scale features
    X_scaled = scaler.transform(X_current.values)
    
    # Predict drift once (expected return per step)
    mu = float(model.predict(X_scaled)[0])
    
    # For short-horizon, assume constant drift (macro regime persists)
    mu_seq = np.full(horizon_steps, mu)
    
    return mu_seq


def run_driver_aware_mc_fast(
    S0: float,
    mu_seq: np.ndarray,
    sigma_annual: float,
    n_sims: int,
    horizon_steps: int,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized Monte Carlo simulation (no progress bar).
    
    Uses arithmetic return per step:
    r_t = μ_t + σ_step * ε
    S_{t+1} = S_t * (1 + r_t)
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    mu_seq : np.ndarray
        Conditional drift sequence (expected returns per step, already monthly)
    sigma_annual : float
        Annualized volatility
    n_sims : int
        Number of simulations
    horizon_steps : int
        Number of steps ahead
    seed : int
        Random seed
    
    Returns
    -------
    paths : np.ndarray
        Price paths (shape: [n_sims, horizon_steps + 1])
    terminals : np.ndarray
        Terminal prices (shape: [n_sims])
    """
    rng = np.random.default_rng(seed)
    
    # Convert annualized sigma to per-step sigma
    sigma_step = sigma_annual / np.sqrt(12)
    
    # Initialize paths
    paths = np.zeros((n_sims, horizon_steps + 1))
    paths[:, 0] = S0
    
    # Generate random shocks
    Z = rng.standard_normal((n_sims, horizon_steps))
    
    # Vectorized: compute all steps in one shot via cumprod
    # rets shape: (n_sims, horizon_steps)
    rets = mu_seq.reshape(1, -1) + sigma_step * Z
    
    # paths[:, 1:] = S0 * cumprod(1.0 + rets, axis=1)
    paths[:, 1:] = S0 * np.cumprod(1.0 + rets, axis=1)
    
    terminals = paths[:, -1]
    
    return paths, terminals


def run_driver_aware_mc_batched(
    S0: float,
    mu_seq: np.ndarray,
    sigma_annual: float,
    n_sims: int,
    horizon_steps: int,
    seed: int = RANDOM_STATE,
    batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched Monte Carlo simulation with progress bar.
    
    Uses arithmetic return per step:
    r_t = μ_t + σ_step * ε
    S_{t+1} = S_t * (1 + r_t)
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    mu_seq : np.ndarray
        Conditional drift sequence (expected returns per step, already monthly)
    sigma_annual : float
        Annualized volatility
    n_sims : int
        Number of simulations
    horizon_steps : int
        Number of steps ahead
    seed : int
        Random seed
    batch_size : int
        Batch size for progress tracking
    
    Returns
    -------
    paths : np.ndarray
        Price paths (shape: [n_sims, horizon_steps + 1])
    terminals : np.ndarray
        Terminal prices (shape: [n_sims])
    """
    rng = np.random.default_rng(seed)
    
    # Convert annualized sigma to per-step sigma
    sigma_step = sigma_annual / np.sqrt(12)
    
    # Initialize paths
    paths = np.zeros((n_sims, horizon_steps + 1))
    paths[:, 0] = S0
    
    # Calculate number of batches
    n_batches = int(np.ceil(n_sims / batch_size))
    
    # Simulate in batches with progress bar
    for b in tqdm(range(n_batches), desc="MC Batches", unit="batch", leave=False):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, n_sims)
        batch_size_actual = end_idx - start_idx
        
        # Generate random shocks for this batch
        Z_batch = rng.standard_normal((batch_size_actual, horizon_steps))
        
        # Simulate paths for this batch
        for t in range(horizon_steps):
            # Arithmetic return per step
            rets = mu_seq[t] + sigma_step * Z_batch[:, t]
            paths[start_idx:end_idx, t + 1] = paths[start_idx:end_idx, t] * (1 + rets)
    
    terminals = paths[:, -1]
    
    return paths, terminals


def summarize_paths(terminals: np.ndarray, S0: float) -> Dict[str, float]:
    """
    Summarize Monte Carlo paths.
    
    Parameters
    ----------
    terminals : np.ndarray
        Terminal prices
    S0 : float
        Initial price
    
    Returns
    -------
    summary : dict
        Summary statistics
    """
    P5 = np.percentile(terminals, 5)
    P50 = np.percentile(terminals, 50)
    P95 = np.percentile(terminals, 95)
    exp_return = np.mean(terminals) / S0 - 1.0
    up_prob = np.mean(terminals > S0)
    
    # VaR (5th percentile loss)
    var_5 = S0 - P5
    
    # CVaR (expected loss beyond VaR)
    losses = S0 - terminals
    cvar_5 = np.mean(losses[losses >= var_5])
    
    return {
        "P5": P5,
        "P50": P50,
        "P95": P95,
        "exp_return": exp_return,
        "up_prob": up_prob,
        "VaR_5": var_5,
        "CVaR_5": cvar_5,
    }


def run_scenario_forecast(
    ticker: str = "NVDA",
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    n_sims: int = DEFAULT_N_SIMS,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    output_dir: str = "outputs",
    random_seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Run full scenario-based forecasting pipeline.
    
    Parameters
    ----------
    ticker : str
        Stock ticker
    horizon_months : int
        Forecast horizon in months
    n_sims : int
        Number of Monte Carlo simulations
    model_path : str, optional
        Path to champion model
    scaler_path : str, optional
        Path to feature scaler
    output_dir : str
        Output directory
    random_seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Results dictionary with paths, summaries, etc.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_last, S0, dates, freq_info = load_latest_features(ticker)
    
    # Determine horizon steps
    if freq_info["is_quarterly"]:
        horizon_steps = horizon_months // 3  # Convert months to quarters
    else:
        horizon_steps = horizon_months
    
    print(f"\nForecast horizon: {horizon_months} months ({horizon_steps} steps)")
    
    # Load model and scaler
    model_path = model_path or DEFAULT_MODEL_PATH
    scaler_path = scaler_path or DEFAULT_SCALER_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✓ Model and scaler loaded")
    
    # Load full history for scenario building
    data_path = None
    for path in DEFAULT_DATA_PATHS:
        if Path(path).exists():
            data_path = Path(path)
            break
    
    if data_path:
        history_df = pd.read_csv(data_path)
        date_col = None
        for col in ["px_date", "date", "Date", "period_end", "timestamp"]:
            if col in history_df.columns:
                date_col = col
                break
        if date_col:
            history_df[date_col] = pd.to_datetime(history_df[date_col])
            history_df = history_df.sort_values(date_col).set_index(date_col)
    else:
        history_df = X_last
    
    # Build scenarios
    scenarios = build_scenarios(X_last, history_df)
    
    # Estimate volatility from model residuals (or use historical)
    # For now, use a default or estimate from data
    if "adj_close" in history_df.columns:
        returns = history_df["adj_close"].pct_change().dropna()
        sigma_annual = returns.std() * np.sqrt(12)  # Annualized
    else:
        sigma_annual = 0.40  # Default 40% annualized volatility
    
    print(f"Estimated volatility (annualized): {sigma_annual:.2%}")
    
    # Run scenarios
    all_results = {}
    forecast_table_rows = []
    
    print(f"\n{'='*60}")
    print(f"Running {len(scenarios)} scenarios with {n_sims:,} simulations each...")
    print(f"{'='*60}\n")
    
    for scenario_name, shock_spec in tqdm(scenarios.items(), desc="Scenarios", unit="scenario"):
        # Apply shock
        X_shocked = apply_shock(X_last, shock_spec)
        
        # Predict conditional drift (simplified: predict once, repeat for all steps)
        mu_seq = predict_conditional_drift(model, scaler, X_shocked, horizon_steps)
        
        # Run MC: use batched version with progress bar for large simulations
        if n_sims >= 1000:
            # For large simulations, use batched MC with progress bar
            paths, terminals = run_driver_aware_mc_batched(
                S0, mu_seq, sigma_annual, n_sims, horizon_steps, random_seed
            )
        else:
            # For small simulations, use fast vectorized MC without progress bar
            paths, terminals = run_driver_aware_mc_fast(
                S0, mu_seq, sigma_annual, n_sims, horizon_steps, random_seed
            )
        
        # Summarize
        summary = summarize_paths(terminals, S0)
        summary["scenario"] = scenario_name
        summary["S0"] = S0
        all_results[scenario_name] = {
            "paths": paths,
            "terminals": terminals,
            "summary": summary,
            "mu_seq": mu_seq,
        }
        
        # Add to forecast table
        forecast_table_rows.append({
            "Scenario": scenario_name,
            "S0": f"${S0:.2f}",
            "P5": f"${summary['P5']:.2f}",
            "P50": f"${summary['P50']:.2f}",
            "P95": f"${summary['P95']:.2f}",
            "Exp Return": f"{summary['exp_return']:.2%}",
            "Up Prob": f"{summary['up_prob']:.2%}",
            "VaR (5%)": f"${summary['VaR_5']:.2f}",
            "CVaR (5%)": f"${summary['CVaR_5']:.2f}",
        })
        
        # Save terminal distribution
        terminals_df = pd.DataFrame({"terminal_price": terminals})
        terminals_df.to_csv(output_path / f"scenario_terminals_{scenario_name}.csv", index=False)
    
    # Save forecast table
    forecast_df = pd.DataFrame(forecast_table_rows)
    forecast_df.to_csv(output_path / "scenario_forecast_table.csv", index=False)
    print(f"\n✓ Saved forecast table: {output_path / 'scenario_forecast_table.csv'}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Fan chart overlay
    plot_fan_chart_overlay(all_results, S0, horizon_steps, output_path / "fan_chart_overlay.png")
    
    # Individual fan charts
    for scenario_name, result in tqdm(all_results.items(), desc="Fan charts", unit="plot", leave=False):
        plot_fan_chart(
            result["paths"],
            S0,
            horizon_steps,
            scenario_name,
            output_path / f"fan_chart_{scenario_name}.png",
        )
    
    # Distribution shift plots
    for scenario_name, result in tqdm(all_results.items(), desc="Distributions", unit="plot", leave=False):
        plot_distribution_shift(
            result["terminals"],
            S0,
            scenario_name,
            output_path / f"distribution_shift_{scenario_name}.png",
        )
    
    return {
        "results": all_results,
        "forecast_table": forecast_df,
        "S0": S0,
        "horizon_steps": horizon_steps,
    }


def plot_fan_chart_overlay(
    all_results: Dict[str, Dict],
    S0: float,
    horizon_steps: int,
    output_path: Path,
) -> None:
    """Plot overlay fan chart for all scenarios."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        "baseline": "black",
        "rate_cut": "green",
        "rate_spike": "red",
        "vix_crash": "blue",
        "vix_spike": "orange",
    }
    
    for scenario_name, result in all_results.items():
        paths = result["paths"]
        
        # Compute percentiles
        percentiles = [5, 25, 50, 75, 95]
        time_steps = np.arange(horizon_steps + 1)
        
        for i, p in enumerate(percentiles):
            pct_values = np.percentile(paths, p, axis=0)
            alpha = 0.3 if p in [25, 75] else 0.5 if p == 50 else 0.2
            linestyle = "-" if p == 50 else "--"
            linewidth = 2 if p == 50 else 1
            
            ax.plot(
                time_steps,
                pct_values,
                color=colors.get(scenario_name, "gray"),
                alpha=alpha,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{scenario_name} P{p}" if i == 0 else "",
            )
    
    ax.axhline(S0, color="black", linestyle=":", linewidth=2, label="Current Price")
    ax.set_xlabel("Months Ahead", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title("Scenario Fan Chart Overlay", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_fan_chart(
    paths: np.ndarray,
    S0: float,
    horizon_steps: int,
    scenario_name: str,
    output_path: Path,
) -> None:
    """Plot individual fan chart for a scenario."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_steps = np.arange(horizon_steps + 1)
    percentiles = [5, 25, 50, 75, 95]
    
    # Fill between percentiles
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    ax.fill_between(time_steps, p5, p95, alpha=0.2, color="blue", label="90% CI")
    ax.fill_between(time_steps, p25, p75, alpha=0.3, color="blue", label="50% CI")
    ax.plot(time_steps, p50, color="darkblue", linewidth=2, label="Median")
    ax.axhline(S0, color="black", linestyle=":", linewidth=2, label="Current Price")
    
    ax.set_xlabel("Months Ahead", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title(f"Fan Chart: {scenario_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_distribution_shift(
    terminals: np.ndarray,
    S0: float,
    scenario_name: str,
    output_path: Path,
) -> None:
    """Plot terminal distribution shift."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(terminals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(S0, color="red", linestyle="--", linewidth=2, label="Current Price")
    ax.axvline(np.median(terminals), color="green", linestyle="--", linewidth=2, label="Median Forecast")
    
    ax.set_xlabel("Terminal Price ($)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Terminal Distribution: {scenario_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 7: Scenario-Based Monte Carlo Forecasting")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker")
    parser.add_argument("--h", type=int, default=DEFAULT_HORIZON_MONTHS, help="Horizon in months")
    parser.add_argument("--n", type=int, default=DEFAULT_N_SIMS, help="Number of simulations")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--model-path", default=None, help="Path to champion model")
    parser.add_argument("--scaler-path", default=None, help="Path to feature scaler")
    
    args = parser.parse_args()
    
    results = run_scenario_forecast(
        ticker=args.ticker,
        horizon_months=args.h,
        n_sims=args.n,
        output_dir=args.output_dir,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )
    
    print("\n" + "="*60)
    print("Step 7 Complete!")
    print("="*60)

