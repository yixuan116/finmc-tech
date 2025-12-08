#!/usr/bin/env python3
"""
Step 7: Scenario-Based Monte Carlo Forecasting with Driver-Aware Shocks.
Step 8: Scenario-Based Risk Engine + HPC Benchmark.

This module implements driver-aware Monte Carlo simulation by:
1. Building macro scenarios aligned with Step 5 drivers (TNX/VIX + interactions)
2. Generating conditional drift from champion models (RF/XGB/ElasticNet per horizon)
3. Running Monte Carlo paths for multiple horizons (1Y, 3Y, 5Y, 10Y)
4. Using driver-aware shocks based on feature importance (Firm/Macro/Interaction)
5. Producing forecast tables, fan charts, and distribution plots
6. Providing an HPC benchmark comparing vectorized NumPy vs Numba parallel execution

Multi-Horizon Driver-Aware Monte Carlo:
- Loads champion models for each horizon (1Y=RF, 3Y=RF, 5Y=XGB, 10Y=ElasticNet)
- Uses feature importance from Step 6 to weight shocks by category
- Implements 4 scenario families: BASE, MACRO_STRESS, FUNDAMENTAL_STRESS, AI_BULL
- Generates comprehensive outputs for each horizon
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import xgboost as xgb

warnings.filterwarnings("ignore")

# Default paths
DEFAULT_DATA_PATHS = [
    "data/processed/nvda_features_extended_v2.csv",
    "data/processed/NVDA_revenue_features.csv",
]
DEFAULT_MODEL_PATH = "models/champion_model.pkl"
DEFAULT_SCALER_PATH = "models/feature_scaler.pkl"
DEFAULT_N_SIMS = 500 #test for simulation
DEFAULT_HORIZON_MONTHS = 12
RANDOM_STATE = 42

#part 2: load latest features
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
    
    # Recompute interaction features only if base features were updated
    if base_features:
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
    if hasattr(scaler, "feature_names_in_"):
        feat_order = list(scaler.feature_names_in_)
    else:
        # Fallback: scaler was fit on numpy array (no names). Enforce same feature count.
        n_expected = int(getattr(scaler, "n_features_in_", X_seq.shape[1]))
        feat_order = list(X_seq.columns)[:n_expected]
        if len(feat_order) < n_expected:
            # pad with dummy names to match expected dim
            feat_order += [f"__pad_{i}__" for i in range(n_expected - len(feat_order))]
        print(f"⚠ scaler has no feature_names_in_; enforcing n_features_in_={n_expected}")
    
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


def predict_horizon_return(
    model: Any,
    scaler: StandardScaler,
    X_one: pd.DataFrame,
) -> float:
    """
    Predict horizon total simple return using a horizon-specific champion model.

    Assumptions:
    - The model was trained with a target of the form:
        y = (P_{t+h} / P_t) - 1
      i.e., total simple return over the horizon.
    - We only need a single scalar prediction for the current feature row.

    Parameters
    ----------
    model : Any
        Trained horizon-specific model (RF/XGB/ElasticNet, etc.).
    scaler : StandardScaler
        Fitted scaler for this horizon.
    X_one : pd.DataFrame
        Single-row feature DataFrame (e.g., X_last).

    Returns
    -------
    float
        Predicted horizon total simple return (mu_horizon).
    """
    # Enforce the same feature order as during training
    if hasattr(scaler, "feature_names_in_"):
        feat_order = list(scaler.feature_names_in_)
    else:
        # Fallback: use current columns order
        feat_order = list(X_one.columns)

    X_current = pd.DataFrame(index=X_one.index, columns=feat_order, dtype=float)
    missing = []

    for f in feat_order:
        if f in X_one.columns:
            X_current[f] = pd.to_numeric(X_one[f], errors="coerce").fillna(0.0).astype(float)
        else:
            X_current[f] = 0.0
            missing.append(f)

    if missing:
        print(f"⚠ Missing {len(missing)} features for horizon model; filled with 0.0")

    X_scaled = scaler.transform(X_current.values)
    mu_horizon = float(model.predict(X_scaled)[0])
    return mu_horizon


# ------------------------------------------------------------------
# Baseline Monte Carlo engine (NumPy, vectorized)
# - Single-process, vectorized over all paths using NumPy/BLAS
# - Serves as the non-HPC baseline for Step 8 benchmarks
# ------------------------------------------------------------------
def run_driver_aware_mc_fast(
    S0: float,
    mu_seq: np.ndarray,
    sigma_annual: float,
    n_sims: int,
    horizon_steps: int,
    seed: int = RANDOM_STATE,
    steps_per_year: int = 12,
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
    steps_per_year : int
        Number of steps per year (12 for monthly, 4 for quarterly)
    
    Returns
    -------
    paths : np.ndarray
        Price paths (shape: [n_sims, horizon_steps + 1])
    terminals : np.ndarray
        Terminal prices (shape: [n_sims])
    """
    rng = np.random.default_rng(seed)
    
    # Convert annualized sigma to per-step sigma, align with literature G–K–X, σ_step = σ_annual / √steps_per_year
    sigma_step = sigma_annual / np.sqrt(steps_per_year)
    
    # Initialize paths
    paths = np.zeros((n_sims, horizon_steps + 1))
    paths[:, 0] = S0
    
    # Generate random shocks
    Z = rng.standard_normal((n_sims, horizon_steps)) 
    
    # Vectorized: compute all steps in one shot via cumprod
    # rets shape: (n_sims, horizon_steps)
    rets = mu_seq.reshape(1, -1) + sigma_step * Z #(n_sims, horizon_steps)
    
    # Clip extreme returns to prevent negative prices
    rets = np.clip(rets, -0.99, None)
    
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
    steps_per_year: int = 12, #by month
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
    steps_per_year : int
        Number of steps per year (12 for monthly, 4 for quarterly)
    
    Returns
    -------
    paths : np.ndarray
        Price paths (shape: [n_sims, horizon_steps + 1])
    terminals : np.ndarray
        Terminal prices (shape: [n_sims])
    """
    rng = np.random.default_rng(seed)
    
    # Convert annualized sigma to per-step sigma
    sigma_step = sigma_annual / np.sqrt(steps_per_year)
    
    # Initialize paths
    paths = np.zeros((n_sims, horizon_steps + 1)) #horizon_steps = 12/36/60/120
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
        
        # Vectorized: compute all steps in one shot via cumprod
        # rets shape: (batch_size_actual, horizon_steps)
        rets = mu_seq.reshape(1, -1) + sigma_step * Z_batch
        
        # Clip extreme returns to prevent negative prices
        rets = np.clip(rets, -0.99, None)
        
        # paths[start_idx:end_idx, 1:] = S0 * cumprod(1.0 + rets, axis=1)
        paths[start_idx:end_idx, 1:] = S0 * np.cumprod(1.0 + rets, axis=1)
    
    terminals = paths[:, -1]
    
    return paths, terminals


# ------------------------------------------------------------------
# HPC Part A: Path-level parallelism with Numba
# - Data-parallel over Monte Carlo paths using prange
# - Multi-core Monte Carlo kernel for Step 8 HPC benchmark
# ------------------------------------------------------------------
# [HPC-OpenMP-Analogy]
# This Numba kernel uses @njit(parallel=True) + prange to parallelize
# over Monte Carlo paths on a single node. Conceptually this mirrors
# the OpenMP parallel-for loop implemented in hpc_demos/openmp_mc_demo.c.
@njit(parallel=True)
def mc_numba_parallel(
    S0: float,
    mu: float,
    sigma_annual: float,
    n_sims: int,
    horizon_steps: int,
    steps_per_year: int,
) -> np.ndarray:
    """
    Numba-parallel Monte Carlo kernel (data-parallel over paths).

    Uses a simple arithmetic return model:
        r_t = mu + sigma_step * eps
        S_{t+1} = S_t * (1 + r_t)
    """
    sigma_step = sigma_annual / np.sqrt(steps_per_year)
    paths = np.zeros((n_sims, horizon_steps + 1))
    paths[:, 0] = S0

    # Parallel loop over simulation paths (one path per core where possible)
    for i in prange(n_sims):
        for t in range(horizon_steps):
            eps = np.random.normal()
            r_t = mu + sigma_step * eps
            if r_t < -0.99:
                r_t = -0.99
            paths[i, t + 1] = paths[i, t] * (1.0 + r_t)

    return paths


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


# Helper used by both sequential and concurrent scenario execution:
# - Apply macro shock, predict conditional drift, run Monte Carlo,
#   and summarize results for a single scenario.
# - Designed to be side-effect free so it can be used safely in threads.
def _run_single_scenario_task(
    scenario_name: str,
    shock_spec: Dict[str, Any],
    X_last: pd.DataFrame,
    S0: float,
    sigma_annual: float,
    horizon_steps: int,
    steps_per_year: int,
    n_sims: int,
    model: RandomForestRegressor,
    scaler: StandardScaler,
    random_seed: int,
) -> Dict[str, Any]:
    """
    Run the full pipeline for a single scenario:
    - apply shock
    - predict conditional drift
    - run Monte Carlo
    - summarize paths

    Returns a dict with keys:
        - "scenario": scenario_name
        - "summary": summary_dict
        - "paths": np.ndarray
        - "terminals": np.ndarray
        - "mu_seq": np.ndarray
    
    This function is side-effect free (no file I/O, no plots, no tqdm).
    """
    # Apply shock
    X_shocked = apply_shock(X_last, shock_spec)
    
    # Predict conditional drift, same as literature G–K–X addictive prediction error model
    mu_seq = predict_conditional_drift(model, scaler, X_shocked, horizon_steps)
    
    # Run MC (using fast vectorized version to avoid tqdm in threads)
    paths, terminals = run_driver_aware_mc_fast(
        S0, mu_seq, sigma_annual, n_sims, horizon_steps, random_seed,
        steps_per_year=steps_per_year
    )
    
    # Summarize
    summary = summarize_paths(terminals, S0)
    summary["scenario"] = scenario_name
    summary["S0"] = S0
    
    return {
        "scenario": scenario_name,
        "summary": summary,
        "paths": paths,
        "terminals": terminals,
        "mu_seq": mu_seq,
    }

#1 year horizon
def run_scenario_forecast(
    ticker: str = "NVDA",
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    n_sims: int = DEFAULT_N_SIMS,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    output_dir: str = "results/step7",
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
    
    # Determine horizon steps and steps per year
    if freq_info["is_quarterly"]:
        horizon_steps = horizon_months // 3  # Convert months to quarters
    else:
        horizon_steps = horizon_months
    
    if freq_info["is_quarterly"] and horizon_months % 3 != 0:
        print(f"⚠ Quarterly data: horizon_months={horizon_months} not divisible by 3; "
              f"using horizon_steps={horizon_steps} (~{horizon_steps*3} months).")
    
    steps_per_year = freq_info["steps_per_year"]
    
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
        sigma_annual = returns.std() * np.sqrt(steps_per_year)  # Annualized with freq-aware scaling
    else:
        sigma_annual = 0.40  # Default 40% annualized volatility
    
    print(f"Estimated volatility (annualized): {sigma_annual:.2%}")
    
    # Run scenarios
    all_results = {}
    forecast_table_rows = []
    
    print(f"\n{'='*60}")
    print(f"Running {len(scenarios)} scenarios with {n_sims:,} simulations each (concurrent across scenarios)...")
    print(f"Monte Carlo is still parallelized at the path level (NumPy / Numba).")
    print(f"Now, scenarios are also run concurrently using ThreadPoolExecutor.")
    print(f"{'='*60}\n")
    
    # Determine max workers
    max_workers = min(len(scenarios), os.cpu_count() or 4)
    
    # HPC Part B (runtime):
    # Run all macro scenarios concurrently using ThreadPoolExecutor.
    # Each worker runs a full scenario (_run_single_scenario_task),
    # while path-level Monte Carlo remains vectorized/parallel inside.
    # [HPC-MPI-Analogy]
    # Scenario-level concurrency uses ThreadPoolExecutor to run macro scenarios
    # concurrently. Conceptually this is similar to coarse-grained task / rank
    # decomposition in MPI, as illustrated in hpc_demos/mpi_mc_demo.py.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_scenario_task,
                scenario_name,
                shock_spec,
                X_last,
                S0,
                sigma_annual,
                horizon_steps,
                steps_per_year,
                n_sims,
                model,
                scaler,
                random_seed,
            ): scenario_name
            for scenario_name, shock_spec in scenarios.items()
        }
        
        for fut in as_completed(futures):
            scenario_name = futures[fut]
            try:
                result = fut.result()
                all_results[scenario_name] = result
                summary = result["summary"]
                
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
                print(f"✓ Scenario completed: {scenario_name}")
            except Exception as e:
                print(f"❌ Scenario failed: {scenario_name} - {e}")
                raise e
    
    # Save forecast table
    forecast_df = pd.DataFrame(forecast_table_rows)
    forecast_df.to_csv(output_path / "scenario_forecast_table.csv", index=False)
    print(f"\n✓ Saved forecast table: {output_path / 'scenario_forecast_table.csv'}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Fan chart overlay
    plot_fan_chart_overlay(all_results, S0, horizon_steps, output_path / "fan_chart_overlay.png")
    
    # Individual fan charts and distributions (Sequential I/O)
    for scenario_name, result in tqdm(all_results.items(), desc="Plots & I/O", unit="scenario"):
        # Save terminal distribution (moved from loop)
        terminals = result["terminals"]
        terminals_df = pd.DataFrame({"terminal_price": terminals})
        terminals_df.to_csv(output_path / f"scenario_terminals_{scenario_name}.csv", index=False)
        
        plot_fan_chart(
            result["paths"],
            S0,
            horizon_steps,
            scenario_name,
            output_path / f"fan_chart_{scenario_name}.png",
        )
        
        plot_distribution_shift(
            terminals,
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
    """Plot overlay fan chart for all scenarios (in percentage returns)."""
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
        
        # Convert price paths to percentage returns
        return_paths = (paths / S0 - 1) * 100
        
        # Compute percentiles
        percentiles = [5, 25, 50, 75, 95]
        time_steps = np.arange(horizon_steps + 1)
        
        for i, p in enumerate(percentiles):
            pct_values = np.percentile(return_paths, p, axis=0)
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
    
    ax.axhline(0, color="black", linestyle=":", linewidth=2, label="Current Price (0%)")
    ax.set_xlabel("Months Ahead", fontsize=12)
    ax.set_ylabel("Total Return (%)", fontsize=12)
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
    """Plot individual fan chart for a scenario (in percentage returns)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_steps = np.arange(horizon_steps + 1)
    percentiles = [5, 25, 50, 75, 95]
    
    # Convert price paths to percentage returns
    return_paths = (paths / S0 - 1) * 100
    
    # Fill between percentiles
    p5 = np.percentile(return_paths, 5, axis=0)
    p25 = np.percentile(return_paths, 25, axis=0)
    p50 = np.percentile(return_paths, 50, axis=0)
    p75 = np.percentile(return_paths, 75, axis=0)
    p95 = np.percentile(return_paths, 95, axis=0)
    
    ax.fill_between(time_steps, p5, p95, alpha=0.2, color="blue", label="90% CI")
    ax.fill_between(time_steps, p25, p75, alpha=0.3, color="blue", label="50% CI")
    ax.plot(time_steps, p50, color="darkblue", linewidth=2, label="Median")
    ax.axhline(0, color="black", linestyle=":", linewidth=2, label="Current Price (0%)")
    
    ax.set_xlabel("Months Ahead", fontsize=12)
    ax.set_ylabel("Total Return (%)", fontsize=12)
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
    """Plot terminal distribution shift (in percentage returns)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert terminal prices to percentage returns
    terminal_returns = (terminals / S0 - 1) * 100
    
    ax.hist(terminal_returns, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Current Price (0%)")
    median_return = np.median(terminal_returns)
    ax.axvline(median_return, color="green", linestyle="--", linewidth=2, label=f"Median Forecast ({median_return:.1f}%)")
    
    ax.set_xlabel("Terminal Return (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Terminal Distribution: {scenario_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def run_step8_scenario_engine(
    ticker: str = "NVDA",
    horizon_months: int = 12,
    n_sims: int = 2000,
    output_dir: str = "results/step8",
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    random_seed: int = RANDOM_STATE,
    multi_horizon: bool = False,
) -> Dict[str, Any]:
    """
    Step 8: final scenario-based risk engine.

    Reuses the Step 7 scenario Monte Carlo engine, but writes
    all outputs into results/step8.
    
    Parameters
    ----------
    multi_horizon : bool, default False
        If True, run multi-horizon driver-aware MC (1Y, 3Y, 5Y, 10Y).
        If False, run single-horizon scenario forecast.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if multi_horizon:
        # Run multi-horizon driver-aware Monte Carlo
        n_paths = n_sims if n_sims > 1000 else 10000  # Use 10k for multi-horizon
        results = run_driver_aware_mc_multi_horizon(
            ticker=ticker,
            n_paths=n_paths,
            output_dir=str(out_dir),
            random_seed=random_seed,
        )
    else:
        # Run single-horizon scenario forecast
        results = run_scenario_forecast(
            ticker=ticker,
            horizon_months=horizon_months,
            n_sims=n_sims,
            output_dir=str(out_dir),
            model_path=model_path,
            scaler_path=scaler_path,
            random_seed=random_seed,
        )
    return results

def run_step8_mc(
    ticker: str = "NVDA",
    horizon_months: int = 12,
    n_sims: int = 2000,
    output_dir: str = "results/step8",
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    random_seed: int = RANDOM_STATE,
    multi_horizon: bool = False,
) -> Dict[str, Any]:
    """
    Wrapper for Step 8 MC engine (MC only).
    
    Parameters
    ----------
    multi_horizon : bool, default False
        If True, run multi-horizon driver-aware MC (1Y, 3Y, 5Y, 10Y).
        If False, run single-horizon scenario forecast.
    """
    return run_step8_scenario_engine(
        ticker=ticker,
        horizon_months=horizon_months,
        n_sims=n_sims,
        output_dir=output_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        random_seed=random_seed,
        multi_horizon=multi_horizon,
    )


# ------------------------------------------------------------------
# HPC Part A Benchmark
# - Compares baseline NumPy vectorized MC vs Numba-parallel kernel
# - Writes hpc_benchmark.csv with runtime and speedup_vs_baseline
# ------------------------------------------------------------------
def benchmark_mc_backends(
    output_dir: str = "results/step8",
    n_sims: int = 100000,
    horizon_steps: int = 36,
    steps_per_year: int = 12,
    sigma_annual: float = 0.40,
    mu: float = 0.01,
    S0: float = 100.0,
    random_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Benchmark baseline NumPy Monte Carlo vs Numba-parallel Monte Carlo.

    Default configuration (when called standalone):
    - 3Y horizon = 36 monthly steps
    - 100,000 simulation paths

    In the multi-horizon benchmark, horizon_steps and n_sims are overridden
    explicitly so that 1Y / 3Y / 5Y share the same per-horizon workload.

    Writes results to results/step8/hpc_benchmark.csv and returns a DataFrame.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline: reuse the existing vectorized engine
    mu_seq = np.full(horizon_steps, mu, dtype=float)

    print(
        f"\n--- Starting Baseline (NumPy) Benchmark "
        f"({n_sims:,} sims, {horizon_steps} steps, steps_per_year={steps_per_year}) ---"
    )
    t0 = time.time()
    paths_base, terminals_base = run_driver_aware_mc_fast(
        S0=S0,
        mu_seq=mu_seq,
        sigma_annual=sigma_annual,
        n_sims=n_sims,
        horizon_steps=horizon_steps,
        seed=random_seed,
        steps_per_year=steps_per_year,
    )
    t1 = time.time()
    baseline_time = t1 - t0
    print(f"✓ Baseline finished in {baseline_time:.4f} seconds")

    # Numba-parallel: warm-up
    print(f"\n--- Starting Numba Parallel Benchmark ({n_sims:,} sims) ---")
    print("Warm-up run...")
    _ = mc_numba_parallel(
        S0=S0,
        mu=mu,
        sigma_annual=sigma_annual,
        n_sims=100,          # small warmup
        horizon_steps=4,
        steps_per_year=steps_per_year,
    )

    # Numba-parallel: timed run
    print("Timed run...")
    t2 = time.time()
    paths_par = mc_numba_parallel(
        S0=S0,
        mu=mu,
        sigma_annual=sigma_annual,
        n_sims=n_sims,
        horizon_steps=horizon_steps,
        steps_per_year=steps_per_year,
    )
    t3 = time.time()
    parallel_time = t3 - t2
    print(f"✓ Numba Parallel finished in {parallel_time:.4f} seconds")

    df = pd.DataFrame({
        "backend": ["baseline_numpy", "numba_parallel"],
        "time_sec": [baseline_time, parallel_time],
        "n_sims": [n_sims, n_sims],
    })
    df["speedup_vs_baseline"] = df["time_sec"].iloc[0] / df["time_sec"]

    csv_path = out_dir / "hpc_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nHPC benchmark written to: {csv_path}")
    print(df)

    return df


# ------------------------------------------------------------------
# HPC Scaling Curve Benchmark: NumPy vs Numba across simulation counts
# ------------------------------------------------------------------
def benchmark_scaling_curve(
    output_dir: str = "results/step8",
    n_sims_grid: Optional[List[int]] = None,
    horizon_steps: int = 36,
    steps_per_year: int = 12,
    sigma_annual: float = 0.40,
    mu: float = 0.01,
    S0: float = 100.0,
    random_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    HPC scaling benchmark: how NumPy vs Numba speedup changes
    as we increase the number of Monte Carlo simulations.
    
    This uses a fixed horizon (default: 3Y = 36 monthly steps) and
    sweeps over n_sims_grid (e.g. 10k → 50k → 100k → 200k → 500k).
    
    Results are written to results/step8/hpc_scaling_curve.csv and
    plotted as results/step8/hpc_scaling_curve.png.
    
    Parameters
    ----------
    output_dir : str
        Output directory for results
    n_sims_grid : list[int] | None
        List of simulation counts to test. If None, defaults to
        [10_000, 50_000, 100_000, 200_000, 500_000]
    horizon_steps : int
        Number of time steps (default: 36 for 3Y)
    steps_per_year : int
        Steps per year (default: 12 for monthly)
    sigma_annual : float
        Annual volatility
    mu : float
        Drift parameter
    S0 : float
        Initial stock price
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: n_sims, horizon_steps, baseline_time,
        numba_time, speedup
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if n_sims_grid is None:
        n_sims_grid = [10_000, 50_000, 100_000, 200_000, 500_000]
    
    # Precompute constant drift sequence for the baseline
    mu_seq = np.full(horizon_steps, mu, dtype=float)
    
    records = []
    
    # Warm-up Numba once (small run) to avoid including compilation time
    _ = mc_numba_parallel(
        S0=S0,
        mu=mu,
        sigma_annual=sigma_annual,
        n_sims=1_000,
        horizon_steps=4,
        steps_per_year=steps_per_year,
    )
    
    print("\n=== HPC Scaling Benchmark: NumPy vs Numba ===")
    print(f"Horizon steps: {horizon_steps}, sigma_annual={sigma_annual:.0%}")
    print(f"n_sims_grid = {n_sims_grid}\n")
    
    for n_sims in n_sims_grid:
        print(f"n_sims = {n_sims:,}")
        
        # Baseline NumPy
        t0 = time.perf_counter()
        _, _ = run_driver_aware_mc_fast(
            S0=S0,
            mu_seq=mu_seq,
            sigma_annual=sigma_annual,
            n_sims=n_sims,
            horizon_steps=horizon_steps,
            seed=random_seed,
            steps_per_year=steps_per_year,
        )
        t1 = time.perf_counter()
        baseline_time = t1 - t0
        print(f"  NumPy baseline:  {baseline_time:.4f} s")
        
        # Numba parallel
        t2 = time.perf_counter()
        _ = mc_numba_parallel(
            S0=S0,
            mu=mu,
            sigma_annual=sigma_annual,
            n_sims=n_sims,
            horizon_steps=horizon_steps,
            steps_per_year=steps_per_year,
        )
        t3 = time.perf_counter()
        numba_time = t3 - t2
        print(f"  Numba parallel:  {numba_time:.4f} s")
        
        speedup = baseline_time / numba_time if numba_time > 0 else np.nan
        print(f"  Speedup (NumPy / Numba): {speedup:.2f}x\n")
        
        records.append(
            {
                "n_sims": n_sims,
                "horizon_steps": horizon_steps,
                "baseline_time": baseline_time,
                "numba_time": numba_time,
                "speedup": speedup,
            }
        )
    
    df = pd.DataFrame(records)
    
    csv_path = out_dir / "hpc_scaling_curve.csv"
    df.to_csv(csv_path, index=False)
    print(f"Scaling benchmark written to: {csv_path}")
    
    # Plot scaling curve (log-log style)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_sims"], df["baseline_time"], marker="o", label="NumPy baseline", linewidth=2)
    ax.plot(df["n_sims"], df["numba_time"], marker="o", label="Numba parallel", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulations (log scale)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=12)
    ax.set_title("HPC Scaling: NumPy vs Numba (fixed horizon)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    
    png_path = out_dir / "hpc_scaling_curve.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Scaling curve plot saved to: {png_path}")
    
    return df


# ------------------------------------------------------------------
# HPC Multi-Horizon Benchmark: NumPy vs Numba for 1Y/3Y/5Y
# ------------------------------------------------------------------
def benchmark_mc_paths_multi_horizon(
    output_dir: str = "results/step8",
    horizons: Optional[Dict[str, int]] = None,
    n_sims: int = 100000,
    sigma_annual: float = 0.40,
    mu: float = 0.01,
    S0: float = 100.0,
    random_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Run benchmark_mc_backends() for multiple horizons (1Y/3Y/5Y)
    and return a concatenated DataFrame with an extra 'horizon'
    and 'n_steps' column.
    
    Parameters
    ----------
    output_dir : str
        Output directory for results
    horizons : dict[str, int] | None
        Dictionary mapping horizon labels to number of steps.
        If None, defaults to {"1Y": 12, "3Y": 36, "5Y": 60}
    n_sims : int
        Number of Monte Carlo simulations per horizon
    sigma_annual : float
        Annual volatility
    mu : float
        Drift parameter
    S0 : float
        Initial stock price
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: backend, time_sec, speedup_vs_baseline,
        horizon, n_steps
    """
    if horizons is None:
        horizons = {"1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    print(f"\n{'='*60}")
    print(f"Starting Multi-Horizon HPC Benchmark")
    print(f"Horizons: {list(horizons.keys())}")
    print(f"Simulations per horizon: {n_sims:,}")
    print(f"{'='*60}\n")
    
    for horizon_label, n_steps in horizons.items():
        print(f"\n--- Benchmarking {horizon_label} ({n_steps} steps) ---")
        
        # Call existing benchmark_mc_backends for this horizon
        df = benchmark_mc_backends(
            output_dir=output_dir,
            n_sims=n_sims,
            horizon_steps=n_steps,
            steps_per_year=12,
            sigma_annual=sigma_annual,
            mu=mu,
            S0=S0,
            random_seed=random_seed,
        )
        
        # Add horizon and n_steps columns
        df["horizon"] = horizon_label
        df["n_steps"] = n_steps
        
        all_dfs.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined results
    csv_path = out_dir / "hpc_benchmark_paths_1y_3y_5y_10y.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Multi-horizon benchmark written to: {csv_path}")
    print(f"{'='*60}\n")
    print(combined_df)
    
    return combined_df


# ------------------------------------------------------------------
# HPC Part B Benchmark: Scenario-level concurrency
# - Compares sequential vs ThreadPoolExecutor-based execution
#   across all macro scenarios
# - Writes hpc_benchmark_scenarios.csv with runtime and speedup
# ------------------------------------------------------------------
def benchmark_scenario_concurrency(
    ticker: str = "NVDA",
    horizon_months: int = 12,
    n_sims: int = 2000,
    output_dir: str = "results/step8",
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    random_seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    HPC Part B: Scenario-level concurrency benchmark.

    Compares:
    - Sequential execution of all macro scenarios
    - Concurrent execution using ThreadPoolExecutor

    Writes results to results/step8/hpc_benchmark_scenarios.csv
    and returns the benchmark DataFrame.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_last, S0, dates, freq_info = load_latest_features(ticker)

    # Determine horizon steps
    if freq_info["is_quarterly"]:
        horizon_steps = horizon_months // 3
    else:
        horizon_steps = horizon_months
    
    steps_per_year = freq_info["steps_per_year"]

    # Load model and scaler
    model_path = model_path or DEFAULT_MODEL_PATH
    scaler_path = scaler_path or DEFAULT_SCALER_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load history for scenarios
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

    # Estimate volatility
    if "adj_close" in history_df.columns:
        returns = history_df["adj_close"].pct_change().dropna()
        sigma_annual = returns.std() * np.sqrt(steps_per_year)
    else:
        sigma_annual = 0.40

    print(f"\n{'='*60}")
    print(f"Starting Scenario Concurrency Benchmark")
    print(f"Scenarios: {len(scenarios)}, Sims: {n_sims}, Horizon: {horizon_months}m")
    print(f"{'='*60}")

    # --- Sequential Run ---
    print("\n1. Running Sequential...")
    t0_seq = time.time()
    
    # Sequential baseline: run all scenarios one-by-one on a single thread.
    for scenario_name, shock_spec in scenarios.items():
        _ = _run_single_scenario_task(
            scenario_name,
            shock_spec,
            X_last,
            S0,
            sigma_annual,
            horizon_steps,
            steps_per_year,
            n_sims,
            model,
            scaler,
            random_seed,
        )
        
    t1_seq = time.time()
    seq_time = t1_seq - t0_seq
    print(f"✓ Sequential finished in {seq_time:.4f} seconds")

    # --- Concurrent Run ---
    print("\n2. Running Concurrent (ThreadPoolExecutor)...")
    max_workers = min(len(scenarios), os.cpu_count() or 4)
    t0_conc = time.time()
    
    # Concurrent run: dispatch all scenarios to the ThreadPoolExecutor.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_scenario_task,
                scenario_name,
                shock_spec,
                X_last,
                S0,
                sigma_annual,
                horizon_steps,
                steps_per_year,
                n_sims,
                model,
                scaler,
                random_seed,
            ): scenario_name
            for scenario_name, shock_spec in scenarios.items()
        }
        
        for fut in as_completed(futures):
            _ = fut.result()
            
    t1_conc = time.time()
    conc_time = t1_conc - t0_conc
    print(f"✓ Concurrent finished in {conc_time:.4f} seconds")

    # --- Results ---
    df = pd.DataFrame([
        {
            "mode": "scenarios_sequential",
            "n_scenarios": len(scenarios),
            "n_sims": n_sims,
            "horizon_steps": horizon_steps,
            "time_sec": seq_time,
        },
        {
            "mode": "scenarios_concurrent",
            "n_scenarios": len(scenarios),
            "n_sims": n_sims,
            "horizon_steps": horizon_steps,
            "time_sec": conc_time,
        },
    ])
    df["speedup_vs_sequential"] = df.loc[df["mode"] == "scenarios_sequential", "time_sec"].iloc[0] / df["time_sec"]

    csv_path = out_dir / "hpc_benchmark_scenarios.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[HPC Part B] Scenario concurrency benchmark written to: {csv_path}")
    print(df)

    return df


# Convenience wrapper for running only the HPC Part A benchmark
# (NumPy vs Numba) without running the full scenario engine.
def run_step8_hpc(
    output_dir: str = "results/step8",
    n_sims: int = 5000,
) -> pd.DataFrame:
    """
    Wrapper for Step 8 HPC benchmark (HPC only).
    """
    return benchmark_mc_backends(
        output_dir=output_dir,
        n_sims=n_sims,
    )


# ------------------------------------------------------------------
# HPC Summary: Combine all benchmark results (1Y/3Y/5Y)
# ------------------------------------------------------------------
def build_hpc_summary_1y_3y_5y_10y(
    output_dir: str = "results/step8",
    paths_csv: str = "hpc_benchmark_paths_1y_3y_5y_10y.csv",
    mpi_csv: str = "hpc_benchmark_mpi.csv",
    openmp_csv: str = "hpc_benchmark_openmp.csv",
) -> pd.DataFrame:
    """
    Build a 1Y / 3Y / 5Y / 10Y summary table combining:
    - NumPy vs Numba (from paths benchmark)
    - MPI (from mpi CSV)
    - OpenMP (from C CSV)
    
    Returns a compact table with one row per horizon:
    ['horizon', 'n_steps', 'baseline_time', 'best_parallel_time', 'speedup']
    
    Parameters
    ----------
    output_dir : str
        Directory containing all benchmark CSV files
    paths_csv : str
        Filename for NumPy/Numba benchmark results
    mpi_csv : str
        Filename for MPI benchmark results
    openmp_csv : str
        Filename for OpenMP benchmark results
    
    Returns
    -------
    pd.DataFrame
        Summary table with one row per horizon
    """
    out_dir = Path(output_dir)
    
    # Map n_steps to horizon labels
    n_steps_to_horizon = {12: "1Y", 36: "3Y", 60: "5Y", 120: "10Y"}
    horizons = ["1Y", "3Y", "5Y", "10Y"]
    n_steps_list = [12, 36, 60, 120]
    
    summary_rows = []
    
    # Load paths CSV (NumPy vs Numba)
    paths_df = None
    paths_path = out_dir / paths_csv
    if paths_path.exists():
        paths_df = pd.read_csv(paths_path)
        print(f"✓ Loaded paths benchmark: {paths_path}")
    else:
        print(f"⚠ Paths benchmark not found: {paths_path}")
    
    # Load MPI CSV
    mpi_df = None
    mpi_path = out_dir / mpi_csv
    if mpi_path.exists():
        mpi_df = pd.read_csv(mpi_path)
        print(f"✓ Loaded MPI benchmark: {mpi_path}")
    else:
        print(f"⚠ MPI benchmark not found: {mpi_path}")
    
    # Load OpenMP CSV
    openmp_df = None
    openmp_path = out_dir / openmp_csv
    if openmp_path.exists():
        openmp_df = pd.read_csv(openmp_path)
        print(f"✓ Loaded OpenMP benchmark: {openmp_path}")
    else:
        print(f"⚠ OpenMP benchmark not found: {openmp_path}")
    
    # Process each horizon
    for horizon, n_steps in zip(horizons, n_steps_list):
        baseline_time = None
        parallel_times = {}
        
        # Get baseline time from paths benchmark
        if paths_df is not None:
            baseline_row = paths_df[
                (paths_df["horizon"] == horizon) & 
                (paths_df["backend"] == "baseline_numpy")
            ]
            if not baseline_row.empty:
                baseline_time = baseline_row["time_sec"].iloc[0]
        
        # Get Numba parallel time
        if paths_df is not None:
            numba_row = paths_df[
                (paths_df["horizon"] == horizon) & 
                (paths_df["backend"] == "numba_parallel")
            ]
            if not numba_row.empty:
                parallel_times["numba"] = numba_row["time_sec"].iloc[0]
        
        # Get MPI time
        if mpi_df is not None:
            mpi_row = mpi_df[
                (mpi_df["n_steps"] == n_steps) &
                (mpi_df["backend"] == "mpi_python") &
                (mpi_df["mode"] == "mpi_parallel")
            ]
            if not mpi_row.empty:
                # Take the most recent run if multiple exist
                parallel_times["mpi"] = mpi_row["time_sec"].iloc[-1]
        
        # Get OpenMP parallel time
        if openmp_df is not None:
            openmp_row = openmp_df[
                (openmp_df["n_steps"] == n_steps) &
                (openmp_df["backend"] == "openmp_c") &
                (openmp_df["mode"] == "openmp_parallel")
            ]
            if not openmp_row.empty:
                # Take the most recent run if multiple exist
                parallel_times["openmp"] = openmp_row["time_sec"].iloc[-1]
        
        # Find best parallel time
        best_parallel_time = None
        best_backend = None
        if parallel_times:
            best_backend = min(parallel_times, key=parallel_times.get)
            best_parallel_time = parallel_times[best_backend]
        
        # Calculate speedup
        speedup = None
        if baseline_time is not None and best_parallel_time is not None:
            speedup = baseline_time / best_parallel_time
        
        summary_rows.append({
            "horizon": horizon,
            "n_steps": n_steps,
            "baseline_time": baseline_time,
            "best_parallel_time": best_parallel_time,
            "best_backend": best_backend,
            "speedup": speedup,
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary
    summary_path = out_dir / "hpc_benchmark_summary_1y_3y_5y_10y.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"HPC Summary written to: {summary_path}")
    print(f"{'='*60}\n")
    print(summary_df.to_string(index=False))
    
    return summary_df


# ============================================================================
# Step 7: Driver-Aware Monte Carlo Engine for Multiple Horizons
# ============================================================================

# Horizon step mapping (monthly steps)
HORIZON_STEPS = {
    "1Y": 12,   # 12 monthly steps
    "3Y": 36,   # 36 monthly steps
    "5Y": 60,   # 60 monthly steps
    "10Y": 120  # 120 monthly steps
}

# Horizon-specific champion model and scaler paths
HORIZON_MODEL_PATHS = {
    "1Y": {
        "model": "models/champion_model_1y.pkl",
        "scaler": "models/feature_scaler_1y.pkl",
    },
    "3Y": {
        "model": "models/champion_model_3y.pkl",
        "scaler": "models/feature_scaler_3y.pkl",
    },
    "5Y": {
        "model": "models/champion_model_5y.pkl",
        "scaler": "models/feature_scaler_5y.pkl",
    },
    "10Y": {
        "model": "models/champion_model_10y.pkl",
        "scaler": "models/feature_scaler_10y.pkl",
    },
}

# Scenario scale factors
MACRO_SCALE_STRESS = 1.5
FIRM_SCALE_STRESS = 1.5
INTERACTION_SCALE_BULL = 1.5
MACRO_SCALE_BULL_CUT = -1.0  # negative for rate cuts

def classify_feature(feature_name: str) -> str:
    """
    Classify a feature into Firm, Macro, or Interaction category.
    """
    if feature_name.startswith("ix_"):
        return "Interaction"
    
    macro_features = [
        "tnx_yield", "tnx_change_3m",
        "vix_level", "vix_change_3m",
        "inflation", "fedfunds",
        "unemployment", "recession"
    ]
    
    if feature_name in macro_features:
        return "Macro"
    
    macro_keywords = ["tnx", "vix", "inflation", "fedfunds", "unemployment", "recession"]
    if any(keyword in feature_name.lower() for keyword in macro_keywords):
        return "Macro"
    
    return "Firm"


def load_champion_models() -> Dict[str, Any]:
    """
    Load champion models for each horizon.
    
    Returns:
        Dictionary mapping horizon to (model, scaler, model_type)
    """
    horizons = {
        "1Y": ("RandomForest", "models/champion_model_1y.pkl", "models/feature_scaler_1y.pkl"),
        "3Y": ("RandomForest", "models/champion_model_3y.pkl", "models/feature_scaler_3y.pkl"),
        "5Y": ("XGBoost", "models/champion_model_5y.pkl", "models/feature_scaler_5y.pkl"),
        "10Y": ("ElasticNet", "models/champion_model_10y.pkl", "models/feature_scaler_10y.pkl"),
    }
    
    models_dict = {}
    
    # Try to load from champion_model_comparison outputs
    model_comparison_path = Path("outputs/feature_importance/results/model_comparison.csv")
    if model_comparison_path.exists():
        df = pd.read_csv(model_comparison_path)
        
        for horizon in ["1Y", "3Y", "5Y", "10Y"]:
            horizon_df = df[df['horizon'] == horizon]
            if len(horizon_df) == 0:
                continue
            
            # Find champion by lowest MAE
            best = horizon_df.loc[horizon_df['mae'].idxmin()]
            model_name = best['model']
            
            # For now, use default paths (models will be trained on-the-fly if not found)
            models_dict[horizon] = {
                "model_type": model_name,
                "model": None,  # Will be loaded/trained on demand
                "scaler": None,
            }
    
    return models_dict


def load_feature_importance(horizon: str) -> Dict[str, float]:
    """
    Load feature importance for a given horizon from Step 6 outputs.
    
    Returns:
        Dictionary mapping feature name to importance (%)
    """
    # Try to load from three_category_feature_importance outputs
    importance_paths = [
        f"outputs/feature_importance/data/long_term/summary/rf_category_importance_3cat.csv",
        f"outputs/feature_importance/data/long_term/summary/xgb_category_importance_3cat.csv",
    ]
    
    importance_dict = {}
    
    # Try to load from rankings
    ranking_paths = [
        f"outputs/feature_importance/rankings/rf_combined_top15.csv",
        f"outputs/feature_importance/rankings/xgb_combined_top15.csv",
    ]
    
    for path_str in ranking_paths:
        path = Path(path_str)
        if path.exists():
            df = pd.read_csv(path)
            # Find column matching horizon
            for col in df.columns:
                if horizon in col and "Importance" in col:
                    feature_col = col.replace(" Importance", " Feature")
                    if feature_col in df.columns:
                        for _, row in df.iterrows():
                            feat = row[feature_col]
                            imp = row[col]
                            if pd.notna(feat) and pd.notna(imp):
                                importance_dict[str(feat)] = float(imp)
                    break
    
    # If no importance found, return empty dict (will use equal weights)
    if len(importance_dict) == 0:
        print(f"⚠ No feature importance found for {horizon}, using equal weights")
    
    return importance_dict


def build_shock_table(
    importance_dict: Dict[str, float],
    history_df: pd.DataFrame,
    horizon: str
) -> Dict[str, Dict[str, float]]:
    """
    Build driver-aware shock table with category weights and volatilities.
    
    Returns:
        Dictionary with weights and sigmas for Macro, Firm, Interaction
    """
    # Aggregate importance by category
    category_importance = {"Firm": 0.0, "Macro": 0.0, "Interaction": 0.0}
    
    for feat, imp in importance_dict.items():
        cat = classify_feature(feat)
        category_importance[cat] += imp
    
    # Normalize to sum to 1
    total = sum(category_importance.values())
    if total > 0:
        for cat in category_importance:
            category_importance[cat] /= total
    else:
        # Equal weights if no importance data
        for cat in category_importance:
            category_importance[cat] = 1.0 / 3.0
    
    # Estimate historical volatilities by category
    # Use returns as proxy for volatility (always monthly)
    if "adj_close" in history_df.columns:
        returns = history_df["adj_close"].pct_change().dropna()
        base_sigma = returns.std() * np.sqrt(12)  # Annualized (monthly data)
    else:
        base_sigma = 0.40  # Default 40% annualized
    
    # Category-specific volatilities (calibrated)
    sigmas = {
        "Macro": base_sigma * 1.2,  # Macro more volatile
        "Firm": base_sigma * 0.8,   # Firm less volatile
        "Interaction": base_sigma * 1.0,  # Interaction medium
    }
    
    return {
        "weights": category_importance,
        "sigmas": sigmas,
    }


def build_shock_components(
    horizon_label: str,
    shock_table: Dict[str, Dict[str, float]],
    history_df: pd.DataFrame,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Build shock components (sigmas and weights) for a given horizon.
    
    Returns:
        sigma_macro, sigma_firm, sigma_interaction, weights
    """
    weights = shock_table["weights"]
    sigmas = shock_table["sigmas"]
    
    # Convert annualized sigmas to per-step (monthly)
    steps_per_year = 12  # Always use monthly steps
    sigma_macro_step = sigmas["Macro"] / np.sqrt(steps_per_year)
    sigma_firm_step = sigmas["Firm"] / np.sqrt(steps_per_year)
    sigma_interaction_step = sigmas["Interaction"] / np.sqrt(steps_per_year)
    
    return sigma_macro_step, sigma_firm_step, sigma_interaction_step, weights


def simulate_paths(
    S0: float,
    mu_horizon: float,
    shock_table: Dict[str, Dict[str, float]],
    horizon_steps: int,
    scenario_label: str,
    history_df: pd.DataFrame,
    n_paths: int = 10000,
    random_seed: int = 42,
    steps_per_year: int = 12,
    sigma_residual_step: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate price paths using driver-aware shocks for a specific scenario.
    
    Formula: return_next = μ_horizon + S + ε
    where S = w_macro * eps_macro + w_firm * eps_firm + w_interaction * eps_interaction
    
    Scenarios modify the base shocks:
    - BASE: normal shocks
    - MACRO_STRESS: eps_macro + 1.5 * sigma_macro
    - FUNDAMENTAL_STRESS: eps_firm + 1.5 * sigma_firm
    - AI_BULL: eps_macro - 1.0 * sigma_macro, eps_interaction + 1.5 * sigma_interaction
    """
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    
    # Get shock components
    sigma_macro_step, sigma_firm_step, sigma_interaction_step, weights = build_shock_components(
        "", shock_table, history_df
    )
    
    # Initialize paths
    paths = np.zeros((n_paths, horizon_steps + 1))
    paths[:, 0] = S0
    
    # Convert annualized mu to per-step (monthly)
    mu_step = mu_horizon / horizon_steps if horizon_steps > 0 else mu_horizon
    
    # Generate shocks for each step
    for t in range(horizon_steps):
        # Base random shocks
        eps_macro_base = rng.normal(0, sigma_macro_step)
        eps_firm_base = rng.normal(0, sigma_firm_step)
        eps_interaction_base = rng.normal(0, sigma_interaction_step)
        
        # Apply scenario-specific modifications
        if scenario_label == "base":
            eps_macro = eps_macro_base
            eps_firm = eps_firm_base
            eps_interaction = eps_interaction_base
        elif scenario_label == "macro_stress":
            eps_macro = eps_macro_base + MACRO_SCALE_STRESS * sigma_macro_step
            eps_firm = eps_firm_base
            eps_interaction = eps_interaction_base
        elif scenario_label == "fundamental_stress":
            eps_macro = eps_macro_base
            eps_firm = eps_firm_base + FIRM_SCALE_STRESS * sigma_firm_step
            eps_interaction = eps_interaction_base
        elif scenario_label == "ai_bull":
            eps_macro = eps_macro_base + MACRO_SCALE_BULL_CUT * sigma_macro_step
            eps_firm = eps_firm_base
            eps_interaction = eps_interaction_base + INTERACTION_SCALE_BULL * sigma_interaction_step
        else:
            # Default to base
            eps_macro = eps_macro_base
            eps_firm = eps_firm_base
            eps_interaction = eps_interaction_base
        
        # Aggregate shock
        S = (weights["Macro"] * eps_macro +
             weights["Firm"] * eps_firm +
             weights["Interaction"] * eps_interaction)
        
        # === GKX-style additive residual noise ===
        epsilon = rng.normal(0, sigma_residual_step)
        
        # Return for this step
        return_step = mu_step + S + epsilon
        
        # Clip extreme returns to prevent negative prices
        return_step = np.clip(return_step, -0.99, None)
        
        # Update prices
        paths[:, t + 1] = paths[:, t] * (1.0 + return_step)
    
    terminals = paths[:, -1]
    
    return paths, terminals


def run_scenarios(
    S0: float,
    mu_horizon: float,
    shock_table: Dict[str, Dict[str, float]],
    horizon_steps: int,
    history_df: pd.DataFrame,
    n_paths: int = 10000,
    random_seed: int = 42,
    steps_per_year: int = 12,
    sigma_residual_step: float = 0.02,
) -> Dict[str, Dict[str, Any]]:
    """
    Run 4 scenario families: base, macro_stress, fundamental_stress, ai_bull.
    """
    scenarios = {}
    scenario_labels = ["base", "macro_stress", "fundamental_stress", "ai_bull"]
    
    for scenario_label in scenario_labels:
        # Use different random seeds for each scenario to ensure independence
        scenario_seed = random_seed + hash(scenario_label) % 1000
        
        paths, terminals = simulate_paths(
            S0, mu_horizon, shock_table, horizon_steps, scenario_label,
            history_df, n_paths, scenario_seed, steps_per_year,
            sigma_residual_step=sigma_residual_step
        )
        
        scenarios[scenario_label] = {
            "paths": paths,
            "terminals": terminals,
            "shock_table": shock_table,
        }
    
    return scenarios


def plot_fan_chart_multi_horizon(
    horizon_label: str,
    scenario_label: str,
    price_paths: np.ndarray,
    current_price: float,
    output_path: Path,
) -> None:
    """Plot fan chart for a single scenario (in percentage returns) - multi-horizon version."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_steps = price_paths.shape[1] - 1
    time_steps = np.arange(n_steps + 1)
    
    # Convert price paths to percentage returns
    # total_return_t = (price_t / S0 - 1) * 100
    return_paths = (price_paths / current_price - 1) * 100
    
    # Compute percentiles from return paths
    p5 = np.percentile(return_paths, 5, axis=0)
    p25 = np.percentile(return_paths, 25, axis=0)
    p50 = np.percentile(return_paths, 50, axis=0)
    p75 = np.percentile(return_paths, 75, axis=0)
    p95 = np.percentile(return_paths, 95, axis=0)
    
    # Fill between percentiles
    ax.fill_between(time_steps, p5, p95, alpha=0.2, color="blue", label="90% CI")
    ax.fill_between(time_steps, p25, p75, alpha=0.3, color="blue", label="50% CI")
    ax.plot(time_steps, p50, color="darkblue", linewidth=2, label="Median")
    ax.axhline(0, color="black", linestyle=":", linewidth=2, label="Current Price (0%)")
    
    ax.set_xlabel("Months Ahead", fontsize=12)
    ax.set_ylabel("Total Return (%)", fontsize=12)
    ax.set_title(
        f"Fan Chart: {horizon_label} - {scenario_label.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_fan_chart_combined_baseline(
    all_results: Dict[str, Dict],
    S0: float,
    output_path: Path,
) -> None:
    """
    Plot combined 2x2 fan chart for baseline scenario across all horizons (1Y, 3Y, 5Y, 10Y).
    This creates a single figure with 4 subplots showing baseline fan charts for each horizon.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    horizons = ["1Y", "3Y", "5Y", "10Y"]
    
    for idx, horizon_name in enumerate(horizons):
        if horizon_name not in all_results:
            continue
            
        ax = axes[idx]
        horizon_data = all_results[horizon_name]
        
        # Get baseline scenario paths
        if "scenarios" not in horizon_data or "base" not in horizon_data["scenarios"]:
            continue
            
        paths = horizon_data["scenarios"]["base"]["paths"]
        n_steps = paths.shape[1] - 1
        time_steps = np.arange(n_steps + 1)
        
        # Convert price paths to percentage returns
        return_paths = (paths / S0 - 1) * 100
        
        # Compute percentiles
        p5 = np.percentile(return_paths, 5, axis=0)
        p25 = np.percentile(return_paths, 25, axis=0)
        p50 = np.percentile(return_paths, 50, axis=0)
        p75 = np.percentile(return_paths, 75, axis=0)
        p95 = np.percentile(return_paths, 95, axis=0)
        
        # Fill between percentiles
        ax.fill_between(time_steps, p5, p95, alpha=0.2, color="blue", label="90% CI")
        ax.fill_between(time_steps, p25, p75, alpha=0.3, color="blue", label="50% CI")
        ax.plot(time_steps, p50, color="darkblue", linewidth=2, label="Median")
        ax.axhline(0, color="black", linestyle=":", linewidth=2, label="Current Price (0%)")
        
        ax.set_xlabel("Months Ahead", fontsize=11)
        ax.set_ylabel("Total Return (%)", fontsize=11)
        ax.set_title(f"Fan Chart: {horizon_name} - Base", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle("7. Monte Carlo Forecasting - Baseline", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_price_paths_multi_horizon(
    horizon_label: str,
    scenario_label: str,
    price_paths: np.ndarray,
    current_price: float,
    output_path: Path,
    n_sample_paths: int = 20,
) -> None:
    """Plot sample price paths for a single scenario - multi-horizon version."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_steps = price_paths.shape[1] - 1
    time_axis = np.arange(n_steps + 1)
    
    # Plot a subset of paths
    n_plot = min(n_sample_paths, price_paths.shape[0])
    for i in range(n_plot):
        ax.plot(time_axis, price_paths[i, :], alpha=0.3, linewidth=0.8, color="steelblue")
    
    # Add S0 reference line
    ax.axhline(current_price, linestyle="--", linewidth=2, color="red", label=f"S0 = ${current_price:.2f}")
    
    ax.set_xlabel("Step (month)", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title(
        f"Sample Price Paths: {horizon_label} - {scenario_label.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_terminal_distribution(
    horizon_label: str,
    scenario_label: str,
    terminal_prices: np.ndarray,
    current_price: float,
    output_path: Path,
) -> None:
    """Plot terminal distribution for a single scenario (in percentage returns)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert terminal prices to percentage returns
    # terminal_return = (terminal_price / S0 - 1) * 100
    terminal_returns = (terminal_prices / current_price - 1) * 100
    
    # Filter out any NaN or inf values
    terminal_returns = terminal_returns[np.isfinite(terminal_returns)]
    
    if len(terminal_returns) == 0:
        print(f"Warning: No valid terminal returns for {horizon_label} - {scenario_label}")
        return
    
    # Check data range and unique values
    data_range = np.max(terminal_returns) - np.min(terminal_returns)
    unique_values = np.unique(terminal_returns)
    
    # If all values are the same or range is extremely small, use a simple visualization
    if data_range < 1e-10 or len(unique_values) == 1:
        # All values are essentially the same - show as a single bar
        ax.bar([np.mean(terminal_returns)], [len(terminal_returns)], 
               width=max(0.1, abs(np.mean(terminal_returns)) * 0.1) if np.mean(terminal_returns) != 0 else 0.1,
               alpha=0.7, color="steelblue", edgecolor="black")
    else:
        # Calculate appropriate number of bins based on data
        # Use Scott's rule as a starting point, but cap it
        n_bins_scott = int(np.ceil(3.49 * np.std(terminal_returns) / (len(terminal_returns) ** (1/3))))
        n_bins = min(50, max(5, n_bins_scott))
        
        # Ensure we don't have more bins than unique values
        n_bins = min(n_bins, len(unique_values))
        
        try:
            ax.hist(terminal_returns, bins=n_bins, alpha=0.7, color="steelblue", edgecolor="black")
        except ValueError:
            # Last resort: use unique values as bins
            ax.hist(terminal_returns, bins=len(unique_values), alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Current Price (0%)")
    median_return = np.median(terminal_returns)
    ax.axvline(median_return, color="red", linestyle="--", linewidth=2, label=f"Median Forecast ({median_return:.1f}%)")
    
    ax.set_xlabel("Terminal Return (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Terminal Distribution: {horizon_label} - {scenario_label.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def debug_print_horizon_drifts(
    ticker: str = "NVDA",
    random_seed: int = 42,
) -> None:
    """
    Debug utility: for the latest feature row X_last, load each horizon model
    (1Y, 3Y, 5Y, 10Y) and print:
      - μ_horizon^(h): model-predicted total simple return over the horizon
      - μ_step^(h): per-month drift used in the Monte Carlo (μ_horizon / horizon_steps)
    This makes the difference across horizons fully explicit in numeric terms.
    """
    # Load latest features
    X_last, S0, dates, freq_info = load_latest_features(ticker)
    
    print(f"Debug horizon drifts for {ticker} (S0 = {S0:.2f}, date = {X_last.index[0].date()})")
    print("=" * 80)
    
    # Iterate over all horizons
    for horizon in ["1Y", "3Y", "5Y", "10Y"]:
        m_path = Path(HORIZON_MODEL_PATHS[horizon]["model"])
        s_path = Path(HORIZON_MODEL_PATHS[horizon]["scaler"])
        
        if not m_path.exists() or not s_path.exists():
            print(f"[{horizon}] model/scaler not found at {m_path} / {s_path} (fallback not shown here).")
            continue
        
        # Load model and scaler
        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        
        # Predict horizon return
        mu_horizon = predict_horizon_return(model, scaler, X_last)
        
        # Get horizon steps
        steps = HORIZON_STEPS[horizon] #1y=12, 3y=36, 5y=60, 10y=120
        
        # Compute per-step drift
        mu_step = mu_horizon / steps
        
        # Print formatted output
        print(f"[{horizon}] μ_horizon = {mu_horizon:.4f} "
              f"({mu_horizon*100:.2f}% total), "
              f"μ_step = {mu_step:.6f} ({mu_step*100:.3f}% per step, steps = {steps})")


def run_driver_aware_mc_multi_horizon(
    ticker: str = "NVDA",
    n_paths: int = 10000,
    output_dir: str = "results/step7",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Main function: Run driver-aware Monte Carlo for multiple horizons (1Y, 3Y, 5Y, 10Y).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_last, S0, dates, freq_info = load_latest_features(ticker)
    
    # Load full history for volatility estimation
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
    
    steps_per_year = 12  # Always use monthly steps
    
    # Pre-load horizon-specific models (if available)
    horizon_models: Dict[str, Tuple[Any, StandardScaler]] = {}
    for h_name, paths in HORIZON_MODEL_PATHS.items():
        m_path = Path(paths["model"])
        s_path = Path(paths["scaler"])
        if m_path.exists() and s_path.exists():
            print(f"Loading {h_name} model from {m_path} and scaler from {s_path} ...")
            model_h = joblib.load(m_path)
            scaler_h = joblib.load(s_path)
            horizon_models[h_name] = (model_h, scaler_h)
        else:
            print(f"⚠ Horizon {h_name}: model or scaler not found at {m_path} / {s_path}; will fall back to historical drift.")
    
    # Horizons configuration - use HORIZON_STEPS mapping
    all_results = {}
    scenario_summary_rows = []
    
    print(f"\n{'='*80}")
    print(f"Driver-Aware Monte Carlo Engine - Multiple Horizons")
    print(f"{'='*80}")
    print(f"Initial Price (S0): ${S0:.2f}")
    print(f"Number of paths per scenario: {n_paths:,}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*80}\n")
    
    for horizon_name in ["1Y", "3Y", "5Y", "10Y"]:
        horizon_steps = HORIZON_STEPS[horizon_name] #12/36/60/120 months
        print(f"\nProcessing {horizon_name} ({horizon_steps} monthly steps)...")
        
        # Load feature importance
        importance_dict = load_feature_importance(horizon_name)
        
        # Build shock table
        shock_table = build_shock_table(importance_dict, history_df, horizon_name)
        
        # Determine horizon drift (mu_horizon):
        # 1) Prefer horizon-specific ML model if available;
        # 2) Otherwise, fall back to historical-mean-based drift.
        if horizon_name in horizon_models:
            model_h, scaler_h = horizon_models[horizon_name]
            mu_horizon = predict_horizon_return(model_h, scaler_h, X_last)
            print(f"  ML-based horizon drift (mu_horizon) for {horizon_name}: {mu_horizon:.2%}")
        else:
            # Fallback: reuse the original historical-mean construction
            if "adj_close" in history_df.columns:
                returns = history_df["adj_close"].pct_change().dropna()
                mu_annual = returns.mean() * steps_per_year  # annualized from monthly mean
            else:
                mu_annual = 0.10  # Default 10% annual return

            horizon_months = horizon_steps  # monthly steps
            mu_horizon = mu_annual * (horizon_months / 12.0)
            print(f"  Historical-mean horizon drift (mu_horizon) for {horizon_name}: {mu_horizon:.2%}")
        
        # === Residual noise estimation (GKX-style additive error term) ===
        if "adj_close" in history_df.columns:
            hist_returns = history_df["adj_close"].pct_change().dropna()
            sigma_residual_annual = hist_returns.std() * np.sqrt(12)   # annualized
        else:
            sigma_residual_annual = 0.40   # fallback
        sigma_residual_step = sigma_residual_annual / np.sqrt(12)
        
        # Run scenarios
        scenarios = run_scenarios(
            S0, mu_horizon, shock_table, horizon_steps, history_df,
            n_paths, random_seed, steps_per_year,
            sigma_residual_step=sigma_residual_step
        )
        
        # Generate plots and summary for each scenario
        for scenario_label, result in scenarios.items():
            paths = result["paths"]
            terminals = result["terminals"]
            
            # Plot price paths (sample trajectories)
            plot_price_paths_multi_horizon(
                horizon_name, scenario_label, paths, S0,
                output_path / f"{horizon_name.lower()}_{scenario_label}_paths.png",
                n_sample_paths=20,
            )
            
            # Plot fan chart
            plot_fan_chart_multi_horizon(
                horizon_name, scenario_label, paths, S0,
                output_path / f"{horizon_name.lower()}_{scenario_label}_fan.png"
            )
            
            # Plot terminal distribution
            plot_terminal_distribution(
                horizon_name, scenario_label, terminals, S0,
                output_path / f"{horizon_name.lower()}_{scenario_label}_terminal.png"
            )
            
            # Add to summary table
            scenario_summary_rows.append({
                "horizon": horizon_name,
                "scenario": scenario_label,
                "mean": float(np.mean(terminals)),
                "p5": float(np.percentile(terminals, 5)),
                "p25": float(np.percentile(terminals, 25)),
                "p50": float(np.percentile(terminals, 50)),
                "p75": float(np.percentile(terminals, 75)),
                "p95": float(np.percentile(terminals, 95)),
            })
        
        all_results[horizon_name] = {
            "scenarios": scenarios,
            "shock_table": shock_table,
            "mu_horizon": mu_horizon,
        }
        
        print(f"✓ {horizon_name} complete")
    
    # Save consolidated scenario summary
    scenario_summary_df = pd.DataFrame(scenario_summary_rows)
    scenario_summary_df.to_csv(
        output_path / "scenario_summary.csv",
        index=False
    )
    print(f"\n✓ Saved scenario summary: {output_path / 'scenario_summary.csv'}")
    
    # Generate combined baseline fan chart (2x2 grid)
    plot_fan_chart_combined_baseline(
        all_results, S0,
        output_path / "fan_chart_combined_baseline.png"
    )
    
    print(f"\n{'='*80}")
    print(f"All horizons complete!")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 7: Scenario-Based Monte Carlo Forecasting")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker")
    parser.add_argument("--h", type=int, default=DEFAULT_HORIZON_MONTHS, help="Horizon in months")
    parser.add_argument("--n", type=int, default=DEFAULT_N_SIMS, help="Number of simulations")
    parser.add_argument("--output-dir", default="results/step7", help="Output directory")
    parser.add_argument("--model-path", default=None, help="Path to champion model")
    parser.add_argument("--scaler-path", default=None, help="Path to feature scaler")
    parser.add_argument("--multi-horizon", action="store_true", help="Run multi-horizon driver-aware MC")
    parser.add_argument("--hpc-multi-horizon", action="store_true", 
                       help="Run HPC multi-horizon benchmark (1Y/3Y/5Y) for NumPy vs Numba")
    parser.add_argument("--hpc-summary", action="store_true",
                       help="Build HPC summary table combining all benchmark results")
    parser.add_argument(
        "--scaling-curve",
        action="store_true",
        help="Run NumPy vs Numba scaling benchmark (Step 8 HPC).",
    )
    parser.add_argument(
        "--debug-horizon-drifts",
        action="store_true",
        help="Print horizon-specific μ_horizon and μ_step for 1Y/3Y/5Y/10Y."
    )
    
    args = parser.parse_args()
    
    if args.debug_horizon_drifts:
        debug_print_horizon_drifts(ticker=args.ticker, random_seed=RANDOM_STATE)
        print("\n" + "="*60)
        print("Horizon drift debug complete.")
        print("="*60)
        exit(0)
    elif args.scaling_curve:
        # Run only the scaling benchmark and exit
        benchmark_scaling_curve(
            output_dir="results/step8",
        )
        print("\n" + "="*60)
        print("HPC Scaling Benchmark Complete!")
        print("="*60)
    elif args.hpc_multi_horizon:
        # Run multi-horizon HPC benchmark
        results = benchmark_mc_paths_multi_horizon(
            output_dir="results/step8",
            n_sims=args.n if args.n >= 10000 else 100000, # Use 100k for multi-horizon
            random_seed=RANDOM_STATE,
        )
        print("\n" + "="*60)
        print("HPC Multi-Horizon Benchmark Complete!")
        print("="*60)
    elif args.hpc_summary:
        # Build HPC summary (1Y/3Y/5Y/10Y)
        summary = build_hpc_summary_1y_3y_5y_10y(output_dir="results/step8")
        print("\n" + "="*60)
        print("HPC Summary Complete!")
        print("="*60)
    elif args.multi_horizon:
        # Run new multi-horizon driver-aware Monte Carlo engine
        results = run_driver_aware_mc_multi_horizon(
            ticker=args.ticker,
            n_paths=args.n if args.n > 1000 else 10000,  # Use 10k for multi-horizon
            output_dir="results/step7",
            random_seed=RANDOM_STATE,
        )
        print("\n" + "="*60)
        print("Step 7 Monte Carlo horizons and 4-scenario engine updated successfully.")
        print("="*60)
    else:
        # Original single-horizon scenario forecast
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

    # Usage Examples:
    # python3 -m finmc_tech.simulation.scenario_mc --ticker NVDA --h 6 --n 50 --output-dir results/step7_sanity
    # python3 -m finmc_tech.simulation.scenario_mc --ticker NVDA --multi-horizon --n 10000
    # python3 -m finmc_tech.simulation.scenario_mc --hpc-multi-horizon --n 50000
    # python3 -m finmc_tech.simulation.scenario_mc --hpc-summary
