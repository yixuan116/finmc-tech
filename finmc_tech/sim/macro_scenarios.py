"""
Generate macro scenarios for Monte Carlo simulation.

Implements path-based Monte Carlo with macro shocks for multi-month horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from finmc_tech.config import cfg, get_logger
from finmc_tech.models.rf_model import RandomForestRegressor

logger = get_logger(__name__)

# Macro column mapping (FRED names to feature names)
MACRO_COLS = ["CPI", "VIX", "DGS10", "FEDFUNDS", "GDP"]


def estimate_macro_vol(
    df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, float]:
    """
    Estimate monthly volatility for macro columns using last 5 years.
    
    Args:
        df: DataFrame with macro data (indexed by date)
        cols: List of macro column names to estimate
    
    Returns:
        Dict mapping column name to monthly standard deviation
    """
    logger.info(f"Estimating macro volatility for {len(cols)} columns (last 5 years)...")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'Date' column")
    
    # Use last 5 years
    df_sorted = df.sort_index()
    cutoff_date = df_sorted.index.max() - pd.DateOffset(years=5)
    df_recent = df_sorted[df_sorted.index >= cutoff_date]
    
    vol_dict = {}
    for col in cols:
        if col not in df_recent.columns:
            logger.warning(f"  Column '{col}' not found, skipping")
            continue
        
        # Calculate monthly changes (pct_change for ratios, diff for levels)
        if col in ["CPI", "GDP"]:
            # Use percentage change for indices
            changes = df_recent[col].pct_change().dropna()
        else:
            # Use absolute change for rates/yields
            changes = df_recent[col].diff().dropna()
        
        # Monthly standard deviation
        vol = changes.std()
        vol_dict[col] = float(vol)
        logger.info(f"  {col}: σ = {vol:.4f}")
    
    return vol_dict


def generate_paths(
    base_row: pd.Series,
    H: int,
    vol_dict: Dict[str, float],
    n_paths: int = 200,
    shock: str = "base",
) -> np.ndarray:
    """
    Generate Monte Carlo paths for macro variables over H months.
    
    Args:
        base_row: Base feature row (latest month) with macro values
        H: Horizon in months
        vol_dict: Dict of monthly volatility for each macro column
        n_paths: Number of paths to generate
        shock: "base" or "stress"
    
    Returns:
        Array of shape (n_paths, H, n_macro_features) with macro paths
    """
    logger.info(f"Generating {n_paths} paths for {H} months (shock: {shock})...")
    
    # Extract base macro values
    macro_base = {}
    for col in MACRO_COLS:
        # Try different possible column names
        for possible_name in [col, col.lower(), f"{col.lower()}_level"]:
            if possible_name in base_row.index:
                macro_base[col] = float(base_row[possible_name])
                break
        if col not in macro_base:
            logger.warning(f"  Base value for {col} not found, using 0")
            macro_base[col] = 0.0
    
    n_macro = len(MACRO_COLS)
    paths = np.zeros((n_paths, H, n_macro))
    
    for path_idx in range(n_paths):
        current_values = {col: macro_base[col] for col in MACRO_COLS}
        
        for month in range(H):
            # Generate shocks
            if shock == "stress" and month < 3:
                # Stress: +1σ for CPI and VIX in first 3 months
                for col in MACRO_COLS:
                    if col in vol_dict:
                        if col in ["CPI", "VIX"]:
                            shock_val = vol_dict[col]  # +1σ
                        else:
                            shock_val = np.random.normal(0, vol_dict[col])
                    else:
                        shock_val = 0.0
                    
                    # Update value (additive for rates, multiplicative for indices)
                    if col in ["CPI", "GDP"]:
                        current_values[col] *= (1 + shock_val)
                    else:
                        current_values[col] += shock_val
            else:
                # Base: random normal shocks
                for col in MACRO_COLS:
                    if col in vol_dict:
                        shock_val = np.random.normal(0, vol_dict[col])
                    else:
                        shock_val = 0.0
                    
                    if col in ["CPI", "GDP"]:
                        current_values[col] *= (1 + shock_val)
                    else:
                        current_values[col] += shock_val
            
            # Store values for this month
            for i, col in enumerate(MACRO_COLS):
                paths[path_idx, month, i] = current_values[col]
    
    logger.info(f"  ✓ Generated {n_paths} paths")
    return paths


def path_to_predictions(
    paths: np.ndarray,
    feature_names: List[str],
    rf_model: RandomForestRegressor,
    prev_values: pd.Series,
    scaler_state: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Convert macro paths to monthly predicted returns using RF model.
    
    Args:
        paths: Array of shape (n_paths, H, n_macro_features) with macro paths
        feature_names: List of feature names expected by model
        rf_model: Fitted RandomForestRegressor
        prev_values: Previous month's feature values (for non-macro features)
        scaler_state: Optional scaler state (not used for RF, kept for compatibility)
    
    Returns:
        DataFrame of shape (H, n_paths) with predicted monthly returns
    """
    logger.info(f"Converting {paths.shape[0]} paths to predictions...")
    
    n_paths, H, n_macro = paths.shape
    
    # Map macro columns to feature names
    macro_to_feature = {
        "CPI": "CPI",
        "VIX": "VIX",
        "DGS10": "DGS10",
        "FEDFUNDS": "FEDFUNDS",
        "GDP": "GDP",
    }
    
    predictions = np.zeros((H, n_paths))
    
    for month in range(H):
        for path_idx in range(n_paths):
            # Build feature vector for this path/month
            features = prev_values.copy()
            
            # Update macro features from path
            for i, macro_col in enumerate(MACRO_COLS):
                feature_name = macro_to_feature.get(macro_col, macro_col)
                if feature_name in features.index:
                    features[feature_name] = paths[path_idx, month, i]
            
            # Ensure feature order matches model
            feature_array = features[feature_names].values.reshape(1, -1)
            
            # Predict
            pred = rf_model.predict(feature_array)[0]
            predictions[month, path_idx] = pred
    
    # Create DataFrame
    pred_df = pd.DataFrame(
        predictions,
        index=pd.date_range(start=pd.Timestamp.now(), periods=H, freq="M"),
        columns=[f"path_{i}" for i in range(n_paths)],
    )
    
    logger.info(f"  ✓ Generated predictions: {pred_df.shape}")
    return pred_df


def run_macro_mc(
    rf_model: RandomForestRegressor,
    base_row: pd.Series,
    macro_df: pd.DataFrame,
    feature_names: List[str],
    H: int = 12,
    n_paths: int = 200,
    shock: str = "base",
    results_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full macro-driven Monte Carlo simulation.
    
    Args:
        rf_model: Fitted RandomForestRegressor
        base_row: Base feature row (latest month)
        macro_df: Historical macro data for volatility estimation
        feature_names: List of feature names
        H: Horizon in months
        n_paths: Number of paths
        shock: "base" or "stress"
        results_dir: Directory to save results
    
    Returns:
        predictions_df: DataFrame (H, n_paths) with monthly returns
        summary_df: DataFrame with summary statistics
    """
    logger.info("=" * 70)
    logger.info("Running Macro-Driven Monte Carlo Simulation")
    logger.info("=" * 70)
    
    # Estimate volatility
    vol_dict = estimate_macro_vol(macro_df, MACRO_COLS)
    
    # Generate paths
    paths = generate_paths(base_row, H, vol_dict, n_paths, shock)
    
    # Convert to predictions
    predictions_df = path_to_predictions(
        paths, feature_names, rf_model, base_row
    )
    
    # Calculate summary statistics
    summary_stats = {
        "month": range(1, H + 1),
        "mean": predictions_df.mean(axis=1).values,
        "p10": predictions_df.quantile(0.10, axis=1).values,
        "p50": predictions_df.quantile(0.50, axis=1).values,
        "p90": predictions_df.quantile(0.90, axis=1).values,
    }
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    if results_dir is None:
        results_dir = Path(cfg.RESULTS_DIR)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_file = results_dir / f"macro_mc_predictions_{shock}_H{H}_n{n_paths}.csv"
    predictions_df.to_csv(pred_file)
    logger.info(f"  ✓ Saved predictions to {pred_file}")
    
    # Save summary
    summary_file = results_dir / f"macro_mc_summary_{shock}_H{H}_n{n_paths}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"  ✓ Saved summary to {summary_file}")
    
    logger.info("=" * 70)
    
    return predictions_df, summary_df


# Backward compatibility functions
@dataclass
class MacroScenario:
    """Macro economic scenario definition (backward compatibility)."""
    name: str
    vix_mean: float
    vix_std: float
    tnx_mean: float
    tnx_std: float
    sp500_return_mean: float
    sp500_return_std: float
    probability: float = 1.0


def generate_macro_scenarios(
    historical_macro: pd.DataFrame,
    n_scenarios: int = 1000,
) -> List[MacroScenario]:
    """Backward compatibility wrapper."""
    scenarios = []
    
    vix_mean = historical_macro.get("VIX", historical_macro.get("vix_level", pd.Series([20]))).mean()
    vix_std = historical_macro.get("VIX", historical_macro.get("vix_level", pd.Series([20]))).std()
    
    tnx_mean = historical_macro.get("DGS10", historical_macro.get("tnx_yield", pd.Series([0.03]))).mean()
    tnx_std = historical_macro.get("DGS10", historical_macro.get("tnx_yield", pd.Series([0.03]))).std()
    
    sp500_mean = historical_macro.get("sp500_returns", pd.Series([0.0])).mean()
    sp500_std = historical_macro.get("sp500_returns", pd.Series([0.01])).std()
    
    scenarios.append(MacroScenario(
        name="baseline",
        vix_mean=vix_mean,
        vix_std=vix_std,
        tnx_mean=tnx_mean,
        tnx_std=tnx_std,
        sp500_return_mean=sp500_mean,
        sp500_return_std=sp500_std,
        probability=0.5,
    ))
    
    return scenarios


def sample_macro_scenario(
    scenario: MacroScenario,
    n_samples: int = 1,
) -> pd.DataFrame:
    """Backward compatibility wrapper."""
    samples = {
        "vix_level": np.random.normal(scenario.vix_mean, scenario.vix_std, n_samples),
        "tnx_yield": np.random.normal(scenario.tnx_mean, scenario.tnx_std, n_samples),
        "sp500_returns": np.random.normal(scenario.sp500_return_mean, scenario.sp500_return_std, n_samples),
    }
    
    samples["vix_level"] = np.maximum(samples["vix_level"], 0)
    samples["tnx_yield"] = np.maximum(samples["tnx_yield"], 0)
    
    return pd.DataFrame(samples)

