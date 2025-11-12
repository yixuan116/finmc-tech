"""
Run macro-driven Monte Carlo simulation pipeline.

Implements complete training + simulation + visualization pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path as PathType

from finmc_tech.config import Settings, cfg, get_logger
from finmc_tech.data.fetch_macro import fetch_macro
from finmc_tech.data.fetch_firm import fetch_firm_data
from finmc_tech.data.align import align_macro_firm
from finmc_tech.features.build_features import build_Xy, train_test_split_time, scale_Xy
from finmc_tech.models.rf_model import fit_rf, evaluate_rf, save_model, plot_feature_importance
from finmc_tech.sim.macro_scenarios import (
    estimate_macro_vol,
    generate_paths,
    path_to_predictions,
    run_macro_mc,
    MACRO_COLS,
)

logger = get_logger(__name__)


def pipeline(
    config: Optional[Settings] = None,
    H: int = 24,
    n_paths: int = 200,
    shock: str = "base",
) -> Dict:
    """
    Complete pipeline: fetch data, train model, run simulation.
    
    Args:
        config: Settings object. If None, uses global cfg
        H: Simulation horizon in months
        n_paths: Number of Monte Carlo paths
        shock: "base" or "stress"
    
    Returns:
        Dict with model, metrics, and simulation results
    """
    if config is None:
        config = cfg
    
    logger.info("=" * 70)
    logger.info("Starting Complete Pipeline")
    logger.info("=" * 70)
    
    # 1. Fetch macro data
    logger.info("\n[1/7] Fetching macro data...")
    macro_df = fetch_macro(
        config.START_DATE,
        config.END_DATE,
        config.cache_dir,
    )
    
    # 2. Fetch firm data (prices + revenue)
    logger.info("\n[2/7] Fetching firm data...")
    firm_df = fetch_firm_data(
        config.TICKER,
        config.START_DATE,
        config.END_DATE,
        config.cache_dir,
    )
    
    # 3. Align macro and firm data
    logger.info("\n[3/7] Aligning macro and firm data...")
    panel = align_macro_firm(macro_df, firm_df)
    
    # Ensure "Ret" column exists (returns)
    if "Ret" not in panel.columns:
        # Try to create from price data
        if "adj_close" in panel.columns:
            panel["Ret"] = panel["adj_close"].pct_change()
        elif "Close" in panel.columns:
            panel["Ret"] = panel["Close"].pct_change()
        elif "price" in panel.columns:
            panel["Ret"] = panel["price"].pct_change()
        else:
            # Try to find any price-like column
            price_cols = [col for col in panel.columns if "price" in col.lower() or "close" in col.lower()]
            if price_cols:
                panel["Ret"] = panel[price_cols[0]].pct_change()
            else:
                logger.warning("Cannot find price column to create 'Ret', using zero returns")
                panel["Ret"] = 0.0
    
    # 4. Build features
    logger.info("\n[4/7] Building features...")
    X, y, feature_names = build_Xy(panel)
    
    # 5. Train/test split
    logger.info("\n[5/7] Splitting data...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_time(
        X, y, config.TRAIN_END
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_Xy(X_train, X_test, y_train, y_test)
    
    # 6. Train model
    logger.info("\n[6/7] Training RandomForest model...")
    model = fit_rf(X_train_scaled, y_train, config.RANDOM_STATE)
    metrics = evaluate_rf(model, X_test_scaled, y_test)
    
    # Save model
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / f"{config.TICKER}_rf_model.pkl"
    save_model(model, model_path)
    
    # Plot feature importance
    importance_path = results_dir / f"{config.TICKER}_feature_importance.png"
    plot_feature_importance(model, feature_names, importance_path, top_n=20)
    
    # 7. Run simulation
    logger.info("\n[7/7] Running Monte Carlo simulation...")
    
    # Get base row (last available feature row from test set)
    base_row = X_test_scaled.iloc[-1].copy()
    base_row.index = feature_names
    
    # Estimate macro volatility
    vol_dict = estimate_macro_vol(panel, MACRO_COLS)
    
    # Generate paths
    paths = generate_paths(base_row, H, vol_dict, n_paths, shock)
    
    # Convert to predictions
    preds_df = path_to_predictions(
        paths, feature_names, model, base_row, scaler_state=None
    )
    
    # Calculate summary statistics
    summary_stats = {
        "month": range(1, H + 1),
        "mean": preds_df.mean(axis=1).values,
        "p10": preds_df.quantile(0.10, axis=1).values,
        "p50": preds_df.quantile(0.50, axis=1).values,
        "p90": preds_df.quantile(0.90, axis=1).values,
    }
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    preds_file = results_dir / f"macro_mc_predictions_{shock}_H{H}_n{n_paths}.csv"
    preds_df.to_csv(preds_file)
    logger.info(f"  ✓ Saved predictions to {preds_file}")
    
    summary_file = results_dir / f"macro_mc_summary_{shock}_H{H}_n{n_paths}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"  ✓ Saved summary to {summary_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)
    
    return {
        "model": model,
        "metrics": metrics,
        "predictions_df": preds_df,
        "summary_df": summary_df,
        "model_path": model_path,
        "importance_path": importance_path,
        "preds_file": preds_file,
        "summary_file": summary_file,
    }


# Backward compatibility
def run_macro_simulation(
    model: any,
    current_features: pd.Series,
    historical_macro: pd.DataFrame,
    config: Settings,
    macro_scenarios: Optional[list] = None,
) -> Dict:
    """Backward compatibility wrapper."""
    from finmc_tech.sim.macro_scenarios import generate_macro_scenarios, sample_macro_scenario
    
    logger.info(f"Running macro-driven simulation ({config.N_SIMULATIONS} paths)...")
    
    if macro_scenarios is None:
        macro_scenarios = generate_macro_scenarios(historical_macro)
    
    all_predictions = []
    scenario_names = []
    
    for scenario in macro_scenarios:
        n_samples = int(config.N_SIMULATIONS * scenario.probability)
        macro_samples = sample_macro_scenario(scenario, n_samples)
        
        for idx, macro_row in macro_samples.iterrows():
            features = current_features.copy()
            features["vix_level"] = macro_row["vix_level"]
            features["tnx_yield"] = macro_row["tnx_yield"]
            if "sp500_returns" in features.index:
                features["sp500_returns"] = macro_row["sp500_returns"]
            
            pred = model.predict(features.values.reshape(1, -1))[0]
            all_predictions.append(pred)
            scenario_names.append(scenario.name)
    
    predictions = np.array(all_predictions)
    
    results = {
        "predictions": predictions.tolist(),
        "scenarios": scenario_names,
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "percentiles": {
            "5": float(np.percentile(predictions, 5)),
            "25": float(np.percentile(predictions, 25)),
            "50": float(np.percentile(predictions, 50)),
            "75": float(np.percentile(predictions, 75)),
            "95": float(np.percentile(predictions, 95)),
        },
    }
    
    logger.info(f"  ✓ Mean prediction: {results['mean']:.4f}")
    logger.info(f"  ✓ Std: {results['std']:.4f}")
    
    return results
