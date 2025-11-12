"""
Visualization functions for model predictions and simulation results.
"""

import sys
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path as PathType

from finmc_tech.config import get_logger

logger = get_logger(__name__)


def plot_pred_vs_actual(
    dates_test: pd.DatetimeIndex,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    outpath: PathType,
) -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        dates_test: Test dates
        y_test: Actual values
        y_pred: Predicted values
        outpath: Path to save plot
    """
    logger.info(f"Plotting predictions vs actual...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    axes[0].plot(dates_test, y_test, label="Actual", alpha=0.7, marker="o", markersize=4)
    axes[0].plot(dates_test, y_pred, label="Predicted", alpha=0.7, marker="s", markersize=4)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Predictions vs Actual (Time Series)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    axes[1].set_xlabel("Actual Return")
    axes[1].set_ylabel("Predicted Return")
    axes[1].set_title("Predictions vs Actual (Scatter)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"  ✓ Saved plot to {outpath}")


def plot_sim_distribution(
    preds_df: pd.DataFrame,
    outpath: PathType,
) -> None:
    """
    Plot histogram of aggregated returns over horizon.
    
    Args:
        preds_df: DataFrame of shape (H, n_paths) with monthly returns
        outpath: Path to save plot
    """
    logger.info(f"Plotting simulation distribution...")
    
    # Aggregate returns over horizon (sum of monthly returns)
    aggregated_returns = preds_df.sum(axis=0).values
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(aggregated_returns, bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(aggregated_returns.mean(), color="r", linestyle="--", 
                    label=f"Mean: {aggregated_returns.mean():.4f}")
    axes[0].axvline(np.median(aggregated_returns), color="g", linestyle="--",
                    label=f"Median: {np.median(aggregated_returns):.4f}")
    axes[0].set_xlabel("Aggregated Return (Sum over Horizon)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Aggregated Returns")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by month
    axes[1].boxplot([preds_df.iloc[i].values for i in range(len(preds_df))],
                    labels=[f"M{i+1}" for i in range(len(preds_df))])
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Monthly Return")
    axes[1].set_title("Monthly Return Distribution by Month")
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"  ✓ Saved plot to {outpath}")


def plot_rolling_corr(
    panel: pd.DataFrame,
    outpath: PathType,
    window: int = 12,
) -> None:
    """
    Plot rolling correlation between Ret and each macro variable.
    
    Args:
        panel: Aligned panel data with Ret and macro columns
        outpath: Path to save plot
        window: Rolling window size in months
    """
    logger.info(f"Plotting rolling correlations...")
    
    if "Ret" not in panel.columns:
        logger.warning("  'Ret' column not found, skipping rolling correlation plot")
        return
    
    macro_cols = ["CPI", "VIX", "DGS10", "FEDFUNDS", "GDP"]
    available_macros = [col for col in macro_cols if col in panel.columns]
    
    if not available_macros:
        logger.warning("  No macro columns found, skipping rolling correlation plot")
        return
    
    n_macros = len(available_macros)
    fig, axes = plt.subplots(n_macros, 1, figsize=(12, 3 * n_macros))
    
    if n_macros == 1:
        axes = [axes]
    
    for i, macro_col in enumerate(available_macros):
        # Calculate rolling correlation
        rolling_corr = panel["Ret"].rolling(window=window).corr(panel[macro_col])
        
        axes[i].plot(panel.index, rolling_corr, label=f"Corr(Ret, {macro_col})")
        axes[i].axhline(0, color="k", linestyle="--", alpha=0.3)
        axes[i].set_ylabel("Correlation")
        axes[i].set_title(f"Rolling Correlation: Ret vs {macro_col} (window={window})")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Date")
    
    plt.tight_layout()
    
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"  ✓ Saved plot to {outpath}")


# Backward compatibility functions
def plot_simulation_results(
    results: dict,
    output_path: Optional[PathType] = None,
) -> None:
    """Backward compatibility wrapper."""
    predictions = np.array(results.get("predictions", []))
    
    if len(predictions) == 0:
        logger.warning("No predictions in results, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(predictions, bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(results.get("mean", 0), color="r", linestyle="--",
                    label=f"Mean: {results.get('mean', 0):.4f}")
    axes[0].set_xlabel("Predicted Return")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Return Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Saved plot to {output_path}")
    else:
        plt.show()


def plot_model_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[PathType] = None,
) -> None:
    """Backward compatibility wrapper."""
    dates = pd.date_range(start="2020-01-01", periods=len(y_test), freq="M")
    plot_pred_vs_actual(dates, y_test, y_pred, output_path or "predictions.png")
