"""
Feature importance analysis using Random Forest.

Identifies which features truly drive predictive performance for NVDA monthly return prediction.
This step ensures interpretability, dimensionality reduction, and robust downstream modeling.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Load .env file before importing other libraries to set environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports to avoid slow module loading at script start
# This allows --help to work quickly without loading heavy dependencies
def _get_logger():
    from finmc_tech.config import get_logger
    return get_logger(__name__)

def _get_cfg():
    from finmc_tech.config import cfg
    return cfg

def _import_build_features():
    from finmc_tech.features.build_features import build_Xy, train_test_split_time
    return build_Xy, train_test_split_time

def _import_rf_model():
    from finmc_tech.models.rf_model import fit_rf, evaluate_rf
    return fit_rf, evaluate_rf


def load_features_data(features_path: str) -> pd.DataFrame:
    """
    Load features data from CSV.
    
    Args:
        features_path: Path to features CSV with columns: date (or px_date/period_end), Ret, and feature columns
    
    Returns:
        DataFrame with features and target
    """
    logger = _get_logger()
    logger.info(f"Loading features from {features_path}...")
    
    # Try to read CSV and detect date column
    df = pd.read_csv(features_path)
    
    # Find date column (try common names)
    date_col = None
    for col in ["date", "px_date", "period_end"]:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    
    # Parse dates and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    # Rename index to 'date' for consistency
    df.index.name = "date"
    
    # Create 'Ret' column if it doesn't exist
    if "Ret" not in df.columns:
        if "adj_close" in df.columns:
            df["Ret"] = df["adj_close"].pct_change()
            logger.info("  Created 'Ret' column from 'adj_close'")
        elif "future_12m_return" in df.columns:
            # Use future return as a proxy (shifted back)
            df["Ret"] = df["future_12m_return"]
            logger.info("  Using 'future_12m_return' as 'Ret'")
        else:
            raise ValueError("Cannot find price column to create 'Ret'. Available columns: " + str(list(df.columns)))
    
    logger.info(f"  Loaded {len(df)} rows")
    logger.info(f"  Date column: {date_col}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    return df


def compute_feature_importance(
    df: pd.DataFrame,
    n_estimators: int = 100,
    n_jobs: int = 1,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, pd.DataFrame, dict]:
    """
    Compute feature importance using Random Forest.
    
    Args:
        df: DataFrame with Ret and feature columns
        n_estimators: Number of trees in Random Forest (default: 100 for faster training)
        n_jobs: Number of parallel jobs (default: 1 to avoid macOS mutex lock issues)
        random_state: Random seed
    
    Returns:
        Fitted model, importance DataFrame, and metrics dict
    """
    logger = _get_logger()
    logger.info("=" * 70)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 70)
    
    # Lazy import (single call to avoid duplicate imports)
    build_Xy, train_test_split_time = _import_build_features()
    
    # Build X and y
    logger.info("\nBuilding features and target...")
    X, y, feature_names = build_Xy(df)
    
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Samples: {len(X)}")
    
    cfg = _get_cfg()
    
    # Train/test split (chronological)
    logger.info(f"\nSplitting data chronologically (train_end={cfg.TRAIN_END})...")
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_time(
        X, y, cfg.TRAIN_END
    )
    
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Test: {len(X_test)} samples")
    
    # Lazy import
    _, evaluate_rf = _import_rf_model()
    
    # Train Random Forest (single fit, no duplicate training)
    logger.info(f"\nTraining Random Forest (n_estimators={n_estimators}, n_jobs={n_jobs})...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    
    # Get feature importance
    importances = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_score": importances,
    }).sort_values("importance_score", ascending=False)
    
    # Compute metrics
    metrics = evaluate_rf(model, X_test, y_test)
    
    logger.info(f"\n  ✓ Model trained")
    logger.info(f"  Test R²: {metrics['R2']:.4f}")
    logger.info(f"  Test RMSE: {metrics['RMSE']:.4f}")
    logger.info(f"  Test MAE: {metrics['MAE']:.4f}")
    
    return model, importance_df, metrics


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature and importance_score columns
        output_path: Path to save plot
        top_n: Number of top features to plot
    """
    logger = _get_logger()
    logger.info(f"\nPlotting top {top_n} features...")
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features["importance_score"], color="steelblue", alpha=0.8)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Random Forest Feature Importance (Top {top_n})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"  ✓ Saved plot to {output_path}")


def print_top_features(importance_df: pd.DataFrame, top_n: int = 10) -> None:
    """Print top N features to console."""
    print("\n" + "=" * 70)
    print(f"Top {top_n} Features by Importance")
    print("=" * 70)
    
    top_features = importance_df.head(top_n)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s}: {row['importance_score']:7.4f}")
    
    print("=" * 70)


def update_readme(
    importance_df: pd.DataFrame,
    readme_path: Path,
    plot_path: Path,
    top_n: int = 10,
) -> None:
    """
    Update README.md with feature importance section.
    
    Args:
        importance_df: DataFrame with feature and importance_score
        readme_path: Path to README.md
        plot_path: Path to feature importance plot (relative to README)
        top_n: Number of top features to include in table
    """
    logger = _get_logger()
    logger.info(f"\nUpdating README.md...")
    
    if not readme_path.exists():
        logger.warning(f"  README.md not found at {readme_path}, skipping update")
        return
    
    # Read README
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    # Check if section already exists
    section_marker = "### Feature Importance Analysis"
    if section_marker in readme_content:
        logger.info("  Feature Importance section already exists, skipping update")
        return
    
    # Prepare table
    top_features = importance_df.head(top_n)
    table_rows = []
    table_rows.append("| Rank | Feature | Importance |")
    table_rows.append("|------|---------|------------|")
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        table_rows.append(f"| {i} | `{row['feature']}` | {row['importance_score']:.4f} |")
    
    table_text = "\n".join(table_rows)
    
    # Prepare section content
    section_content = f"""
### Feature Importance Analysis

**Why Random Forest?**
Random Forest is used for feature importance analysis because:
1. It captures **nonlinear relationships** and **interactions** between macro and firm-level features
2. It is **ensemble-based** and robust to outliers, noise, and overfitting
3. It provides **direct, interpretable importance scores** through mean decrease in impurity
4. It does **not assume linearity** or stationarity—important since macro-financial data often violate those assumptions
5. Compared to deep networks (LSTM) or SVMs, Random Forest offers explainability and speed for feature screening

**Why Feature Importance?**
Feature importance helps to:
1. Identify the **most predictive signals** (rev_yoy, VIX, FedFunds, etc.)
2. Remove redundant or noisy features before training LSTM (faster convergence, better generalization)
3. Provide a **transparent feature ranking** for the project
4. Build trust in the ML pipeline—by knowing "what the model is learning"

**Top {top_n} Features:**

{table_text}

![RF Feature Importance]({plot_path})
"""
    
    # Find insertion point (after "Prediction Target & Features" section)
    insertion_marker = "## Data Pipeline & Alignment"
    if insertion_marker in readme_content:
        # Insert before "Data Pipeline & Alignment"
        readme_content = readme_content.replace(
            insertion_marker,
            section_content + "\n" + insertion_marker
        )
    else:
        # Append at end
        readme_content += section_content
    
    # Write updated README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    logger.info(f"  ✓ Updated README.md with feature importance section")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Feature importance analysis using Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to features CSV (required). If not provided, script will exit with error.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to plot (default: 20)",
    )
    parser.add_argument(
        "--save-readme",
        action="store_true",
        help="Auto-update README.md with feature importance results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Plots directory (default: plots)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest (default: 100 for faster training)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Random Forest (default: 1 to avoid macOS mutex lock issues, use -1 for all cores)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger = _get_logger()
    cfg = _get_cfg()
    
    logger.info("=" * 70)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 70)
    
    # Load features (required - no auto-generation to avoid slow pipeline)
    if not args.features or not Path(args.features).exists():
        logger.error(f"Features file not found: {args.features}")
        logger.error("Please provide a valid --features path. Example:")
        logger.error("  python feature_importance_rf.py --features data/processed/NVDA_revenue_features.csv")
        sys.exit(1)
    
    df = load_features_data(args.features)
    
    # Compute feature importance
    model, importance_df, metrics = compute_feature_importance(
        df,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        random_state=cfg.RANDOM_STATE,
    )
    
    # Save importance CSV
    importance_csv = output_dir / "rf_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)
    logger.info(f"\n✓ Saved importance scores to {importance_csv}")
    
    # Plot
    plot_path = plots_dir / "rf_feature_importance.png"
    plot_feature_importance(importance_df, plot_path, top_n=args.top_n)
    
    # Print top features
    print_top_features(importance_df, top_n=10)
    
    # Update README if requested
    if args.save_readme:
        readme_path = Path("README.md")
        # Use relative path for plot in README
        plot_rel_path = plots_dir / "rf_feature_importance.png"
        update_readme(importance_df, readme_path, plot_rel_path, top_n=10)
    
    logger.info("\n" + "=" * 70)
    logger.info("Analysis Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

