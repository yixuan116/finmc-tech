"""
Feature importance analysis using Random Forest (Light Version).

Ultra-light, fast, self-contained script that computes feature importance for NVDA
monthly return prediction using only existing CSVs (no project-internal imports).

Input: CSVs with prices, macro, and firm features
Output: Importance table, bar plot, optional README update
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def read_csv_detect_date(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Read CSV and detect date column, set index to monthly date.
    
    Args:
        path: Path to CSV file
        date_col: Optional explicit date column name
    
    Returns:
        DataFrame with 'date' index (month-end frequency)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Detect date column
    if date_col is None:
        for col in ["date", "px_date", "period_end", "Date"]:
            if col in df.columns:
                date_col = col
                break
    
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    
    # Parse dates and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    
    # Ensure month-end frequency
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Date index must be DatetimeIndex")
    
    # Resample to month-end if needed (if daily data)
    if len(df) > 0:
        freq = pd.infer_freq(df.index)
        if freq and "D" in freq:  # Daily frequency
            df = df.resample("M").last()  # Resample to month-end
            print(f"  Resampled daily data to monthly (M) frequency")
    
    return df


def load_and_prepare_data(
    prices_path: str,
    macro_path: Optional[str],
    firm_path: Optional[str],
    row_subsample: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load prices, macro, and firm CSVs, merge, and prepare X/y.
    
    Args:
        prices_path: Path to prices CSV
        macro_path: Path to macro CSV (optional)
        firm_path: Path to firm-aligned CSV (optional)
        row_subsample: Fraction of rows to keep (1.0 = all, 0.5 = last 50%)
    
    Returns:
        X (features DataFrame), y (target Series)
    """
    print("=" * 70)
    print("Loading and Preparing Data")
    print("=" * 70)
    
    # Load prices
    print(f"\n1. Loading prices from {prices_path}...")
    prices_df = read_csv_detect_date(prices_path)
    
    # Get price column
    price_col = None
    for col in ["adj_close", "close", "Close"]:
        if col in prices_df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"No price column found. Available: {list(prices_df.columns)}")
    
    # Compute monthly returns: Ret_t = (P_t / P_{t-1}) - 1
    prices_df["Ret"] = prices_df[price_col].pct_change()
    print(f"  ✓ Computed monthly returns from {price_col}")
    print(f"  ✓ Prices: {len(prices_df)} rows")
    
    # Start with prices_df (includes Ret and any other feature columns)
    # Exclude non-feature columns: price column, date columns, metadata
    exclude_from_prices = {price_col, "date", "Date", "ticker", "fy", "fp", "form", "tag_used"}
    prices_feature_cols = [col for col in prices_df.columns if col not in exclude_from_prices]
    merged_df = prices_df[prices_feature_cols].copy()
    print(f"  ✓ Included {len(prices_feature_cols)} columns from prices file (including Ret)")
    
    # Merge macro data
    if macro_path and Path(macro_path).exists():
        print(f"\n2. Loading macro data from {macro_path}...")
        macro_df = read_csv_detect_date(macro_path)
        # Inner join on date index
        merged_df = merged_df.join(macro_df, how="inner")
        print(f"  ✓ Macro: {len(macro_df)} rows, {len(macro_df.columns)} columns")
    else:
        print(f"\n2. Skipping macro data (file not found or not provided)")
    
    # Merge firm data
    if firm_path and Path(firm_path).exists():
        print(f"\n3. Loading firm data from {firm_path}...")
        firm_df = read_csv_detect_date(firm_path)
        # Inner join on date index
        merged_df = merged_df.join(firm_df, how="inner")
        print(f"  ✓ Firm: {len(firm_df)} rows, {len(firm_df.columns)} columns")
    else:
        print(f"\n3. Skipping firm data (file not found or not provided)")
    
    print(f"\n4. Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Identify feature columns (exclude Ret, y, price columns, and non-numeric columns)
    exclude_cols = {
        "Ret", "y", "adj_close", "close", "Close", price_col,
        "date", "Date", "px_date", "period_end",  # Date columns
        "ticker", "fy", "fp", "form", "tag_used",  # Metadata columns
        "future_12m_price", "future_12m_return", "future_12m_logprice",  # Future targets
    }
    
    # Filter feature columns: exclude non-numeric columns
    feature_cols = []
    for col in merged_df.columns:
        if col in exclude_cols:
            continue
        # Try to convert to numeric - if successful, it's a valid feature
        try:
            test_series = pd.to_numeric(merged_df[col], errors="coerce")
            if not test_series.isna().all():  # At least some valid numeric values
                feature_cols.append(col)
                merged_df[col] = test_series.astype("float32")
        except:
            continue  # Skip this column
    
    print(f"  ✓ Identified {len(feature_cols)} numeric feature columns")
    
    # Create target: y = Ret.shift(-1) (next-month return)
    merged_df["y"] = merged_df["Ret"].shift(-1)
    
    # Drop rows with NaN
    merged_df = merged_df.dropna()
    
    # Row subsampling (keep last N% rows)
    if row_subsample < 1.0:
        n_keep = int(len(merged_df) * row_subsample)
        merged_df = merged_df.tail(n_keep)
        print(f"  ✓ Subsampled to last {n_keep} rows ({row_subsample*100:.0f}%)")
    
    # Assert minimum data requirement
    if len(merged_df) < 12:
        raise ValueError(
            f"Insufficient data after merging: {len(merged_df)} rows. "
            "Need at least 12 months. Check CSV paths and date alignment."
        )
    
    # Extract X and y
    X = merged_df[feature_cols].copy()
    y = merged_df["y"].copy()
    
    print(f"\n5. Final dataset:")
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Samples: {len(X)}")
    print(f"  ✓ Date range: {X.index.min()} to {X.index.max()}")
    
    return X, y


def train_rf_and_compute_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 150,
    n_jobs: int = 4,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    """
    Train Random Forest and compute feature importance.
    
    Args:
        X: Feature DataFrame
        y: Target Series (next-month return)
        n_estimators: Number of trees
        n_jobs: Number of parallel jobs
        random_state: Random seed
    
    Returns:
        Fitted model, importance DataFrame
    """
    print("\n" + "=" * 70)
    print("Training Random Forest")
    print("=" * 70)
    
    print(f"\nParameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  n_jobs: {n_jobs}")
    print(f"  max_depth: None")
    print(f"  min_samples_leaf: 3")
    print(f"  max_features: sqrt")
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
    )
    
    print(f"\nTraining on {len(X)} samples...")
    model.fit(X, y)
    
    # Compute feature importance
    importances = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_score": importances,
    }).sort_values("importance_score", ascending=False)
    
    # Compute metrics
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    print(f"\n✓ Model trained")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return model, importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature and importance_score
        output_path: Path to save plot
        top_n: Number of top features to plot
    """
    print(f"\nPlotting top {top_n} features...")
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features["importance_score"], color="steelblue", alpha=0.8)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature"], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(
        f"Random Forest Feature Importance (Top {top_n}) — Monthly NVDA Returns",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved plot to {output_path}")


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
    print(f"\nUpdating README.md...")
    
    if not readme_path.exists():
        print(f"  ⚠ README.md not found at {readme_path}, skipping update")
        return
    
    # Read README
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
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
### Feature Importance Analysis (Light)

**Why Random Forest?**
Random Forest is used for feature importance analysis because:
1. It captures **nonlinear relationships** and **interactions** between macro and firm-level features
2. It is **ensemble-based** and robust to outliers, noise, and overfitting
3. It provides **direct, interpretable importance scores** through mean decrease in impurity
4. It does **not assume linearity** or stationarity—important since macro-financial data often violate those assumptions
5. Compared to deep networks (LSTM) or SVMs, Random Forest offers explainability and speed for feature screening

**Why Feature Importance before forecasting?**
Feature importance helps to:
1. Identify the **most predictive signals** (rev_yoy, VIX, FedFunds, etc.)
2. Remove redundant or noisy features before training LSTM (faster convergence, better generalization)
3. Provide a **transparent feature ranking** for the project
4. Build trust in the ML pipeline—by knowing "what the model is learning"

**Top {top_n} Features:**

{table_text}

![RF Feature Importance]({plot_path})
"""
    
    # Check if section markers exist
    start_marker = "<!-- FEAT_IMPORT_START -->"
    end_marker = "<!-- FEAT_IMPORT_END -->"
    
    if start_marker in readme_content and end_marker in readme_content:
        # Replace existing section
        import re
        pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
        replacement = start_marker + section_content + "\n" + end_marker
        readme_content = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)
        print(f"  ✓ Replaced existing feature importance section")
    else:
        # Insert new section (after "Prediction Target & Features" or at end)
        insertion_marker = "## Data Pipeline & Alignment"
        if insertion_marker in readme_content:
            readme_content = readme_content.replace(
                insertion_marker,
                start_marker + section_content + "\n" + end_marker + "\n" + insertion_marker
            )
            print(f"  ✓ Inserted feature importance section before '{insertion_marker}'")
        else:
            # Append at end
            readme_content += "\n" + start_marker + section_content + "\n" + end_marker
            print(f"  ✓ Appended feature importance section at end")
    
    # Write updated README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"  ✓ Updated README.md")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Feature importance analysis using Random Forest (Light Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run with defaults:
  python3 feature_importance_rf_light.py --save-readme

  # Faster test (subsample + fewer trees):
  python3 feature_importance_rf_light.py --row-subsample 0.5 --n-estimators 120 --save-readme

  # Custom paths:
  python3 feature_importance_rf_light.py --prices data/prices/NVDA.csv --macro data/processed/macro.csv --firm data/processed/firm_aligned.csv --save-readme
        """,
    )
    
    parser.add_argument(
        "--prices",
        type=str,
        default="data/prices/nvda_prices.csv",
        help="Path to NVDA prices CSV (default: data/prices/nvda_prices.csv)",
    )
    parser.add_argument(
        "--macro",
        type=str,
        default="data/processed/macro.csv",
        help="Path to macro CSV (default: data/processed/macro.csv)",
    )
    parser.add_argument(
        "--firm",
        type=str,
        default="data/processed/firm_aligned.csv",
        help="Path to firm-aligned CSV (default: data/processed/firm_aligned.csv)",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Name of date column if autodetect fails",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to plot (default: 20)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Number of trees in Random Forest (default: 150)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (default: 4, use 1 to avoid macOS mutex issues)",
    )
    parser.add_argument(
        "--row-subsample",
        type=float,
        default=1.0,
        help="Fraction of rows to keep (default: 1.0 = all, 0.5 = last 50%% for quick run)",
    )
    parser.add_argument(
        "--save-readme",
        action="store_true",
        help="Auto-update README.md with feature importance results",
    )
    
    args = parser.parse_args()
    
    # Create output directories
    results_dir = Path("results")
    plots_dir = Path("plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data(
            prices_path=args.prices,
            macro_path=args.macro,
            firm_path=args.firm,
            row_subsample=args.row_subsample,
        )
        
        # Train RF and compute importance
        model, importance_df = train_rf_and_compute_importance(
            X,
            y,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=42,
        )
        
        # Save importance CSV
        importance_csv = results_dir / "rf_feature_importance.csv"
        importance_df.to_csv(importance_csv, index=False)
        print(f"\n✓ Saved importance scores to {importance_csv}")
        
        # Plot
        plot_path = plots_dir / "rf_feature_importance.png"
        plot_feature_importance(importance_df, plot_path, top_n=args.top_n)
        
        # Print top features
        print_top_features(importance_df, top_n=10)
        
        # Update README if requested
        if args.save_readme:
            readme_path = Path("README.md")
            plot_rel_path = Path("plots") / "rf_feature_importance.png"
            update_readme(importance_df, readme_path, plot_rel_path, top_n=10)
        
        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

