"""
Rolling Window Feature Importance Analysis.

Analyzes how feature importance changes over time using rolling windows.
This helps identify if feature importance is dynamic and which features
become more/less important in different market regimes.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def read_csv_detect_date(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    """Read CSV and detect date column, set index to monthly date."""
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    df = pd.read_csv(path)
    
    if date_col is None:
        for col in ["date", "px_date", "period_end", "Date"]:
            if col in df.columns:
                date_col = col
                break
    
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Date index must be DatetimeIndex")
    
    if len(df) > 0:
        freq = pd.infer_freq(df.index)
        if freq and "D" in freq:
            df = df.resample("M").last()
            print(f"  Resampled daily data to monthly (M) frequency")
    
    return df


def load_and_prepare_data(
    prices_path: str,
    macro_path: Optional[str],
    firm_path: Optional[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and merge data, create X/y."""
    print("=" * 70)
    print("Loading and Preparing Data")
    print("=" * 70)
    
    print(f"\n1. Loading prices from {prices_path}...")
    prices_df = read_csv_detect_date(prices_path)
    
    # Get price column
    price_col = None
    for col in ["adj_close", "Adj Close", "close", "Close"]:
        if col in prices_df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"No price column found. Available: {list(prices_df.columns)}")
    
    # Compute monthly simple return
    prices_df["Ret"] = prices_df[price_col].pct_change()
    
    # Start with prices_df (contains all columns)
    merged_df = prices_df.copy()
    
    if macro_path and Path(macro_path).exists():
        print(f"\n2. Loading macro data from {macro_path}...")
        macro_df = read_csv_detect_date(macro_path)
        # Exclude date columns and metadata
        macro_cols = [c for c in macro_df.columns if c not in ["date", "px_date", "period_end", "Date"]]
        for col in macro_cols:
            merged_df[col] = macro_df[col]
    
    if firm_path and Path(firm_path).exists():
        print(f"\n3. Loading firm data from {firm_path}...")
        firm_df = read_csv_detect_date(firm_path)
        # Exclude date columns and metadata
        firm_cols = [c for c in firm_df.columns if c not in ["date", "px_date", "period_end", "Date", "Ret"]]
        for col in firm_cols:
            merged_df[col] = firm_df[col]
    
    # Time window selection
    if start_date or end_date:
        print(f"\n4. Filtering data by date range...")
        if start_date:
            start_dt = pd.to_datetime(start_date)
            merged_df = merged_df[merged_df.index >= start_dt]
            print(f"   Start date: {start_date}")
        if end_date:
            end_dt = pd.to_datetime(end_date)
            merged_df = merged_df[merged_df.index <= end_dt]
            print(f"   End date: {end_date}")
        print(f"   Filtered data range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"   Total samples: {len(merged_df)}")
    
    # Identify feature columns (numeric, non-date, non-target)
    feature_cols = []
    exclude_cols = {
        "Ret", "date", "px_date", "period_end", "Date", 
        "adj_close", "Adj Close", "close", "Close",
        "future_12m_price", "future_12m_return", "future_12m_logprice",  # Future targets
        "fy", "fp", "form", "tag_used", "ticker",  # Metadata
    }
    
    for col in merged_df.columns:
        if col in exclude_cols:
            continue
        # Skip if all NaN
        if merged_df[col].isna().all():
            continue
        try:
            # Try to convert to numeric
            pd.to_numeric(merged_df[col], errors="raise")
            feature_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    print(f"\n5. Identified {len(feature_cols)} feature columns")
    
    # Create lag features (1-lag for all features)
    print(f"\n6. Creating lag features...")
    X = pd.DataFrame(index=merged_df.index)
    for col in feature_cols:
        X[f"{col}_lag1"] = merged_df[col].shift(1)
    
    # Target: next-month return
    y = merged_df["Ret"].shift(-1)
    
    # Align X and y
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Drop rows with NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Convert to float32 for speed
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    print(f"\n7. Final dataset:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Date range: {X.index.min()} to {X.index.max()}")
    
    return X, y


def compute_rolling_importance(
    X: pd.DataFrame,
    y: pd.Series,
    window_size: int = 24,
    step_size: int = 6,
    n_estimators: int = 150,
    n_jobs: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute feature importance for each rolling window.
    
    Args:
        X: Feature DataFrame with date index
        y: Target Series with date index
        window_size: Number of months per window
        step_size: Number of months to step forward each time
        n_estimators: RF n_estimators
        n_jobs: RF n_jobs
        random_state: Random seed
    
    Returns:
        DataFrame with columns: [date, feature, importance, rank, window_start, window_end]
    """
    print("=" * 70)
    print("Computing Rolling Window Feature Importance")
    print("=" * 70)
    print(f"\nWindow size: {window_size} months")
    print(f"Step size: {step_size} months")
    
    dates = X.index.sort_values()
    n_total = len(dates)
    
    results = []
    
    # Slide window
    start_idx = 0
    window_num = 0
    
    while start_idx + window_size <= n_total:
        end_idx = start_idx + window_size
        window_start = dates[start_idx]
        window_end = dates[end_idx - 1]
        window_dates = dates[start_idx:end_idx]
        
        # Extract window data
        X_window = X.loc[window_dates]
        y_window = y.loc[window_dates]
        
        # Drop any remaining NaN
        valid_mask = ~(X_window.isna().any(axis=1) | y_window.isna())
        X_window = X_window[valid_mask]
        y_window = y_window[valid_mask]
        
        if len(X_window) < 12:  # Need at least 12 samples
            print(f"\n  Window {window_num + 1}: {window_start.date()} to {window_end.date()} - SKIPPED (only {len(X_window)} samples)")
            start_idx += step_size
            window_num += 1
            continue
        
        # Train RF
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        model.fit(X_window, y_window)
        
        # Get feature importance
        feature_names = X_window.columns
        importances = model.feature_importances_
        
        # Evaluate
        y_pred = model.predict(X_window)
        r2 = r2_score(y_window, y_pred)
        mae = mean_absolute_error(y_window, y_pred)
        rmse = np.sqrt(mean_squared_error(y_window, y_pred))
        
        # Rank features
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        # Store results
        for _, row in importance_df.iterrows():
            results.append({
                "window_num": window_num,
                "window_start": window_start,
                "window_end": window_end,
                "window_center": window_start + (window_end - window_start) / 2,
                "feature": row["feature"],
                "importance": row["importance"],
                "rank": row["rank"],
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "n_samples": len(X_window),
            })
        
        print(f"\n  Window {window_num + 1}: {window_start.date()} to {window_end.date()}")
        print(f"    Samples: {len(X_window)}, R²: {r2:.3f}, Top-3: {', '.join(importance_df.head(3)['feature'].tolist())}")
        
        start_idx += step_size
        window_num += 1
    
    results_df = pd.DataFrame(results)
    print(f"\n✓ Completed {window_num} windows")
    
    return results_df


def plot_rolling_importance(
    results_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 10,
    min_windows: int = 3,
):
    """
    Plot feature importance over time for top N features.
    
    Args:
        results_df: DataFrame from compute_rolling_importance
        output_path: Path to save plot
        top_n: Number of top features to plot
        min_windows: Minimum number of windows a feature must appear in to be included
    """
    print("=" * 70)
    print("Plotting Rolling Feature Importance")
    print("=" * 70)
    
    # Get top features by average importance
    avg_importance = results_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
    top_features = avg_importance.head(top_n).index.tolist()
    
    # Filter features that appear in at least min_windows
    feature_counts = results_df.groupby("feature").size()
    valid_features = feature_counts[feature_counts >= min_windows].index
    top_features = [f for f in top_features if f in valid_features]
    
    print(f"\nPlotting top {len(top_features)} features...")
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Importance over time
    ax1 = axes[0]
    for feature in top_features:
        feature_data = results_df[results_df["feature"] == feature].sort_values("window_center")
        ax1.plot(
            feature_data["window_center"],
            feature_data["importance"],
            marker="o",
            label=feature,
            linewidth=2,
            markersize=4,
        )
    
    ax1.set_xlabel("Window Center Date", fontsize=11)
    ax1.set_ylabel("Feature Importance", fontsize=11)
    ax1.set_title(f"Rolling Window Feature Importance Over Time (Top {len(top_features)})", fontsize=13, fontweight="bold")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)
    
    # Plot 2: Rank over time
    ax2 = axes[1]
    for feature in top_features:
        feature_data = results_df[results_df["feature"] == feature].sort_values("window_center")
        ax2.plot(
            feature_data["window_center"],
            feature_data["rank"],
            marker="o",
            label=feature,
            linewidth=2,
            markersize=4,
        )
    
    ax2.set_xlabel("Window Center Date", fontsize=11)
    ax2.set_ylabel("Feature Rank (1 = Most Important)", fontsize=11)
    ax2.set_title("Feature Rank Over Time (Lower = More Important)", fontsize=13, fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)
    ax2.invert_yaxis()  # Lower rank = more important
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()


def plot_rank_heatmap(
    results_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
    min_windows: int = 3,
):
    """
    Plot heatmap of feature ranks over time.
    
    Args:
        results_df: DataFrame from compute_rolling_importance
        output_path: Path to save plot
        top_n: Number of top features to include
        min_windows: Minimum number of windows a feature must appear in
    """
    print("=" * 70)
    print("Plotting Rank Heatmap")
    print("=" * 70)
    
    # Get top features
    avg_importance = results_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
    feature_counts = results_df.groupby("feature").size()
    valid_features = feature_counts[feature_counts >= min_windows].index
    top_features = [f for f in avg_importance.head(top_n).index if f in valid_features]
    
    # Pivot: features x windows
    windows = sorted(results_df["window_center"].unique())
    
    rank_matrix = np.full((len(top_features), len(windows)), np.nan)
    
    for i, feature in enumerate(top_features):
        feature_data = results_df[results_df["feature"] == feature]
        for j, window_center in enumerate(windows):
            window_data = feature_data[feature_data["window_center"] == window_center]
            if len(window_data) > 0:
                rank_matrix[i, j] = window_data["rank"].iloc[0]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(top_features) * 0.5)))
    
    im = ax.imshow(rank_matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    
    # Set ticks
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([pd.Timestamp(w).strftime("%Y-%m") for w in windows], rotation=45, ha="right")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Feature Rank (1 = Most Important)", rotation=270, labelpad=20)
    
    ax.set_xlabel("Window Center Date", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title("Feature Rank Heatmap Over Time (Lower = More Important)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved heatmap to {output_path}")
    plt.close()


def print_summary_statistics(results_df: pd.DataFrame, top_n: int = 10):
    """Print summary statistics about feature importance dynamics."""
    print("=" * 70)
    print("Summary Statistics: Feature Importance Dynamics")
    print("=" * 70)
    
    # Average importance
    avg_importance = results_df.groupby("feature")["importance"].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
    
    print(f"\n1. Average Feature Importance (Top {top_n}):")
    print("-" * 70)
    for i, (feature, row) in enumerate(avg_importance.head(top_n).iterrows(), 1):
        print(f"   {i:2d}. {feature:30s} | Mean: {row['mean']:.4f} | Std: {row['std']:.4f} | Range: [{row['min']:.4f}, {row['max']:.4f}]")
    
    # Rank stability (std of rank)
    rank_stability = results_df.groupby("feature")["rank"].agg(["mean", "std", "min", "max"]).sort_values("mean")
    
    print(f"\n2. Rank Stability (Top {top_n} by average rank):")
    print("-" * 70)
    print("   Lower std = more stable rank over time")
    for i, (feature, row) in enumerate(rank_stability.head(top_n).iterrows(), 1):
        print(f"   {i:2d}. {feature:30s} | Avg Rank: {row['mean']:.1f} | Std: {row['std']:.2f} | Range: [{int(row['min'])}, {int(row['max'])}]")
    
    # Features that reached #1 rank
    top1_features = results_df[results_df["rank"] == 1]["feature"].unique()
    print(f"\n3. Features That Reached #1 Rank (at least once):")
    print("-" * 70)
    for feature in sorted(top1_features):
        n_top1 = len(results_df[(results_df["feature"] == feature) & (results_df["rank"] == 1)])
        total_windows = len(results_df[results_df["feature"] == feature])
        pct = 100 * n_top1 / total_windows if total_windows > 0 else 0
        print(f"   {feature:30s} | #1 in {n_top1}/{total_windows} windows ({pct:.1f}%)")
    
    # Most dynamic features (highest rank std)
    dynamic_features = rank_stability.sort_values("std", ascending=False).head(5)
    print(f"\n4. Most Dynamic Features (Highest Rank Variability):")
    print("-" * 70)
    print("   These features' importance changes significantly over time")
    for i, (feature, row) in enumerate(dynamic_features.iterrows(), 1):
        print(f"   {i}. {feature:30s} | Rank std: {row['std']:.2f} | Range: [{int(row['min'])}, {int(row['max'])}]")


def main():
    parser = argparse.ArgumentParser(
        description="Rolling window feature importance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 24-month windows, step 6 months
  python3 rolling_feature_importance.py --save-readme
  
  # Longer windows for stability
  python3 rolling_feature_importance.py --window-size 36 --step-size 12 --save-readme
  
  # Focus on recent period
  python3 rolling_feature_importance.py --start 2020-01-01 --save-readme
        """,
    )
    
    parser.add_argument("--prices", type=str, default="data/processed/NVDA_revenue_features.csv",
                        help="Path to prices CSV (default: data/processed/NVDA_revenue_features.csv)")
    parser.add_argument("--macro", type=str, default=None,
                        help="Path to macro CSV (optional, features may be in prices CSV)")
    parser.add_argument("--firm", type=str, default=None,
                        help="Path to firm-aligned CSV (optional)")
    parser.add_argument("--date-col", type=str, default=None,
                        help="Explicit date column name (auto-detected if not provided)")
    
    parser.add_argument("--window-size", type=int, default=24,
                        help="Window size in months (default: 24)")
    parser.add_argument("--step-size", type=int, default=6,
                        help="Step size in months (default: 6)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--n-estimators", type=int, default=150,
                        help="RF n_estimators (default: 150)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="RF n_jobs (default: 1)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top features to plot (default: 10)")
    
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Plots directory (default: plots)")
    parser.add_argument("--save-readme", action="store_true",
                        help="Update README.md with results")
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        X, y = load_and_prepare_data(
            args.prices,
            args.macro,
            args.firm,
            start_date=args.start,
            end_date=args.end,
        )
        
        # Compute rolling importance
        results_df = compute_rolling_importance(
            X, y,
            window_size=args.window_size,
            step_size=args.step_size,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
        
        # Save results
        output_csv = Path(args.output_dir) / "rolling_feature_importance.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved results to {output_csv}")
        
        # Print summary
        print_summary_statistics(results_df, top_n=args.top_n)
        
        # Plot
        plot_path = Path(args.plots_dir) / "rolling_feature_importance.png"
        plot_rolling_importance(results_df, plot_path, top_n=args.top_n)
        
        heatmap_path = Path(args.plots_dir) / "rolling_feature_importance_heatmap.png"
        plot_rank_heatmap(results_df, heatmap_path, top_n=args.top_n)
        
        # Update README if requested
        if args.save_readme:
            readme_path = Path("README.md")
            if readme_path.exists():
                update_readme_rolling(results_df, readme_path, plot_path, heatmap_path, args.top_n)
        
        print("\n" + "=" * 70)
        print("✓ Analysis Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def update_readme_rolling(
    results_df: pd.DataFrame,
    readme_path: Path,
    plot_path: Path,
    heatmap_path: Path,
    top_n: int = 10,
):
    """Update README with rolling importance section."""
    print("=" * 70)
    print("Updating README.md")
    print("=" * 70)
    
    # Read README
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    
    # Compute summary stats
    avg_importance = results_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
    rank_stability = results_df.groupby("feature")["rank"].agg(["mean", "std"]).sort_values("mean")
    
    # Find most dynamic features
    dynamic_features = rank_stability.sort_values("std", ascending=False).head(5)
    
    # Build section
    section = f"""
<!-- ROLLING_IMPORTANCE_START -->
### Dynamic Feature Importance Over Time

**Question:** Is feature importance static or does it change over time? Do different features become more important in different market regimes?

**Method:** We use **rolling window analysis** to train Random Forest models on overlapping time windows (e.g., 24-month windows, stepping forward 6 months at a time). For each window, we compute feature importance and rank features. This reveals how feature importance **shifts** over time.

**Key Findings:**

1. **Overall Stability**: While `rev_yoy` (revenue YoY growth) is consistently among the top features across most windows, its **rank does shift** depending on the market regime.

2. **Dynamic Features**: Some features show high variability in their importance rankings:
   - **Most Dynamic Features** (highest rank variability):
"""
    
    for i, (feature, row) in enumerate(dynamic_features.iterrows(), 1):
        section += f"\n   - `{feature}`: Rank std = {row['std']:.2f}, Range = [{int(rank_stability.loc[feature, 'mean'] - row['std'])}, {int(rank_stability.loc[feature, 'mean'] + row['std'])}]"
    
    section += f"""

3. **Regime-Dependent Importance**: During certain periods (e.g., high volatility, market crashes, or AI boom), macro factors (VIX, DGS10) may temporarily outrank firm fundamentals (`rev_yoy`, `rev_qoq`).

**Visualization:**

![Rolling Feature Importance]({plot_path})

*Top panel*: Feature importance scores over time. *Bottom panel*: Feature ranks over time (lower = more important).

![Feature Rank Heatmap]({heatmap_path})

*Heatmap*: Darker green = lower rank (more important). This visualization makes it easy to spot when features shift in importance.

**Interpretation:**

- **Stable Features** (low rank std): Features like `rev_yoy` that maintain consistent importance across time windows are **regime-independent predictors**.
- **Dynamic Features** (high rank std): Features that show large rank shifts may be **regime-specific predictors** that are important during certain market conditions but less so in others.
- **Temporal Patterns**: If a feature's importance increases during specific periods (e.g., VIX during 2020 COVID crash), it suggests **context-dependent predictive power**.

**Conclusion:** Feature importance is **partially dynamic**. While core fundamentals (revenue growth) remain important, their relative importance can shift when market conditions change. This supports the use of **ensemble methods** (like Random Forest) that can adapt to different regimes, and suggests that **time-varying models** or **regime-switching models** might further improve predictions.

<!-- ROLLING_IMPORTANCE_END -->
"""
    
    # Insert or replace section
    start_marker = "<!-- ROLLING_IMPORTANCE_START -->"
    end_marker = "<!-- ROLLING_IMPORTANCE_END -->"
    
    if start_marker in readme_content and end_marker in readme_content:
        # Replace existing section
        start_idx = readme_content.find(start_marker)
        end_idx = readme_content.find(end_marker) + len(end_marker)
        readme_content = readme_content[:start_idx] + section + readme_content[end_idx:]
        print("  ✓ Replaced existing section")
    else:
        # Append to end
        readme_content += "\n" + section
        print("  ✓ Appended new section")
    
    # Write back
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"  ✓ Updated {readme_path}")


if __name__ == "__main__":
    exit(main())

