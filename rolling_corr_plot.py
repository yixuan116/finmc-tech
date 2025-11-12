"""
Rolling correlation analysis between model predictions and macro factors.

Computes and visualizes rolling correlations between predictions and macro variables
over time to identify regime-dependent relationships.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_merge_data(
    predictions_path: str,
    macro_path: str,
) -> pd.DataFrame:
    """
    Load predictions and macro data, merge on date.
    
    Args:
        predictions_path: Path to predictions CSV (columns: date, y_pred, y_true)
        macro_path: Path to macro CSV (columns: date or index, VIX, FedFunds, CPI_yoy, etc.)
    
    Returns:
        Merged DataFrame sorted chronologically
    """
    # Load predictions
    pred_df = pd.read_csv(predictions_path, parse_dates=["date"])
    print(f"Loaded predictions: {len(pred_df)} rows")
    
    # Load macro data - handle both date column and index
    macro_df = pd.read_csv(macro_path)
    
    # Check if date is in index or column
    if "date" in macro_df.columns:
        macro_df["date"] = pd.to_datetime(macro_df["date"])
    elif "Unnamed: 0" in macro_df.columns:
        # Date is in first column (index)
        macro_df["date"] = pd.to_datetime(macro_df["Unnamed: 0"])
        macro_df = macro_df.drop(columns=["Unnamed: 0"])
    elif macro_df.index.name is None or macro_df.index.name == "Unnamed: 0":
        # Try to parse index as date
        try:
            macro_df["date"] = pd.to_datetime(macro_df.index)
            macro_df = macro_df.reset_index(drop=True)
        except:
            raise ValueError("Cannot find date column or index in macro data")
    
    print(f"Loaded macro data: {len(macro_df)} rows")
    
    # Merge on date (inner join)
    merged = pd.merge(
        pred_df,
        macro_df,
        on="date",
        how="inner",
        sort=True,
    )
    
    print(f"Merged data: {len(merged)} rows")
    if len(merged) > 0:
        print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    
    return merged


def compute_rolling_correlations(
    df: pd.DataFrame,
    window: int = 90,
) -> pd.DataFrame:
    """
    Compute rolling correlations between y_pred and each macro feature.
    
    Args:
        df: Merged DataFrame with date, y_pred, and macro columns
        window: Rolling window size in days
    
    Returns:
        DataFrame with date and correlation columns (corr_VIX, corr_FedFunds, etc.)
    """
    # Identify macro columns (exclude date, y_pred, y_true)
    exclude_cols = {"date", "y_pred", "y_true"}
    macro_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not macro_cols:
        raise ValueError("No macro columns found in merged data")
    
    print(f"\nComputing rolling correlations (window={window} days) for {len(macro_cols)} macro variables:")
    for col in macro_cols:
        print(f"  • {col}")
    
    # Initialize result DataFrame
    result = pd.DataFrame({"date": df["date"]})
    
    # Compute rolling correlation for each macro variable
    for col in macro_cols:
        corr_name = f"corr_{col}"
        result[corr_name] = df["y_pred"].rolling(window=window).corr(df[col])
    
    # Drop rows with all NaN correlations (before window is filled)
    result = result.dropna(how="all")
    
    print(f"\nComputed correlations: {len(result)} valid rows")
    
    return result


def plot_rolling_correlations(
    corr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot rolling correlation time series.
    
    Args:
        corr_df: DataFrame with date and correlation columns
        output_path: Path to save figure
    """
    # Get correlation columns
    corr_cols = [col for col in corr_df.columns if col.startswith("corr_")]
    
    if not corr_cols:
        raise ValueError("No correlation columns found")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each correlation series
    colors = plt.cm.tab10(np.linspace(0, 1, len(corr_cols)))
    
    for i, col in enumerate(corr_cols):
        macro_name = col.replace("corr_", "")
        ax.plot(
            corr_df["date"],
            corr_df[col],
            label=macro_name,
            linewidth=1.5,
            alpha=0.8,
            color=colors[i],
        )
    
    # Add zero line
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Zero")
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rolling Correlation Coefficient", fontsize=12)
    ax.set_title("Rolling Correlation of Model Predictions vs Macro Factors", fontsize=14, fontweight="bold")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()


def analyze_correlations(corr_df: pd.DataFrame) -> None:
    """
    Print correlation statistics and top periods.
    
    Args:
        corr_df: DataFrame with correlation columns
    """
    corr_cols = [col for col in corr_df.columns if col.startswith("corr_")]
    
    print("\n" + "=" * 70)
    print("Correlation Analysis")
    print("=" * 70)
    
    # Average correlation per macro variable
    print("\nAverage Correlation (Full Sample):")
    print("-" * 70)
    avg_corrs = {}
    for col in corr_cols:
        macro_name = col.replace("corr_", "")
        avg_corr = corr_df[col].mean()
        avg_corrs[macro_name] = avg_corr
        print(f"  {macro_name:20s}: {avg_corr:7.4f}")
    
    # Top 3 periods with highest correlation magnitude
    print("\nTop 3 Periods (Highest Correlation Magnitude):")
    print("-" * 70)
    
    for col in corr_cols:
        macro_name = col.replace("corr_", "")
        
        # Compute absolute correlation
        abs_corr = corr_df[col].abs()
        
        # Find top 3 periods
        top_indices = abs_corr.nlargest(3).index
        
        print(f"\n  {macro_name}:")
        for idx in top_indices:
            date = corr_df.loc[idx, "date"]
            corr_value = corr_df.loc[idx, col]
            
            # Find date range (window size)
            window_start = max(0, idx - 90)
            window_end = min(len(corr_df) - 1, idx + 90)
            start_date = corr_df.loc[window_start, "date"]
            end_date = corr_df.loc[window_end, "date"]
            
            print(f"    {start_date.date()} to {end_date.date()}: {corr_value:7.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute and plot rolling correlations between predictions and macro factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--pred",
        type=str,
        default="results/predictions.csv",
        help="Path to predictions CSV (default: results/predictions.csv)",
    )
    parser.add_argument(
        "--macro",
        type=str,
        default="results/macro_data_combined.csv",
        help="Path to macro CSV (default: results/macro_data_combined.csv)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=90,
        help="Rolling window size in days (default: 90)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/rolling_corr.png",
        help="Output path for plot (default: plots/rolling_corr.png)",
    )
    
    args = parser.parse_args()
    
    # Validate input files
    pred_path = Path(args.pred)
    macro_path = Path(args.macro)
    
    if not pred_path.exists():
        print(f"Error: Predictions file not found: {pred_path}")
        sys.exit(1)
    
    if not macro_path.exists():
        print(f"Error: Macro file not found: {macro_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Rolling Correlation Analysis")
    print("=" * 70)
    
    # Load and merge data
    merged_df = load_and_merge_data(str(pred_path), str(macro_path))
    
    if len(merged_df) == 0:
        print("Error: No overlapping dates between predictions and macro data")
        sys.exit(1)
    
    # Compute rolling correlations
    corr_df = compute_rolling_correlations(merged_df, window=args.window)
    
    # Plot
    output_path = Path(args.output)
    plot_rolling_correlations(corr_df, output_path)
    
    # Analyze
    analyze_correlations(corr_df)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

