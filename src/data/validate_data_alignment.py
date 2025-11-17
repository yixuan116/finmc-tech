"""
Validate data alignment and consistency across all features.

This module checks:
1. All features are aligned to the same px_date
2. Data frequency consistency (quarterly)
3. Unit consistency
4. Missing data patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def validate_alignment(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate that all features are properly aligned.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe with px_date, end_date, and all features
    
    Returns
    -------
    Dict[str, any]
        Validation results with alignment status and issues
    """
    results = {
        "aligned": True,
        "issues": [],
        "warnings": [],
        "stats": {},
    }
    
    # Check 1: px_date consistency
    if "px_date" not in df.columns:
        results["aligned"] = False
        results["issues"].append("Missing px_date column")
        return results
    
    # Check 2: All features should be aligned to px_date
    # (This is a conceptual check - we can't verify the actual alignment logic here)
    px_date_coverage = df["px_date"].notna().sum() / len(df)
    results["stats"]["px_date_coverage"] = px_date_coverage
    
    if px_date_coverage < 0.9:
        results["warnings"].append(f"Low px_date coverage: {px_date_coverage:.1%}")
    
    # Check 3: Frequency consistency
    # All data should be quarterly (one record per quarter)
    if "end_date" in df.columns:
        df_dates = df[df["end_date"].notna()].copy()
        df_dates["end_date"] = pd.to_datetime(df_dates["end_date"])
        df_dates = df_dates.sort_values("end_date")
        
        # Check if dates are roughly quarterly spaced
        date_diffs = df_dates["end_date"].diff().dt.days
        median_diff = date_diffs.median()
        
        results["stats"]["median_days_between_records"] = median_diff
        
        # Quarterly should be ~90 days
        if not (60 <= median_diff <= 120):
            results["warnings"].append(
                f"Date spacing not quarterly: median {median_diff:.0f} days "
                f"(expected ~90 days)"
            )
    
    # Check 4: Feature availability
    feature_categories = {
        "revenue": ["rev_qoq", "rev_yoy", "rev_accel"],
        "macro": ["vix_level", "tnx_yield", "vix_change_3m", "tnx_change_3m"],
        "price_momentum": [
            "price_returns_1m", "price_returns_3m", "price_returns_6m",
            "price_returns_12m", "price_momentum", "price_volatility"
        ],
        "technical": ["rsi_14", "macd", "macd_signal", "bb_position", "stoch_k", "atr"],
        "market": ["sp500_level", "sp500_returns"],
        "time": ["quarter", "month", "year"],
    }
    
    feature_availability = {}
    for category, features in feature_categories.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        
        feature_availability[category] = {
            "available": len(available),
            "total": len(features),
            "missing": missing,
            "coverage": len(available) / len(features) if len(features) > 0 else 0,
        }
    
    results["stats"]["feature_availability"] = feature_availability
    
    # Check 5: Missing data patterns
    missing_data = {}
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            missing_pct = df[available_features].isna().sum().sum() / (len(df) * len(available_features))
            missing_data[category] = {
                "missing_pct": missing_pct,
                "features_with_missing": df[available_features].isna().sum().to_dict(),
            }
    
    results["stats"]["missing_data"] = missing_data
    
    # Check 6: Unit consistency (conceptual)
    # Prices should be in USD
    price_features = ["adj_close", "sp500_level"]
    for feat in price_features:
        if feat in df.columns:
            # Check if values are reasonable (not negative, not too large)
            values = df[feat].dropna()
            if len(values) > 0:
                if values.min() < 0:
                    results["warnings"].append(f"{feat} has negative values")
                if values.max() > 1e6:
                    results["warnings"].append(f"{feat} has unusually large values")
    
    # Check 7: Ratio consistency
    # Ratios should be between reasonable bounds
    ratio_features = ["rev_qoq", "rev_yoy", "price_returns_1m", "price_returns_12m"]
    for feat in ratio_features:
        if feat in df.columns:
            values = df[feat].dropna()
            if len(values) > 0:
                # Ratios can be negative or positive, but should be reasonable
                if abs(values.max()) > 10 or abs(values.min()) < -0.9:
                    results["warnings"].append(
                        f"{feat} has extreme values: min={values.min():.2f}, max={values.max():.2f}"
                    )
    
    return results


def print_validation_report(results: Dict[str, any]) -> None:
    """Print validation report in readable format."""
    print("=" * 70)
    print("Data Alignment Validation Report")
    print("=" * 70)
    
    if results["aligned"]:
        print("\n✓ Basic alignment check passed")
    else:
        print("\n✗ Alignment check failed")
        for issue in results["issues"]:
            print(f"  - {issue}")
    
    # Stats
    print("\nStatistics:")
    print("-" * 70)
    
    if "px_date_coverage" in results["stats"]:
        coverage = results["stats"]["px_date_coverage"]
        print(f"px_date coverage: {coverage:.1%}")
    
    if "median_days_between_records" in results["stats"]:
        median = results["stats"]["median_days_between_records"]
        print(f"Median days between records: {median:.0f} days (quarterly should be ~90 days)")
    
    # Feature availability
    if "feature_availability" in results["stats"]:
        print("\nFeature Availability:")
        print("-" * 70)
        for category, info in results["stats"]["feature_availability"].items():
            coverage = info["coverage"]
            status = "✓" if coverage >= 0.8 else "⚠" if coverage >= 0.5 else "✗"
            print(f"{status} {category:20s}: {info['available']}/{info['total']} ({coverage:.1%})")
            if info["missing"]:
                print(f"    Missing: {', '.join(info['missing'])}")
    
    # Missing data
    if "missing_data" in results["stats"]:
        print("\nMissing Data Statistics:")
        print("-" * 70)
        for category, info in results["stats"]["missing_data"].items():
            missing_pct = info["missing_pct"]
            status = "✓" if missing_pct < 0.1 else "⚠" if missing_pct < 0.3 else "✗"
            print(f"{status} {category:20s}: {missing_pct:.1%} missing")
            if info["features_with_missing"]:
                high_missing = {
                    k: v for k, v in info["features_with_missing"].items()
                    if v > 0
                }
                if high_missing:
                    print(f"    Main missing features: {', '.join(high_missing.keys())}")
    
    # Warnings
    if results["warnings"]:
        print("\nWarnings:")
        print("-" * 70)
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Load data
    data_file = Path("outputs") / "nvda_revenue_features.csv"
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("Please run create_nvda_revenue_features.py first")
        exit(1)
    
    df = pd.read_csv(data_file)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["px_date"] = pd.to_datetime(df["px_date"])
    
    # Validate
    results = validate_alignment(df)
    
    # Print report
    print_validation_report(results)

