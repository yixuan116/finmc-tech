#!/usr/bin/env python3
"""
Create extended features dataset for AMD (similar to NVDA).

This script:
1. Loads AMD revenue data (or creates base structure)
2. Adds price data from Yahoo Finance
3. Creates all extended features (price, macro, time, interactions)
4. Exports to amd_features_extended.csv

NOTE: This script does NOT modify or overwrite NVDA data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

from src.data.create_extended_features import (
    create_all_extended_features,
    create_price_momentum_features,
    create_market_macro_features,
    create_time_features,
    create_interaction_features,
)
from src.data.create_nvda_revenue_features import (
    load_revenue_data,
    add_price_data,
    create_revenue_features,
    create_forward_return_target,
)


def load_amd_base_data() -> pd.DataFrame:
    """
    Load AMD base data (revenue + price).

    Tries to load from existing files, or creates base structure from scratch.

    Returns
    -------
    pd.DataFrame
        Base data with revenue, price, and date columns
    """
    print("=" * 80)
    print("Loading AMD Base Data")
    print("=" * 80)

    # Try to load from existing revenue files
    outputs_dir = Path("outputs")
    possible_files = [
        outputs_dir / "revenues_panel_amd.csv",
        outputs_dir / "revenues_amd_with_prices.csv",
        outputs_dir / "amd_revenue_features.csv",
    ]

    df = None
    for file_path in possible_files:
        if file_path.exists():
            print(f"✓ Found existing file: {file_path}")
            df = pd.read_csv(file_path)

            # Ensure we have required columns
            if "px_date" not in df.columns and "date" in df.columns:
                df["px_date"] = pd.to_datetime(df["date"])
            elif "px_date" not in df.columns:
                raise ValueError(f"File {file_path} missing date column")
            break

    # If no existing file, create base structure from scratch
    if df is None:
        print("⚠ No existing AMD revenue file found. Creating base structure from Yahoo Finance...")
        df = create_amd_base_from_yahoo()

    # Ensure required columns exist
    required_cols = ["px_date", "adj_close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠ Missing columns: {missing}. Adding price data...")
        df = add_price_data(df, ticker="AMD")

    # Ensure ticker column
    if "ticker" not in df.columns:
        df["ticker"] = "AMD"

    # Sort by date
    df["px_date"] = pd.to_datetime(df["px_date"])
    df = df.sort_values("px_date").reset_index(drop=True)

    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Date range: {df['px_date'].min()} to {df['px_date'].max()}")

    return df


def create_amd_base_from_yahoo() -> pd.DataFrame:
    """
    Create base AMD data structure by downloading from Yahoo Finance.

    This is a fallback if no revenue data file exists.
    """
    print("  Downloading AMD data from Yahoo Finance...")
    ticker = "AMD"
    stock = yf.Ticker(ticker)

    # Get historical data (max period)
    hist = stock.history(period="max")
    if hist.empty:
        raise ValueError(f"Could not download data for {ticker}")

    # Reset index to get Date as column
    hist.reset_index(inplace=True)
    hist.columns = [col.lower().strip() for col in hist.columns]

    # Create quarterly aggregation
    hist["date"] = pd.to_datetime(hist["date"])
    hist["year"] = hist["date"].dt.year
    hist["quarter"] = hist["date"].dt.quarter

    # Find adj_close column (could be "adj close" or "adj_close")
    adj_close_col = None
    for col in hist.columns:
        if "adj" in col.lower() and "close" in col.lower():
            adj_close_col = col
            break

    if adj_close_col is None:
        # Fallback to close if adj_close not found
        adj_close_col = "close"
        print(f"  ⚠ Using 'close' instead of adjusted close")

    # Aggregate to quarterly (use last trading day of quarter)
    agg_dict = {
        "close": "last",
        "date": "last",
    }
    if adj_close_col != "close":
        agg_dict[adj_close_col] = "last"

    quarterly = hist.groupby(["year", "quarter"]).agg(agg_dict).reset_index()

    # Rename columns
    quarterly.rename(
        columns={
            adj_close_col: "adj_close",
            "date": "px_date",
        },
        inplace=True,
    )

    # Add period_end (approximate as quarter end)
    quarterly["period_end"] = pd.to_datetime(
        quarterly.apply(
            lambda row: f"{row['year']}-{row['quarter']*3:02d}-01",
            axis=1,
        )
    ) + pd.offsets.QuarterEnd()

    # Add revenue placeholder (will need to be filled from SEC data)
    quarterly["revenue"] = np.nan
    quarterly["ticker"] = ticker

    print(f"  ✓ Created {len(quarterly)} quarterly records")
    print(f"  ⚠ Note: Revenue data is missing - you may need to add SEC XBRL data")

    return quarterly


def main():
    """Main function to create AMD extended features dataset."""
    print("=" * 80)
    print("Creating AMD Extended Features Dataset")
    print("=" * 80)
    print("\nNOTE: This will NOT modify or overwrite NVDA data files.\n")

    # Output path
    output_path = Path("data/processed/amd_features_extended.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if output_path.exists():
        response = input(f"⚠ File {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted. Existing file preserved.")
            return

    # Load base data
    df = load_amd_base_data()

    # Create revenue features (if revenue data exists)
    if "revenue" in df.columns and not df["revenue"].isna().all():
        print("\n" + "=" * 80)
        print("Creating Revenue Features")
        print("=" * 80)
        df = create_revenue_features(df)
        df = create_forward_return_target(df, trading_days=252)
    else:
        print("\n⚠ Revenue data missing. Skipping revenue features.")
        print("  You may need to add SEC XBRL revenue data manually.")

        # Add placeholder columns to match structure
        for col in ["rev_qoq", "rev_yoy", "rev_accel"]:
            if col not in df.columns:
                df[col] = np.nan

    # Always create future return targets (even without revenue data)
    # Ensure end_date column exists (function expects it)
    if "end_date" not in df.columns:
        if "period_end" in df.columns:
            df["end_date"] = pd.to_datetime(df["period_end"])
        else:
            df["end_date"] = pd.to_datetime(df["px_date"])

    if "future_12m_return" not in df.columns or df["future_12m_return"].isna().all():
        print("\n" + "=" * 80)
        print("Creating Future Return Targets")
        print("=" * 80)
        df = create_forward_return_target(df, trading_days=252)

    # Create all extended features
    print("\n" + "=" * 80)
    print("Creating Extended Features")
    print("=" * 80)
    df = create_all_extended_features(df, ticker="AMD")

    # Ensure all required columns exist (match NVDA structure)
    print("\n" + "=" * 80)
    print("Validating Column Structure")
    print("=" * 80)

    # Load NVDA to compare structure
    nvda_path = Path("data/processed/nvda_features_extended.csv")
    if nvda_path.exists():
        nvda_df = pd.read_csv(nvda_path, nrows=1)
        nvda_cols = set(nvda_df.columns)
        amd_cols = set(df.columns)

        missing_in_amd = nvda_cols - amd_cols
        extra_in_amd = amd_cols - nvda_cols

        if missing_in_amd:
            print(f"⚠ Missing columns in AMD ({len(missing_in_amd)}):")
            for col in sorted(missing_in_amd)[:10]:
                print(f"  - {col}")
            if len(missing_in_amd) > 10:
                print(f"  ... and {len(missing_in_amd) - 10} more")

        if extra_in_amd:
            print(f"⚠ Extra columns in AMD ({len(extra_in_amd)}):")
            for col in sorted(extra_in_amd)[:10]:
                print(f"  - {col}")
            if len(extra_in_amd) > 10:
                print(f"  ... and {len(extra_in_amd) - 10} more")

        print(f"\n✓ Common columns: {len(nvda_cols & amd_cols)}")
    else:
        print("⚠ NVDA file not found for comparison")

    # Save
    print("\n" + "=" * 80)
    print("Saving Dataset")
    print("=" * 80)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    print(f"✓ Rows: {len(df)}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"✓ Date range: {df['px_date'].min()} to {df['px_date'].max()}")

    print("\n" + "=" * 80)
    print("✓ AMD Extended Features Dataset Created Successfully!")
    print("=" * 80)
    print(f"\nNext step: Run compare_nvda_amd.py to compare NVDA vs AMD")


if __name__ == "__main__":
    main()
