"""
Create revenue features for NVDA modeling analysis.

This script:
1. Loads revenue data (from revenues_panel_nvda.csv or revenues_nvda_with_prices.csv)
2. Adds price data if not present
3. Creates revenue features (YoY, QoQ, acceleration)
4. Creates 12-month forward return target
5. Exports to nvda_revenue_features.csv
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


def load_revenue_data(input_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load revenue data from CSV file.
    
    Tries to load from revenues_panel_nvda.csv first, then falls back to
    revenues_nvda_with_prices.csv if available.
    
    Parameters
    ----------
    input_file : Optional[str]
        Explicit input file path. If None, tries default locations.
    
    Returns
    -------
    pd.DataFrame
        Revenue data with standard columns
    """
    outputs_dir = Path("outputs")
    
    if input_file:
        file_path = Path(input_file)
    else:
        # Try revenues_panel_nvda.csv first
        file_path = outputs_dir / "revenues_panel_nvda.csv"
        if not file_path.exists():
            # Fall back to revenues_nvda_with_prices.csv
            file_path = outputs_dir / "revenues_nvda_with_prices.csv"
            if not file_path.exists():
                raise FileNotFoundError(
                    "Neither revenues_panel_nvda.csv nor revenues_nvda_with_prices.csv found in outputs/"
                )
    
    print(f"Loading revenue data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Standardize column names
    column_mapping = {
        "period_end": "end_date",
        "fp": "period",
        "tag_used": "tag",
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Ensure required columns exist
    required_cols = ["ticker", "end_date", "revenue"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert end_date to datetime
    df["end_date"] = pd.to_datetime(df["end_date"])
    
    # Convert revenue to numeric
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    
    # Keep only USD rows (if unit column exists)
    if "unit" in df.columns:
        df = df[df["unit"] == "USD"].copy()
        print(f"  Kept {len(df)} rows with USD units")
    
    # Drop rows with missing revenue
    df = df[df["revenue"].notna()].copy()
    
    print(f"  Loaded {len(df)} revenue records")
    print(f"  Date range: {df['end_date'].min().date()} to {df['end_date'].max().date()}")
    
    return df


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by end_date and drop duplicates per (ticker, end_date).
    
    If both 10-K and 10-Q exist for the same end_date, keep 10-K.
    
    Parameters
    ----------
    df : pd.DataFrame
        Revenue data
    
    Returns
    -------
    pd.DataFrame
        Cleaned data with duplicates removed
    """
    print("\nCleaning and deduplicating data...")
    
    initial_count = len(df)
    
    # Sort by end_date
    df = df.sort_values("end_date").copy()
    
    # If form column exists, prioritize 10-K over 10-Q
    if "form" in df.columns:
        # Create priority: 10-K = 2, 10-Q = 1, other = 0
        df["form_priority"] = df["form"].map(
            lambda x: 2 if x == "10-K" else (1 if x == "10-Q" else 0)
        )
        # Sort by end_date, then by form_priority (descending)
        df = df.sort_values(["end_date", "form_priority"], ascending=[True, False])
        df = df.drop(columns=["form_priority"])
    
    # Drop duplicates per (ticker, end_date), keeping first (which is 10-K if available)
    df = df.drop_duplicates(subset=["ticker", "end_date"], keep="first")
    
    final_count = len(df)
    removed = initial_count - final_count
    
    print(f"  Removed {removed} duplicate(s), kept {final_count} unique records")
    
    return df


def add_price_data(df: pd.DataFrame, ticker: str = "NVDA") -> pd.DataFrame:
    """
    Download Adjusted Close prices and merge with revenue data.
    
    If price data already exists (px_date, adj_close), skip download.
    
    Parameters
    ----------
    df : pd.DataFrame
        Revenue data
    ticker : str
        Stock ticker symbol
    
    Returns
    -------
    pd.DataFrame
        Data with price columns added
    """
    # Check if price data already exists
    if "px_date" in df.columns and "adj_close" in df.columns:
        price_matched = df["adj_close"].notna().sum()
        if price_matched > 0:
            print(f"\nPrice data already exists: {price_matched}/{len(df)} rows matched")
            return df
    
    print(f"\nDownloading price data for {ticker}...")
    
    # Get date range with ±10 day buffer
    min_date = df["end_date"].min() - timedelta(days=10)
    max_date = df["end_date"].max() + timedelta(days=10)
    
    print(f"  Date range: {min_date.date()} to {max_date.date()}")
    
    # Download price data
    stock = yf.Ticker(ticker)
    try:
        price_data = stock.history(start=min_date, end=max_date + timedelta(days=1))
    except Exception as e:
        raise ConnectionError(f"Failed to fetch price data: {str(e)}")
    
    if price_data.empty:
        raise ValueError(f"No price data found for {ticker}")
    
    # Reset index to get Date as column
    price_data = price_data.reset_index()
    price_data["Date"] = pd.to_datetime(price_data["Date"]).dt.date
    
    print(f"  Downloaded {len(price_data)} trading days")
    
    # For each revenue end_date, align to nearest next trading day
    def get_next_trading_day(date: pd.Timestamp, max_days: int = 5) -> Optional[datetime.date]:
        """Find next trading day (weekday) within max_days."""
        for i in range(1, max_days + 1):
            candidate = date + timedelta(days=i)
            if candidate.weekday() < 5:  # Monday=0, Friday=4
                return candidate.date()
        return None
    
    merged_data = []
    for _, row in df.iterrows():
        period_end = pd.Timestamp(row["end_date"])
        next_trading_day = get_next_trading_day(period_end, max_days=5)
        
        if next_trading_day is None:
            merged_data.append({
                **row.to_dict(),
                "px_date": None,
                "adj_close": None,
            })
            continue
        
        # Find matching price
        matching_prices = price_data[price_data["Date"] == next_trading_day]
        
        if len(matching_prices) > 0:
            adj_close = matching_prices.iloc[0]["Close"]  # yfinance uses Close for adjusted close
            merged_data.append({
                **row.to_dict(),
                "px_date": next_trading_day,
                "adj_close": adj_close,
            })
        else:
            merged_data.append({
                **row.to_dict(),
                "px_date": next_trading_day,
                "adj_close": None,
            })
    
    result_df = pd.DataFrame(merged_data)
    
    # Count successful matches
    successful_matches = result_df["adj_close"].notna().sum()
    print(f"  Successfully matched prices for {successful_matches}/{len(result_df)} records")
    
    return result_df


def create_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create revenue features grouped by ticker.
    
    Features:
    - rev_yoy: Year-over-year revenue change (pct_change(4) for quarterly)
    - rev_qoq: Quarter-over-quarter revenue change (pct_change(1))
    - rev_accel: Acceleration of YoY growth (diff of rev_yoy)
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with revenue and price columns
    
    Returns
    -------
    pd.DataFrame
        Data with revenue features added
    """
    print("\nCreating revenue features...")
    
    df = df.copy()
    df = df.sort_values(["ticker", "end_date"])
    
    # Group by ticker for feature calculation
    grouped = df.groupby("ticker")
    
    # rev_yoy: Year-over-year change
    # For quarterly data, use pct_change(4) to compare same quarter year-over-year
    # For annual data, we need to handle differently
    df["rev_yoy"] = grouped["revenue"].pct_change(4)
    
    # rev_qoq: Quarter-over-quarter change
    df["rev_qoq"] = grouped["revenue"].pct_change(1)
    
    # rev_accel: Acceleration of YoY growth
    df["rev_accel"] = grouped["rev_yoy"].diff(1)
    
    print(f"  Created features: rev_yoy, rev_qoq, rev_accel")
    
    return df


def create_forward_return_target(df: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Create 12-month forward return target using specified trading days.
    
    For each revenue record, find the price 252 trading days after px_date.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with price columns
    trading_days : int
        Number of trading days for forward return (default: 252 for 12 months)
    
    Returns
    -------
    pd.DataFrame
        Data with future_12m_return column added
    """
    print(f"\nCreating {trading_days}-day forward return target...")
    
    df = df.copy()
    
    # Filter to rows with valid price data
    df_with_price = df[df["adj_close"].notna() & df["px_date"].notna()].copy()
    
    if len(df_with_price) == 0:
        print("  Warning: No price data available for forward return calculation")
        df["future_12m_return"] = None
        return df
    
    # Download full price history to find future prices
    ticker = df_with_price["ticker"].iloc[0] if "ticker" in df_with_price.columns else "NVDA"
    
    # Convert px_date to datetime if it's a string
    df_with_price["px_date_dt"] = pd.to_datetime(df_with_price["px_date"])
    
    print(f"  Downloading full price history for {ticker}...")
    min_date = df_with_price["px_date_dt"].min() - timedelta(days=10)
    max_date = df_with_price["px_date_dt"].max() + timedelta(days=365)  # Extra buffer for forward dates
    
    stock = yf.Ticker(ticker)
    try:
        price_history = stock.history(start=min_date, end=max_date + timedelta(days=1))
    except Exception as e:
        print(f"  Warning: Could not download price history: {e}")
        df["future_12m_return"] = None
        return df
    
    if price_history.empty:
        print("  Warning: No price history available")
        df["future_12m_return"] = None
        return df
    
    # Reset index and prepare price lookup
    price_history = price_history.reset_index()
    price_history["Date"] = pd.to_datetime(price_history["Date"]).dt.date
    price_history = price_history.sort_values("Date")
    price_history = price_history[["Date", "Close"]].copy()
    price_history.columns = ["date", "price"]
    
    print(f"  Loaded {len(price_history)} trading days")
    
    # Calculate forward return for each revenue record
    future_returns = []
    
    for _, row in df_with_price.iterrows():
        px_date = pd.to_datetime(row["px_date"]).date()
        current_price = row["adj_close"]
        
        # Find the row in price_history with this date
        matching_rows = price_history[price_history["date"] == px_date]
        
        if len(matching_rows) == 0:
            future_returns.append(None)
            continue
        
        # Find index of this date
        current_idx = matching_rows.index[0]
        future_idx = current_idx + trading_days
        
        # Check if future date exists
        if future_idx < len(price_history):
            future_date = price_history.iloc[future_idx]["date"]
            future_price = price_history.iloc[future_idx]["price"]
            forward_return = (future_price / current_price) - 1
            future_returns.append(forward_return)
        else:
            future_returns.append(None)
    
    df_with_price["future_12m_return"] = future_returns
    
    # Merge back to original dataframe
    df = df.merge(
        df_with_price[["end_date", "future_12m_return"]],
        on="end_date",
        how="left",
        suffixes=("", "_new")
    )
    
    # Use the merged future_12m_return if it exists
    if "future_12m_return_new" in df.columns:
        df["future_12m_return"] = df["future_12m_return_new"]
        df = df.drop(columns=["future_12m_return_new"])
    
    # Count valid forward returns
    valid_returns = df["future_12m_return"].notna().sum()
    print(f"  Created future_12m_return for {valid_returns}/{len(df)} records")
    
    return df


def main():
    """Main function to create revenue features."""
    print("=" * 70)
    print("NVDA Revenue Features Creation")
    print("=" * 70)
    
    # Load revenue data
    df = load_revenue_data()
    
    # Clean and deduplicate
    df = clean_and_deduplicate(df)
    
    # Add price data if not present
    df = add_price_data(df, ticker="NVDA")
    
    # Create revenue features
    df = create_revenue_features(df)
    
    # Create forward return target
    df = create_forward_return_target(df, trading_days=252)
    
    # Ensure form column exists (if not present, infer from period type)
    if "form" not in df.columns:
        # Infer form: FY periods are typically from 10-K, quarterly from 10-Q
        df["form"] = df["period"].map(lambda x: "10-K" if x == "FY" else "10-Q")
        print("\n  Inferred form column from period type (FY -> 10-K, quarterly -> 10-Q)")
    
    # Select and reorder output columns
    output_columns = [
        "ticker",
        "period",
        "end_date",
        "revenue",
        "form",
        "tag",
        "px_date",
        "adj_close",
        "rev_qoq",
        "rev_yoy",
        "rev_accel",
        "future_12m_return",
    ]
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in df.columns]
    df_output = df[available_columns].copy()
    
    # Save to CSV
    output_file = Path("outputs") / "nvda_revenue_features.csv"
    df_output.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved {len(df_output)} records to {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Revenue rows kept: {len(df_output)}")
    print(f"First end_date: {df_output['end_date'].min().date()}")
    print(f"Last end_date: {df_output['end_date'].max().date()}")
    
    price_matched = df_output["adj_close"].notna().sum() if "adj_close" in df_output.columns else 0
    price_match_rate = price_matched / len(df_output) * 100 if len(df_output) > 0 else 0
    print(f"Price-matched rows: {price_matched}/{len(df_output)} ({price_match_rate:.1f}%)")
    
    if "period" in df_output.columns:
        fy_count = len(df_output[df_output["period"] == "FY"])
        quarterly_count = len(df_output[df_output["period"].isin(["Q1", "Q2", "Q3", "Q4"])])
        print(f"FY rows: {fy_count}")
        print(f"Quarterly rows: {quarterly_count}")
    
    print("\n" + "=" * 70)
    print("Sample data:")
    print("=" * 70)
    print(df_output.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()

