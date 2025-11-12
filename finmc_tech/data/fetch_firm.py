"""
Fetch firm-specific data (revenue, prices).

Reuses existing code from src/data/fetch.py and src/data/fetch_sec_revenue.py.
Checks for existing processed data files to avoid redundant fetching.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.fetch import fetch_stock_data
from src.data.fetch_sec_revenue import fetch_revenue_data
from src.data.create_nvda_revenue_features import (
    add_price_data,
    create_revenue_features,
    create_forward_return_target,
)


def fetch_firm_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch firm-specific data (revenue from SEC, prices from yfinance).
    
    Checks for existing data files first to avoid redundant API calls.
    See README.md "Data Pipeline & Alignment" section for file priority order.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format. If None, fetches to latest.
        cache_dir: Directory to cache data. If None, no caching.
    
    Returns:
        DataFrame with revenue and price data aligned to quarterly frequency
    """
    print(f"Fetching firm data for {ticker}...")
    
    # 1. Check cache first (highest priority)
    if cache_dir:
        cache_file = cache_dir / f"{ticker}_firm_{start_date}_{end_date or 'latest'}.csv"
        if cache_file.exists():
            print(f"  Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
    
    # 2. Check for existing processed data files
    project_root = Path(__file__).parent.parent.parent
    processed_files = [
        project_root / "data" / "processed" / f"{ticker}_revenue_features.csv",
        project_root / "outputs" / f"{ticker.lower()}_revenue_features.csv",
        project_root / "outputs" / f"revenues_{ticker.lower()}_with_prices.csv",
    ]
    
    for file_path in processed_files:
        if file_path.exists():
            print(f"  Loading from existing processed file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if not all(col in df.columns for col in ["revenue", "end_date"]):
                continue
            
            # Check if price data exists
            if "px_date" in df.columns and "adj_close" in df.columns:
                price_matched = df["adj_close"].notna().sum()
                if price_matched > 0:
                    print(f"  ✓ Found existing processed data with {price_matched} price records")
                    
                    # Create missing features if needed
                    if "rev_qoq" not in df.columns:
                        print("  Creating revenue features...")
                        df = create_revenue_features(df)
                    if "future_12m_return" not in df.columns:
                        print("  Creating forward return target...")
                        df = create_forward_return_target(df)
                    
                    # Cache if requested
                    if cache_dir:
                        cache_file = cache_dir / f"{ticker}_firm_{start_date}_{end_date or 'latest'}.csv"
                        df.to_csv(cache_file, index=False)
                        print(f"  ✓ Cached to {cache_file}")
                    
                    return df
    
    # 3. Check for raw revenue data files
    raw_files = [
        project_root / "data" / "raw" / f"{ticker}_revenue.csv",
        project_root / "outputs" / f"revenues_{ticker.lower()}.csv",
    ]
    
    revenue_df = None
    for file_path in raw_files:
        if file_path.exists():
            print(f"  Loading revenue from existing file: {file_path}")
            revenue_df = pd.read_csv(file_path)
            
            if "revenue" in revenue_df.columns and "end_date" in revenue_df.columns:
                print(f"  ✓ Found existing revenue data with {len(revenue_df)} records")
                break
            revenue_df = None
    
    # 4. Fetch from APIs if no existing data found
    if revenue_df is None or revenue_df.empty:
        print("  No existing revenue data found, fetching from SEC XBRL API...")
        revenue_df = fetch_revenue_data([ticker], start_date=start_date, end_date=end_date)
        
        if revenue_df.empty:
            raise ValueError(f"No revenue data found for {ticker}")
    
    # 5. Add price data and create features
    print("  Fetching/checking price data...")
    revenue_df = add_price_data(revenue_df, ticker)
    
    print("  Creating revenue features...")
    revenue_df = create_revenue_features(revenue_df)
    
    print("  Creating forward return target...")
    revenue_df = create_forward_return_target(revenue_df)
    
    # 6. Cache if requested
    if cache_dir:
        cache_file = cache_dir / f"{ticker}_firm_{start_date}_{end_date or 'latest'}.csv"
        revenue_df.to_csv(cache_file, index=False)
        print(f"  ✓ Cached to {cache_file}")
    
    return revenue_df

