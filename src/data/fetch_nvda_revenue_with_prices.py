"""
Fetch NVIDIA quarterly and annual revenue from SEC XBRL API and merge with adjusted close prices.

This script:
1. Maps NVDA ticker to CIK using SEC company_tickers.json
2. Fetches revenue data from SEC Company Concept API
3. Downloads adjusted close prices from yfinance
4. Merges revenue with prices aligned to next trading day
5. Exports to revenues_nvda_with_prices.csv
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


# SEC API configuration
SEC_HEADERS = {
    "User-Agent": "FINMC-TECH Research Project contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Revenue tags to try in order
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
]

# Valid fiscal periods
VALID_PERIODS = {"Q1", "Q2", "Q3", "Q4", "FY"}

# Rate limiting
RATE_LIMIT_DELAY = 0.2
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0


def fetch_with_retry(
    url: str, headers: Dict[str, str], max_retries: int = MAX_RETRIES
) -> requests.Response:
    """
    Fetch URL with exponential backoff retry for HTTP 429/5xx errors.

    Parameters
    ----------
    url : str
        URL to fetch
    headers : Dict[str, str]
        HTTP headers
    max_retries : int
        Maximum number of retry attempts

    Returns
    -------
    requests.Response
        Response object

    Raises
    ------
    requests.exceptions.RequestException
        If all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                # Rate limited, wait longer
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                print(f"  Rate limited (429), waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                continue
            elif 500 <= response.status_code < 600:
                # Server error, retry
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                print(f"  Server error ({response.status_code}), waiting {wait_time:.1f}s before retry...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = BASE_RETRY_DELAY * (2 ** attempt)
            print(f"  Request failed, waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
    
    raise requests.exceptions.RequestException(f"Failed after {max_retries} retries")


def get_ticker_to_cik() -> Dict[str, str]:
    """
    Fetch ticker to CIK mapping from SEC.

    Returns
    -------
    Dict[str, str]
        Mapping of ticker -> CIK (as 10-digit string with leading zeros)

    Raises
    ------
    ConnectionError
        If unable to fetch ticker mapping
    """
    print("Fetching ticker to CIK mapping from SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"
    
    try:
        response = fetch_with_retry(url, SEC_HEADERS)
        data = response.json()
        
        ticker_cik = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = entry.get("cik_str", "")
            if ticker and cik:
                cik_str = str(cik).zfill(10)
                ticker_cik[ticker] = cik_str
        
        print(f"Loaded {len(ticker_cik)} ticker mappings")
        return ticker_cik
    
    except Exception as e:
        raise ConnectionError(f"Failed to fetch ticker mapping: {str(e)}")


def get_company_concept(cik: str, tag: str) -> Optional[List[Dict]]:
    """
    Fetch company concept data from SEC API for a specific tag.

    Parameters
    ----------
    cik : str
        10-digit CIK with leading zeros
    tag : str
        XBRL tag name

    Returns
    -------
    Optional[List[Dict]]
        List of concept records, or None if not available
    """
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    
    try:
        response = fetch_with_retry(url, SEC_HEADERS)
        data = response.json()
        
        units = data.get("units", {})
        if "USD" not in units:
            return None
        
        usd_data = units["USD"]
        records = []
        
        for record in usd_data:
            period = record.get("fp", "")
            if period not in VALID_PERIODS:
                continue
            
            end_date = record.get("end", "")
            revenue = record.get("val")
            fy = record.get("fy")
            form = record.get("form", "")
            filed_date = record.get("filed", "")
            
            if revenue is not None and end_date:
                records.append({
                    "period_end": end_date,
                    "revenue": revenue,
                    "fy": fy,
                    "fp": period,
                    "form": form,
                    "filed_date": filed_date,
                })
        
        return records if records else None
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise
    except Exception as e:
        print(f"  Warning: Error fetching concept {tag} for CIK {cik}: {str(e)}")
        return None


def fetch_revenue_data(ticker: str) -> Tuple[pd.DataFrame, str]:
    """
    Fetch revenue data for a ticker from SEC API.
    
    Tries all revenue tags and merges data, preferring newer tags for overlapping periods.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol

    Returns
    -------
    Tuple[pd.DataFrame, str]
        DataFrame with revenue data and tags used (comma-separated)

    Raises
    ------
    ValueError
        If no revenue data found for any tag
    """
    ticker = ticker.upper()
    print(f"\nProcessing {ticker}...")
    
    # Get CIK
    ticker_cik = get_ticker_to_cik()
    if ticker not in ticker_cik:
        raise ValueError(f"Ticker {ticker} not found in SEC mapping")
    
    cik = ticker_cik[ticker]
    print(f"  CIK: {cik}")
    
    # Try all revenue tags and collect data
    all_records = []
    tags_found = []
    
    for tag in REVENUE_TAGS:
        print(f"  Trying tag: {tag}...")
        records = get_company_concept(cik, tag)
        time.sleep(RATE_LIMIT_DELAY)
        
        if records:
            for record in records:
                record["tag_used"] = tag
            all_records.extend(records)
            tags_found.append(tag)
            print(f"  ✓ Found {len(records)} records with tag: {tag}")
    
    if not all_records:
        raise ValueError(f"No revenue data found for {ticker} with any tag")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    df["ticker"] = ticker
    
    # Convert period_end to datetime
    df["period_end"] = pd.to_datetime(df["period_end"])
    
    # Convert filed_date to datetime for sorting
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    
    # Tag priority for deduplication (prefer newer tags)
    tag_priority = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": 3,
        "SalesRevenueNet": 2,
        "Revenues": 1,
    }
    df["tag_priority"] = df["tag_used"].map(lambda x: tag_priority.get(x, 0))
    
    # Form priority: 10-K > 10-Q (10-K is more authoritative for annual data)
    df["form_priority"] = df["form"].map(lambda x: 2 if x == "10-K" else (1 if x == "10-Q" else 0))
    
    # Drop duplicates by period_end with smart selection:
    # 1. Prefer higher tag priority (newer tags)
    # 2. If same tag, prefer 10-K over 10-Q
    # 3. If same form, prefer most recently filed
    df = df.sort_values(
        ["period_end", "tag_priority", "form_priority", "filed_date"],
        ascending=[True, False, False, False],
        na_position="last"
    )
    df = df.drop_duplicates(subset=["period_end"], keep="first")
    
    # Remove helper columns (keep form and filed_date for reference, but don't include in output)
    df = df.drop(columns=["tag_priority", "form_priority"])
    
    tags_used_str = ", ".join(tags_found)
    print(f"  After deduplication: {len(df)} unique records")
    print(f"  Tags used: {tags_used_str}")
    print(f"  Period range: {df['period_end'].min().date()} to {df['period_end'].max().date()}")
    
    return df, tags_used_str


def get_next_trading_day(date: pd.Timestamp, max_days: int = 5) -> Optional[pd.Timestamp]:
    """
    Find the next trading day (excluding weekends) from a given date.

    Parameters
    ----------
    date : pd.Timestamp
        Starting date
    max_days : int
        Maximum number of calendar days to look ahead

    Returns
    -------
    Optional[pd.Timestamp]
        Next trading day, or None if not found within max_days
    """
    for i in range(1, max_days + 1):
        candidate = date + timedelta(days=i)
        # Check if it's a weekday (Monday=0, Sunday=6)
        if candidate.weekday() < 5:
            return candidate
    return None


def fetch_and_merge_prices(
    revenue_df: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """
    Fetch adjusted close prices and merge with revenue data.

    Parameters
    ----------
    revenue_df : pd.DataFrame
        DataFrame with revenue data including period_end column
    ticker : str
        Stock ticker symbol

    Returns
    -------
    pd.DataFrame
        DataFrame with merged revenue and price data
    """
    print(f"\nFetching adjusted close prices for {ticker}...")
    
    # Get date range ±10 days
    min_date = revenue_df["period_end"].min() - timedelta(days=10)
    max_date = revenue_df["period_end"].max() + timedelta(days=10)
    
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
    
    # Merge revenue with prices
    merged_data = []
    
    for _, row in revenue_df.iterrows():
        period_end = pd.Timestamp(row["period_end"])
        
        # Find next trading day (up to 5 days ahead)
        next_trading_day = get_next_trading_day(period_end, max_days=5)
        
        if next_trading_day is None:
            print(f"  Warning: No trading day found for {period_end.date()}")
            merged_data.append({
                **row.to_dict(),
                "px_date": None,
                "adj_close": None,
            })
            continue
        
        # Find matching price
        next_trading_day_date = next_trading_day.date()
        matching_prices = price_data[price_data["Date"] == next_trading_day_date]
        
        if len(matching_prices) > 0:
            adj_close = matching_prices.iloc[0]["Close"]  # yfinance uses Close for adjusted close
            merged_data.append({
                **row.to_dict(),
                "px_date": next_trading_day_date,
                "adj_close": adj_close,
            })
        else:
            print(f"  Warning: No price found for {next_trading_day_date}")
            merged_data.append({
                **row.to_dict(),
                "px_date": None,
                "adj_close": None,
            })
    
    result_df = pd.DataFrame(merged_data)
    
    # Drop form and filed_date columns before final output (they were only for deduplication)
    result_df = result_df.drop(columns=["form", "filed_date"], errors="ignore")
    
    # Reorder columns
    column_order = [
        "ticker",
        "period_end",
        "fy",
        "fp",
        "revenue",
        "tag_used",
        "px_date",
        "adj_close",
    ]
    result_df = result_df[column_order]
    
    # Count successful merges
    successful_merges = result_df["adj_close"].notna().sum()
    print(f"  Successfully merged prices for {successful_merges} of {len(result_df)} revenue records")
    
    return result_df


def main():
    """Main function to fetch and merge revenue and price data."""
    print("=" * 70)
    print("NVIDIA Revenue and Price Data Fetcher")
    print("=" * 70)
    
    ticker = "NVDA"
    
    try:
        # Fetch revenue data
        revenue_df, tag_used = fetch_revenue_data(ticker)
        print(f"\n✓ Revenue data: {len(revenue_df)} rows")
        print(f"  Tag used: {tag_used}")
        print(f"  First period_end: {revenue_df['period_end'].min().date()}")
        print(f"  Last period_end: {revenue_df['period_end'].max().date()}")
        
        # Fetch and merge prices
        merged_df = fetch_and_merge_prices(revenue_df, ticker)
        
        # Save to CSV in outputs directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "revenues_nvda_with_prices.csv"
        merged_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved {len(merged_df)} records to {output_file}")
        print(f"  Columns: {', '.join(merged_df.columns)}")
        print(f"  Successfully merged prices: {merged_df['adj_close'].notna().sum()} rows")
        
        print("\n" + "=" * 70)
        print("Sample data:")
        print("=" * 70)
        print(merged_df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

