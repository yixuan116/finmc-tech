"""Fetch quarterly revenue data from SEC XBRL API."""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import requests


# SEC requires a User-Agent header
SEC_HEADERS = {
    "User-Agent": "FINMC-TECH Research Project contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Revenue tags to try in order
# Note: Newer companies use RevenueFromContractWithCustomerExcludingAssessedTax (ASC 606, 2018+)
# Older data may use SalesRevenueNet or Revenues
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",  # New standard (2018+)
    "SalesRevenueNet",  # Older standard, covers 2007-2018 period
    "Revenues",  # Most generic, covers earliest periods
]

# Valid fiscal periods
VALID_PERIODS = {"Q1", "Q2", "Q3", "Q4", "FY"}


def get_ticker_to_cik() -> Dict[str, str]:
    """
    Fetch ticker to CIK mapping from SEC.

    Returns
    -------
    Dict[str, str]
        Mapping of ticker -> CIK (as 10-digit string with leading zeros)
    """
    print("Fetching ticker to CIK mapping from SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"
    
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to ticker -> CIK mapping
        ticker_cik = {}
        for entry in data.values():
            ticker = entry.get("ticker", "").upper()
            cik = entry.get("cik_str", "")
            if ticker and cik:
                # Format CIK as 10-digit string with leading zeros
                cik_str = str(cik).zfill(10)
                ticker_cik[ticker] = cik_str
        
        print(f"Loaded {len(ticker_cik)} ticker mappings")
        return ticker_cik
    
    except Exception as e:
        raise ConnectionError(f"Failed to fetch ticker mapping: {str(e)}")


def get_company_facts(cik: str) -> Optional[Dict]:
    """
    Fetch company facts (XBRL data) from SEC API.

    Parameters
    ----------
    cik : str
        10-digit CIK with leading zeros

    Returns
    -------
    Optional[Dict]
        Company facts JSON data, or None if not available
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if response.status_code == 404:
            print(f"  Warning: No company facts found for CIK {cik}")
            return None
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Error fetching facts for CIK {cik}: {str(e)}")
        return None


def extract_revenue_data(
    company_facts: Dict, ticker: str, revenue_tags: List[str]
) -> List[Dict]:
    """
    Extract revenue data from company facts for a given ticker.

    Parameters
    ----------
    company_facts : Dict
        Company facts JSON data
    ticker : str
        Stock ticker symbol
    revenue_tags : List[str]
        List of revenue tags to try in order

    Returns
    -------
    List[Dict]
        List of revenue records with columns: ticker, period, end_date, revenue, unit
    """
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    
    revenue_data = []
    all_tags_data = {}  # Collect data from all tags
    
    # Try each revenue tag and collect data from all
    for tag in revenue_tags:
        if tag not in us_gaap:
            continue
        
        tag_data = us_gaap[tag]
        units = tag_data.get("units", {})
        
        # Look for USD units
        if "USD" not in units:
            continue
        
        usd_data = units["USD"]
        tag_records = []
        
        for record in usd_data:
            # Check if it's a valid period
            period = record.get("fp", "")
            if period not in VALID_PERIODS:
                continue
            
            # Extract data
            end_date = record.get("end", "")
            start_date = record.get("start", "")
            revenue = record.get("val")
            form = record.get("form", "")
            filed_date = record.get("filed", "")
            
            if revenue is not None and end_date:
                tag_records.append({
                    "ticker": ticker,
                    "period": period,
                    "end_date": end_date,
                    "start_date": start_date,
                    "revenue": revenue,
                    "unit": "USD",
                    "form": form,
                    "tag": tag,
                    "filed_date": filed_date,
                })
        
        if tag_records:
            all_tags_data[tag] = tag_records
            print(f"  Found {len(tag_records)} records with tag: {tag}")
    
    # Merge data from all tags, prioritizing newer tags for overlapping periods
    # Strategy: Use RevenueFromContractWithCustomerExcludingAssessedTax for 2018+,
    #           use SalesRevenueNet/Revenues for earlier periods
    if not all_tags_data:
        return revenue_data
    
    # Collect all records
    all_records = []
    for tag, records in all_tags_data.items():
        all_records.extend(records)
    
    # Deduplicate by (ticker, period, end_date), with smart selection
    # Strategy:
    # 1. For same (period, end_date), prefer single-quarter values over cumulative
    #    (single-quarter has start_date close to end_date, cumulative has start_date far from end_date)
    # 2. Prefer newer tags
    # 3. Prefer most recently filed records
    seen = {}
    tag_priority = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": 3,
        "SalesRevenueNet": 2,
        "Revenues": 1,
    }
    
    for record in all_records:
        key = (record["ticker"], record["period"], record["end_date"])
        priority = tag_priority.get(record["tag"], 0)
        
        # Calculate if this is a single-quarter record (start_date close to end_date)
        # For quarterly records, start_date should be within ~4 months of end_date
        start_date = record.get("start_date", "")
        end_date = record.get("end_date", "")
        
        is_single_quarter = False
        if start_date and end_date:
            try:
                from datetime import datetime
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                days_diff = (end_dt - start_dt).days
                # Single quarter is typically 60-120 days, cumulative can be 90-365+ days
                is_single_quarter = 60 <= days_diff <= 120
            except:
                pass
        
        filed_date = record.get("filed_date", "")
        
        if key not in seen:
            seen[key] = record
        else:
            existing = seen[key]
            existing_priority = tag_priority.get(existing["tag"], 0)
            existing_is_single = False
            
            existing_start = existing.get("start_date", "")
            existing_end = existing.get("end_date", "")
            if existing_start and existing_end:
                try:
                    from datetime import datetime
                    start_dt = datetime.strptime(existing_start, "%Y-%m-%d")
                    end_dt = datetime.strptime(existing_end, "%Y-%m-%d")
                    days_diff = (end_dt - start_dt).days
                    existing_is_single = 60 <= days_diff <= 120
                except:
                    pass
            
            # Prefer: single-quarter > cumulative, newer tag > older tag, newer filed > older filed
            if is_single_quarter and not existing_is_single:
                seen[key] = record
            elif not is_single_quarter and existing_is_single:
                pass  # Keep existing
            elif priority > existing_priority:
                seen[key] = record
            elif priority == existing_priority and filed_date > existing.get("filed_date", ""):
                seen[key] = record
    
    revenue_data = list(seen.values())
    
    # Remove helper fields before returning
    for record in revenue_data:
        record.pop("start_date", None)
        record.pop("filed_date", None)
    
    if revenue_data:
        tags_used = set(r["tag"] for r in revenue_data)
        print(f"  Merged {len(revenue_data)} unique records from tags: {', '.join(tags_used)}")
    
    return revenue_data


def fetch_revenues_panel(tickers: List[str], output_path: str = None) -> pd.DataFrame:
    """
    Fetch quarterly revenue data for a list of tickers from SEC XBRL API.

    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols (e.g., ["AAPL", "MSFT", "NVDA"])
    output_path : str, optional
        Path to save CSV file. If None, uses outputs/revenues_panel.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, period, end_date, revenue, unit
    """
    # Set default output path if not provided
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "outputs" / "revenues_panel.csv"
    
    # Get ticker to CIK mapping
    ticker_cik = get_ticker_to_cik()
    
    all_revenue_data = []
    
    # Fetch data for each ticker
    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\nProcessing {ticker}...")
        
        if ticker not in ticker_cik:
            print(f"  Warning: Ticker {ticker} not found in SEC mapping")
            continue
        
        cik = ticker_cik[ticker]
        print(f"  CIK: {cik}")
        
        # Fetch company facts
        company_facts = get_company_facts(cik)
        if company_facts is None:
            continue
        
        # Extract revenue data
        revenue_data = extract_revenue_data(company_facts, ticker, REVENUE_TAGS)
        
        if revenue_data:
            all_revenue_data.extend(revenue_data)
            print(f"  ✓ Found {len(revenue_data)} revenue records")
        else:
            print(f"  ✗ No revenue data found for {ticker}")
        
        # Rate limiting: be polite to SEC servers
        time.sleep(0.2)
    
    if not all_revenue_data:
        raise ValueError("No revenue data found for any ticker")
    
    # Create DataFrame
    df = pd.DataFrame(all_revenue_data)
    
    # Sort by ticker, end_date
    df = df.sort_values(["ticker", "end_date"])
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} records to {output_path}")
    
    return df


def main():
    """Main function to fetch and save revenue data."""
    tickers = ["AAPL", "MSFT", "NVDA"]
    
    print("=" * 70)
    print("SEC XBRL Revenue Data Fetcher")
    print("=" * 70)
    print(f"Fetching quarterly revenue for: {', '.join(tickers)}")
    
    try:
        df = fetch_revenues_panel(tickers)
        print(f"\n✓ Successfully fetched revenue data")
        print(f"  Total records: {len(df)}")
        print(f"  Tickers: {df['ticker'].unique().tolist()}")
        print(f"  Date range: {df['end_date'].min()} to {df['end_date'].max()}")
        print(f"\nSample data:")
        print(df.head(10).to_string(index=False))
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

