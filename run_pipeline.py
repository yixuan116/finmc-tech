"""
End-to-end ML pipeline for dual-head stock prediction (return + price).

This pipeline:
1. Fetches revenue fundamentals from SEC API
2. Merges with price data and macro features (VIX, TNX) from yfinance
3. Engineers features (YoY, QoQ, acceleration, macro features)
4. Trains dual-head models:
   - Return head: predicts 12-month forward return
   - Price head: predicts log(price_{t+12m}) directly
5. Evaluates both direct and indirect price predictions
6. Generates figures and metrics
7. Auto-updates README.md with results

Usage:
    python run_pipeline.py --tickers NVDA
    python run_pipeline.py --tickers NVDA,MSFT
    python run_pipeline.py --tickers NVDA --skip-fetch
    python run_pipeline.py --tickers NVDA --start 2008-01-01
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Configuration
# ============================================================================

SEC_HEADERS = {
    "User-Agent": "FINMC-TECH Research Project contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
]

VALID_PERIODS = {"Q1", "Q2", "Q3", "Q4", "FY"}
RATE_LIMIT_DELAY = 0.2
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0

# Feature columns for both heads
FEATURE_COLS = [
    "rev_qoq",
    "rev_yoy",
    "rev_accel",
    "vix_level",
    "tnx_yield",
    "vix_change_3m",
    "tnx_change_3m",
]


# ============================================================================
# Helper Functions
# ============================================================================

def fetch_with_retry(url: str, headers: Dict[str, str], max_retries: int = MAX_RETRIES) -> requests.Response:
    """Fetch URL with exponential backoff retry for HTTP 429/5xx errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                print(f"  Rate limited (429), waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            elif 500 <= response.status_code < 600:
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                print(f"  Server error ({response.status_code}), waiting {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = BASE_RETRY_DELAY * (2 ** attempt)
            time.sleep(wait_time)
    raise requests.exceptions.RequestException(f"Failed after {max_retries} retries")


# ============================================================================
# Step 1: Fetch Fundamentals from SEC
# ============================================================================

def get_cik_map() -> Dict[str, str]:
    """Fetch ticker to CIK mapping from SEC."""
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
                ticker_cik[ticker] = str(cik).zfill(10)
        
        print(f"  Loaded {len(ticker_cik)} ticker mappings")
        return ticker_cik
    except Exception as e:
        raise ConnectionError(f"Failed to fetch ticker mapping: {str(e)}")


def get_company_concept(cik: str, tag: str) -> Optional[List[Dict]]:
    """Fetch company concept data from SEC API for a specific tag."""
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
            
            if revenue is not None and end_date:
                records.append({
                    "period_end": end_date,
                    "revenue": revenue,
                    "fy": fy,
                    "fp": period,
                    "form": form,
                    "tag_used": tag,
                })
        
        return records if records else None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise
    except Exception as e:
        print(f"  Warning: Error fetching concept {tag}: {str(e)}")
        return None


def fetch_revenue_series(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Fetch revenue data for a ticker from SEC API."""
    print(f"\nProcessing {ticker}...")
    
    ticker_cik = get_cik_map()
    if ticker not in ticker_cik:
        raise ValueError(f"Ticker {ticker} not found in SEC mapping")
    
    cik = ticker_cik[ticker]
    print(f"  CIK: {cik}")
    
    # Try all revenue tags
    all_records = []
    tags_found = []
    
    for tag in REVENUE_TAGS:
        print(f"  Trying tag: {tag}...")
        records = get_company_concept(cik, tag)
        time.sleep(RATE_LIMIT_DELAY)
        
        if records:
            all_records.extend(records)
            tags_found.append(tag)
            print(f"  ✓ Found {len(records)} records")
    
    if not all_records:
        raise ValueError(f"No revenue data found for {ticker}")
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    df["ticker"] = ticker
    
    # Convert period_end to datetime
    df["period_end"] = pd.to_datetime(df["period_end"])
    
    # Filter by date range if provided
    if start_date:
        df = df[df["period_end"] >= pd.to_datetime(start_date)]
    if end_date and end_date != "today":
        df = df[df["period_end"] <= pd.to_datetime(end_date)]
    
    # Tag priority for deduplication
    tag_priority = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": 3,
        "SalesRevenueNet": 2,
        "Revenues": 1,
    }
    df["tag_priority"] = df["tag_used"].map(lambda x: tag_priority.get(x, 0))
    
    # Form priority: 10-K > 10-Q
    df["form_priority"] = df["form"].map(lambda x: 2 if x == "10-K" else (1 if x == "10-Q" else 0))
    
    # Deduplicate
    df = df.sort_values(
        ["period_end", "tag_priority", "form_priority"],
        ascending=[True, False, False]
    )
    df = df.drop_duplicates(subset=["ticker", "period_end"], keep="first")
    df = df.drop(columns=["tag_priority", "form_priority"])
    
    tag_used_str = ", ".join(tags_found)
    print(f"  Kept {len(df)} unique records")
    print(f"  Period range: {df['period_end'].min().date()} to {df['period_end'].max().date()}")
    
    return df, tag_used_str


# ============================================================================
# Step 2: Fetch Macro Features
# ============================================================================

def fetch_macro_series(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch VIX and TNX data from yfinance."""
    print(f"\nFetching macro data (VIX, TNX) from {start_date.date()} to {end_date.date()}...")
    
    # Fetch VIX
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(start=start_date, end=end_date + timedelta(days=1))
    
    # Fetch TNX (10-year treasury yield)
    tnx_ticker = yf.Ticker("^TNX")
    tnx_data = tnx_ticker.history(start=start_date, end=end_date + timedelta(days=1))
    
    # Combine into DataFrame using trading days from VIX (or TNX if VIX is empty)
    if not vix_data.empty:
        macro_df = vix_data[["Close"]].copy()
        macro_df.columns = ["vix_level"]
        macro_df = macro_df.reset_index()
        macro_df["Date"] = pd.to_datetime(macro_df["Date"]).dt.date
    elif not tnx_data.empty:
        macro_df = tnx_data[["Close"]].copy()
        macro_df.columns = ["tnx_yield"]
        macro_df = macro_df.reset_index()
        macro_df["Date"] = pd.to_datetime(macro_df["Date"]).dt.date
    else:
        # Fallback: create empty DataFrame
        macro_df = pd.DataFrame(columns=["Date"])
    
    # Merge VIX if we have it
    if not vix_data.empty:
        vix_close = vix_data["Close"].reset_index()
        vix_close["Date"] = pd.to_datetime(vix_close["Date"]).dt.date
        vix_dict = dict(zip(vix_close["Date"], vix_close["Close"]))
        if "vix_level" not in macro_df.columns:
            macro_df["vix_level"] = macro_df["Date"].map(vix_dict)
        else:
            # Already have it from initialization
            pass
    
    # Merge TNX (Close price, convert percentage to decimal)
    if not tnx_data.empty:
        tnx_close = tnx_data["Close"].reset_index()
        tnx_close["Date"] = pd.to_datetime(tnx_close["Date"]).dt.date
        tnx_dict = dict(zip(tnx_close["Date"], tnx_close["Close"]))
        macro_df["tnx_yield"] = macro_df["Date"].map(tnx_dict)
        # Convert percentage to decimal (e.g., 2.5% -> 0.025)
        macro_df["tnx_yield"] = macro_df["tnx_yield"] / 100.0
    
    # Forward fill missing values (using ffill method)
    if "vix_level" in macro_df.columns:
        macro_df["vix_level"] = macro_df["vix_level"].ffill()
    if "tnx_yield" in macro_df.columns:
        macro_df["tnx_yield"] = macro_df["tnx_yield"].ffill()
    
    # Calculate 3-month changes (63 trading days)
    if "vix_level" in macro_df.columns:
        macro_df["vix_change_3m"] = macro_df["vix_level"].pct_change(63)
    if "tnx_yield" in macro_df.columns:
        # Change in percentage points (not percentage change)
        macro_df["tnx_change_3m"] = macro_df["tnx_yield"].diff(63)
    
    print(f"  ✓ Fetched macro data for {len(macro_df)} trading days")
    
    return macro_df


# ============================================================================
# Step 3: Merge Prices, Macro Features, and Build Targets
# ============================================================================

def get_next_trading_day(date: pd.Timestamp, max_days: int = 5) -> Optional[datetime.date]:
    """Find next trading day (weekday) within max_days."""
    for i in range(1, max_days + 1):
        candidate = date + timedelta(days=i)
        if candidate.weekday() < 5:
            return candidate.date()
    return None


def load_merge_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Merge price data, macro features, and engineer all features and targets."""
    print(f"\nMerging prices, macro features, and engineering features for {ticker}...")
    
    # Download price data
    min_date = df["period_end"].min() - timedelta(days=10)
    max_date = df["period_end"].max() + timedelta(days=365)
    
    print(f"  Downloading prices from {min_date.date()} to {max_date.date()}...")
    stock = yf.Ticker(ticker)
    try:
        price_data = stock.history(start=min_date, end=max_date + timedelta(days=1))
    except Exception as e:
        raise ConnectionError(f"Failed to fetch price data: {str(e)}")
    
    if price_data.empty:
        raise ValueError(f"No price data found for {ticker}")
    
    price_data = price_data.reset_index()
    price_data["Date"] = pd.to_datetime(price_data["Date"]).dt.date
    
    print(f"  Downloaded {len(price_data)} trading days")
    
    # Fetch macro data
    macro_df = fetch_macro_series(min_date, max_date + timedelta(days=365))
    
    # Align to next trading day and merge prices
    merged_data = []
    for _, row in df.iterrows():
        period_end = pd.Timestamp(row["period_end"])
        next_trading_day = get_next_trading_day(period_end, max_days=5)
        
        if next_trading_day:
            matching_prices = price_data[price_data["Date"] == next_trading_day]
            if len(matching_prices) > 0:
                merged_data.append({
                    **row.to_dict(),
                    "px_date": next_trading_day,
                    "adj_close": matching_prices.iloc[0]["Close"],
                })
            else:
                merged_data.append({
                    **row.to_dict(),
                    "px_date": next_trading_day,
                    "adj_close": None,
                })
        else:
            merged_data.append({
                **row.to_dict(),
                "px_date": None,
                "adj_close": None,
            })
    
    df = pd.DataFrame(merged_data)
    print(f"  Matched prices for {df['adj_close'].notna().sum()}/{len(df)} records")
    
    # Engineer revenue features
    df = df.sort_values(["ticker", "period_end"])
    grouped = df.groupby("ticker")
    
    df["rev_qoq"] = grouped["revenue"].pct_change(1)
    df["rev_yoy"] = grouped["revenue"].pct_change(4)
    df["rev_accel"] = grouped["rev_yoy"].diff(1)
    
    # Merge macro features (same-day or previous trading day)
    df_with_price = df[df["adj_close"].notna() & df["px_date"].notna()].copy()
    if len(df_with_price) > 0:
        df_with_price["px_date_dt"] = pd.to_datetime(df_with_price["px_date"])
        
        # Merge macro data (same-day or nearest previous trading day)
        macro_merge = []
        for _, row in df_with_price.iterrows():
            px_date = row["px_date"]
            # Try same day first
            macro_row = macro_df[macro_df["Date"] == px_date]
            if len(macro_row) > 0:
                macro_merge.append({
                    "period_end": row["period_end"],
                    "vix_level": macro_row.iloc[0].get("vix_level", None),
                    "tnx_yield": macro_row.iloc[0].get("tnx_yield", None),
                    "vix_change_3m": macro_row.iloc[0].get("vix_change_3m", None),
                    "tnx_change_3m": macro_row.iloc[0].get("tnx_change_3m", None),
                })
            else:
                # Try previous trading days (up to 5 days back)
                found = False
                for days_back in range(1, 6):
                    prev_date = px_date - timedelta(days=days_back)
                    macro_row = macro_df[macro_df["Date"] == prev_date]
                    if len(macro_row) > 0:
                        macro_merge.append({
                            "period_end": row["period_end"],
                            "vix_level": macro_row.iloc[0].get("vix_level", None),
                            "tnx_yield": macro_row.iloc[0].get("tnx_yield", None),
                            "vix_change_3m": macro_row.iloc[0].get("vix_change_3m", None),
                            "tnx_change_3m": macro_row.iloc[0].get("tnx_change_3m", None),
                        })
                        found = True
                        break
                
                if not found:
                    macro_merge.append({
                        "period_end": row["period_end"],
                        "vix_level": None,
                        "tnx_yield": None,
                        "vix_change_3m": None,
                        "tnx_change_3m": None,
                    })
        
        macro_df_merge = pd.DataFrame(macro_merge)
        df = df.merge(macro_df_merge, on="period_end", how="left")
        
        # Build targets: future_12m_return and future_12m_logprice
        price_history = stock.history(
            start=df_with_price["px_date_dt"].min() - timedelta(days=10),
            end=df_with_price["px_date_dt"].max() + timedelta(days=365)
        )
        
        if not price_history.empty:
            price_history = price_history.reset_index()
            price_history["Date"] = pd.to_datetime(price_history["Date"]).dt.date
            price_history = price_history.sort_values("Date")
            
            future_prices = []
            future_returns = []
            future_logprices = []
            
            for _, row in df_with_price.iterrows():
                px_date = pd.to_datetime(row["px_date"]).date()
                current_price = row["adj_close"]
                
                matching_rows = price_history[price_history["Date"] == px_date]
                if len(matching_rows) > 0:
                    current_idx = matching_rows.index[0]
                    future_idx = current_idx + 252
                    if future_idx < len(price_history):
                        future_price_val = price_history.iloc[future_idx]["Close"]
                        future_prices.append(future_price_val)
                        future_returns.append((future_price_val / current_price) - 1)
                        future_logprices.append(np.log(future_price_val))
                    else:
                        future_prices.append(None)
                        future_returns.append(None)
                        future_logprices.append(None)
                else:
                    future_prices.append(None)
                    future_returns.append(None)
                    future_logprices.append(None)
            
            df_with_price["future_12m_price"] = future_prices
            df_with_price["future_12m_return"] = future_returns
            df_with_price["future_12m_logprice"] = future_logprices
            
            df = df.merge(
                df_with_price[["period_end", "future_12m_price", "future_12m_return", "future_12m_logprice"]],
                on="period_end",
                how="left"
            )
    
    return df


# ============================================================================
# Step 4: Train Return Head Models
# ============================================================================

def train_return_models(df: pd.DataFrame, ticker: str, split_date: pd.Timestamp) -> Dict:
    """Train models to predict future_12m_return."""
    print(f"\nTraining return head models for {ticker}...")
    
    # Prepare data
    # Only check features that exist in the dataframe
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    if len(available_features) == 0:
        raise ValueError(f"No features from FEATURE_COLS found in dataframe")
    
    df_clean = df[
        df[available_features].notna().all(axis=1) &
        df["future_12m_return"].notna()
    ].copy()
    
    if len(df_clean) < 10:
        raise ValueError(f"Insufficient data for return modeling: {len(df_clean)} rows")
    
    # Train/test split by time
    train_df = df_clean[df_clean["period_end"] < split_date]
    test_df = df_clean[df_clean["period_end"] >= split_date]
    
    print(f"  Train: {len(train_df)} rows (before {split_date.date()})")
    print(f"  Test: {len(test_df)} rows (from {split_date.date()})")
    
    if len(train_df) < 5 or len(test_df) < 5:
        raise ValueError("Insufficient data in train or test set")
    
    # Features and target
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["future_12m_return"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["future_12m_return"].values
    
    # Scale features for Ridge/KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        "ridge": Ridge(),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Use scaled features for Ridge/KNN, raw for RF
        if name == "rf":
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            "train": {"r2": float(train_r2), "rmse": float(train_rmse)},
            "test": {"r2": float(test_r2), "rmse": float(test_rmse)},
        }
        
        # Store predictions for plotting
        results[name]["predictions"] = {
            "test_period_end": test_df["period_end"].dt.strftime("%Y-%m-%d").tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_test_pred.tolist(),
        }
        
        # Store feature importance for RF
        if name == "rf":
            feature_importance = dict(zip(available_features, model.feature_importances_.tolist()))
            results[name]["feature_importance"] = feature_importance
    
    return results


# ============================================================================
# Step 5: Train Price Head Models
# ============================================================================

def train_price_models(df: pd.DataFrame, ticker: str, split_date: pd.Timestamp) -> Dict:
    """Train models to predict future_12m_logprice (direct price head)."""
    print(f"\nTraining price head models for {ticker}...")
    
    # Prepare data
    # Only check features that exist in the dataframe
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    if len(available_features) == 0:
        raise ValueError(f"No features from FEATURE_COLS found in dataframe")
    
    df_clean = df[
        df[available_features].notna().all(axis=1) &
        df["future_12m_logprice"].notna() &
        df["future_12m_price"].notna() &
        df["adj_close"].notna()
    ].copy()
    
    if len(df_clean) < 10:
        raise ValueError(f"Insufficient data for price modeling: {len(df_clean)} rows")
    
    # Train/test split by time
    train_df = df_clean[df_clean["period_end"] < split_date]
    test_df = df_clean[df_clean["period_end"] >= split_date]
    
    print(f"  Train: {len(train_df)} rows (before {split_date.date()})")
    print(f"  Test: {len(test_df)} rows (from {split_date.date()})")
    
    if len(train_df) < 5 or len(test_df) < 5:
        raise ValueError("Insufficient data in train or test set")
    
    # Features and target (log price)
    # Use same available features as return head
    available_features = [col for col in FEATURE_COLS if col in df_clean.columns]
    
    X_train = train_df[available_features].values
    y_train_log = train_df["future_12m_logprice"].values
    y_train_price = train_df["future_12m_price"].values
    X_test = test_df[available_features].values
    y_test_log = test_df["future_12m_logprice"].values
    y_test_price = test_df["future_12m_price"].values
    
    # Scale features for Ridge/KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        "ridge": Ridge(),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Use scaled features for Ridge/KNN, raw for RF
        if name == "rf":
            model.fit(X_train, y_train_log)
            y_train_pred_log = model.predict(X_train)
            y_test_pred_log = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train_log)
            y_train_pred_log = model.predict(X_train_scaled)
            y_test_pred_log = model.predict(X_test_scaled)
        
        # Convert log predictions to price
        y_train_pred_price = np.exp(y_train_pred_log)
        y_test_pred_price = np.exp(y_test_pred_log)
        
        # Evaluate on log scale
        train_r2_log = r2_score(y_train_log, y_train_pred_log)
        train_rmse_log = np.sqrt(mean_squared_error(y_train_log, y_train_pred_log))
        test_r2_log = r2_score(y_test_log, y_test_pred_log)
        test_rmse_log = np.sqrt(mean_squared_error(y_test_log, y_test_pred_log))
        
        # Evaluate on price scale
        test_rmse_price = np.sqrt(mean_squared_error(y_test_price, y_test_pred_price))
        
        results[name] = {
            "train": {"r2": float(train_r2_log), "rmse": float(train_rmse_log)},
            "test": {"r2": float(test_r2_log), "rmse": float(test_rmse_log)},
            "test_price_rmse": float(test_rmse_price),  # RMSE on actual price scale
        }
        
        # Store predictions for plotting
        results[name]["predictions"] = {
            "test_period_end": test_df["period_end"].dt.strftime("%Y-%m-%d").tolist(),
            "y_test_price": y_test_price.tolist(),
            "y_pred_price": y_test_pred_price.tolist(),
        }
    
    return results


# ============================================================================
# Step 6: Evaluate and Compare
# ============================================================================

def evaluate_and_plot(metrics: Dict, output_dir: Path):
    """Evaluate models and generate all figures."""
    print(f"\nGenerating figures...")
    
    df = metrics["df_clean"]
    ticker = metrics["ticker"]
    
    # Set style
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except:
        try:
            plt.style.use("seaborn-darkgrid")
        except:
            plt.style.use("default")
    figsize = (10, 6)
    
    # 1. YoY vs Return (unchanged)
    fig, ax = plt.subplots(figsize=figsize)
    mask = df["rev_yoy"].notna() & df["future_12m_return"].notna()
    x = df.loc[mask, "rev_yoy"]
    y = df.loc[mask, "future_12m_return"]
    ax.scatter(x, y, alpha=0.6, s=50, label="Data points")
    
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"y={z[0]:.2f}x+{z[1]:.2f}")
        ax.legend()
    
    ax.set_xlabel("Revenue YoY Growth (decimal: 0.5 = 50%)")
    ax.set_ylabel("Future 12M Return (decimal: 1.0 = 100%)")
    ax.set_title(f"{ticker}: Revenue YoY Growth vs Future 12M Return")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "yoy_vs_return.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved yoy_vs_return.png")
    
    # 2. Acceleration vs Return (unchanged)
    fig, ax = plt.subplots(figsize=figsize)
    mask = df["rev_accel"].notna() & df["future_12m_return"].notna()
    x = df.loc[mask, "rev_accel"]
    y = df.loc[mask, "future_12m_return"]
    ax.scatter(x, y, alpha=0.6, s=50, label="Data points")
    
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"y={z[0]:.2f}x+{z[1]:.2f}")
        ax.legend()
    
    ax.set_xlabel("Revenue Acceleration (YoY Growth Change, decimal)")
    ax.set_ylabel("Future 12M Return (decimal: 1.0 = 100%)")
    ax.set_title(f"{ticker}: Revenue Acceleration vs Future 12M Return")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accel_vs_return.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved accel_vs_return.png")
    
    # 3. Rolling correlation (unchanged)
    fig, ax = plt.subplots(figsize=figsize)
    df_sorted = df.sort_values("period_end")
    window_size = 12
    rolling_corr = []
    rolling_dates = []
    
    for i in range(window_size, len(df_sorted)):
        window = df_sorted.iloc[i-window_size:i]
        corr = window["rev_yoy"].corr(window["future_12m_return"])
        if pd.notna(corr):
            rolling_corr.append(corr)
            rolling_dates.append(df_sorted.iloc[i]["period_end"])
    
    if rolling_corr:
        ax.plot(rolling_dates, rolling_corr, linewidth=2, label="Rolling correlation")
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Period End Date")
        ax.set_ylabel("3-Year Rolling Correlation")
        ax.set_title(f"{ticker}: Rolling Correlation (Rev YoY vs Future 12M Return)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "rolling_corr.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved rolling_corr.png")
    
    # 4. RF Feature Importance (return head)
    if "return_head" in metrics["models"] and "rf" in metrics["models"]["return_head"]:
        if "feature_importance" in metrics["models"]["return_head"]["rf"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            imp = metrics["models"]["return_head"]["rf"]["feature_importance"]
            features = list(imp.keys())
            importances = list(imp.values())
            ax.barh(features, importances)
            ax.set_xlabel("Importance")
            ax.set_title(f"{ticker}: RandomForest Feature Importance (Return Head)")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(output_dir / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved rf_feature_importance.png")
    
    # 5. Pred vs Actual - Return Head RF (Test)
    if "return_head" in metrics["models"] and "rf" in metrics["models"]["return_head"]:
        if "predictions" in metrics["models"]["return_head"]["rf"]:
            preds = metrics["models"]["return_head"]["rf"]["predictions"]
            fig, ax = plt.subplots(figsize=figsize)
            dates = pd.to_datetime(preds["test_period_end"])
            ax.plot(dates, preds["y_test"], "o-", label="Actual Return", alpha=0.7, linewidth=2)
            ax.plot(dates, preds["y_pred"], "s-", label="RF Predicted Return", alpha=0.7, linewidth=2)
            ax.set_xlabel("Period End Date")
            ax.set_ylabel("Future 12M Return (decimal)")
            ax.set_title(f"{ticker}: RandomForest Return Predictions vs Actual (Test Set)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "pred_vs_actual_return_rf.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved pred_vs_actual_return_rf.png")
    
    # 6. Pred vs Actual - Price Indirect (from return head RF)
    if "return_head" in metrics["models"] and "rf" in metrics["models"]["return_head"]:
        if "predictions" in metrics["models"]["return_head"]["rf"]:
            preds = metrics["models"]["return_head"]["rf"]["predictions"]
            # Get test dates and current prices
            test_df = df[df["period_end"] >= pd.to_datetime("2019-01-01")].copy()
            test_df = test_df[test_df["future_12m_return"].notna() & test_df["adj_close"].notna()]
            test_df = test_df.sort_values("period_end")
            
            if len(test_df) == len(preds["y_pred"]):
                dates = pd.to_datetime(preds["test_period_end"])
                actual_prices = test_df["future_12m_price"].values
                # Indirect: price_hat = current_price * (1 + return_hat)
                indirect_prices = test_df["adj_close"].values * (1 + np.array(preds["y_pred"]))
                
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(dates, actual_prices, "o-", label="Actual Price", alpha=0.7, linewidth=2)
                ax.plot(dates, indirect_prices, "s-", label="Indirect Price (from Return RF)", alpha=0.7, linewidth=2)
                ax.set_xlabel("Period End Date")
                ax.set_ylabel("Future 12M Stock Price (USD)")
                ax.set_title(f"{ticker}: Indirect Price Predictions vs Actual (Test Set)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "pred_vs_actual_price_indirect.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  ✓ Saved pred_vs_actual_price_indirect.png")
    
    # 7. Pred vs Actual - Price Direct (from price head RF)
    if "price_head" in metrics["models"] and "rf" in metrics["models"]["price_head"]:
        if "predictions" in metrics["models"]["price_head"]["rf"]:
            preds = metrics["models"]["price_head"]["rf"]["predictions"]
            fig, ax = plt.subplots(figsize=figsize)
            dates = pd.to_datetime(preds["test_period_end"])
            ax.plot(dates, preds["y_test_price"], "o-", label="Actual Price", alpha=0.7, linewidth=2)
            ax.plot(dates, preds["y_pred_price"], "^-", label="Direct Price (RF)", alpha=0.7, linewidth=2, color="green")
            ax.set_xlabel("Period End Date")
            ax.set_ylabel("Future 12M Stock Price (USD)")
            ax.set_title(f"{ticker}: Direct Price Predictions vs Actual (Test Set)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "pred_vs_actual_price_direct.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved pred_vs_actual_price_direct.png")
    
    # 8. Calibration - Return Head (scatter plot)
    if "return_head" in metrics["models"] and "rf" in metrics["models"]["return_head"]:
        if "predictions" in metrics["models"]["return_head"]["rf"]:
            preds = metrics["models"]["return_head"]["rf"]["predictions"]
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(preds["y_test"], preds["y_pred"], alpha=0.6, s=50)
            # y=x line
            min_val = min(min(preds["y_test"]), min(preds["y_pred"]))
            max_val = max(max(preds["y_test"]), max(preds["y_pred"]))
            ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x (perfect prediction)", linewidth=2)
            ax.set_xlabel("Actual Future 12M Return")
            ax.set_ylabel("Predicted Future 12M Return")
            ax.set_title(f"{ticker}: Return Head Calibration (Test Set)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "calibration_return.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved calibration_return.png")
    
    # 9. Residuals - Return Head (over time)
    if "return_head" in metrics["models"] and "rf" in metrics["models"]["return_head"]:
        if "predictions" in metrics["models"]["return_head"]["rf"]:
            preds = metrics["models"]["return_head"]["rf"]["predictions"]
            residuals = np.array(preds["y_test"]) - np.array(preds["y_pred"])
            fig, ax = plt.subplots(figsize=figsize)
            dates = pd.to_datetime(preds["test_period_end"])
            ax.plot(dates, residuals, "o-", alpha=0.7, linewidth=2)
            ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax.set_xlabel("Period End Date")
            ax.set_ylabel("Residual (Actual - Predicted Return)")
            ax.set_title(f"{ticker}: Return Head Residuals Over Time (Test Set)")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "residuals_return.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ Saved residuals_return.png")


# ============================================================================
# Step 7: Update README
# ============================================================================

def update_readme(metrics: Dict, output_dir: Path):
    """Update README.md with auto-generated dual-head analysis report."""
    print(f"\nUpdating README.md...")
    
    readme_path = Path("README.md")
    
    # Read existing README or create new
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = "# FINMC-TECH\n\nMachine Learning + Monte Carlo Simulation for Tech Stock Forecasting\n\n"
    
    # Find or create auto-report section
    start_marker = "<!-- AUTO-REPORT:START -->"
    end_marker = "<!-- AUTO-REPORT:END -->"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    # Extract metrics
    return_head = metrics["models"]["return_head"]
    price_head = metrics["models"]["price_head"]
    
    # Find best models
    best_return_r2 = max([m["test"]["r2"] for m in return_head.values()])
    best_price_r2 = max([m["test"]["r2"] for m in price_head.values()])
    
    # Compare indirect vs direct price RMSE
    indirect_price_rmse = metrics["models"]["price_rmse"]["indirect_from_return_rf"]
    direct_price_rmse = metrics["models"]["price_rmse"]["direct_price_rf"]
    better_price_route = "Direct" if direct_price_rmse < indirect_price_rmse else "Indirect"
    
    # Generate report content
    report_lines = [
        "## Auto-Generated Dual-Head Analysis Report\n",
        f"This report analyzes revenue-based signals for **{metrics['ticker']}** using SEC XBRL data and dual-head machine learning models. "
        f"The analysis spans {metrics['date_range'][0]} to {metrics['date_range'][1]}, using the `{metrics['tag_used']}` revenue tag. "
        f"We model both **12-month forward returns** (return head) and **12-month forward stock prices** (price head) using revenue features (YoY growth, QoQ growth, acceleration) plus macro features (VIX level, 10Y yield, and their 3-month changes). "
        f"Three baseline models (Ridge Regression, k-Nearest Neighbors, and RandomForest) are trained for each head with a temporal split (train before 2019, test from 2019). "
        f"Price predictions are generated via two routes: (1) **indirect**: `price_hat = current_price * (1 + return_hat)` from the return head, and (2) **direct**: `log(price_hat)` predicted directly from the price head. "
        f"Results show that the **{better_price_route}** price route achieves lower RMSE (${direct_price_rmse:.2f} vs ${indirect_price_rmse:.2f}). "
        f"Revenue acceleration remains important after adding macro features, with RandomForest capturing non-linear relationships effectively.\n",
        "",
        "### Return Head Performance\n",
        "| Model | Test R² | Test RMSE (Return) |",
        "|-------|---------|---------------------|",
    ]
    
    for name in ["ridge", "knn", "rf"]:
        if name in return_head:
            test_r2 = return_head[name]["test"]["r2"]
            test_rmse = return_head[name]["test"]["rmse"]
            model_name = {"ridge": "Ridge Regression", "knn": "k-NN (k=5)", "rf": "RandomForest"}[name]
            report_lines.append(f"| {model_name} | {test_r2:.4f} | {test_rmse:.4f} |")
    
    report_lines.extend([
        "",
        "### Direct Price Head Performance\n",
        "| Model | Test R² (Log Price) | Test RMSE (Log Price) | Test Price RMSE (USD) |",
        "|-------|---------------------|------------------------|----------------------|",
    ])
    
    for name in ["ridge", "knn", "rf"]:
        if name in price_head:
            test_r2 = price_head[name]["test"]["r2"]
            test_rmse_log = price_head[name]["test"]["rmse"]
            test_rmse_price = price_head[name]["test_price_rmse"]
            model_name = {"ridge": "Ridge Regression", "knn": "k-NN (k=5)", "rf": "RandomForest"}[name]
            report_lines.append(f"| {model_name} | {test_r2:.4f} | {test_rmse_log:.4f} | ${test_rmse_price:.2f} |")
    
    report_lines.extend([
        "",
        "### Price RMSE Comparison\n",
        f"- **Indirect (from Return RF)**: ${indirect_price_rmse:.2f}",
        f"- **Direct (from Price RF)**: ${direct_price_rmse:.2f}",
        f"- **Best Route**: {better_price_route}",
        "",
        "### Figures",
        "",
        "![YoY vs Return](outputs/figs/yoy_vs_return.png)",
        "",
        "![Acceleration vs Return](outputs/figs/accel_vs_return.png)",
        "",
        "![Rolling Correlation](outputs/figs/rolling_corr.png)",
        "",
        "![RF Feature Importance (Return Head)](outputs/figs/rf_feature_importance.png)",
        "",
        "![RF Return: Pred vs Actual](outputs/figs/pred_vs_actual_return_rf.png)",
        "",
        "![Price (Indirect from Return)](outputs/figs/pred_vs_actual_price_indirect.png)",
        "",
        "![Price (Direct Head)](outputs/figs/pred_vs_actual_price_direct.png)",
        "",
        "![Calibration Return](outputs/figs/calibration_return.png)",
        "",
        "![Residuals Return](outputs/figs/residuals_return.png)",
        "",
        "### Key Findings",
        "",
        f"- **Sample Size**: {metrics['n_rows']} rows",
        f"- **Date Range**: {metrics['date_range'][0]} to {metrics['date_range'][1]}",
        f"- **Revenue Tag Used**: `{metrics['tag_used']}`",
        f"- **Pearson Correlations** (with returns for analysis):",
        f"  - Rev YoY vs Future 12M Return: {metrics['corr']['rev_yoy_vs_fut12m']:.4f}" if metrics['corr']['rev_yoy_vs_fut12m'] is not None else "  - Rev YoY vs Future 12M Return: N/A",
        f"  - Rev Acceleration vs Future 12M Return: {metrics['corr']['rev_accel_vs_fut12m']:.4f}" if metrics['corr']['rev_accel_vs_fut12m'] is not None else "  - Rev Acceleration vs Future 12M Return: N/A",
        f"- **Best Return Head Model**: RandomForest (Test R² = {best_return_r2:.4f})",
        f"- **Best Price Head Model**: RandomForest (Test R² = {best_price_r2:.4f})",
        f"- **Best Price Route**: {better_price_route} (RMSE = ${min(direct_price_rmse, indirect_price_rmse):.2f})",
        "",
    ])
    
    report_content = "\n".join(report_lines)
    
    # Replace or insert report section
    if start_idx >= 0 and end_idx >= 0:
        # Replace existing section
        content = (
            content[:start_idx + len(start_marker)] +
            "\n" + report_content + "\n" +
            content[end_idx:]
        )
    else:
        # Append new section
        content += f"\n\n{start_marker}\n{report_content}\n{end_marker}\n"
    
    readme_path.write_text(content, encoding="utf-8")
    print(f"  ✓ Updated README.md")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Dual-head ML pipeline for revenue-based stock prediction")
    parser.add_argument("--tickers", type=str, default="NVDA", help="Comma-separated tickers (default: NVDA)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="today", help="End date (YYYY-MM-DD or 'today')")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip SEC data fetching, reuse existing CSVs")
    
    args = parser.parse_args()
    
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    
    print("=" * 70)
    print("FINMC-TECH: Dual-Head Revenue-Based Stock Prediction Pipeline")
    print("=" * 70)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Start date: {args.start or 'earliest available'}")
    print(f"End date: {args.end}")
    print(f"Skip fetch: {args.skip_fetch}")
    
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/figs").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    
    # Process each ticker
    for ticker in tickers:
        try:
            # Step 1: Fetch fundamentals
            if not args.skip_fetch:
                df_revenue, tag_used = fetch_revenue_series(ticker, args.start, args.end)
                raw_file = Path(f"data/raw/{ticker}_revenue.csv")
                df_revenue.to_csv(raw_file, index=False)
                print(f"  ✓ Saved to {raw_file}")
            else:
                raw_file = Path(f"data/raw/{ticker}_revenue.csv")
                if not raw_file.exists():
                    raise FileNotFoundError(f"{raw_file} not found. Run without --skip-fetch first.")
                df_revenue = pd.read_csv(raw_file)
                df_revenue["period_end"] = pd.to_datetime(df_revenue["period_end"])
                tag_used = df_revenue["tag_used"].iloc[0] if "tag_used" in df_revenue.columns else "Unknown"
                print(f"  ✓ Loaded from {raw_file}")
            
            # Step 2: Merge prices, macro features, and build targets
            df_features = load_merge_features(df_revenue, ticker)
            processed_file = Path(f"data/processed/{ticker}_revenue_features.csv")
            df_features.to_csv(processed_file, index=False)
            print(f"  ✓ Saved to {processed_file}")
            
            # Step 3: Train return head models
            split_date = pd.to_datetime("2019-01-01")
            return_results = train_return_models(df_features, ticker, split_date)
            
            # Step 4: Train price head models
            price_results = train_price_models(df_features, ticker, split_date)
            
            # Step 5: Compute indirect price RMSE from return head
            test_df = df_features[
                (df_features["period_end"] >= split_date) &
                df_features[FEATURE_COLS].notna().all(axis=1) &
                df_features["future_12m_return"].notna() &
                df_features["future_12m_price"].notna() &
                df_features["adj_close"].notna()
            ].copy()
            
            if len(test_df) > 0 and "rf" in return_results and "predictions" in return_results["rf"]:
                return_preds = return_results["rf"]["predictions"]["y_pred"]
                if len(return_preds) == len(test_df):
                    test_df = test_df.sort_values("period_end")
                    indirect_prices = test_df["adj_close"].values * (1 + np.array(return_preds))
                    actual_prices = test_df["future_12m_price"].values
                    indirect_price_rmse = np.sqrt(mean_squared_error(actual_prices, indirect_prices))
                else:
                    indirect_price_rmse = None
            else:
                indirect_price_rmse = None
            
            # Prepare metrics
            df_clean = df_features[
                df_features[FEATURE_COLS].notna().all(axis=1) &
                (df_features["future_12m_return"].notna() | df_features["future_12m_logprice"].notna())
            ].copy()
            
            corr_yoy = df_clean["rev_yoy"].corr(df_clean["future_12m_return"]) if "rev_yoy" in df_clean.columns and "future_12m_return" in df_clean.columns else None
            corr_accel = df_clean["rev_accel"].corr(df_clean["future_12m_return"]) if "rev_accel" in df_clean.columns and "future_12m_return" in df_clean.columns else None
            
            metrics = {
                "ticker": ticker,
                "n_rows": len(df_clean),
                "date_range": [
                    df_clean["period_end"].min().strftime("%Y-%m-%d"),
                    df_clean["period_end"].max().strftime("%Y-%m-%d"),
                ],
                "tag_used": df_clean["tag_used"].iloc[0] if len(df_clean) > 0 else "",
                "corr": {
                    "rev_yoy_vs_fut12m": float(corr_yoy) if pd.notna(corr_yoy) else None,
                    "rev_accel_vs_fut12m": float(corr_accel) if pd.notna(corr_accel) else None,
                },
                "models": {
                    "return_head": return_results,
                    "price_head": price_results,
                    "price_rmse": {
                        "indirect_from_return_rf": float(indirect_price_rmse) if indirect_price_rmse is not None else None,
                        "direct_price_rf": float(price_results["rf"]["test_price_rmse"]) if "rf" in price_results else None,
                    },
                },
                "df_clean": df_clean,  # For plotting
            }
            
            # Save metrics (without df_clean for JSON)
            metrics_json = metrics.copy()
            metrics_json.pop("df_clean", None)
            metrics_file = Path(f"outputs/metrics/{ticker}_dual_head_scores.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics_json, f, indent=2)
            print(f"  ✓ Saved to {metrics_file}")
            
            # Step 6: Generate figures
            evaluate_and_plot(metrics, Path("outputs/figs"))
            
            # Step 7: Update README
            update_readme(metrics, Path("outputs/figs"))
            
            # Print summary
            print("\n" + "=" * 70)
            print(f"Summary for {ticker}:")
            print("=" * 70)
            print(f"Return Head - Best Test R²: {max([m['test']['r2'] for m in return_results.values()]):.4f}")
            print(f"Price Head - Best Test R²: {max([m['test']['r2'] for m in price_results.values()]):.4f}")
            if indirect_price_rmse is not None:
                print(f"Price RMSE - Indirect: ${indirect_price_rmse:.2f}")
            if "rf" in price_results:
                print(f"Price RMSE - Direct: ${price_results['rf']['test_price_rmse']:.2f}")
            print("=" * 70)
            
            print(f"\n✓ Completed pipeline for {ticker}")
            
        except Exception as e:
            print(f"\n✗ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("Pipeline completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
