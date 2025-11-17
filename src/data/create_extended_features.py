"""
Create extended features beyond revenue for ML modeling.

This module adds:
- Price momentum features
- Technical indicators
- Market macro features
- Time features
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, List
from datetime import datetime
from pathlib import Path

# Import technical indicators from ml/features
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.ml.features import compute_technical_indicators, compute_rsi, compute_atr


def create_price_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price momentum features from adj_close.
    
    Features:
    - price_returns_1m: 1-month return
    - price_returns_3m: 3-month return
    - price_returns_6m: 6-month return
    - price_returns_12m: 12-month return
    - price_momentum: Short-term momentum
    - price_volatility: Rolling volatility
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with adj_close and px_date columns
    
    Returns
    -------
    pd.DataFrame
        Data with price momentum features added
    """
    df = df.copy()
    
    if "adj_close" not in df.columns:
        print("  Warning: adj_close not found, skipping price momentum features")
        return df
    
    # Sort by date
    df = df.sort_values("px_date").copy()
    
    # Calculate returns at different horizons
    # For quarterly data, approximate months as quarters
    df["price_returns_1m"] = df["adj_close"].pct_change(1)  # QoQ ≈ 3 months
    df["price_returns_3m"] = df["adj_close"].pct_change(1)  # Same as 1m for quarterly
    df["price_returns_6m"] = df["adj_close"].pct_change(2)  # 2 quarters
    df["price_returns_12m"] = df["adj_close"].pct_change(4)  # 4 quarters (YoY)
    
    # Momentum (current vs previous)
    df["price_momentum"] = df["adj_close"] / df["adj_close"].shift(1) - 1
    
    # Rolling volatility (using returns)
    df["price_volatility"] = df["price_returns_1m"].rolling(window=4, min_periods=2).std()
    
    # Price ratios (relative to moving averages)
    df["price_ma_4q"] = df["adj_close"].rolling(window=4, min_periods=1).mean()
    df["price_to_ma_4q"] = df["adj_close"] / df["price_ma_4q"]
    
    print("  Created price momentum features: price_returns_1m, price_returns_3m, "
          "price_returns_6m, price_returns_12m, price_momentum, price_volatility, "
          "price_to_ma_4q")
    
    return df


def create_technical_indicators_from_prices(df: pd.DataFrame, ticker: str = "NVDA") -> pd.DataFrame:
    """
    Create technical indicators by downloading daily price data.
    
    For quarterly revenue data, we download daily prices and aggregate to quarterly.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with px_date and adj_close columns
    ticker : str
        Stock ticker symbol
    
    Returns
    -------
    pd.DataFrame
        Data with technical indicators added
    """
    df = df.copy()
    
    if "px_date" not in df.columns or "adj_close" not in df.columns:
        print("  Warning: px_date or adj_close not found, skipping technical indicators")
        return df
    
    print("  Downloading daily price data for technical indicators...")
    
    # Get date range
    min_date = pd.to_datetime(df["px_date"]).min() - pd.Timedelta(days=300)
    max_date = pd.to_datetime(df["px_date"]).max() + pd.Timedelta(days=30)
    
    try:
        stock = yf.Ticker(ticker)
        price_data = stock.history(start=min_date, end=max_date)
        
        if price_data.empty:
            print("  Warning: No price data downloaded")
            return df
        
        # Reset index
        price_data = price_data.reset_index()
        price_data["Date"] = pd.to_datetime(price_data["Date"])
        
        # Compute technical indicators
        price_data = compute_technical_indicators(
            price_data,
            close_col="Close",
            high_col="High",
            low_col="Low",
            volume_col="Volume"
        )
        
        # For each revenue record, find the closest trading day and get indicators
        tech_features = []
        for _, row in df.iterrows():
            px_date = pd.to_datetime(row["px_date"])
            
            # Find closest trading day
            closest_idx = (price_data["Date"] - px_date).abs().idxmin()
            closest_row = price_data.iloc[closest_idx]
            
            tech_features.append({
                "rsi_14": closest_row.get("rsi", None),
                "macd": closest_row.get("macd", None),
                "macd_signal": closest_row.get("macd_signal", None),
                "bb_position": closest_row.get("bb_position", None),
                "stoch_k": closest_row.get("stoch_k", None),
                "atr": closest_row.get("atr", None),
            })
        
        tech_df = pd.DataFrame(tech_features, index=df.index)
        df = pd.concat([df, tech_df], axis=1)
        
        print("  Created technical indicators: rsi_14, macd, macd_signal, "
              "bb_position, stoch_k, atr")
        
    except Exception as e:
        print(f"  Warning: Could not create technical indicators: {e}")
    
    return df


def create_market_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create market macro features.
    
    Features:
    - sp500_level: S&P 500 index level
    - sp500_returns: S&P 500 returns
    - dxy_level: Dollar index (if available)
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with px_date column
    
    Returns
    -------
    pd.DataFrame
        Data with market macro features added
    """
    df = df.copy()
    
    if "px_date" not in df.columns:
        print("  Warning: px_date not found, skipping market macro features")
        return df
    
    print("  Downloading market macro data (SPY for S&P 500)...")
    
    # Get date range
    min_date = pd.to_datetime(df["px_date"]).min() - pd.Timedelta(days=30)
    max_date = pd.to_datetime(df["px_date"]).max() + pd.Timedelta(days=30)
    
    try:
        # Download S&P 500 data (using SPY ETF as proxy)
        spy = yf.Ticker("SPY")
        spy_data = spy.history(start=min_date, end=max_date)
        
        if not spy_data.empty:
            spy_data = spy_data.reset_index()
            spy_data["Date"] = pd.to_datetime(spy_data["Date"]).dt.date
            
            # For each revenue record, find closest trading day
            sp500_features = []
            for _, row in df.iterrows():
                px_date = pd.to_datetime(row["px_date"]).date()
                
                # Find closest date
                closest_idx = (pd.to_datetime(spy_data["Date"]) - pd.to_datetime(px_date)).abs().idxmin()
                closest_row = spy_data.iloc[closest_idx]
                
                sp500_features.append({
                    "sp500_level": closest_row["Close"],
                    "sp500_returns": closest_row["Close"] / spy_data.iloc[max(0, closest_idx-1)]["Close"] - 1 if closest_idx > 0 else None,
                })
            
            macro_df = pd.DataFrame(sp500_features, index=df.index)
            df = pd.concat([df, macro_df], axis=1)
            
            print("  Created market macro features: sp500_level, sp500_returns")
        else:
            print("  Warning: No SPY data downloaded")
    
    except Exception as e:
        print(f"  Warning: Could not create market macro features: {e}")
    
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    
    Features:
    - quarter: Quarter of year (1-4)
    - month: Month of year (1-12)
    - year: Year
    - days_since_start: Days since first record
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with px_date or end_date column
    
    Returns
    -------
    pd.DataFrame
        Data with time features added
    """
    df = df.copy()
    
    date_col = "px_date" if "px_date" in df.columns else "end_date"
    if date_col not in df.columns:
        print("  Warning: No date column found, skipping time features")
        return df
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    df["quarter"] = df[date_col].dt.quarter
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    
    # Days since start
    start_date = df[date_col].min()
    df["days_since_start"] = (df[date_col] - start_date).dt.days
    
    print("  Created time features: quarter, month, year, days_since_start")
    
    return df


def create_interaction_features(
    df: pd.DataFrame,
    use_kronecker: bool = True,
    macro_cols: Optional[List[str]] = None,
    micro_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create interaction features (multiplicative combinations).
    
    Following Gu-Kelly-Xiu (2020) RFS methodology, these interactions represent
    state-dependent effects: the impact of a firm characteristic depends on the
    macro environment (macro × micro = Kronecker product structure).
    
    Two modes:
    1. Simple mode (use_kronecker=False): Creates a few fixed interactions
    2. Kronecker mode (use_kronecker=True): Creates all macro × micro interactions
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with base features
    use_kronecker : bool
        If True, creates comprehensive macro × micro interactions (Kronecker product)
        If False, creates only a few fixed interactions (backward compatible)
    macro_cols : Optional[List[str]]
        List of macro feature columns. If None, auto-detects from common macro names.
    micro_cols : Optional[List[str]]
        List of firm (micro) feature columns. If None, auto-detects from common firm names.
    
    Returns
    -------
    pd.DataFrame
        Data with interaction features added
    """
    df = df.copy()
    
    if not use_kronecker:
        # Simple mode: backward compatible with fixed interactions
        # Revenue × VIX
        if "rev_yoy" in df.columns and "vix_level" in df.columns:
            df["rev_yoy_x_vix"] = df["rev_yoy"] * df["vix_level"]
        
        # Revenue × SP500
        if "rev_qoq" in df.columns and "sp500_returns" in df.columns:
            df["rev_qoq_x_sp500"] = df["rev_qoq"] * df["sp500_returns"]
        
        # Price momentum × volatility
        if "price_momentum" in df.columns and "price_volatility" in df.columns:
            df["price_momentum_x_volatility"] = df["price_momentum"] * df["price_volatility"]
        
        # VIX × Treasury yield
        if "vix_level" in df.columns and "tnx_yield" in df.columns:
            df["vix_x_tnx"] = df["vix_level"] * df["tnx_yield"]
        
        print("  Created simple interaction features (backward compatible)")
        return df
    
    # Kronecker mode: comprehensive macro × micro interactions
    # Auto-detect macro columns if not provided
    if macro_cols is None:
        macro_candidates = [
            'cpi_yoy', 'cpi_qoq', 'vix_level', 'vix_q_avg', 'vix_q_end',
            'tnx_yield', 'y10', 'fedfunds', 'FEDFUNDS', 'DGS10',
            'gdp_yoy', 'gdp_qoq', 'GDP', 'GDPC1',
            'sp500_level', 'sp500_returns', 'vix_change_3m', 'tnx_change_3m'
        ]
        macro_cols = [col for col in macro_candidates if col in df.columns]
    
    # Auto-detect micro (firm) columns if not provided
    if micro_cols is None:
        micro_candidates = [
            'rev_yoy', 'rev_qoq', 'rev_accel', 'revenue',
            'gross_margin', 'operating_margin', 'net_margin',
            'dc_share', 'gaming_share',
            'eps_gaap', 'eps_ng',
            'capex', 'rnd', 'fcf',
            'price_mom_3m', 'price_mom_6m', 'price_mom_12m',
            'price_returns_1m', 'price_returns_3m', 'price_returns_6m', 'price_returns_12m',
            'price_momentum', 'price_volatility', 'price_vol_4q',
            'inventory', 'dio', 'dso', 'roe_ttm', 'roa_ttm', 'leverage'
        ]
        micro_cols = [col for col in micro_candidates if col in df.columns]
    
    if not macro_cols or not micro_cols:
        print(f"  Warning: Insufficient macro/micro columns for Kronecker interactions")
        print(f"    Available macro: {macro_cols}")
        print(f"    Available micro: {micro_cols}")
        # Fall back to simple mode
        return create_interaction_features(df, use_kronecker=False)
    
    # Create all macro × micro interactions (Kronecker product)
    interaction_count = 0
    prefix = "ix_"  # Interaction prefix
    
    for macro_col in macro_cols:
        for micro_col in micro_cols:
            # Skip if either column has too many NaN
            if df[macro_col].isna().sum() > len(df) * 0.5:
                continue
            if df[micro_col].isna().sum() > len(df) * 0.5:
                continue
            
            # Create interaction: macro × micro
            interaction_name = f"{prefix}{macro_col}__{micro_col}"
            df[interaction_name] = df[macro_col] * df[micro_col]
            interaction_count += 1
    
    print(f"  Created {interaction_count} Kronecker interaction features (macro × micro)")
    print(f"    Macro features: {len(macro_cols)}")
    print(f"    Micro features: {len(micro_cols)}")
    print(f"    Total interactions: {len(macro_cols)} × {len(micro_cols)} = {interaction_count}")
    
    return df


def create_all_extended_features(df: pd.DataFrame, ticker: str = "NVDA") -> pd.DataFrame:
    """
    Create all extended features beyond revenue.
    
    This function combines all feature creation functions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Base data with revenue and price columns
    ticker : str
        Stock ticker symbol
    
    Returns
    -------
    pd.DataFrame
        Data with all extended features added
    """
    print("\nCreating extended features (beyond revenue)...")
    
    # Price momentum features
    df = create_price_momentum_features(df)
    
    # Technical indicators (from daily prices)
    df = create_technical_indicators_from_prices(df, ticker=ticker)
    
    # Market macro features
    df = create_market_macro_features(df)
    
    # Time features
    df = create_time_features(df)
    
    # Interaction features
    df = create_interaction_features(df)
    
    print(f"  Extended features created. Total columns: {len(df.columns)}")
    
    return df

