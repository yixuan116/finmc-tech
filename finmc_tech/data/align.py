"""
Align macro and firm data to common time index.

Provides two alignment functions:
1. align_data() - Legacy quarterly alignment (backward compatibility)
2. align_macro_firm() - Monthly panel alignment with outer-join by MonthEnd
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.create_extended_features import create_all_extended_features


def align_macro_firm(
    macro_df: pd.DataFrame,
    firm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align macro and firm data by MonthEnd using outer-join.
    
    Forward-fills macro quarterly items (e.g., GDP) to monthly frequency.
    Ensures returns column "Ret" remains untouched.
    
    Args:
        macro_df: Monthly macro data (indexed by MonthEnd or has Date column)
        firm_df: Firm data (monthly or quarterly, with date column)
    
    Returns:
        DataFrame with MonthEnd index, outer-joined macro and firm data
    """
    from finmc_tech.config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Aligning macro and firm data by MonthEnd...")
    
    # Prepare macro_df: ensure MonthEnd index
    macro_df = macro_df.copy()
    if isinstance(macro_df.index, pd.DatetimeIndex):
        # Already has datetime index, convert to MonthEnd
        macro_df.index = macro_df.index.to_period("M").to_timestamp("M")
    elif "Date" in macro_df.columns:
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        macro_df = macro_df.set_index("Date")
        macro_df.index = macro_df.index.to_period("M").to_timestamp("M")
    else:
        raise ValueError("macro_df must have DatetimeIndex or 'Date' column")
    
    # Prepare firm_df: identify date column and convert to MonthEnd
    firm_df = firm_df.copy()
    
    # Find date column in firm_df
    date_col = None
    for col in ["Date", "date", "px_date", "end_date", "period_end"]:
        if col in firm_df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("firm_df must have a date column (Date, date, px_date, end_date, or period_end)")
    
    firm_df[date_col] = pd.to_datetime(firm_df[date_col])
    firm_df = firm_df.set_index(date_col)
    firm_df.index = firm_df.index.to_period("M").to_timestamp("M")
    
    # Outer-join on MonthEnd index
    aligned_df = macro_df.join(firm_df, how="outer", sort=True)
    
    # Forward-fill macro quarterly items (GDP is quarterly, others are monthly)
    # Identify quarterly columns (those with many NaN values)
    quarterly_cols = []
    for col in macro_df.columns:
        if col in ["GDP", "GDPC1"]:  # Known quarterly series
            quarterly_cols.append(col)
        elif aligned_df[col].isna().sum() > len(aligned_df) * 0.3:  # >30% NaN suggests quarterly
            quarterly_cols.append(col)
    
    if quarterly_cols:
        logger.info(f"  Forward-filling quarterly columns: {quarterly_cols}")
        for col in quarterly_cols:
            if col in aligned_df.columns:
                aligned_df[col] = aligned_df[col].ffill()
    
    # Ensure "Ret" column remains untouched (if it exists)
    if "Ret" in aligned_df.columns:
        logger.info("  Preserving 'Ret' column (returns)")
    
    logger.info(f"  ✓ Aligned {len(aligned_df)} monthly observations")
    logger.info(f"  Date range: {aligned_df.index.min().date()} to {aligned_df.index.max().date()}")
    
    return aligned_df


def align_data(
    firm_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    ticker: str,
    align_to: str = "px_date",
) -> pd.DataFrame:
    """
    Align macro and firm data to quarterly time index.
    
    Args:
        firm_df: Quarterly firm data (revenue, prices)
        macro_df: Daily macro data (VIX, TNX, SP500)
        ticker: Stock ticker symbol
        align_to: Column to align to ("px_date" or "end_date")
    
    Returns:
        DataFrame with all features aligned to quarterly frequency
    """
    print(f"Aligning data to {align_to}...")
    
    # Validate input
    if align_to not in firm_df.columns:
        raise ValueError(f"Column {align_to} not found in firm_df")
    
    # Prepare data: convert dates
    firm_df = firm_df.copy()
    firm_df[align_to] = pd.to_datetime(firm_df[align_to])
    macro_df = macro_df.copy()
    
    # Handle different macro_df formats:
    # 1. DataFrame with "Date" column (from old fetch_macro_data)
    # 2. DataFrame with MonthEnd index (from new fetch_macro)
    if "Date" in macro_df.columns:
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        date_col = "Date"
    elif isinstance(macro_df.index, pd.DatetimeIndex):
        # New format: index is already datetime
        macro_df = macro_df.reset_index()
        if "index" in macro_df.columns:
            macro_df.rename(columns={"index": "Date"}, inplace=True)
        elif "DATE" in macro_df.columns:
            macro_df.rename(columns={"DATE": "Date"}, inplace=True)
        date_col = "Date"
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
    else:
        raise ValueError("macro_df must have 'Date' column or DatetimeIndex")
    
    # Align macro data to quarterly firm records
    # This enables lag operations (shift, pct_change) on all features
    aligned_features = []
    for idx, row in firm_df.iterrows():
        target_date = row[align_to]
        closest_idx = (macro_df[date_col] - target_date).abs().idxmin()
        closest_row = macro_df.iloc[closest_idx]
        macro_features = closest_row.drop(date_col).to_dict()
        aligned_features.append(macro_features)
    
    # Merge macro features into firm data
    macro_features_df = pd.DataFrame(aligned_features, index=firm_df.index)
    aligned_df = pd.concat([firm_df, macro_features_df], axis=1)
    
    # Create extended features (price momentum, technical indicators, etc.)
    print("  Creating extended features...")
    aligned_df = create_all_extended_features(aligned_df, ticker)
    
    print(f"  ✓ Aligned {len(aligned_df)} records")
    return aligned_df

