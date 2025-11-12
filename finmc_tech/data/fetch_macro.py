"""
Fetch monthly macro economic indicators from FRED.

Fetches: CPI, VIX, 10Y Treasury, Fed Funds Rate, Real GDP
Resamples to monthly frequency and aligns all series.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pandas_datareader.data as web

from finmc_tech.config import cfg, get_logger

logger = get_logger(__name__)


# FRED series definitions
FRED_SERIES = {
    "CPIAUCSL": "CPI",      # Consumer Price Index (monthly)
    "VIXCLS": "VIX",         # VIX (daily → monthly mean)
    "DGS10": "DGS10",        # 10Y Treasury (daily → monthly mean)
    "FEDFUNDS": "FEDFUNDS",  # Fed Funds Rate (monthly)
    "GDPC1": "GDP",          # Real GDP (quarterly → monthly ffill)
}


def _fetch_fred_series(
    series_id: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str] = None,
) -> Optional[pd.Series]:
    """Fetch a single FRED series with error handling."""
    try:
        # pandas_datareader DataReader signature:
        # DataReader(name, data_source, start, end, api_key=None, ...)
        data = web.DataReader(series_id, "fred", start, end, api_key=api_key)
        
        if data.empty:
            logger.warning(f"  {series_id}: Empty data returned")
            return None
        
        # Extract series (DataReader returns DataFrame with series_id as column)
        if isinstance(data, pd.DataFrame):
            series = data[series_id] if series_id in data.columns else data.iloc[:, 0]
        else:
            series = data
        
        logger.info(f"  ✓ {series_id}: {len(series)} observations")
        return series
    
    except Exception as e:
        logger.warning(f"  {series_id}: Failed to fetch - {e}")
        return None


def _resample_to_monthly(series: pd.Series, method: str = "mean") -> pd.Series:
    """Resample series to month-end frequency."""
    if method == "mean":
        return series.resample("M").mean()
    elif method == "last":
        return series.resample("M").last()
    elif method == "ffill":
        # Forward fill for quarterly data
        monthly_index = pd.date_range(series.index.min(), series.index.max(), freq="M")
        return series.reindex(monthly_index).ffill()
    else:
        raise ValueError(f"Unknown resample method: {method}")


def _load_cached_series(series_id: str, cache_dir: Path, start: str, end: str) -> Optional[pd.Series]:
    """Load cached series from CSV if available."""
    cache_file = cache_dir / f"fred_{series_id}_{start}_{end}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            series = df.iloc[:, 0]
            logger.info(f"  ✓ {series_id}: Loaded from cache ({len(series)} observations)")
            return series
        except Exception as e:
            logger.warning(f"  {series_id}: Failed to load cache - {e}")
    return None


def _save_cached_series(series: pd.Series, series_id: str, cache_dir: Path, start: str, end: str) -> None:
    """Save series to cache CSV."""
    cache_file = cache_dir / f"fred_{series_id}_{start}_{end}.csv"
    try:
        series.to_frame().to_csv(cache_file)
        logger.debug(f"  ✓ {series_id}: Cached to {cache_file}")
    except Exception as e:
        logger.warning(f"  {series_id}: Failed to cache - {e}")


def fetch_macro(
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch monthly macro economic indicators from FRED.
    
    Fetches and aligns:
    - CPIAUCSL (CPI, monthly)
    - VIXCLS (VIX, daily → monthly mean)
    - DGS10 (10Y Treasury, daily → monthly mean)
    - FEDFUNDS (Fed Funds Rate, monthly)
    - GDPC1 (Real GDP, quarterly → monthly ffill)
    
    Args:
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        cache_dir: Directory to cache data. If None, uses cfg.CACHE_DIR
    
    Returns:
        DataFrame indexed by MonthEnd with columns: ["CPI", "VIX", "DGS10", "FEDFUNDS", "GDP"]
    """
    if cache_dir is None:
        cache_dir = cfg.cache_dir
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    logger.info(f"Fetching macro data from {start} to {end}...")
    
    # Fetch each series
    series_data = {}
    
    for fred_id, col_name in FRED_SERIES.items():
        logger.info(f"Fetching {fred_id} ({col_name})...")
        
        # Try cache first
        cached = _load_cached_series(fred_id, cache_dir, start, end)
        if cached is not None:
            series_data[col_name] = cached
            continue
        
        # Fetch from FRED
        series = _fetch_fred_series(fred_id, start_dt, end_dt, cfg.FRED_API_KEY)
        
        if series is None:
            logger.warning(f"  {fred_id}: Skipping (fetch failed and no cache)")
            continue
        
        # Resample to monthly based on series type
        if fred_id == "CPIAUCSL":
            # Already monthly, just align to month-end
            monthly = series.resample("M").last()
        elif fred_id == "VIXCLS":
            # Daily → monthly mean
            monthly = _resample_to_monthly(series, method="mean")
        elif fred_id == "DGS10":
            # Daily → monthly mean
            monthly = _resample_to_monthly(series, method="mean")
        elif fred_id == "FEDFUNDS":
            # Already monthly, align to month-end
            monthly = series.resample("M").last()
        elif fred_id == "GDPC1":
            # Quarterly → monthly ffill
            monthly = _resample_to_monthly(series, method="ffill")
        else:
            monthly = series.resample("M").last()
        
        # Convert to MonthEnd index
        monthly.index = monthly.index.to_period("M").to_timestamp("M")
        
        # Cache the resampled series
        _save_cached_series(monthly, fred_id, cache_dir, start, end)
        
        series_data[col_name] = monthly
    
    if not series_data:
        logger.error("No macro series fetched successfully")
        return pd.DataFrame()
    
    # Combine into single DataFrame
    df = pd.DataFrame(series_data)
    
    # Forward fill missing values (especially for GDP which is quarterly)
    df = df.ffill()
    
    # Drop rows with any remaining NA (at the very end only)
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with remaining NA values")
    
    # Ensure index is MonthEnd
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp("M")
    
    # Unit-like assertions
    assert df.index.is_monotonic_increasing, "Index must be monotonic increasing"
    assert not df.index.duplicated().any(), "Index must not have duplicates"
    
    logger.info(f"✓ Fetched {len(df)} monthly observations")
    logger.info(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    return df


# Backward compatibility alias
def fetch_macro_data(
    indicators: list,
    start_date: str,
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Backward compatibility wrapper for fetch_macro.
    
    Maps old indicator names to FRED series and calls fetch_macro.
    """
    if end_date is None:
        end_date = cfg.END_DATE
    
    # Fetch all macro series
    df = fetch_macro(start_date, end_date, cache_dir)
    
    # Map old indicator names if needed
    if "VIX" in indicators and "VIX" not in df.columns:
        logger.warning("VIX not available in FRED data")
    if "TNX" in indicators and "DGS10" in df.columns:
        df["tnx_yield"] = df["DGS10"] / 100.0  # Convert to decimal
    
    return df

