"""
Aggregate NVDA daily WRDS data to monthly frequency.

Reads daily data from nvda_daily_wrds_1999_2025.csv and aggregates to monthly
features for long-cycle ML / Monte Carlo analysis.

Outputs monthly features:
- Month-end price (adj_close)
- Monthly return (ret_1m)
- Annualized realized volatility (realized_vol_21d_annual)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from finmc_tech.config import get_logger

logger = get_logger(__name__)


def load_daily_data(input_path: Path) -> pd.DataFrame:
    """
    Load daily WRDS data from CSV file.
    
    Args:
        input_path: Path to input CSV file
    
    Returns:
        DataFrame with date as index, sorted by date
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading daily data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Auto-detect date column (case-insensitive)
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'datadate', 'period_end']:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(
            f"Could not find date column in {list(df.columns)}. "
            "Expected 'date', 'datadate', or 'period_end'"
        )
    
    # Convert date to datetime and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    logger.info(f"Loaded {len(df)} daily rows")
    logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data to monthly frequency (month-end).
    
    Computes:
    - Month-end price (adj_close)
    - Monthly return (ret_1m)
    - Annualized realized volatility (realized_vol_21d_annual)
    
    Args:
        df: Daily DataFrame with date index and adj_close, ret columns
    
    Returns:
        Monthly DataFrame indexed by month-end dates
    """
    logger.info("Aggregating daily data to monthly frequency...")
    
    # Detect price and return columns (case-insensitive)
    price_col = None
    ret_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['adj_close', 'adjclose', 'close', 'prc']:
            price_col = col
        elif col_lower in ['ret', 'return', 'returns']:
            ret_col = col
    
    if price_col is None:
        raise ValueError(
            f"Could not find price column. Available columns: {list(df.columns)}"
        )
    
    logger.info(f"Using price column: {price_col}")
    if ret_col:
        logger.info(f"Using return column: {ret_col}")
    else:
        logger.warning("No return column found, will compute from prices")
    
    # Resample to month-end: take last trading day of each month
    monthly = pd.DataFrame()
    monthly['adj_close'] = df[price_col].resample('ME').last()
    
    # Compute monthly return: month-end price / previous month-end price - 1
    monthly['ret_1m'] = monthly['adj_close'].pct_change()
    
    # Compute annualized realized volatility
    # Use 21 trading days rolling window, then annualize with sqrt(252)
    if ret_col and ret_col in df.columns:
        # Use provided returns if available
        daily_ret = df[ret_col].dropna()
    else:
        # Compute returns from prices
        daily_ret = df[price_col].pct_change().dropna()
    
    if len(daily_ret) > 0:
        # Rolling 21-day standard deviation
        rolling_std = daily_ret.rolling(window=21, min_periods=1).std()
        
        # Resample to month-end (take last value of each month)
        monthly['realized_vol_21d_annual'] = (
            rolling_std.resample('ME').last() * np.sqrt(252)
        )
    else:
        logger.warning("No daily returns available, realized_vol will be NaN")
        monthly['realized_vol_21d_annual'] = np.nan
    
    # Drop rows where adj_close is NaN (months with no data)
    monthly = monthly.dropna(subset=['adj_close'])
    
    logger.info(f"Aggregated to {len(monthly)} monthly observations")
    logger.info(f"Date range: {monthly.index.min().date()} to {monthly.index.max().date()}")
    
    return monthly


def save_monthly_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save monthly aggregated data to CSV.
    
    Args:
        df: Monthly DataFrame
        output_path: Path to output CSV file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Reset index to make date a column
    df_to_save = df.reset_index()
    
    # Rename index column if it exists
    if df_to_save.columns[0] not in df.columns:
        df_to_save = df_to_save.rename(columns={df_to_save.columns[0]: 'month_end'})
    
    df_to_save.to_csv(output_path, index=False)
    logger.info(f"✓ Saved monthly data to {output_path}")
    logger.info(f"  Rows: {len(df_to_save)}")
    logger.info(f"  Columns: {list(df_to_save.columns)}")


def main() -> None:
    """
    Main function to aggregate NVDA daily WRDS data to monthly frequency.
    
    Reads: data/processed/nvda_daily_wrds_1999_2025.csv
    Writes: data/processed/nvda_monthly_wrds_1999_2025.csv
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Input and output paths
    input_path = project_root / "data" / "processed" / "nvda_daily_wrds_1999_2025.csv"
    output_path = project_root / "data" / "processed" / "nvda_monthly_wrds_1999_2025.csv"
    
    logger.info("=" * 70)
    logger.info("Aggregating NVDA daily WRDS data to monthly frequency")
    logger.info("=" * 70)
    
    try:
        # Load daily data
        daily_df = load_daily_data(input_path)
        
        # Aggregate to monthly
        monthly_df = aggregate_to_monthly(daily_df)
        
        # Save monthly data
        save_monthly_data(monthly_df, output_path)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("✓ Monthly aggregation completed successfully!")
        logger.info("=" * 70)
        logger.info(
            f"Monthly rows: {len(monthly_df)}, "
            f"from {monthly_df.index.min().strftime('%Y-%m')} to {monthly_df.index.max().strftime('%Y-%m')}"
        )
        logger.info(f"\nPreview:\n{monthly_df.head(10).to_string()}")
        
    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        logger.error("Please run finmc_tech.data.fetch_nvda_wrds first to generate daily data")
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

