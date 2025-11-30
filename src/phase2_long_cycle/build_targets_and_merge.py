"""
Build mid-horizon target variables (log-returns) and merge with long-cycle features.

This script:
1. Loads long-cycle features from CSV
2. Loads NVDA price data from CSV
3. Resamples prices to quarter-end
4. Merges prices with features
5. Creates log-return targets for 4, 8, 12 quarter horizons
6. Saves final training dataset
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_long_cycle_features(features_path: Path) -> pd.DataFrame:
    """
    Load long-cycle features CSV.
    
    Args:
        features_path: Path to features CSV
        
    Returns:
        DataFrame with period_end_date as index
    """
    logger.info(f"Loading long-cycle features from {features_path}")
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    df = pd.read_csv(
        features_path,
        index_col=0,
        parse_dates=True
    )
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} features")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def load_price_data(price_path: Path) -> pd.DataFrame:
    """
    Load NVDA price data and resample to quarter-end.
    
    Args:
        price_path: Path to price CSV
        
    Returns:
        DataFrame with quarter-end prices, indexed by date
    """
    logger.info(f"Loading price data from {price_path}")
    
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")
    
    # Load price data
    df_price = pd.read_csv(price_path)
    
    # Convert date to datetime
    if 'date' in df_price.columns:
        df_price['date'] = pd.to_datetime(df_price['date'])
    else:
        raise ValueError("Price data must have 'date' column")
    
    # Set date as index
    df_price = df_price.set_index('date')
    
    # Find close price column (prefer 'close', fall back to 'adj_close')
    if 'close' in df_price.columns:
        price_col = 'close'
    elif 'adj_close' in df_price.columns:
        price_col = 'adj_close'
        logger.warning("Using 'adj_close' instead of 'close'")
    else:
        raise ValueError("Price data must have 'close' or 'adj_close' column")
    
    # Select only price column
    df_price = df_price[[price_col]].copy()
    df_price.columns = ['price']
    
    logger.info(f"Loaded {len(df_price)} daily price records")
    logger.info(f"Date range: {df_price.index.min()} to {df_price.index.max()}")
    
    # Resample to quarter-end using last price in quarter
    # Use 'QE' (quarter-end) instead of deprecated 'Q'
    df_price_q = df_price.resample('QE').last()
    df_price_q.columns = ['price_q']
    
    logger.info(f"Resampled to {len(df_price_q)} quarterly records")
    logger.info(f"Quarter-end date range: {df_price_q.index.min()} to {df_price_q.index.max()}")
    
    return df_price_q


def merge_features_and_prices(
    df_features: pd.DataFrame,
    df_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features and prices on period_end_date using nearest match.
    
    Since features use actual quarter-end dates and prices use standard
    quarter-end dates, we use merge_asof to find the nearest price.
    
    Args:
        df_features: Features DataFrame with period_end_date index
        df_prices: Prices DataFrame with date index
        
    Returns:
        Merged DataFrame
    """
    logger.info("Merging features and prices")
    
    # Reset index to columns for merge_asof
    df_features_reset = df_features.reset_index()
    df_prices_reset = df_prices.reset_index()
    
    # Sort both by date for merge_asof
    df_features_reset = df_features_reset.sort_values('period_end_date')
    df_prices_reset = df_prices_reset.sort_values('date')
    
    # Use merge_asof to find nearest price (backward direction)
    # This finds the most recent price on or before the feature date
    df_merged = pd.merge_asof(
        df_features_reset,
        df_prices_reset,
        left_on='period_end_date',
        right_on='date',
        direction='backward',
        tolerance=pd.Timedelta(days=90)  # Allow up to 90 days difference
    )
    
    # Set period_end_date back as index
    df_merged = df_merged.set_index('period_end_date')
    
    # Drop the date column from prices (we only need price_q)
    if 'date' in df_merged.columns:
        df_merged = df_merged.drop('date', axis=1)
    
    # Check how many rows have prices
    has_price = df_merged['price_q'].notna().sum()
    logger.info(f"Merged dataset: {len(df_merged)} rows")
    logger.info(f"Rows with prices: {has_price}/{len(df_merged)}")
    logger.info(f"Features: {len(df_features.columns)} columns")
    logger.info(f"Price column: price_q")
    
    return df_merged


def build_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mid-horizon log-return targets for 4, 8, 12, 16, 20 quarters.
    
    Args:
        df: DataFrame with price_q column
        
    Returns:
        DataFrame with target variables added
    """
    logger.info("Building target variables")
    
    if 'price_q' not in df.columns:
        raise ValueError("DataFrame must have 'price_q' column")
    
    # Create log-return targets for different horizons
    # y_log_hq = ln(price_q.shift(-h) / price_q)
    # 4q = 1 year, 8q = 2 years, 12q = 3 years, 16q = 4 years, 20q = 5 years
    
    df['y_log_4q'] = np.log(df['price_q'].shift(-4) / df['price_q'])
    df['y_log_8q'] = np.log(df['price_q'].shift(-8) / df['price_q'])
    df['y_log_12q'] = np.log(df['price_q'].shift(-12) / df['price_q'])
    df['y_log_16q'] = np.log(df['price_q'].shift(-16) / df['price_q'])
    df['y_log_20q'] = np.log(df['price_q'].shift(-20) / df['price_q'])
    
    # Replace inf and -inf with NaN
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    for col in target_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Count non-null targets
    for col in target_cols:
        non_null = df[col].notna().sum()
        logger.info(f"  {col}: {non_null}/{len(df)} non-null values")
    
    return df


def drop_rows_with_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows at the end with NaN target values.
    
    Args:
        df: DataFrame with target columns
        
    Returns:
        DataFrame with rows containing NaN targets dropped
    """
    logger.info("Dropping rows with missing targets")
    
    before = len(df)
    
    # Drop rows where all targets are NaN
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    df = df.dropna(subset=target_cols, how='all')
    
    after = len(df)
    dropped = before - after
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with all missing targets")
    else:
        logger.info("No rows dropped (all targets have at least one non-null value)")
    
    return df


def save_training_data(df: pd.DataFrame, output_path: Path):
    """
    Save final training dataset to CSV.
    
    Args:
        df: Final training DataFrame
        output_path: Path to output CSV
    """
    logger.info(f"Saving training data to {output_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path)
    
    logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns")


def print_summary(df: pd.DataFrame):
    """
    Print summary of final training dataset.
    
    Args:
        df: Final training DataFrame
    """
    logger.info("=" * 60)
    logger.info("TRAINING DATASET SUMMARY")
    logger.info("=" * 60)
    
    # Number of rows
    logger.info(f"\nNumber of rows: {len(df)}")
    
    # Feature columns (X)
    feature_cols = [col for col in df.columns 
                   if not col.startswith('y_') and col != 'price_q']
    logger.info(f"\nFeature columns (X): {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        logger.info(f"  {i:2d}. {col:30s} ({non_null}/{len(df)} = {pct:.1f}% non-null)")
    
    # Target columns (y)
    target_cols = [col for col in df.columns if col.startswith('y_')]
    # Sort by horizon length
    target_cols = sorted(target_cols, key=lambda x: int(x.split('_')[-1].replace('q', '')))
    
    logger.info(f"\nTarget columns (y): {len(target_cols)}")
    horizon_map = {'4q': '1 year', '8q': '2 years', '12q': '3 years', 
                   '16q': '4 years', '20q': '5 years'}
    for col in target_cols:
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        mean_val = df[col].mean()
        std_val = df[col].std()
        horizon = horizon_map.get(col.split('_')[-1], '')
        logger.info(f"  - {col:20s} ({horizon:8s}): {non_null}/{len(df)} ({pct:.1f}% non-null), "
                   f"mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Date range
    logger.info(f"\nDate range: {df.index.min()} to {df.index.max()}")
    
    # Price info
    if 'price_q' in df.columns:
        logger.info(f"\nPrice range: ${df['price_q'].min():.2f} to ${df['price_q'].max():.2f}")
    
    logger.info("=" * 60)


def build_targets_and_merge(
    features_path: Path,
    price_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Main function: build targets and merge with features.
    
    Args:
        features_path: Path to long-cycle features CSV
        price_path: Path to price data CSV
        output_path: Path to output training CSV
        
    Returns:
        Final training DataFrame
    """
    logger.info("=" * 60)
    logger.info("Building mid-horizon targets and merging with features")
    logger.info("=" * 60)
    
    # Load data
    df_features = load_long_cycle_features(features_path)
    df_prices = load_price_data(price_path)
    
    # Merge
    df_merged = merge_features_and_prices(df_features, df_prices)
    
    # Build targets
    df_merged = build_target_variables(df_merged)
    
    # Drop rows with missing targets
    df_merged = drop_rows_with_missing_targets(df_merged)
    
    # Save
    save_training_data(df_merged, output_path)
    
    # Print summary
    print_summary(df_merged)
    
    return df_merged


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build mid-horizon targets and merge with long-cycle features"
    )
    parser.add_argument(
        '--features',
        type=Path,
        default=Path('data/processed/nvda_long_horizon_firm_features.csv'),
        help='Path to long-cycle features CSV'
    )
    parser.add_argument(
        '--prices',
        type=Path,
        default=Path('data/raw/nvda_yf_daily_1999_2025.csv'),
        help='Path to price data CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/nvda_long_cycle_train.csv'),
        help='Path to output training CSV'
    )
    
    args = parser.parse_args()
    
    # Try alternative price file if default doesn't exist
    if not args.prices.exists():
        alt_price = Path('data/raw/NVDA_data_2010_2025.csv')
        if alt_price.exists():
            logger.info(f"Price file not found at {args.prices}, trying {alt_price}")
            args.prices = alt_price
        else:
            # Try CRSP file
            alt_price = Path('data/raw/nvda_crsp_daily_1999_2025.csv')
            if alt_price.exists():
                logger.info(f"Trying CRSP file: {alt_price}")
                args.prices = alt_price
    
    build_targets_and_merge(args.features, args.prices, args.output)


if __name__ == '__main__':
    main()

