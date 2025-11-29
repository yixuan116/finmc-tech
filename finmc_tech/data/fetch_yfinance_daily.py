"""
Download daily price data using yfinance for NVDA (and optional peers).

Provides CRSP-like columns to replace WRDS/CRSP in the pipeline.

Outputs:
- data/raw/nvda_yf_daily_1999_2025.csv (or custom path)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from finmc_tech.config import get_logger

logger = get_logger(__name__)


def download_daily_prices(
    tickers: List[str],
    start: str = "1999-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV + Adj Close for given tickers from yfinance.
    
    Parameters
    ----------
    tickers : list of str
        e.g. ["NVDA", "AMD", "^GSPC"]
    start : str
        Start date, "YYYY-MM-DD"
    end : str or None
        End date, "YYYY-MM-DD". If None, uses today.
    
    Returns
    -------
    DataFrame with columns:
    date, ticker, open, high, low, close, adj_close, volume, ret_1d
    """
    # yfinance prefers space-separated tickers for multi-download
    tickers_str = " ".join(tickers)
    
    logger.info(f"Downloading data for {tickers} from {start} to {end or 'latest'}...")
    
    try:
        data = yf.download(
            tickers_str,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,  # Keep original and adjusted prices
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        raise ConnectionError(f"Failed to download data from yfinance: {str(e)}")
    
    if data.empty:
        raise ValueError(f"No data returned from yfinance for tickers: {tickers}")
    
    frames = []
    
    for ticker in tickers:
        # yfinance returns MultiIndex columns when group_by="ticker"
        # Structure: (Ticker, Price) - Level 0 is Ticker, Level 1 is Price
        
        if isinstance(data.columns, pd.MultiIndex):
            # Check which level is Ticker and which is Price
            level_names = data.columns.names
            if 'Ticker' in level_names:
                ticker_level = level_names.index('Ticker')
                price_level = level_names.index('Price')
            else:
                # Fallback: assume first level is ticker
                ticker_level = 0
                price_level = 1
            
            # Check if ticker exists
            available_tickers = data.columns.get_level_values(ticker_level).unique()
            if ticker not in available_tickers:
                logger.warning(f"No data returned for {ticker}")
                continue
            
            # Select columns for this ticker using xs
            df_t = data.xs(ticker, level=ticker_level, axis=1).copy()
        else:
            # Single level columns (shouldn't happen with group_by="ticker")
            logger.warning(f"Unexpected column structure for {ticker}")
            continue
        
        if df_t.empty:
            logger.warning(f"No data returned for {ticker}")
            continue
        
        # Reset index to get date as column
        df_t = df_t.reset_index()
        
        # Rename date column
        if "Date" in df_t.columns:
            df_t = df_t.rename(columns={"Date": "date"})
        else:
            # Use index as date
            df_t["date"] = df_t.index
        
        # Rename columns to lowercase
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Adj. Close": "adj_close",
            "Volume": "volume",
        }
        
        # Apply renaming
        for old_col, new_col in column_mapping.items():
            if old_col in df_t.columns:
                df_t = df_t.rename(columns={old_col: new_col})
        
        # If adj_close not available, use close as adj_close
        if "adj_close" not in df_t.columns and "close" in df_t.columns:
            df_t["adj_close"] = df_t["close"]
            logger.info(f"  Using 'close' as 'adj_close' for {ticker}")
        
        # Add ticker column
        df_t["ticker"] = ticker
        
        # Select and reorder columns
        available_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        df_t = df_t[[col for col in available_cols if col in df_t.columns]]
        
        # Daily return based on adj_close, like CRSP RET
        df_t = df_t.sort_values("date").reset_index(drop=True)
        if "adj_close" in df_t.columns:
            df_t["ret_1d"] = df_t["adj_close"].pct_change()
        elif "close" in df_t.columns:
            df_t["ret_1d"] = df_t["close"].pct_change()
            logger.warning(f"Using 'close' instead of 'adj_close' for returns for {ticker}")
        
        frames.append(df_t)
        logger.info(f"  ✓ {ticker}: {len(df_t)} rows")
    
    if not frames:
        raise ValueError("No data downloaded for any ticker.")
    
    out = pd.concat(frames, axis=0, ignore_index=True)
    
    logger.info(f"Total rows downloaded: {len(out):,}")
    logger.info(f"Date range: {out['date'].min().date()} to {out['date'].max().date()}")
    
    return out


def main() -> None:
    """
    Main function to download daily prices via yfinance.
    
    Command-line interface for downloading stock data.
    """
    parser = argparse.ArgumentParser(
        description="Download daily prices via yfinance (CRSP-like)."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="NVDA",
        help='Comma-separated tickers, e.g. "NVDA,AMD,^GSPC"',
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1999-01-01",
        help="Start date YYYY-MM-DD (default: 1999-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: data/raw/nvda_yf_daily_1999_2025.csv)",
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    
    if not tickers:
        logger.error("No valid tickers provided")
        return
    
    # Set default output path if not provided
    if args.out is None:
        project_root = Path(__file__).parent.parent.parent
        # Use first ticker for default filename
        ticker_name = tickers[0].replace("^", "").lower()
        args.out = project_root / "data" / "raw" / f"{ticker_name}_yf_daily_1999_2025.csv"
    
    out_path = Path(args.out)
    
    logger.info("=" * 70)
    logger.info("Downloading daily prices via yfinance")
    logger.info("=" * 70)
    
    try:
        # Download data
        df = download_daily_prices(tickers=tickers, start=args.start, end=args.end)
        
        # Save to CSV
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        
        logger.info("=" * 70)
        logger.info(f"✓ Saved {len(df):,} rows to {out_path}")
        logger.info("=" * 70)
        logger.info(f"\nPreview:\n{df.head(5).to_string()}")
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

