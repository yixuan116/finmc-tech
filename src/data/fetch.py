"""Fetch stock data from Yahoo Finance API."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def fetch_stock_data(
    ticker: str,
    period: str = None,
    interval: str = "1d",
    start: Optional[str] = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance API.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "NVDA")
    period : str, optional
        Period to fetch data for. Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        If None, uses start and end dates instead.
    interval : str, default "1d"
        Data interval. Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    start : str, default "2010-01-01"
        Start date in YYYY-MM-DD format. If provided, takes precedence over period.
    end : str, optional
        End date in YYYY-MM-DD format. If None, fetches up to latest available data.
        If provided, takes precedence over period.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Additional computed columns: Returns, Log_Returns, Volatility

    Raises
    ------
    ValueError
        If ticker data cannot be fetched or is empty

    Examples
    --------
    >>> data = fetch_stock_data("NVDA")  # Default: 2010 to latest
    >>> data = fetch_stock_data("AAPL", start="2020-01-01", end="2023-12-31")
    >>> data = fetch_stock_data("NVDA", period="max")  # Use period instead
    """
    # Download data from Yahoo Finance
    stock = yf.Ticker(ticker)
    
    try:
        if start and end:
            data = stock.history(start=start, end=end, interval=interval)
        elif start:
            # Fetch from start to latest available
            data = stock.history(start=start, interval=interval)
        else:
            data = stock.history(period=period, interval=interval)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    if data.empty:
        raise ValueError(f"No data retrieved for ticker {ticker}")
    
    # Clean column names (make lowercase, reset index)
    data.reset_index(inplace=True)
    data.columns = [col.lower().strip() for col in data.columns]
    
    # Ensure date column exists
    if "date" not in data.columns and len(data) > 0:
        data.reset_index(inplace=True)
    
    # Compute returns
    data = compute_returns(data)
    
    # Compute rolling volatility
    data = compute_volatility(data)
    
    return data


def compute_returns(data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """
    Compute simple returns and log returns.

    Parameters
    ----------
    data : pd.DataFrame
        Stock price data
    price_column : str, default "close"
        Column name for price data

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'returns' and 'log_returns' columns
    """
    # Simple returns
    data["returns"] = data[price_column].pct_change()
    
    # Log returns (more suitable for Monte Carlo simulations)
    data["log_returns"] = np.log(data[price_column] / data[price_column].shift(1))
    
    return data


def compute_volatility(
    data: pd.DataFrame,
    returns_column: str = "log_returns",
    window: int = 20,
) -> pd.DataFrame:
    """
    Compute rolling volatility.

    Parameters
    ----------
    data : pd.DataFrame
        Stock returns data
    returns_column : str, default "log_returns"
        Column name for returns data
    window : int, default 20
        Rolling window size for volatility calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'volatility' column (annualized)
    """
    # Rolling volatility (daily std)
    data["volatility"] = data[returns_column].rolling(window=window).std()
    
    # Annualize (assuming 252 trading days)
    data["volatility"] = data["volatility"] * np.sqrt(252)
    
    return data


def compute_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for stock data.

    Parameters
    ----------
    data : pd.DataFrame
        Stock data with returns and volatility columns

    Returns
    -------
    Dict[str, float]
        Dictionary with mean return, volatility, and other statistics
    """
    log_returns = data["log_returns"].dropna()
    
    stats = {
        "mean_return": log_returns.mean(),
        "annualized_return": log_returns.mean() * 252,
        "volatility": log_returns.std() * np.sqrt(252),
        "sharpe_ratio": None,  # Will compute if risk-free rate provided
        "min_return": log_returns.min(),
        "max_return": log_returns.max(),
        "skewness": log_returns.skew(),
        "kurtosis": log_returns.kurtosis(),
    }
    
    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    if stats["volatility"] > 0:
        stats["sharpe_ratio"] = stats["annualized_return"] / stats["volatility"]
    
    return stats


def get_latest_price(data: pd.DataFrame) -> Tuple[float, str]:
    """
    Get the latest closing price and date.

    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'close' and 'date' columns

    Returns
    -------
    Tuple[float, str]
        (latest_price, latest_date)
    """
    latest_idx = data.index[-1]
    latest_price = data.loc[latest_idx, "close"]
    latest_date = str(data.loc[latest_idx, "date"])
    
    return float(latest_price), latest_date

