"""Feature engineering for ML models."""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_technical_indicators(
    data: pd.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Compute technical indicators for stock data.

    Parameters
    ----------
    data : pd.DataFrame
        Stock price data
    close_col : str, default "close"
        Column name for close price
    high_col : str, default "high"
        Column name for high price
    low_col : str, default "low"
        Column name for low price
    volume_col : str, default "volume"
        Column name for volume

    Returns
    -------
    pd.DataFrame
        Original data with added technical indicator columns
    """
    df = data.copy()
    
    # Simple Moving Averages
    df["sma_5"] = df[close_col].rolling(window=5).mean()
    df["sma_10"] = df[close_col].rolling(window=10).mean()
    df["sma_20"] = df[close_col].rolling(window=20).mean()
    df["sma_50"] = df[close_col].rolling(window=50).mean()
    df["sma_200"] = df[close_col].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df["ema_12"] = df[close_col].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df[close_col].ewm(span=26, adjust=False).mean()
    
    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    # RSI
    df["rsi"] = compute_rsi(df[close_col], period=14)
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_sma = df[close_col].rolling(window=bb_period).mean()
    bb_stdev = df[close_col].rolling(window=bb_period).std()
    df["bb_upper"] = bb_sma + (bb_stdev * bb_std)
    df["bb_lower"] = bb_sma - (bb_stdev * bb_std)
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df[close_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # Stochastic Oscillator
    stoch_k = 14
    stoch_d = 3
    low_min = df[low_col].rolling(window=stoch_k).min()
    high_max = df[high_col].rolling(window=stoch_k).max()
    df["stoch_k"] = 100 * (df[close_col] - low_min) / (high_max - low_min)
    df["stoch_d"] = df["stoch_k"].rolling(window=stoch_d).mean()
    
    # Volume indicators
    df["volume_sma"] = df[volume_col].rolling(window=20).mean()
    df["volume_ratio"] = df[volume_col] / df["volume_sma"]
    
    # ATR (Average True Range)
    df["atr"] = compute_atr(df[high_col], df[low_col], df[close_col], period=14)
    
    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int, default 14
        RSI period

    Returns
    -------
    pd.Series
        RSI values
    """
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default 14
        ATR period

    Returns
    -------
    pd.Series
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def engineer_features(
    data: pd.DataFrame,
    target_horizon: int = 1,
    include_returns: bool = True,
    include_volatility: bool = True,
) -> pd.DataFrame:
    """
    Engineer features for ML model training.

    Parameters
    ----------
    data : pd.DataFrame
        Stock price data with computed indicators
    target_horizon : int, default 1
        Number of periods ahead to predict
    include_returns : bool, default True
        Whether to include return-based features
    include_volatility : bool, default True
        Whether to include volatility features

    Returns
    -------
    pd.DataFrame
        Feature matrix with engineered features
    """
    df = data.copy()
    
    # Add technical indicators
    if "sma_20" not in df.columns:
        df = compute_technical_indicators(df)
    
    # Return-based features
    if include_returns:
        # Lagged returns
        for lag in [1, 3, 5, 10, 20]:
            df[f"return_lag_{lag}"] = df["log_returns"].shift(lag)
        
        # Momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        # Price change features
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]
    
    # Volatility features
    if include_volatility:
        if "volatility" in df.columns:
            for lag in [1, 5, 10, 20]:
                df[f"volatility_lag_{lag}"] = df["volatility"].shift(lag)
        
        # Rolling volatility ratios
        df["volatility_ratio_5_20"] = (
            df["log_returns"].rolling(window=5).std()
            / df["log_returns"].rolling(window=20).std()
        )
    
    # Trend features
    df["price_to_sma20"] = df["close"] / df["sma_20"]
    df["price_to_sma50"] = df["close"] / df["sma_50"]
    df["sma20_to_sma50"] = df["sma_20"] / df["sma_50"]
    
    # Target variable (future return)
    df["target"] = df["log_returns"].shift(-target_horizon)
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df


def select_features(
    data: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select features for model training.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix
    feature_cols : List[str], optional
        List of feature columns to include. If None, uses all relevant columns.

    Returns
    -------
    pd.DataFrame
        Selected features
    """
    if feature_cols is None:
        # Default feature selection
        feature_cols = [
            # Technical indicators
            "sma_5",
            "sma_10",
            "sma_20",
            "sma_50",
            "macd",
            "macd_signal",
            "rsi",
            "stoch_k",
            "stoch_d",
            # Return features
            "return_lag_1",
            "return_lag_5",
            "return_lag_20",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            # Volatility features
            "volatility",
            "volatility_ratio_5_20",
            # Price ratios
            "price_to_sma20",
            "price_to_sma50",
            "sma20_to_sma50",
            "high_low_ratio",
            "close_open_ratio",
            # Volume
            "volume_ratio",
        ]
    
    # Select available features
    available_features = [col for col in feature_cols if col in data.columns]
    
    return data[available_features + ["target"]]

