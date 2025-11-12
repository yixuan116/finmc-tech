"""
Build features from aligned data.

Provides two main functions:
1. build_features() - Legacy feature engineering (backward compatibility)
2. build_Xy() - New feature engineering with lag features and scaling
"""

import sys
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from sklearn.preprocessing import RobustScaler

from finmc_tech.config import Settings, cfg, get_logger

# Import existing modules
from src.data.create_extended_features import (
    create_price_momentum_features,
    create_technical_indicators_from_prices,
    create_market_macro_features,
    create_time_features,
    create_interaction_features,
)

logger = get_logger(__name__)


def build_features(
    df: pd.DataFrame,
    config: Settings,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build features from aligned data based on config.
    
    Parameters
    ----------
    df : pd.DataFrame
        Aligned data with firm and macro features
    config : Config
        Configuration object
    feature_cols : Optional[List[str]]
        Specific features to include. If None, uses config settings.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features
    """
    print("Building features...")
    
    df = df.copy()
    
    # Base features (always included)
    base_features = [
        "rev_qoq",
        "rev_yoy",
        "rev_accel",
        "vix_level",
        "tnx_yield",
        "vix_change_3m",
        "tnx_change_3m",
    ]
    
    # Extended features based on config
    if config.INCLUDE_PRICE_FEATURES:
        df = create_price_momentum_features(df)
    
    if config.INCLUDE_TECHNICAL_FEATURES:
        df = create_technical_indicators_from_prices(df, config.TICKER)
    
    if config.INCLUDE_MARKET_FEATURES:
        df = create_market_macro_features(df)
    
    if config.INCLUDE_TIME_FEATURES:
        df = create_time_features(df)
    
    if config.INCLUDE_INTERACTION_FEATURES:
        df = create_interaction_features(df)
    
    # Select features
    if feature_cols is None:
        # Build feature list from config
        feature_cols = base_features.copy()
        
        if config.INCLUDE_PRICE_FEATURES:
            feature_cols.extend([
                "price_returns_1m", "price_returns_3m", "price_returns_6m",
                "price_returns_12m", "price_momentum", "price_volatility",
                "price_to_ma_4q",
            ])
        
        if config.INCLUDE_TECHNICAL_FEATURES:
            feature_cols.extend([
                "rsi_14", "macd", "macd_signal", "bb_position",
                "stoch_k", "atr",
            ])
        
        if config.INCLUDE_MARKET_FEATURES:
            feature_cols.extend(["sp500_level", "sp500_returns"])
        
        if config.INCLUDE_TIME_FEATURES:
            feature_cols.extend(["quarter", "month", "year", "days_since_start"])
        
        if config.INCLUDE_INTERACTION_FEATURES:
            feature_cols.extend([
                "rev_yoy_x_vix", "rev_qoq_x_sp500",
                "price_momentum_x_volatility", "vix_x_tnx",
            ])
    
    # Select only available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    print(f"  ✓ Built {len(available_features)} features")
    print(f"  Available: {', '.join(available_features[:10])}...")
    
    return df, available_features


def build_Xy(
    df: pd.DataFrame,
    scale_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build feature matrix X and target y from aligned data.
    
    Inputs: df with columns like ["Ret", "CPI", "VIX", "DGS10", "FEDFUNDS", 
                                   "GDP", "Revenue", "R_and_D", "Capex", "Close"]
    
    Steps:
    1. Create 1-lag versions for all macro & firm features except "Ret": suffix "_L1"
    2. Drop rows with NA after lag
    3. y = next-month return: y = df["Ret"].shift(-1). Drop tail NA
    
    Note: Scaling should be done after train_test_split_time using scale_Xy() 
    to ensure fit on train only.
    
    Args:
        df: DataFrame with aligned macro and firm data
        scale_features: Whether to scale features (deprecated, use scale_Xy instead)
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Target series (next-month return)
        feature_names: List of feature names
    """
    logger.info("Building X and y from aligned data...")
    
    df = df.copy()
    
    # Identify feature columns (all except "Ret")
    if "Ret" not in df.columns:
        raise ValueError("DataFrame must contain 'Ret' column (returns)")
    
    feature_cols = [col for col in df.columns if col != "Ret"]
    
    # Create 1-lag versions for all macro & firm features
    lagged_cols = []
    for col in feature_cols:
        lag_col = f"{col}_L1"
        df[lag_col] = df[col].shift(1)
        lagged_cols.append(lag_col)
        logger.debug(f"  Created lag feature: {lag_col}")
    
    # Combine original and lagged features
    all_feature_cols = feature_cols + lagged_cols
    
    # Create target: next-month return
    y = df["Ret"].shift(-1)
    
    # Drop rows with NA (after lag operations)
    valid_mask = df[all_feature_cols].notna().all(axis=1) & y.notna()
    df_clean = df[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    logger.info(f"  Dropped {len(df) - len(df_clean)} rows with NA values")
    logger.info(f"  Final sample size: {len(df_clean)}")
    
    # Select features
    X = df_clean[all_feature_cols].copy()
    feature_names = all_feature_cols.copy()
    
    logger.info(f"  ✓ Built X: {X.shape}, y: {len(y_clean)}")
    logger.info(f"  Feature count: {len(feature_names)}")
    
    return X, y_clean, feature_names


def scale_Xy(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Scale continuous features with RobustScaler (fit on train only).
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target (unused, kept for consistency)
        y_test: Test target (unused, kept for consistency)
    
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        scaler: Fitted RobustScaler
    """
    logger.info("Scaling features with RobustScaler (fit on train only)...")
    
    # Identify continuous features
    continuous_cols = []
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            continuous_cols.append(col)
    
    if not continuous_cols:
        logger.warning("  No continuous features found to scale")
        return X_train, X_test, None
    
    # Fit scaler on train only
    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    logger.info(f"  Scaled {len(continuous_cols)} continuous features")
    
    return X_train_scaled, X_test_scaled, scaler


def train_test_split_time(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
    """
    Split data by time, ensuring no data leakage.
    
    Args:
        X: Feature matrix (DataFrame with datetime index)
        y: Target series (with datetime index)
        train_end: Training end date (YYYY-MM-DD). If None, uses cfg.TRAIN_END
    
    Returns:
        X_train, X_test, y_train, y_test, train_idx, test_idx
        train_idx and test_idx are the original indices for alignment
    """
    if train_end is None:
        train_end = cfg.TRAIN_END
    
    train_end_dt = pd.to_datetime(train_end)
    
    # Ensure X and y have datetime index
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have DatetimeIndex")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("y must have DatetimeIndex")
    
    # Create masks
    train_mask = X.index < train_end_dt
    test_mask = X.index >= train_end_dt
    
    # Split data
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    
    # Get indices for future alignment
    train_idx = X_train.index
    test_idx = X_test.index
    
    logger.info(f"  Train: {len(X_train)} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
    logger.info(f"  Test: {len(X_test)} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
    
    return X_train, X_test, y_train, y_test, train_idx, test_idx

