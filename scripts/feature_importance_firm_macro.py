#!/usr/bin/env python3
"""
Comprehensive Feature Importance Analysis: Firm + Macro + Interaction Features

This script performs feature importance analysis across 1-year, 3-year, and 7-year
horizons using firm-level features, macro features, and their Kronecker interactions.

Outputs:
- Feature importance plots for RF and XGB models
- SHAP summary plots for XGB models
- Consolidated importance heatmap
- Summary tables
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_firm_features(firm_path: Path) -> pd.DataFrame:
    """Load firm-level features."""
    logger.info(f"Loading firm features from {firm_path}")
    df = pd.read_csv(firm_path)
    
    # Convert date column to datetime
    if 'period_end' in df.columns:
        df['date'] = pd.to_datetime(df['period_end'])
        df = df.drop('period_end', axis=1)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No date column found in firm features")
    
    df = df.sort_values('date').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def load_macro_features(macro_path: Path) -> pd.DataFrame:
    """Load macro features (optional, can be None if already in firm features)."""
    logger.info(f"Loading macro features from {macro_path}")
    
    if not macro_path.exists():
        logger.info("Macro features file not found, will extract from firm features")
        return None
    
    df = pd.read_csv(macro_path)
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No date column found in macro features")
    
    df = df.sort_values('date').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def extract_macro_from_firm(firm_df: pd.DataFrame) -> pd.DataFrame:
    """Extract macro features from firm features if they exist."""
    macro_keywords = ['vix', 'tnx', 'yield', 'cpi', 'fedfunds']
    macro_cols = [c for c in firm_df.columns if any(kw in c.lower() for kw in macro_keywords) and '_x_' not in c]
    
    if len(macro_cols) == 0:
        return None
    
    macro_df = firm_df[['date'] + macro_cols].copy()
    logger.info(f"Extracted {len(macro_cols)} macro features from firm data")
    
    return macro_df


def merge_firm_and_macro(
    firm_df: pd.DataFrame,
    macro_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Merge firm and macro features on date (monthly frequency)."""
    logger.info("Merging firm and macro features")
    
    # If macro_df is None, extract from firm_df
    if macro_df is None:
        macro_df = extract_macro_from_firm(firm_df)
    
    # Resample firm features to quarterly (since they're already quarterly)
    # Just ensure date is properly set
    firm_df = firm_df.copy()
    firm_df['date'] = pd.to_datetime(firm_df['date'])
    firm_monthly = firm_df.copy()  # Keep as quarterly, don't resample
    
    # If we have separate macro data, merge it
    if macro_df is not None and len(macro_df) > 0:
        macro_df = macro_df.copy()
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        
        # Merge on date
        merged = pd.merge(
            firm_monthly,
            macro_df,
            on='date',
            how='inner',
            suffixes=('_firm', '_macro')
        )
        
        # If merge resulted in 0 rows, try merge_asof
        if len(merged) == 0:
            logger.warning("Exact date match failed, trying nearest match")
            firm_monthly_sorted = firm_monthly.sort_values('date')
            macro_sorted = macro_df.sort_values('date')
            merged = pd.merge_asof(
                firm_monthly_sorted,
                macro_sorted,
                on='date',
                direction='nearest',
                tolerance=pd.Timedelta(days=15),
                suffixes=('_firm', '_macro')
            )
    else:
        # Use firm_monthly directly (macro already included)
        merged = firm_monthly
    
    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    if len(merged) > 0:
        logger.info(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    else:
        logger.error("Merge resulted in 0 rows! Check date formats.")
    
    return merged


def build_interaction_features(
    df: pd.DataFrame,
    firm_cols: List[str],
    macro_cols: List[str],
    missing_threshold: float = 0.2,
    max_interactions: int = 500
) -> pd.DataFrame:
    """
    Build Kronecker interaction features: F_x_M = F * M
    
    Args:
        df: DataFrame with firm and macro features
        firm_cols: List of firm feature column names
        macro_cols: List of macro feature column names
        missing_threshold: Skip interaction if either column has >threshold missing
        max_interactions: Maximum number of interactions to create
        
    Returns:
        DataFrame with interaction features added
    """
    logger.info("Building interaction features (Kronecker layer)")
    
    if len(df) == 0:
        logger.warning("DataFrame is empty, cannot build interactions")
        return df
    
    df = df.copy()
    interaction_data = {}
    interaction_count = 0
    skipped_count = 0
    
    # Filter firm and macro cols by missing data first
    valid_firm_cols = []
    for firm_col in firm_cols:
        if firm_col not in df.columns:
            continue
        firm_missing_pct = df[firm_col].isna().sum() / len(df) if len(df) > 0 else 1.0
        if firm_missing_pct <= missing_threshold:
            valid_firm_cols.append(firm_col)
        else:
            skipped_count += 1
    
    valid_macro_cols = []
    for macro_col in macro_cols:
        if macro_col not in df.columns:
            continue
        macro_missing_pct = df[macro_col].isna().sum() / len(df) if len(df) > 0 else 1.0
        if macro_missing_pct <= missing_threshold:
            valid_macro_cols.append(macro_col)
        else:
            skipped_count += 1
    
    # Limit interactions if too many
    if len(valid_firm_cols) * len(valid_macro_cols) > max_interactions:
        # Select top firm and macro features by variance
        firm_var = df[valid_firm_cols].var().sort_values(ascending=False)
        macro_var = df[valid_macro_cols].var().sort_values(ascending=False)
        
        n_firm = min(len(valid_firm_cols), max_interactions // len(valid_macro_cols) + 1)
        n_macro = min(len(valid_macro_cols), max_interactions // n_firm + 1)
        
        valid_firm_cols = firm_var.head(n_firm).index.tolist()
        valid_macro_cols = macro_var.head(n_macro).index.tolist()
        logger.info(f"Limiting to top {n_firm} firm × {n_macro} macro = {n_firm * n_macro} interactions")
    
    # Build interactions using vectorized operations
    for firm_col in valid_firm_cols:
        for macro_col in valid_macro_cols:
            if interaction_count >= max_interactions:
                break
            
            interaction_name = f"{firm_col}_x_{macro_col}"
            interaction_data[interaction_name] = df[firm_col] * df[macro_col]
            interaction_count += 1
        
        if interaction_count >= max_interactions:
            break
    
    # Add all interactions at once using pd.concat
    if interaction_data:
        interaction_df = pd.DataFrame(interaction_data, index=df.index)
        df = pd.concat([df, interaction_df], axis=1)
    
    logger.info(f"Created {interaction_count} interaction features")
    logger.info(f"Skipped {skipped_count} interactions due to missing data")
    
    return df


def build_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build future return targets for 1-year, 2-year, 3-year, 4-year, 5-year, 7-year, and 10-year horizons.
    
    Args:
        df: DataFrame with 'adj_close' or 'close' column and 'date' index
        
    Returns:
        DataFrame with return targets added
    """
    logger.info("Building return targets")
    
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Find price column
    if 'adj_close' in df.columns:
        price_col = 'adj_close'
    elif 'close' in df.columns:
        price_col = 'close'
    else:
        raise ValueError("No price column found (adj_close or close)")
    
    # Build return targets (using quarters since data is quarterly)
    # ret_1y = pct_change over 4 quarters, shifted -4 quarters (future)
    df['ret_1y'] = df[price_col].pct_change(4, fill_method=None).shift(-4)
    # ret_2y = pct_change over 8 quarters, shifted -8 quarters
    df['ret_2y'] = df[price_col].pct_change(8, fill_method=None).shift(-8)
    # ret_3y = pct_change over 12 quarters, shifted -12 quarters
    df['ret_3y'] = df[price_col].pct_change(12, fill_method=None).shift(-12)
    # ret_4y = pct_change over 16 quarters, shifted -16 quarters
    df['ret_4y'] = df[price_col].pct_change(16, fill_method=None).shift(-16)
    # ret_5y = pct_change over 20 quarters, shifted -20 quarters
    df['ret_5y'] = df[price_col].pct_change(20, fill_method=None).shift(-20)
    # ret_7y = pct_change over 28 quarters, shifted -28 quarters
    df['ret_7y'] = df[price_col].pct_change(28, fill_method=None).shift(-28)
    # ret_10y = pct_change over 40 quarters, shifted -40 quarters
    df['ret_10y'] = df[price_col].pct_change(40, fill_method=None).shift(-40)
    
    logger.info(f"Return targets created:")
    logger.info(f"  ret_1y: {df['ret_1y'].notna().sum()} non-null values")
    logger.info(f"  ret_2y: {df['ret_2y'].notna().sum()} non-null values")
    logger.info(f"  ret_3y: {df['ret_3y'].notna().sum()} non-null values")
    logger.info(f"  ret_4y: {df['ret_4y'].notna().sum()} non-null values")
    logger.info(f"  ret_5y: {df['ret_5y'].notna().sum()} non-null values")
    logger.info(f"  ret_7y: {df['ret_7y'].notna().sum()} non-null values")
    logger.info(f"  ret_10y: {df['ret_10y'].notna().sum()} non-null values")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Prepare feature set, excluding specified columns.
    
    Returns:
        X: Feature matrix
        firm_cols: List of firm feature names
        macro_cols: List of macro feature names
        interaction_cols: List of interaction feature names
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'adj_close', 'close', 'open', 'high', 'low', 
                       'volume', 'ret_1y', 'ret_2y', 'ret_3y', 'ret_4y', 'ret_5y', 'ret_7y', 'ret_10y', 'price_q',
                       'period_end', 'px_date', 'fy', 'fp', 'form', 'tag_used', 
                       'ticker', 'quarter', 'month', 'year', 'days_since_start',
                       'future_12m_price', 'future_12m_return', 'future_12m_logprice']
    
    # Get all feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Categorize features
    firm_cols = [c for c in feature_cols if '_x_' not in c and 
                 not any(kw in c.lower() for kw in ['vix', 'tnx', 'yield', 'cpi', 'fedfunds'])]
    macro_cols = [c for c in feature_cols if '_x_' not in c and 
                  any(kw in c.lower() for kw in ['vix', 'tnx', 'yield', 'cpi', 'fedfunds'])]
    interaction_cols = [c for c in feature_cols if '_x_' in c]
    
    # Select numeric features only
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    logger.info(f"Feature preparation:")
    logger.info(f"  Firm features: {len(firm_cols)}")
    logger.info(f"  Macro features: {len(macro_cols)}")
    logger.info(f"  Interaction features: {len(interaction_cols)}")
    logger.info(f"  Total features: {len(X.columns)}")
    
    return X, firm_cols, macro_cols, interaction_cols


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestRegressor, XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Train RF and XGB models and return feature importance.
    
    Returns:
        rf_model: Trained RandomForest model
        xgb_model: Trained XGBoost model
        rf_importance: DataFrame with RF feature importance
        xgb_importance: DataFrame with XGB feature importance
    """
    logger.info("Training models")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train RandomForest
    logger.info("Training RandomForest...")
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    rf_train_r2 = rf.score(X_train, y_train)
    rf_test_r2 = rf.score(X_test, y_test)
    logger.info(f"RF - Train R²: {rf_train_r2:.4f}, Test R²: {rf_test_r2:.4f}")
    
    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_train_r2 = xgb.score(X_train, y_train)
    xgb_test_r2 = xgb.score(X_test, y_test)
    logger.info(f"XGB - Train R²: {xgb_train_r2:.4f}, Test R²: {xgb_test_r2:.4f}")
    
    # Get feature importance
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return rf, xgb, rf_importance, xgb_importance, rf_train_r2, rf_test_r2, xgb_train_r2, xgb_test_r2


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    title: str = "Feature Importance"
):
    """Plot top-N feature importance bar chart."""
    logger.info(f"Plotting feature importance to {output_path}")
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    ax.barh(range(len(top_features)), top_features['importance'].values[::-1], 
            color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values[::-1])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_shap_values(
    model: XGBRegressor,
    X: pd.DataFrame,
    output_path: Path,
    max_samples: int = 100
):
    """Compute and plot SHAP summary."""
    logger.info(f"Computing SHAP values for {len(X)} samples")
    
    # Sample data if too large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
        logger.info(f"Sampling {max_samples} instances for SHAP")
    else:
        X_sample = X
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create summary plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP plot saved to {output_path}")


def create_consolidated_heatmap(
    importance_dict: Dict[str, pd.DataFrame],
    output_path: Path,
    firm_cols: List[str],
    macro_cols: List[str],
    interaction_cols: List[str],
    top_n: int = 20
):
    """
    Create consolidated importance heatmaps across all horizons.
    Generates two heatmaps:
    1. Overall heatmap (all features: firm + macro + interaction)
    2. Separated heatmap (firm and macro features shown separately)
    
    Args:
        importance_dict: Dict mapping horizon names to importance DataFrames
        output_path: Path to save overall heatmap (separated will be saved with _separated suffix)
        firm_cols: List of firm feature names
        macro_cols: List of macro feature names
        interaction_cols: List of interaction feature names
        top_n: Number of top features to include
    """
    logger.info("Creating consolidated importance heatmaps")
    
    # Ensure horizons are in order: 1Y, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y
    horizon_order = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']
    importance_dict_ordered = {h: importance_dict[h] for h in horizon_order if h in importance_dict}
    
    # Get union of top features across all horizons
    all_top_features = set()
    for horizon, df in importance_dict_ordered.items():
        top_features = df.head(top_n)['feature'].tolist()
        all_top_features.update(top_features)
    
    all_top_features = sorted(list(all_top_features))
    logger.info(f"Union of top features: {len(all_top_features)}")
    
    # Create DataFrame with importance values (overall)
    heatmap_data = []
    for feature in all_top_features:
        row = {'feature': feature}
        for horizon in horizon_order:
            if horizon in importance_dict_ordered:
                df = importance_dict_ordered[horizon]
                feature_importance = df[df['feature'] == feature]['importance'].values
                if len(feature_importance) > 0:
                    row[horizon] = feature_importance[0]
                else:
                    row[horizon] = 0.0
            else:
                row[horizon] = 0.0
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('feature')
    
    # Sort by average importance
    heatmap_df['avg'] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values('avg', ascending=False).drop('avg', axis=1)
    
    # Take top N
    heatmap_df = heatmap_df.head(top_n)
    
    # ===== Plot 1: Overall Heatmap (All Features) =====
    fig, ax = plt.subplots(figsize=(12, max(12, len(heatmap_df) * 0.4)))
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Feature Importance'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title('Feature Importance Heatmap Across Horizons (RF) - All Features', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Overall heatmap saved to {output_path}")
    
    # ===== Plot 2: Separated Heatmap (Firm vs Macro) =====
    # Separate features by category
    # Include pure firm/macro features AND interaction features (classified by naming)
    firm_features = [f for f in all_top_features if f in firm_cols]
    macro_features = [f for f in all_top_features if f in macro_cols]
    interaction_features = [f for f in all_top_features if f in interaction_cols]
    
    # Classify interaction features by their naming convention
    # Features ending with _firm or containing firm-related terms go to firm
    # Features ending with _macro or containing macro-related terms go to macro
    interaction_firm = []
    interaction_macro = []
    
    for feature in interaction_features:
        # Check naming convention: _firm or _macro suffix
        if feature.endswith('_firm') or (feature.endswith('_Firm')):
            interaction_firm.append(feature)
        elif feature.endswith('_macro') or (feature.endswith('_Macro')):
            interaction_macro.append(feature)
        else:
            # Default: if contains macro keywords, classify as macro; otherwise firm
            if any(kw in feature.lower() for kw in ['vix', 'tnx', 'yield', 'cpi', 'fedfunds']):
                interaction_macro.append(feature)
            else:
                interaction_firm.append(feature)
    
    # Combine pure and interaction features
    all_firm_features = firm_features + interaction_firm
    all_macro_features = macro_features + interaction_macro
    
    logger.info(f"Separated heatmap feature counts:")
    logger.info(f"  Firm (pure + interaction): {len(firm_features)} + {len(interaction_firm)} = {len(all_firm_features)}")
    logger.info(f"  Macro (pure + interaction): {len(macro_features)} + {len(interaction_macro)} = {len(all_macro_features)}")
    
    # Create separate heatmaps for firm and macro
    firm_data = []
    macro_data = []
    
    for feature in all_firm_features:
        row = {'feature': feature}
        for horizon in horizon_order:
            if horizon in importance_dict_ordered:
                df = importance_dict_ordered[horizon]
                feature_importance = df[df['feature'] == feature]['importance'].values
                if len(feature_importance) > 0:
                    row[horizon] = feature_importance[0]
                else:
                    row[horizon] = 0.0
            else:
                row[horizon] = 0.0
        firm_data.append(row)
    
    for feature in all_macro_features:
        row = {'feature': feature}
        for horizon in horizon_order:
            if horizon in importance_dict_ordered:
                df = importance_dict_ordered[horizon]
                feature_importance = df[df['feature'] == feature]['importance'].values
                if len(feature_importance) > 0:
                    row[horizon] = feature_importance[0]
                else:
                    row[horizon] = 0.0
            else:
                row[horizon] = 0.0
        macro_data.append(row)
    
    firm_df = pd.DataFrame(firm_data).set_index('feature') if firm_data else pd.DataFrame()
    macro_df = pd.DataFrame(macro_data).set_index('feature') if macro_data else pd.DataFrame()
    
    # Sort by average importance
    if len(firm_df) > 0:
        firm_df['avg'] = firm_df.mean(axis=1)
        firm_df = firm_df.sort_values('avg', ascending=False).drop('avg', axis=1)
        firm_df = firm_df.head(min(top_n, len(firm_df)))
    
    if len(macro_df) > 0:
        macro_df['avg'] = macro_df.mean(axis=1)
        macro_df = macro_df.sort_values('avg', ascending=False).drop('avg', axis=1)
        macro_df = macro_df.head(min(top_n, len(macro_df)))
    
    # Combine firm and macro with labels
    if len(firm_df) > 0 and len(macro_df) > 0:
        # Add category prefix to feature names
        firm_df_labeled = firm_df.copy()
        firm_df_labeled.index = ['[Firm] ' + str(idx) for idx in firm_df_labeled.index]
        
        macro_df_labeled = macro_df.copy()
        macro_df_labeled.index = ['[Macro] ' + str(idx) for idx in macro_df_labeled.index]
        
        separated_df = pd.concat([firm_df_labeled, macro_df_labeled])
    elif len(firm_df) > 0:
        firm_df_labeled = firm_df.copy()
        firm_df_labeled.index = ['[Firm] ' + str(idx) for idx in firm_df_labeled.index]
        separated_df = firm_df_labeled
    elif len(macro_df) > 0:
        macro_df_labeled = macro_df.copy()
        macro_df_labeled.index = ['[Macro] ' + str(idx) for idx in macro_df_labeled.index]
        separated_df = macro_df_labeled
    else:
        logger.warning("No firm or macro features found for separated heatmap")
        return
    
    # Create separated heatmap
    fig, ax = plt.subplots(figsize=(12, max(12, len(separated_df) * 0.4)))
    
    sns.heatmap(
        separated_df,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Feature Importance'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title('Feature Importance Heatmap Across Horizons (RF) - Firm vs Macro', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Feature Category', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.tight_layout()
    separated_path = output_path.parent / (output_path.stem + '_separated' + output_path.suffix)
    plt.savefig(separated_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Separated heatmap saved to {separated_path}")


def generate_firm_vs_macro_summary(
    importance_dict: Dict[str, pd.DataFrame],
    firm_cols: List[str],
    macro_cols: List[str],
    interaction_cols: List[str],
    results: Dict[str, Dict],
    output_path: Path
):
    """
    Generate firm vs macro importance summary CSV, including interaction features.
    
    Interaction features are classified by naming convention:
    - Features ending with _firm -> Firm
    - Features ending with _macro -> Macro
    """
    logger.info("Generating firm vs macro importance summary")
    
    summary_data = []
    
    # Map horizon names to years
    horizon_to_years = {
        '1Y': 1.0,
        '2Y': 2.0,
        '3Y': 3.0,
        '4Y': 4.0,
        '5Y': 5.0,
        '7Y': 7.0,
        '10Y': 10.0
    }
    
    for horizon_name, importance_df in importance_dict.items():
        years = horizon_to_years.get(horizon_name, float(horizon_name.replace('Y', '')))
        
        # Classify all features (pure + interaction)
        firm_features = [f for f in importance_df['feature'] if f in firm_cols]
        macro_features = [f for f in importance_df['feature'] if f in macro_cols]
        
        # Classify interaction features
        interaction_firm = []
        interaction_macro = []
        for feature in interaction_cols:
            if feature in importance_df['feature'].values:
                if feature.endswith('_firm') or feature.endswith('_Firm'):
                    interaction_firm.append(feature)
                elif feature.endswith('_macro') or feature.endswith('_Macro'):
                    interaction_macro.append(feature)
                else:
                    # Default classification based on keywords
                    if any(kw in feature.lower() for kw in ['vix', 'tnx', 'yield', 'cpi', 'fedfunds']):
                        interaction_macro.append(feature)
                    else:
                        interaction_firm.append(feature)
        
        # Combine pure and interaction features
        all_firm_features = firm_features + interaction_firm
        all_macro_features = macro_features + interaction_macro
        
        # Calculate total importance for each category
        firm_importance = importance_df[importance_df['feature'].isin(all_firm_features)]['importance'].sum()
        macro_importance = importance_df[importance_df['feature'].isin(all_macro_features)]['importance'].sum()
        total_importance = importance_df['importance'].sum()
        
        # Calculate percentages
        firm_pct = (firm_importance / total_importance * 100) if total_importance > 0 else 0.0
        macro_pct = (macro_importance / total_importance * 100) if total_importance > 0 else 0.0
        
        # Other features (if any)
        other_features = [f for f in importance_df['feature'] 
                         if f not in all_firm_features and f not in all_macro_features]
        other_importance = importance_df[importance_df['feature'].isin(other_features)]['importance'].sum()
        other_pct = (other_importance / total_importance * 100) if total_importance > 0 else 0.0
        
        # Get model performance
        horizon_key = horizon_name.lower()
        rf_train_r2 = results.get(horizon_key, {}).get('rf_train_r2', 0.0)
        rf_test_r2 = results.get(horizon_key, {}).get('rf_test_r2', 0.0)
        n_samples = len(importance_df)
        
        summary_data.append({
            'horizon': f"{int(years)} year{'s' if years > 1 else ''}",
            'years': years,
            'firm_level_importance': firm_importance,
            'firm_level_pct': firm_pct,
            'macro_importance': macro_importance,
            'macro_pct': macro_pct,
            'other_importance': other_importance,
            'other_pct': other_pct,
            'n_firm_level': len(all_firm_features),
            'n_macro': len(all_macro_features),
            'n_other': len(other_features),
            'train_r2': rf_train_r2,
            'test_r2': rf_test_r2,
            'n_samples': n_samples
        })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('years')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"Firm vs Macro summary saved to {output_path}")
    logger.info(f"  Total horizons: {len(summary_df)}")
    logger.info(f"  Average Firm importance: {summary_df['firm_level_pct'].mean():.1f}%")
    logger.info(f"  Average Macro importance: {summary_df['macro_pct'].mean():.1f}%")


def print_summary_table(
    results: Dict[str, Dict],
    horizons: List[str] = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']
):
    """Print summary table of top features for each horizon and model."""
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 100)
    
    for horizon in horizons:
        horizon_key = horizon.lower()
        if horizon_key not in results:
            continue
        
        result = results[horizon_key]
        
        print(f"\n{horizon} Horizon:")
        print("-" * 100)
        
        print("\nTop 10 Features - RandomForest:")
        print(f"{'Rank':<6} {'Feature':<50} {'Importance':<15}")
        print("-" * 100)
        for i, row in result['rf_importance'].head(10).iterrows():
            print(f"{i+1:<6} {row['feature']:<50} {row['importance']:<15.6f}")
        
        print("\nTop 10 Features - XGBoost:")
        print(f"{'Rank':<6} {'Feature':<50} {'Importance':<15}")
        print("-" * 100)
        for i, row in result['xgb_importance'].head(10).iterrows():
            print(f"{i+1:<6} {row['feature']:<50} {row['importance']:<15.6f}")
        
        print(f"\nModel Performance:")
        print(f"  RF - Train R²: {result.get('rf_train_r2', 'N/A'):.4f}, Test R²: {result.get('rf_test_r2', 'N/A'):.4f}")
        print(f"  XGB - Train R²: {result.get('xgb_train_r2', 'N/A'):.4f}, Test R²: {result.get('xgb_test_r2', 'N/A'):.4f}")
    
    print("\n" + "=" * 100)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Firm + Macro + Interaction Feature Importance Analysis"
    )
    parser.add_argument(
        '--firm-features',
        type=Path,
        default=Path('data/processed/nvda_features_extended.csv'),
        help='Path to firm features CSV'
    )
    parser.add_argument(
        '--macro-features',
        type=Path,
        default=Path('data/processed/macro_features_monthly.csv'),
        help='Path to macro features CSV (optional if already in firm features)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory for plots and results'
    )
    parser.add_argument(
        '--missing-threshold',
        type=float,
        default=0.2,
        help='Skip interaction if either column has >threshold missing'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Firm + Macro + Interaction Feature Importance Analysis")
    logger.info("=" * 80)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    firm_df = load_firm_features(args.firm_features)
    macro_df = load_macro_features(args.macro_features) if args.macro_features.exists() else None
    
    # 2. Merge firm and macro
    merged_df = merge_firm_and_macro(firm_df, macro_df)
    
    if len(merged_df) == 0:
        logger.error("No data after merge. Exiting.")
        return
    
    # 3. Build interaction features
    # Identify firm and macro columns
    firm_cols = [c for c in merged_df.columns if c != 'date' and 
                 not any(kw in c.lower() for kw in ['vix', 'tnx', 'yield', 'cpi', 'fedfunds', '_x_']) and
                 merged_df[c].dtype in [np.float64, np.int64]]
    macro_cols = [c for c in merged_df.columns if c != 'date' and 
                  any(kw in c.lower() for kw in ['vix', 'tnx', 'yield', 'cpi', 'fedfunds']) and
                  '_x_' not in c and
                  merged_df[c].dtype in [np.float64, np.int64]]
    
    logger.info(f"Identified {len(firm_cols)} firm features and {len(macro_cols)} macro features for interactions")
    
    merged_df = build_interaction_features(
        merged_df, firm_cols, macro_cols, 
        missing_threshold=args.missing_threshold
    )
    
    # 4. Build return targets
    merged_df = build_return_targets(merged_df)
    
    # 5. Prepare features
    X, firm_cols_final, macro_cols_final, interaction_cols_final = prepare_features(merged_df)
    
    # 6. Analyze each horizon
    horizons = {
        '1Y': 'ret_1y',
        '2Y': 'ret_2y',
        '3Y': 'ret_3y',
        '4Y': 'ret_4y',
        '5Y': 'ret_5y',
        '7Y': 'ret_7y',
        '10Y': 'ret_10y'
    }
    
    results = {}
    rf_importance_dict = {}
    
    for horizon_name, target_col in horizons.items():
        if target_col not in merged_df.columns:
            logger.warning(f"Target {target_col} not found, skipping {horizon_name}")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing {horizon_name} Horizon ({target_col})")
        logger.info(f"{'='*80}")
        
        # Prepare target
        y = merged_df[target_col].dropna()
        X_horizon = X.loc[y.index]
        
        if len(X_horizon) < 20:
            logger.warning(f"Insufficient samples ({len(X_horizon)}), skipping {horizon_name}")
            continue
        
        # Handle missing values
        X_horizon = X_horizon.fillna(X_horizon.median())
        
        # Train models
        rf_model, xgb_model, rf_importance, xgb_importance, rf_train_r2, rf_test_r2, xgb_train_r2, xgb_test_r2 = train_models(
            X_horizon, y
        )
        
        # Store results
        results[horizon_name.lower()] = {
            'rf_importance': rf_importance,
            'xgb_importance': xgb_importance,
            'rf_train_r2': rf_train_r2,
            'rf_test_r2': rf_test_r2,
            'xgb_train_r2': xgb_train_r2,
            'xgb_test_r2': xgb_test_r2,
        }
        rf_importance_dict[horizon_name] = rf_importance
        
        # 7. Save importance plots
        plot_feature_importance(
            rf_importance,
            args.output_dir / f"importance_rf_{horizon_name.lower()}.png",
            top_n=20,
            title=f"RandomForest Feature Importance - {horizon_name}"
        )
        
        plot_feature_importance(
            xgb_importance,
            args.output_dir / f"importance_xgb_{horizon_name.lower()}.png",
            top_n=20,
            title=f"XGBoost Feature Importance - {horizon_name}"
        )
        
        # 8. SHAP analysis (XGB only)
        try:
            compute_shap_values(
                xgb_model,
                X_horizon,
                args.output_dir / f"shap_{horizon_name.lower()}.png",
                max_samples=100
            )
        except Exception as e:
            logger.warning(f"SHAP computation failed for {horizon_name}: {e}")
    
    # 9. Create consolidated heatmaps
    if rf_importance_dict:
        create_consolidated_heatmap(
            rf_importance_dict,
            args.output_dir / "importance_heatmap_all_horizons.png",
            firm_cols_final,
            macro_cols_final,
            interaction_cols_final,
            top_n=20
        )
    
    # 10. Print summary table
    print_summary_table(results, horizons=list(horizons.keys()))
    
    # 11. Generate firm vs macro importance summary (including interaction features)
    if rf_importance_dict:
        # Save to summary subdirectory
        summary_path = Path("outputs/feature_importance/data/long_term/summary/firm_vs_macro_importance_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        generate_firm_vs_macro_summary(
            rf_importance_dict,
            firm_cols_final,
            macro_cols_final,
            interaction_cols_final,
            results,
            summary_path
        )
    
    logger.info("=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()


