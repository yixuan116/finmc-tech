"""
Three-Category Feature Importance Analysis (Firm / Macro / Interaction)

This script:
1. Loads extended feature table (nvda_features_extended_v2.csv)
2. Classifies features into Firm, Macro, or Interaction categories
3. Trains RF and XGB models for 1y, 3y, 5y, 10y horizons
4. Aggregates importance by category
5. Generates heatmaps and summary reports
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================
# (2) Define Feature Classification
# =============================================

def classify_feature(feature_name: str) -> str:
    """
    Classify a feature into one of three categories:
    - Interaction: starts with "ix_"
    - Macro: pure macro variables
    - Firm: everything else
    """
    # Rule 1: Interaction features (most important check first)
    if feature_name.startswith("ix_"):
        return "Interaction"
    
    # Rule 2: Pure macro features
    macro_features = [
        "tnx_yield", "tnx_change_3m",
        "vix_level", "vix_change_3m",
        "inflation", "fedfunds",
        "unemployment", "recession"
    ]
    
    # Check exact match or if macro keyword is in the name (but not as part of interaction)
    if feature_name in macro_features:
        return "Macro"
    
    # Check for macro keywords in the name (for variations)
    macro_keywords = ["tnx", "vix", "inflation", "fedfunds", "unemployment", "recession"]
    if any(keyword in feature_name.lower() for keyword in macro_keywords):
        # But make sure it's not an interaction feature (already checked above)
        return "Macro"
    
    # Rule 3: Everything else is Firm
    return "Firm"


# =============================================
# (1) Load Extended Feature Table
# =============================================

def load_features(csv_path: Path) -> pd.DataFrame:
    """Load extended feature table."""
    logger.info(f"Loading features from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    if 'period_end' in df.columns:
        df['date'] = pd.to_datetime(df['period_end'])
        df = df.set_index('date')
    elif 'px_date' in df.columns:
        df['date'] = pd.to_datetime(df['px_date'])
        df = df.set_index('date')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        logger.warning("No date column found, using index")
        df.index = pd.to_datetime(df.index)
    
    df = df.sort_index()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Verify cash flow features exist
    cash_flow_features = ['ocf_ttm', 'capex_ttm', 'fcf_ttm']
    for cf in cash_flow_features:
        if cf in df.columns:
            logger.info(f"  ✓ Found cash flow feature: {cf}")
        else:
            logger.warning(f"  ✗ Missing cash flow feature: {cf}")
    
    return df


# =============================================
# (3) Prepare Training Data
# =============================================

def create_target_variables(df: pd.DataFrame, horizons: Dict[str, int]) -> pd.DataFrame:
    """Create target variables for different horizons."""
    logger.info("Creating target variables")
    
    df = df.copy()
    
    # Find price column
    if 'adj_close' in df.columns:
        price_col = 'adj_close'
    elif 'close' in df.columns:
        price_col = 'close'
    else:
        raise ValueError("No price column found (adj_close or close)")
    
    # Create log returns for each horizon
    for horizon_name, quarters in horizons.items():
        target_name = f'ret_{horizon_name}'
        # Log return: log(price_t+h / price_t)
        df[target_name] = np.log(df[price_col].shift(-quarters) / df[price_col])
        non_null = df[target_name].notna().sum()
        logger.info(f"  {target_name}: {non_null} non-null values")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare feature matrix, excluding targets and non-numeric columns."""
    logger.info("Preparing feature matrix")
    
    # Exclude columns
    exclude_cols = [
        'period_end', 'px_date', 'fiscal_year', 'fiscal_quarter',
        'adj_close', 'close', 'open', 'high', 'low',
        'fy', 'fp', 'form', 'tag_used', 'ticker',
        'future_12m_return', 'future_12m_price', 'future_12m_logprice'
    ]
    
    # Add target columns to exclude
    target_cols = [c for c in df.columns if c.startswith('ret_')]
    exclude_cols.extend(target_cols)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Only keep numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]
    
    # Remove columns with >50% missing
    valid_cols = []
    for col in feature_cols:
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct < 0.5:
            valid_cols.append(col)
        else:
            logger.debug(f"Dropping {col}: {missing_pct:.1%} missing")
    
    logger.info(f"Using {len(valid_cols)} features")
    
    # Classify features
    classifications = {col: classify_feature(col) for col in valid_cols}
    firm_count = sum(1 for v in classifications.values() if v == "Firm")
    macro_count = sum(1 for v in classifications.values() if v == "Macro")
    interaction_count = sum(1 for v in classifications.values() if v == "Interaction")
    
    logger.info(f"Feature classification:")
    logger.info(f"  Firm: {firm_count}")
    logger.info(f"  Macro: {macro_count}")
    logger.info(f"  Interaction: {interaction_count}")
    
    return df[valid_cols], valid_cols, classifications


# =============================================
# (4) Train Random Forest Models
# =============================================

def train_rf_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Dict[str, int],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Train Random Forest models for each horizon.
    
    Returns:
        Dictionary mapping horizon to DataFrame with feature importance
    """
    logger.info("=" * 60)
    logger.info("Training Random Forest Models")
    logger.info("=" * 60)
    
    importance_dict = {}
    
    for horizon_name, quarters in horizons.items():
        logger.info(f"\n--- Horizon: {horizon_name} ({quarters} quarters) ---")
        
        target_col = f'ret_{horizon_name}'
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not found, skipping")
            continue
        
        y = df[target_col]
        X = df[feature_cols].copy()
        
        # Drop rows where target is NaN
        valid_mask = y.notna() & X.notna().all(axis=1)
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        if len(X_clean) < 10:
            logger.warning(f"Not enough samples: {len(X_clean)}, skipping")
            continue
        
        # Fill NaN with median (numeric columns only)
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
        
        # Time-based split (first 80% train, last 20% test)
        split_idx = int(len(X_clean) * (1 - test_size))
        if split_idx < 10:
            split_idx = max(5, len(X_clean) - 5)
        
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Compute importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Normalize to sum to 100%
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        # Metrics
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        importance_dict[horizon_name] = importance_df
    
    return importance_dict


# =============================================
# (5) Train XGBoost Models
# =============================================

def train_xgb_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Dict[str, int],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Train XGBoost models for each horizon.
    
    Returns:
        Dictionary mapping horizon to DataFrame with feature importance
    """
    logger.info("=" * 60)
    logger.info("Training XGBoost Models")
    logger.info("=" * 60)
    
    importance_dict = {}
    
    for horizon_name, quarters in horizons.items():
        logger.info(f"\n--- Horizon: {horizon_name} ({quarters} quarters) ---")
        
        target_col = f'ret_{horizon_name}'
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not found, skipping")
            continue
        
        y = df[target_col]
        X = df[feature_cols].copy()
        
        # Drop rows where target is NaN
        valid_mask = y.notna() & X.notna().all(axis=1)
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        if len(X_clean) < 10:
            logger.warning(f"Not enough samples: {len(X_clean)}, skipping")
            continue
        
        # Fill NaN with median (numeric columns only)
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
        
        # Time-based split (first 80% train, last 20% test)
        split_idx = int(len(X_clean) * (1 - test_size))
        if split_idx < 10:
            split_idx = max(5, len(X_clean) - 5)
        
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        # Get importance (gain)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Normalize to sum to 100%
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        # Metrics
        train_pred = xgb_model.predict(X_train)
        test_pred = xgb_model.predict(X_test)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        importance_dict[horizon_name] = importance_df
    
    return importance_dict


# =============================================
# (6) Aggregate Importance by Category
# =============================================

def aggregate_by_category(
    importance_dict: Dict[str, pd.DataFrame],
    classifications: Dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate feature importance by category for each horizon.
    
    Returns:
        DataFrame with rows = categories, columns = horizons
    """
    logger.info("Aggregating importance by category")
    
    categories = ['Firm', 'Macro', 'Interaction']
    # Sort horizons in the correct order: 1y, 3y, 5y, 10y (not alphabetical)
    horizon_order = ['1y', '3y', '5y', '10y']
    horizons = [h for h in horizon_order if h in importance_dict.keys()]
    # Add any remaining horizons that might not be in the standard list
    for h in importance_dict.keys():
        if h not in horizons:
            horizons.append(h)
    
    category_df = pd.DataFrame(index=categories, columns=horizons)
    
    for horizon in horizons:
        importance_df = importance_dict[horizon]
        
        for category in categories:
            # Filter features by category
            mask = importance_df['feature'].apply(lambda f: classifications.get(f, 'Firm') == category)
            category_importance = importance_df[mask]['importance_pct'].sum()
            category_df.loc[category, horizon] = category_importance
    
    category_df = category_df.astype(float)
    
    # Normalize each column to sum to 100% (in case of rounding)
    for col in category_df.columns:
        total = category_df[col].sum()
        if total > 0:
            category_df[col] = category_df[col] / total * 100
    
    logger.info("\nCategory Importance Summary:")
    logger.info(category_df.to_string())
    
    return category_df


# =============================================
# (7) Plot Three-Category Heatmap
# =============================================

def plot_category_heatmap(
    category_df: pd.DataFrame,
    model_name: str,
    output_dir: Path
):
    """Create heatmap showing category importance across horizons."""
    logger.info(f"Creating category heatmap for {model_name}")
    
    if len(category_df) == 0:
        logger.warning("No data for heatmap")
        return
    
    # Rename columns for display and ensure correct order
    display_df = category_df.copy()
    # Ensure columns are in the correct order: 1Y, 3Y, 5Y, 10Y
    horizon_order = ['1y', '3y', '5y', '10y']
    existing_horizons = [h for h in horizon_order if h in display_df.columns]
    # Add any remaining horizons
    for h in display_df.columns:
        if h not in existing_horizons:
            existing_horizons.append(h)
    # Reorder columns
    display_df = display_df[existing_horizons]
    # Rename for display
    display_df.columns = [f"{h.replace('y', 'Y')}" for h in display_df.columns]
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        display_df,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Importance (%)'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'{model_name} Feature Importance by Category Across Horizons', fontsize=14, pad=20)
    plt.ylabel('Category', fontsize=12)
    plt.xlabel('Horizon', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / f'importance_categories_{model_name.lower()}_3cat.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Also save to project root for user access
    root_output = Path.cwd() / f'ROOT_importance_categories_{model_name.lower()}_3cat.png'
    plt.savefig(root_output, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    logger.info(f"Saved copy to root: {root_output}")


# =============================================
# (8) Export Top Features
# =============================================

def export_top_features(
    importance_dict: Dict[str, pd.DataFrame],
    classifications: Dict[str, str],
    model_name: str
):
    """Print top 10 features for each horizon with category labels."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Top 10 Features by Horizon ({model_name})")
    logger.info("=" * 60)
    
    cash_flow_features = ['ocf_ttm', 'capex_ttm', 'fcf_ttm']
    
    for horizon in sorted(importance_dict.keys()):
        logger.info(f"\n{horizon}:")
        df = importance_dict[horizon].head(10)
        
        for idx, row in df.iterrows():
            feature = row['feature']
            importance = row['importance_pct']
            category = classifications.get(feature, 'Firm')
            
            # Highlight cash flow features
            highlight = " ⭐" if feature in cash_flow_features else ""
            logger.info(f"  {importance:6.2f}% - {feature:40s} ({category}){highlight}")


# =============================================
# (9) Print Explanation
# =============================================

def print_explanation(rf_category_df: pd.DataFrame, xgb_category_df: pd.DataFrame):
    """Print brief explanation of results."""
    logger.info("\n" + "=" * 60)
    logger.info("Key Insights")
    logger.info("=" * 60)
    
    insights = [
        "• Short horizons (1-2 years): Macro + Interaction features tend to dominate as market sentiment and regime effects are more immediate.",
        "• Medium horizons (3-5 years): Interaction features increase in importance, representing how firm fundamentals respond to macro conditions.",
        "• Long horizons (5-10 years): Firm fundamentals become the anchor, as company-specific factors drive long-term value creation.",
        "• Interaction features represent regime dependence (macro × firm sensitivity), showing how different firms perform under different economic conditions.",
        "• Cash-flow features (OCF_TTM, FCF_TTM) typically rise in importance at longer horizons, as cash generation becomes critical for sustained growth."
    ]
    
    for insight in insights:
        logger.info(insight)
    
    # Print specific observations
    if len(rf_category_df) > 0:
        logger.info("\nRF Model Observations:")
        for horizon in rf_category_df.columns:
            firm_pct = rf_category_df.loc['Firm', horizon]
            macro_pct = rf_category_df.loc['Macro', horizon]
            interaction_pct = rf_category_df.loc['Interaction', horizon]
            logger.info(f"  {horizon}: Firm={firm_pct:.1f}%, Macro={macro_pct:.1f}%, Interaction={interaction_pct:.1f}%")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Three-category feature importance analysis')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to extended features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance/plots',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create detailed top-features plots
    top_k_script = Path('scripts/generate_topk_feature_heatmaps.py')
    if top_k_script.exists():
        import subprocess
        logger.info("\nGenerating detailed Top-K feature heatmaps...")
        # Use subprocess to run the script, capturing output to ensure it runs
        result = subprocess.run(['python3', str(top_k_script), 
                       '--features-csv', args.features_csv,
                       '--output-dir', str(output_dir)],
                       capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Top-K heatmap generation failed:\n{result.stderr}")
        else:
            logger.info("Top-K heatmaps generated successfully.")
                       
    logger.info("=" * 80)
    logger.info("Three-Category Feature Importance Analysis")
    logger.info("=" * 80)
    
    # (1) Load features
    logger.info("\n[Step 1] Loading extended feature table...")
    df = load_features(Path(args.features_csv))
    
    # (2) Classify features (done in prepare_features)
    
    # (3) Prepare training data
    logger.info("\n[Step 3] Preparing training data...")
    horizons = {
        '1y': 4,   # 4 quarters = 1 year
        '3y': 12,  # 12 quarters = 3 years
        '5y': 20,  # 20 quarters = 5 years
        '10y': 40  # 40 quarters = 10 years
    }
    
    df = create_target_variables(df, horizons)
    X, feature_cols, classifications = prepare_features(df)
    
    # Add feature columns back to df for training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = X[col]
    
    # (4) Train RF models
    logger.info("\n[Step 4] Training Random Forest models...")
    rf_importance = train_rf_models(df, feature_cols, horizons)
    
    # (5) Train XGB models
    logger.info("\n[Step 5] Training XGBoost models...")
    xgb_importance = train_xgb_models(df, feature_cols, horizons)
    
    # (6) Aggregate by category
    logger.info("\n[Step 6] Aggregating importance by category...")
    rf_category_df = aggregate_by_category(rf_importance, classifications)
    xgb_category_df = aggregate_by_category(xgb_importance, classifications)
    
    # (7) Plot heatmaps
    logger.info("\n[Step 7] Generating heatmaps...")
    plot_category_heatmap(rf_category_df, 'RF', output_dir)
    plot_category_heatmap(xgb_category_df, 'XGB', output_dir)
    
    # (8) Export top features
    logger.info("\n[Step 8] Exporting top features...")
    export_top_features(rf_importance, classifications, 'RF')
    export_top_features(xgb_importance, classifications, 'XGB')
    
    # (9) Print explanation
    print_explanation(rf_category_df, xgb_category_df)
    
    # Save category tables
    logger.info("\n[Step 10] Saving category tables...")
    summary_dir = output_dir.parent / 'data' / 'long_term' / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    rf_path = summary_dir / 'rf_category_importance_3cat.csv'
    xgb_path = summary_dir / 'xgb_category_importance_3cat.csv'
    
    rf_category_df.to_csv(rf_path)
    xgb_category_df.to_csv(xgb_path)
    
    logger.info(f"Saved RF category table: {rf_path}")
    logger.info(f"Saved XGB category table: {xgb_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All steps completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

