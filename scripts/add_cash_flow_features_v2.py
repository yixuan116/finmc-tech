"""
Add Cash Flow Features (OCF / FCF) + Regenerate RF/XGB Importance

This script:
1. Loads current features and cash flow data from JSON
2. Creates monthly OCF_TTM, CAPEX_TTM, FCF_TTM features
3. Adds Macro × Cash Flow interaction features
4. Saves updated feature table
5. Retrains RF & XGB models for feature importance
6. Generates three-level heatmaps (Firm/Macro/Interaction)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================
# (1) Load Current Features & Cash Flow Data
# =============================================

def load_fundamentals_json(json_path: Path) -> pd.DataFrame:
    """Load quarterly fundamentals from JSON and extract cash flow data."""
    logger.info(f"Loading fundamentals from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'NVDA_Firm_Fundamentals' not in data:
        raise ValueError("JSON must have 'NVDA_Firm_Fundamentals' key")
    
    quarters = data['NVDA_Firm_Fundamentals']
    logger.info(f"Loaded {len(quarters)} quarters")
    
    # Extract cash flow data
    rows = []
    for q in quarters:
        row = {
            'fiscal_year': q.get('fiscal_year'),
            'fiscal_quarter': q.get('fiscal_quarter'),
            'period_end': q.get('period_end'),
            'report_date': q.get('report_date'),
        }
        
        # Extract financials
        fin = q.get('financials', {})
        cash_flows = fin.get('cash_flows', {})
        
        row['operating_cash_flow'] = cash_flows.get('operating_cash_flow')
        row['investing_cash_flow'] = cash_flows.get('investing_cash_flow')
        row['financing_cash_flow'] = cash_flows.get('financing_cash_flow')
        row['capital_expenditures'] = fin.get('capital_expenditures')
        row['revenue'] = fin.get('revenue')
        row['net_income_gaap'] = fin.get('net_income_gaap')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert period_end to datetime
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    df = df.sort_values('period_end').reset_index(drop=True)
    df = df.set_index('period_end')
    
    return df


def load_current_features(csv_path: Path) -> pd.DataFrame:
    """Load current extended features CSV."""
    logger.info(f"Loading current features from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert period_end to datetime
    if 'period_end' in df.columns:
        df['period_end'] = pd.to_datetime(df['period_end'])
        df = df.set_index('period_end')
    elif 'px_date' in df.columns:
        df['px_date'] = pd.to_datetime(df['px_date'])
        df = df.set_index('px_date')
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================
# (2) Create Monthly Cash Flow Features
# =============================================

def create_monthly_cash_flow_features(
    quarterly_df: pd.DataFrame,
    monthly_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create monthly OCF_TTM, CAPEX_TTM, FCF_TTM features.
    
    Strategy:
    - Forward-fill quarterly values to monthly
    - Compute rolling sum over last 4 quarters (12 months)
    """
    logger.info("Creating monthly cash flow features")
    
    # Prepare quarterly cash flow data
    q_cf = quarterly_df[['operating_cash_flow', 'investing_cash_flow', 'capital_expenditures']].copy()
    
    # Resample quarterly to monthly and forward-fill
    q_cf_monthly = q_cf.resample('ME').last().ffill()
    
    # Align with monthly feature dataframe
    monthly_index = monthly_df.index
    
    # Create monthly series aligned with feature dataframe
    ocf_monthly = pd.Series(index=monthly_index, dtype=float)
    capex_monthly = pd.Series(index=monthly_index, dtype=float)
    
    for date in monthly_index:
        # Find the most recent quarterly value up to this date
        available = q_cf_monthly[q_cf_monthly.index <= date]
        if len(available) > 0:
            ocf_monthly[date] = available['operating_cash_flow'].iloc[-1]
            # CapEx: use capital_expenditures if available, otherwise use -investing_cash_flow
            if pd.notna(available['capital_expenditures'].iloc[-1]):
                capex_monthly[date] = available['capital_expenditures'].iloc[-1]
            else:
                # Investing cash flow is typically negative, so we take absolute value
                capex_monthly[date] = abs(available['investing_cash_flow'].iloc[-1])
    
    # Compute TTM (trailing 12 months = 4 quarters)
    # For monthly data, we need to sum over last 12 months
    # But since we forward-filled quarterly data, we need to be careful
    
    # Better approach: compute TTM from quarterly data, then forward-fill
    q_ocf_ttm = q_cf['operating_cash_flow'].rolling(window=4, min_periods=1).sum()
    q_capex_ttm = q_cf['capital_expenditures'].fillna(
        q_cf['investing_cash_flow'].abs()
    ).rolling(window=4, min_periods=1).sum()
    
    # Resample TTM to monthly and forward-fill
    q_ocf_ttm_monthly = q_ocf_ttm.resample('ME').last().ffill()
    q_capex_ttm_monthly = q_capex_ttm.resample('ME').last().ffill()
    
    # Align with monthly feature dataframe
    ocf_ttm = pd.Series(index=monthly_index, dtype=float)
    capex_ttm = pd.Series(index=monthly_index, dtype=float)
    
    for date in monthly_index:
        available_ocf = q_ocf_ttm_monthly[q_ocf_ttm_monthly.index <= date]
        available_capex = q_capex_ttm_monthly[q_capex_ttm_monthly.index <= date]
        
        if len(available_ocf) > 0:
            ocf_ttm[date] = available_ocf.iloc[-1]
        if len(available_capex) > 0:
            capex_ttm[date] = available_capex.iloc[-1]
    
    # FCF_TTM = OCF_TTM - CAPEX_TTM
    fcf_ttm = ocf_ttm - capex_ttm
    
    # Add to monthly dataframe
    monthly_df = monthly_df.copy()
    monthly_df['ocf_ttm'] = ocf_ttm
    monthly_df['capex_ttm'] = capex_ttm
    monthly_df['fcf_ttm'] = fcf_ttm
    
    logger.info(f"OCF_TTM: {monthly_df['ocf_ttm'].notna().sum()}/{len(monthly_df)} non-null")
    logger.info(f"CAPEX_TTM: {monthly_df['capex_ttm'].notna().sum()}/{len(monthly_df)} non-null")
    logger.info(f"FCF_TTM: {monthly_df['fcf_ttm'].notna().sum()}/{len(monthly_df)} non-null")
    
    return monthly_df


# =============================================
# (4) Generate Macro × Cash Flow Interactions
# =============================================

def create_cash_flow_interactions(
    df: pd.DataFrame,
    macro_features: List[str],
    cash_flow_features: List[str]
) -> pd.DataFrame:
    """
    Create Kronecker product interactions: macro × cash_flow.
    
    Format: ix_<macro>__<cashflow_feature>
    """
    logger.info("Creating Macro × Cash Flow interaction features")
    
    df = df.copy()
    interaction_cols = []
    
    for macro in macro_features:
        if macro not in df.columns:
            logger.warning(f"Macro feature '{macro}' not found, skipping")
            continue
        
        for cf in cash_flow_features:
            if cf not in df.columns:
                logger.warning(f"Cash flow feature '{cf}' not found, skipping")
                continue
            
            # Create interaction
            interaction_name = f"ix_{macro}__{cf}"
            df[interaction_name] = df[macro] * df[cf]
            interaction_cols.append(interaction_name)
            
            # Check for too many missing values
            missing_pct = df[interaction_name].isna().sum() / len(df)
            if missing_pct > 0.2:
                logger.warning(f"{interaction_name}: {missing_pct:.1%} missing, but keeping")
    
    logger.info(f"Created {len(interaction_cols)} interaction features")
    return df, interaction_cols


# =============================================
# (6) Train Models & Compute Importance
# =============================================

def create_target_variables(df: pd.DataFrame, horizons: Dict[str, int]) -> pd.DataFrame:
    """Create target variables for different horizons from price data."""
    logger.info("Creating target variables for different horizons")
    
    df = df.copy()
    
    # Use adj_close if available, otherwise use price columns
    if 'adj_close' in df.columns:
        price_col = 'adj_close'
    elif 'close' in df.columns:
        price_col = 'close'
    else:
        logger.warning("No price column found, cannot create target variables")
        return df
    
    # Create log returns for different horizons
    for horizon_name, months in horizons.items():
        target_name = f'ret_{horizon_name}'
        # Log return: log(price_t+h / price_t)
        df[target_name] = np.log(df[price_col].shift(-months) / df[price_col])
        logger.info(f"Created {target_name}: {df[target_name].notna().sum()} non-null values")
    
    return df


def train_and_compute_importance(
    df: pd.DataFrame,
    horizons: Dict[str, int],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Train RF and XGB models for each horizon and compute feature importance.
    
    Returns:
        rf_importance_dict: {horizon: DataFrame with feature importance}
        xgb_importance_dict: {horizon: DataFrame with feature importance}
    """
    logger.info("=" * 60)
    logger.info("Training models and computing feature importance")
    logger.info("=" * 60)
    
    rf_importance_dict = {}
    xgb_importance_dict = {}
    
    # Create target variables
    df = create_target_variables(df, horizons)
    
    # Prepare features (exclude target and date columns)
    exclude_cols = [
        'period_end', 'px_date', 'fiscal_year', 'fiscal_quarter',
        'future_12m_return', 'future_12m_price', 'future_12m_logprice',
        'adj_close', 'close', 'open', 'high', 'low',
        'fy', 'fp', 'form', 'tag_used', 'ticker'  # Non-numeric metadata columns
    ]
    # Add target columns to exclude
    for horizon_name in horizons.keys():
        exclude_cols.append(f'ret_{horizon_name}')
    
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
    
    feature_cols = valid_cols
    logger.info(f"Using {len(feature_cols)} features")
    
    for horizon_name, months in horizons.items():
        logger.info(f"\n--- Horizon: {horizon_name} ({months} months) ---")
        
        # Get target column name
        target_col = f'ret_{horizon_name}'
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found, skipping {horizon_name}")
            continue
        
        # Get target
        y = df[target_col]
        
        # Prepare X and y
        X = df[feature_cols].copy()
        
        # Drop rows where target is NaN
        valid_mask = y.notna() & X.notna().all(axis=1)
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        if len(X_clean) < 10:
            logger.warning(f"Not enough samples for {horizon_name}: {len(X_clean)}, skipping")
            continue
        
        # Fill remaining NaN with median (only for numeric columns)
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
        
        # Train/test split (temporal split)
        split_idx = int(len(X_clean) * (1 - test_size))
        if split_idx < 10:
            logger.warning(f"Split index too small ({split_idx}), using all data for training")
            X_train = X_clean
            X_test = X_clean.iloc[-min(5, len(X_clean)):]  # Use last 5 for test
            y_train = y_clean
            y_test = y_clean.iloc[-min(5, len(y_clean)):]
        else:
            X_train = X_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_train = y_clean.iloc[:split_idx]
            y_test = y_clean.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # RF importance
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        rf_importance['importance_pct'] = rf_importance['importance'] / rf_importance['importance'].sum() * 100
        
        # RF metrics
        rf_train_pred = rf.predict(X_train)
        rf_test_pred = rf.predict(X_test)
        rf_train_r2 = r2_score(y_train, rf_train_pred)
        rf_test_r2 = r2_score(y_test, rf_test_pred)
        
        logger.info(f"RF - Train R²: {rf_train_r2:.4f}, Test R²: {rf_test_r2:.4f}")
        
        rf_importance_dict[horizon_name] = rf_importance
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        # XGB importance
        xgb_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        xgb_importance['importance_pct'] = xgb_importance['importance'] / xgb_importance['importance'].sum() * 100
        
        # XGB metrics
        xgb_train_pred = xgb_model.predict(X_train)
        xgb_test_pred = xgb_model.predict(X_test)
        xgb_train_r2 = r2_score(y_train, xgb_train_pred)
        xgb_test_r2 = r2_score(y_test, xgb_test_pred)
        
        logger.info(f"XGB - Train R²: {xgb_train_r2:.4f}, Test R²: {xgb_test_r2:.4f}")
        
        xgb_importance_dict[horizon_name] = xgb_importance
    
    return rf_importance_dict, xgb_importance_dict


# =============================================
# (7) Generate Three-Level Heatmap
# =============================================

def classify_feature(feature_name: str) -> str:
    """Classify feature as Firm, Macro, or Interaction."""
    if feature_name.startswith('ix_'):
        return 'Interaction'
    elif any(macro in feature_name for macro in ['vix', 'tnx', 'inflation', 'fedfunds', 'unemployment', 'macro']):
        return 'Macro'
    else:
        return 'Firm'


def create_category_heatmap(
    importance_dict: Dict[str, pd.DataFrame],
    model_name: str,
    output_dir: Path
):
    """Create heatmap showing importance by category (Firm/Macro/Interaction) across horizons."""
    logger.info(f"Creating category heatmap for {model_name}")
    
    horizons = sorted(importance_dict.keys())
    if len(horizons) == 0:
        logger.warning(f"No horizons available for {model_name}, skipping heatmap")
        return pd.DataFrame()
    
    # Aggregate by category
    categories = ['Firm', 'Macro', 'Interaction']
    category_importance = pd.DataFrame(index=categories, columns=horizons)
    
    for horizon in horizons:
        df = importance_dict[horizon]
        
        for category in categories:
            # Filter features by category
            mask = df['feature'].apply(classify_feature) == category
            category_importance.loc[category, horizon] = df[mask]['importance_pct'].sum()
    
    category_importance = category_importance.astype(float)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        category_importance,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Importance (%)'}
    )
    plt.title(f'{model_name} Feature Importance by Category Across Horizons')
    plt.ylabel('Category')
    plt.xlabel('Horizon')
    plt.tight_layout()
    
    output_path = output_dir / f'importance_categories_{model_name.lower()}_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    return category_importance


def create_top_features_plot(
    importance_dict: Dict[str, pd.DataFrame],
    model_name: str,
    output_dir: Path,
    top_n: int = 20
):
    """Create bar plot of top N features for each horizon."""
    logger.info(f"Creating top features plot for {model_name}")
    
    horizons = sorted(importance_dict.keys())
    n_horizons = len(horizons)
    
    if n_horizons == 0:
        logger.warning(f"No horizons available for {model_name}, skipping plot")
        return
    
    fig, axes = plt.subplots(1, n_horizons, figsize=(5 * n_horizons, 8))
    if n_horizons == 1:
        axes = [axes]
    
    for idx, horizon in enumerate(horizons):
        df = importance_dict[horizon].head(top_n)
        
        ax = axes[idx]
        colors = [classify_feature(f) for f in df['feature']]
        color_map = {'Firm': '#2E86AB', 'Macro': '#A23B72', 'Interaction': '#F18F01'}
        bar_colors = [color_map.get(c, '#6C757D') for c in colors]
        
        ax.barh(range(len(df)), df['importance_pct'], color=bar_colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'], fontsize=8)
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'{horizon}')
        ax.invert_yaxis()
    
    plt.suptitle(f'{model_name} Top {top_n} Features by Horizon', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f'importance_{model_name.lower()}_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Add cash flow features and retrain models')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended.csv',
                       help='Path to current features CSV')
    parser.add_argument('--fundamentals-json', type=str,
                       default='outputs/data/fundamentals/nvda_firm_fundamentals_master.json',
                       help='Path to fundamentals JSON')
    parser.add_argument('--output-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to output CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance/plots',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Add Cash Flow Features + Regenerate Feature Importance")
    logger.info("=" * 80)
    
    # (1) Load data
    logger.info("\n[Step 1] Loading data...")
    features_df = load_current_features(Path(args.features_csv))
    fundamentals_df = load_fundamentals_json(Path(args.fundamentals_json))
    
    # (2) Create monthly cash flow features
    logger.info("\n[Step 2] Creating monthly cash flow features...")
    features_df = create_monthly_cash_flow_features(fundamentals_df, features_df)
    
    # (3) Add cash flow features to feature table
    logger.info("\n[Step 3] Cash flow features added to feature table")
    
    # (4) Generate Macro × Cash Flow interactions
    logger.info("\n[Step 4] Generating Macro × Cash Flow interactions...")
    macro_features = ['tnx_yield', 'tnx_change_3m', 'vix_level', 'vix_change_3m']
    # Add more macro features if available
    for col in features_df.columns:
        if any(m in col.lower() for m in ['inflation', 'fedfunds', 'unemployment']):
            if col not in macro_features:
                macro_features.append(col)
    
    cash_flow_features = ['ocf_ttm', 'capex_ttm', 'fcf_ttm']
    features_df, interaction_cols = create_cash_flow_interactions(
        features_df, macro_features, cash_flow_features
    )
    
    # (5) Save updated feature table
    logger.info("\n[Step 5] Saving updated feature table...")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Reset index for CSV
    features_df_save = features_df.reset_index()
    features_df_save.to_csv(output_csv, index=False)
    logger.info(f"Saved: {output_csv}")
    logger.info(f"Total features: {len(features_df.columns)}")
    logger.info(f"  - Cash flow features: 3")
    logger.info(f"  - Interaction features: {len(interaction_cols)}")
    
    # (6) Retrain models
    logger.info("\n[Step 6] Retraining models...")
    # Use quarters instead of months since data is quarterly
    horizons = {
        '12m': 4,   # 4 quarters = 12 months
        '36m': 12,  # 12 quarters = 36 months
        '60m': 20,  # 20 quarters = 60 months
        '120m': 40  # 40 quarters = 120 months
    }
    
    rf_importance, xgb_importance = train_and_compute_importance(
        features_df, horizons
    )
    
    # (7) Generate plots
    logger.info("\n[Step 7] Generating plots...")
    
    # Top features plots
    if len(rf_importance) > 0:
        create_top_features_plot(rf_importance, 'RF', output_dir, top_n=20)
    if len(xgb_importance) > 0:
        create_top_features_plot(xgb_importance, 'XGB', output_dir, top_n=20)
    
    # Category heatmaps
    rf_category = create_category_heatmap(rf_importance, 'RF', output_dir)
    xgb_category = create_category_heatmap(xgb_importance, 'XGB', output_dir)
    
    # (8) Print summary table
    logger.info("\n[Step 8] Summary Table:")
    logger.info("=" * 60)
    if len(rf_category) > 0:
        logger.info("Category | " + " | ".join(rf_category.columns.tolist()))
        logger.info("-" * 60)
        for category in ['Firm', 'Macro', 'Interaction']:
            if category in rf_category.index:
                row = [category]
                for horizon in rf_category.columns:
                    val = rf_category.loc[category, horizon]
                    if pd.notna(val):
                        row.append(f"{val:.1f}%")
                    else:
                        row.append("N/A")
                logger.info(" | ".join(row))
    else:
        logger.warning("No category data available for summary table")
    
    # (9) Top 10 features by horizon
    if len(rf_importance) > 0:
        logger.info("\n[Step 9] Top 10 Features by Horizon (RF):")
        for horizon in sorted(rf_importance.keys()):
            logger.info(f"\n{horizon}:")
            top10 = rf_importance[horizon].head(10)
            for idx, row in top10.iterrows():
                category = classify_feature(row['feature'])
                logger.info(f"  {row['importance_pct']:.2f}% - {row['feature']} ({category})")
    
    # (10) Cash flow feature ranks
    if len(rf_importance) > 0:
        logger.info("\n[Step 10] Cash Flow Feature Ranks:")
        cash_flow_feature_names = ['ocf_ttm', 'capex_ttm', 'fcf_ttm']
        for horizon in sorted(rf_importance.keys()):
            logger.info(f"\n{horizon}:")
            df = rf_importance[horizon]
            for cf_feat in cash_flow_feature_names:
                if cf_feat in df['feature'].values:
                    rank = df[df['feature'] == cf_feat].index[0] + 1
                    importance = df[df['feature'] == cf_feat]['importance_pct'].iloc[0]
                    logger.info(f"  {cf_feat}: Rank {rank}, Importance {importance:.2f}%")
                else:
                    logger.info(f"  {cf_feat}: Not found")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All steps completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

