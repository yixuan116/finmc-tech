"""
Unified Model Evaluation: Consistent Evaluation Across All Horizons

This script evaluates all models (Linear, Ridge, ElasticNet, RandomForest, XGBoost, NeuralNetwork)
across all horizons (1Y, 3Y, 5Y, 10Y) using the SAME evaluation criteria:
- Same time-based split (train < 2020, val 2020-2022, test > 2022)
- Same model configurations
- Same metrics (MAE, RMSE, R²)
- Same feature preprocessing

This allows fair comparison and identifies the true champion model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.three_category_feature_importance import (
    load_features, create_target_variables, prepare_features,
    classify_feature
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================
# (1) Data Loading & Preparation
# =============================================

def load_and_prepare_data(
    csv_path: str,
    date_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load features and prepare target variables.
    
    Returns:
        df: Full dataframe with features and targets
        X: Feature matrix
        feature_cols: List of feature column names
    """
    logger.info(f"Loading features from {csv_path}")
    df = load_features(Path(csv_path))
    
    # Create target variables (1Y, 3Y, 5Y, 10Y)
    horizon_quarters = {
        '1y': 4,
        '3y': 12,
        '5y': 20,
        '10y': 40
    }
    df = create_target_variables(df, horizon_quarters)
    
    # Prepare features
    X, feature_cols, classifications = prepare_features(df)
    
    # Add feature columns back to df
    for col in feature_cols:
        if col not in df.columns:
            df[col] = X[col]
    
    # Auto-detect date column if not provided
    if date_column is None:
        date_candidates = ['date', 'Date', 'px_date', 'period_end', 'timestamp']
        for col in date_candidates:
            if col in df.columns:
                date_column = col
                break
    
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).set_index(date_column)
    else:
        logger.warning("No date column found, using integer index")
        df.index = pd.RangeIndex(len(df))
    
    logger.info(f"Loaded {len(df)} rows, {len(feature_cols)} features")
    logger.info(f"Target variables: ret_1y ({df['ret_1y'].notna().sum()} non-null), "
                f"ret_3y ({df['ret_3y'].notna().sum()} non-null), "
                f"ret_5y ({df['ret_5y'].notna().sum()} non-null), "
                f"ret_10y ({df['ret_10y'].notna().sum()} non-null)")
    
    return df, X, feature_cols


# =============================================
# (2) Time-Based Split (Same as train_models.py)
# =============================================

def time_based_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data using adaptive time-based split:
    - For 1Y: Train < 2020-12-31, Test > 2022-12-31 (same as train_models.py)
    - For 3Y+: Use earlier test split to ensure test set has data
    """
    # Determine horizon from target column
    if '1y' in target_col.lower():
        train_end = pd.Timestamp("2020-12-31")
        test_start = pd.Timestamp("2022-12-31")
    elif '3y' in target_col.lower():
        train_end = pd.Timestamp("2018-12-31")
        test_start = pd.Timestamp("2020-12-31")
    elif '5y' in target_col.lower():
        train_end = pd.Timestamp("2016-12-31")
        test_start = pd.Timestamp("2018-12-31")
    elif '10y' in target_col.lower():
        train_end = pd.Timestamp("2012-12-31")
        test_start = pd.Timestamp("2014-12-31")
    else:
        # Default: use 80/20 split
        train_end = None
        test_start = None
    
    if isinstance(df.index, pd.DatetimeIndex) and train_end is not None:
        train_mask = df.index < train_end + pd.Timedelta(days=1)
        test_mask = df.index > test_start
        val_mask = pd.Series([False] * len(df), index=df.index)  # No validation set for simplicity
    else:
        # Fallback: use 80/20 split if no datetime index or no specific dates
        split_idx = int(len(df) * 0.8)
        train_mask = pd.Series([True] * split_idx + [False] * (len(df) - split_idx), index=df.index)
        val_mask = pd.Series([False] * len(df), index=df.index)
        test_mask = ~train_mask
        logger.warning("No datetime index found or no specific dates, using 80/20 split instead")
    
    # Filter valid rows (non-null target)
    train_df = df[train_mask & df[target_col].notna()].copy()
    val_df = df[val_mask & df[target_col].notna()].copy()
    test_df = df[test_mask & df[target_col].notna()].copy()
    
    # If test set is empty, fall back to 80/20 split
    if len(test_df) == 0:
        logger.warning(f"Test set empty for {target_col}, falling back to 80/20 split")
        split_idx = int(len(df) * 0.8)
        train_mask = pd.Series([True] * split_idx + [False] * (len(df) - split_idx), index=df.index)
        test_mask = ~train_mask
        train_df = df[train_mask & df[target_col].notna()].copy()
        test_df = df[test_mask & df[target_col].notna()].copy()
        val_df = pd.DataFrame(columns=feature_cols)
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Time split failed for {target_col}. Train: {len(train_df)}, Test: {len(test_df)}")
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[feature_cols].copy() if len(val_df) > 0 else pd.DataFrame(columns=feature_cols)
    y_val = val_df[target_col].copy() if len(val_df) > 0 else pd.Series(dtype=float)
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    
    logger.info(f"Split sizes for {target_col}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================
# (3) Model Definitions (Same Configurations)
# =============================================

def get_models():
    """
    Return models with same configurations as train_models.py and champion_model_comparison.py.
    """
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
        'RandomForest': RandomForestRegressor(
            n_estimators=500,  # Same as train_models.py
            max_depth=None,     # Same as train_models.py
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500,   # Same as train_models.py
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42
        ),
        'NeuralNetwork': MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
    }
    return models


# =============================================
# (4) Evaluation Function
# =============================================

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = None,
    scaler: StandardScaler = None
) -> Dict[str, float]:
    """
    Train and evaluate a model.
    
    Neural Network requires feature scaling.
    """
    # Neural Network requires scaling
    if model_name == 'NeuralNetwork':
        if scaler is None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


# =============================================
# (5) Main Evaluation Pipeline
# =============================================

def evaluate_all_models_across_horizons(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Dict[str, str]
) -> pd.DataFrame:
    """
    Evaluate all models across all horizons using unified evaluation criteria.
    """
    logger.info("=" * 80)
    logger.info("Unified Model Evaluation Across All Horizons")
    logger.info("=" * 80)
    
    models = get_models()
    results = []
    
    for horizon_name, target_col in horizons.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Horizon: {horizon_name} (target: {target_col})")
        logger.info(f"{'='*80}")
        
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not found, skipping {horizon_name}")
            continue
        
        try:
            # Time-based split
            X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
                df, feature_cols, target_col
            )
            
            # Fill NaN in features
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
            X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
            
            # Create scaler for Neural Network
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            # Evaluate each model
            for model_name, model in models.items():
                logger.info(f"  Training {model_name}...")
                
                try:
                    metrics = evaluate_model(
                        model, X_train, y_train, X_test, y_test,
                        model_name=model_name, scaler=scaler
                    )
                    
                    results.append({
                        'model': model_name,
                        'horizon': horizon_name,
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics['r2']
                    })
                    
                    logger.info(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"    Error training {model_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {horizon_name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df


# =============================================
# (6) Champion Identification
# =============================================

def identify_champion_models(results_df: pd.DataFrame):
    """
    Identify champion models using the same criteria as train_models.py:
    - Best R² on test set
    """
    logger.info("\n" + "=" * 80)
    logger.info("Champion Model Identification (Based on Test R²)")
    logger.info("=" * 80)
    
    horizons = sorted(results_df['horizon'].unique())
    
    for horizon in horizons:
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        # Best R² (same as train_models.py)
        best_r2 = horizon_df.loc[horizon_df['r2'].idxmax()]
        
        logger.info(f"\n{horizon}:")
        logger.info(f"  Champion: {best_r2['model']:15s} (R² = {best_r2['r2']:.4f})")
        logger.info(f"  MAE: {best_r2['mae']:.4f}, RMSE: {best_r2['rmse']:.4f}")
        
        # Show all models sorted by R²
        logger.info(f"\n  All models (sorted by R²):")
        sorted_df = horizon_df.sort_values('r2', ascending=False)
        for _, row in sorted_df.iterrows():
            marker = "🏆" if row['model'] == best_r2['model'] else "  "
            logger.info(f"    {marker} {row['model']:15s} - R²: {row['r2']:8.4f}, "
                       f"MAE: {row['mae']:.4f}, RMSE: {row['rmse']:.4f}")
    
    # Overall champion (best average R² across all horizons)
    overall_champion = results_df.groupby('model')['r2'].mean().idxmax()
    overall_r2 = results_df.groupby('model')['r2'].mean().max()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Overall Champion (Best Average R²): {overall_champion} (R² = {overall_r2:.4f})")
    logger.info(f"{'='*80}")


# =============================================
# (7) Visualization
# =============================================

def plot_unified_results(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization of unified evaluation results."""
    logger.info("\nGenerating unified comparison plots...")
    
    horizons = sorted(results_df['horizon'].unique())
    models = sorted(results_df['model'].unique())
    
    # Plot R² comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, horizon in enumerate(horizons):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        # Sort by R² (descending)
        horizon_df = horizon_df.sort_values('r2', ascending=False)
        
        bars = ax.bar(horizon_df['model'], horizon_df['r2'], color='coral', alpha=0.7)
        ax.set_title(f'{horizon} - R² Comparison (Unified Evaluation)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('R²', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best (highest R²)
        best_idx = horizon_df['r2'].idxmax()
        bars[horizon_df.index.get_loc(best_idx)].set_color('green')
        bars[horizon_df.index.get_loc(best_idx)].set_alpha(1.0)
    
    plt.suptitle('Unified Model Comparison: R² Across Horizons\n(Same Time Split: Train<2020, Test>2022)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    r2_path = output_dir / 'unified_model_comparison_r2.png'
    plt.savefig(r2_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {r2_path}")
    
    # Plot MAE comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, horizon in enumerate(horizons):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        # Sort by MAE (ascending - lower is better)
        horizon_df = horizon_df.sort_values('mae', ascending=True)
        
        bars = ax.bar(horizon_df['model'], horizon_df['mae'], color='steelblue', alpha=0.7)
        ax.set_title(f'{horizon} - MAE Comparison (Unified Evaluation)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('MAE', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best (lowest MAE)
        best_idx = horizon_df['mae'].idxmin()
        bars[horizon_df.index.get_loc(best_idx)].set_color('green')
        bars[horizon_df.index.get_loc(best_idx)].set_alpha(1.0)
    
    plt.suptitle('Unified Model Comparison: MAE Across Horizons\n(Same Time Split: Train<2020, Test>2022)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    mae_path = output_dir / 'unified_model_comparison_mae.png'
    plt.savefig(mae_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {mae_path}")
    
    # Plot RMSE comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, horizon in enumerate(horizons):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        # Sort by RMSE (ascending - lower is better)
        horizon_df = horizon_df.sort_values('rmse', ascending=True)
        
        bars = ax.bar(horizon_df['model'], horizon_df['rmse'], color='mediumpurple', alpha=0.7)
        ax.set_title(f'{horizon} - RMSE Comparison (Unified Evaluation)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best (lowest RMSE)
        best_idx = horizon_df['rmse'].idxmin()
        bars[horizon_df.index.get_loc(best_idx)].set_color('green')
        bars[horizon_df.index.get_loc(best_idx)].set_alpha(1.0)
    
    plt.suptitle('Unified Model Comparison: RMSE Across Horizons\n(Same Time Split: Train<2020, Test>2022)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    rmse_path = output_dir / 'unified_model_comparison_rmse.png'
    plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {rmse_path}")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Unified model evaluation across horizons')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to extended features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Unified Model Evaluation")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n[Step 1] Loading data...")
    df, X, feature_cols = load_and_prepare_data(args.features_csv)
    
    horizons = {
        '1Y': 'ret_1y',
        '3Y': 'ret_3y',
        '5Y': 'ret_5y',
        '10Y': 'ret_10y'
    }
    
    # Evaluate all models
    logger.info("\n[Step 2] Evaluating all models across horizons...")
    results_df = evaluate_all_models_across_horizons(df, feature_cols, horizons)
    
    # Save results
    logger.info("\n[Step 3] Saving results...")
    results_path = results_dir / 'unified_model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved: {results_path}")
    
    # Identify champions
    logger.info("\n[Step 4] Identifying champion models...")
    identify_champion_models(results_df)
    
    # Create plots
    logger.info("\n[Step 5] Creating visualization...")
    plot_unified_results(results_df, plots_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Unified evaluation complete!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Plots saved to: {plots_dir}")


if __name__ == '__main__':
    main()

