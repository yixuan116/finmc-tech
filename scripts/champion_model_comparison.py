"""
Champion Model Comparison Across Multiple Model Families and Horizons

This script compares multiple models (Linear, Ridge, ElasticNet, RF, XGB, NeuralNetwork) across
different horizons (1Y, 3Y, 5Y, 10Y) to identify the champion model for each horizon.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.three_category_feature_importance import (
    load_features, create_target_variables, prepare_features,
    classify_feature
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================
# (1) Data & Horizons
# =============================================

def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare training and test data with time-based split.
    
    Args:
        df: Full dataframe with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        test_size: Proportion of data to use for testing (last portion)
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Get target
    y = df[target_col]
    X = df[feature_cols].copy()
    
    # Drop rows where target is NaN
    valid_mask = y.notna() & X.notna().all(axis=1)
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    if len(X_clean) < 10:
        raise ValueError(f"Not enough samples: {len(X_clean)}")
    
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
    
    return X_train, y_train, X_test, y_test


# =============================================
# (2) Candidate Models
# =============================================

def get_models():
    """
    Return a dictionary of model name → model instance.
    
    Models are initialized with simple, comparable hyperparameters.
    """
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
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
# (3) Evaluation Procedure
# =============================================

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = None
) -> Dict[str, float]:
    """
    Train a model and evaluate on test set.
    
    Returns:
        Dictionary with 'mae', 'rmse', 'r2'
    """
    # Neural Network requires feature scaling
    if model_name == 'NeuralNetwork':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
    else:
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def compare_models_across_horizons(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Dict[str, str],
    test_size: float = 0.2
) -> pd.DataFrame:
    """
    Compare all models across all horizons.
    
    Returns:
        DataFrame with columns: model, horizon, mae, rmse, r2
    """
    logger.info("=" * 80)
    logger.info("Model Comparison Across Horizons")
    logger.info("=" * 80)
    
    models = get_models()
    results = []
    
    for horizon_name, target_col in horizons.items():
        logger.info(f"\n--- Horizon: {horizon_name} (target: {target_col}) ---")
        
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not found, skipping {horizon_name}")
            continue
        
        try:
            # Prepare data
            X_train, y_train, X_test, y_test = prepare_training_data(
                df, feature_cols, target_col, test_size
            )
            
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Evaluate each model
            for model_name, model in models.items():
                logger.info(f"  Training {model_name}...")
                
                try:
                    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, model_name=model_name)
                    
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
# (4) Champion Model Identification
# =============================================

def identify_champion_models(results_df: pd.DataFrame):
    """
    Identify champion models for each horizon based on MAE and R².
    
    Prints a summary of best models per horizon.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Champion Model Identification")
    logger.info("=" * 80)
    
    horizons = sorted(results_df['horizon'].unique())
    
    for horizon in horizons:
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        logger.info(f"\n{horizon}:")
        
        # Best MAE (lower is better)
        best_mae = horizon_df.loc[horizon_df['mae'].idxmin()]
        logger.info(f"  Best MAE: {best_mae['model']:15s} = {best_mae['mae']:.4f}")
        
        # Best RMSE (lower is better)
        best_rmse = horizon_df.loc[horizon_df['rmse'].idxmin()]
        logger.info(f"  Best RMSE: {best_rmse['model']:15s} = {best_rmse['rmse']:.4f}")
        
        # Best R² (higher is better)
        best_r2 = horizon_df.loc[horizon_df['r2'].idxmax()]
        logger.info(f"  Best R² : {best_r2['model']:15s} = {best_r2['r2']:.4f}")
        
        # Show all models for comparison
        logger.info(f"\n  All models (sorted by R²):")
        sorted_df = horizon_df.sort_values('r2', ascending=False)
        for _, row in sorted_df.iterrows():
            logger.info(f"    {row['model']:15s} - MAE: {row['mae']:.4f}, RMSE: {row['rmse']:.4f}, R²: {row['r2']:.4f}")


# =============================================
# (5) Plots
# =============================================

def plot_model_comparison(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """Create bar plots comparing models across horizons."""
    logger.info("\nGenerating comparison plots...")
    
    horizons = sorted(results_df['horizon'].unique())
    models = sorted(results_df['model'].unique())
    
    # Plot MAE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, horizon in enumerate(horizons):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        # Sort by MAE (ascending)
        horizon_df = horizon_df.sort_values('mae')
        
        bars = ax.bar(horizon_df['model'], horizon_df['mae'], color='steelblue', alpha=0.7)
        ax.set_title(f'{horizon} - MAE Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('MAE', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best (lowest MAE)
        best_idx = horizon_df['mae'].idxmin()
        bars[horizon_df.index.get_loc(best_idx)].set_color('green')
        bars[horizon_df.index.get_loc(best_idx)].set_alpha(1.0)
    
    plt.suptitle('Model Comparison: MAE Across Horizons', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    mae_path = output_dir / 'model_comparison_mae.png'
    plt.savefig(mae_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {mae_path}")
    
    # Plot RMSE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, horizon in enumerate(horizons):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        horizon_df = results_df[results_df['horizon'] == horizon]
        
        if len(horizon_df) == 0:
            continue
        
        # Sort by RMSE (ascending)
        horizon_df = horizon_df.sort_values('rmse')
        
        bars = ax.bar(horizon_df['model'], horizon_df['rmse'], color='crimson', alpha=0.7)
        ax.set_title(f'{horizon} - RMSE Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best (lowest RMSE)
        best_idx = horizon_df['rmse'].idxmin()
        bars[horizon_df.index.get_loc(best_idx)].set_color('green')
        bars[horizon_df.index.get_loc(best_idx)].set_alpha(1.0)
    
    plt.suptitle('Model Comparison: RMSE Across Horizons', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    rmse_path = output_dir / 'model_comparison_rmse.png'
    plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {rmse_path}")
    
    # Plot R²
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
        ax.set_title(f'{horizon} - R² Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('R²', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best (highest R²)
        best_idx = horizon_df['r2'].idxmax()
        bars[horizon_df.index.get_loc(best_idx)].set_color('green')
        bars[horizon_df.index.get_loc(best_idx)].set_alpha(1.0)
    
    plt.suptitle('Model Comparison: R² Across Horizons', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    r2_path = output_dir / 'model_comparison_r2.png'
    plt.savefig(r2_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {r2_path}")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Champion model comparison')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to extended features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance',
                       help='Base output directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Champion Model Comparison")
    logger.info("=" * 80)
    
    # (1) Load data
    logger.info("\n[Step 1] Loading data...")
    df = load_features(Path(args.features_csv))
    
    horizons = {
        '1Y': 'ret_1y',
        '3Y': 'ret_3y',
        '5Y': 'ret_5y',
        '10Y': 'ret_10y'
    }
    
    horizon_quarters = {
        '1y': 4,
        '3y': 12,
        '5y': 20,
        '10y': 40
    }
    
    df = create_target_variables(df, horizon_quarters)
    X, feature_cols, classifications = prepare_features(df)
    
    # Add feature columns back to df
    for col in feature_cols:
        if col not in df.columns:
            df[col] = X[col]
    
    # (2) Compare models
    logger.info("\n[Step 2] Comparing models across horizons...")
    results_df = compare_models_across_horizons(
        df, feature_cols, horizons, test_size=args.test_size
    )
    
    # (3) Save results
    logger.info("\n[Step 3] Saving results...")
    results_path = results_dir / 'model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved: {results_path}")
    
    # (4) Identify champion models
    logger.info("\n[Step 4] Identifying champion models...")
    identify_champion_models(results_df)
    
    # (5) Create plots
    logger.info("\n[Step 5] Creating comparison plots...")
    plot_model_comparison(results_df, plots_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All steps completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

