"""
Unified Model Evaluation: Publication-Grade Protocol (Gu, Kelly, Xiu 2020)

This script implements a rigorous Train/Validation/Test protocol:
1. TRAIN (<= 2020): Parameter estimation
2. VALIDATION (2021-2022): Hyperparameter tuning & Champion Model Selection
3. TEST (>= 2023): Ex post performance reporting (not used for selection)

Metrics:
- R²_OOS: Computed relative to the TRAINING set mean (benchmark).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.three_category_feature_importance import (
    load_features, create_target_variables, prepare_features
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================
# Code Task 2: Define Out-of-Sample R²
# =============================================

def calculate_oos_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Compute Out-of-Sample R² relative to the Training Mean benchmark.
    
    R²_OOS = 1 - (SSE / SST)
    where:
      SSE = sum((y_true - y_pred)²)
      SST = sum((y_true - mean(y_train))²)
      
    This penalizes models that perform worse than the historical historical mean.
    """
    if len(y_true) == 0:
        return np.nan
        
    y_train_mean = np.mean(y_train)
    sse = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - y_train_mean)**2)
    
    if sst == 0:
        return 0.0
        
    return 1 - (sse / sst)


# =============================================
# (1) Data Loading & Preparation
# =============================================

def load_and_prepare_data(
    csv_path: str,
    date_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load features and prepare target variables."""
    logger.info(f"Loading features from {csv_path}")
    df = load_features(Path(csv_path))
    
    # Create target variables
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
    
    # Auto-detect date column
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
        # Code Task 1 Requirement: Must be DatetimeIndex
        raise ValueError("Publication protocol requires a strict DatetimeIndex for temporal splitting.")
    
    return df, X, feature_cols


# =============================================
# Code Task 1: Time-Based Split Refactor
# =============================================

def time_based_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Implement strict calendar-based Train/Validation/Test split.
    
    Windows:
    - TRAIN:      <= 2020-12-31
    - VALIDATION: 2021-01-01 to 2022-12-31
    - TEST:       >= 2023-01-01
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # Define fixed windows
    TRAIN_END = pd.Timestamp("2020-12-31")
    VAL_START = pd.Timestamp("2021-01-01")
    VAL_END = pd.Timestamp("2022-12-31")
    TEST_START = pd.Timestamp("2023-01-01")

    # Base masks
    train_mask = df.index <= TRAIN_END
    val_mask = (df.index >= VAL_START) & (df.index <= VAL_END)
    test_mask = df.index >= TEST_START

    # Apply target availability filter
    # (Horizon dependence comes ONLY from whether the target exists)
    has_target = df[target_col].notna()
    
    train_df = df[train_mask & has_target].copy()
    val_df = df[val_mask & has_target].copy()
    test_df = df[test_mask & has_target].copy()

    # Logging split info
    logger.info(f"  TRAIN : {len(train_df)} samples "
                f"({train_df.index.min().date()} to {train_df.index.max().date()})")
    
    if len(val_df) > 0:
        logger.info(f"  VAL   : {len(val_df)} samples "
                    f"({val_df.index.min().date()} to {val_df.index.max().date()})")
    else:
        logger.warning("  VAL   : 0 samples (Targets not available for this horizon in 2021-2022)")

    if len(test_df) > 0:
        logger.info(f"  TEST  : {len(test_df)} samples "
                    f"({test_df.index.min().date()} to {test_df.index.max().date()})")
    else:
        logger.warning("  TEST  : 0 samples (Targets not available for this horizon >= 2023)")

    # Separate X and y
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols] if not val_df.empty else pd.DataFrame(columns=feature_cols)
    y_val = val_df[target_col] if not val_df.empty else pd.Series(dtype=float)
    
    X_test = test_df[feature_cols] if not test_df.empty else pd.DataFrame(columns=feature_cols)
    y_test = test_df[target_col] if not test_df.empty else pd.Series(dtype=float)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================
# (3) Model Definitions
# =============================================

def get_models():
    """Return models with standard configurations."""
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
        'RandomForest': RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500,
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
# Code Task 3: Dual-Split Evaluation
# =============================================

def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = None
) -> Dict[str, float]:
    """
    Train on TRAIN only.
    Evaluate separately on VALIDATION and TEST.
    Return metrics for both.
    """
    # 1. Scaling (fit on train, transform all)
    if model_name == 'NeuralNetwork':
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_val_use = scaler.transform(X_val) if len(X_val) > 0 else X_val
        X_test_use = scaler.transform(X_test) if len(X_test) > 0 else X_test
    else:
        X_train_use = X_train
        X_val_use = X_val
        X_test_use = X_test

    # 2. Train (ONLY on training set)
    model.fit(X_train_use, y_train)

    metrics = {}

    # 3. Validation Metrics
    if len(X_val_use) > 0:
        y_val_pred = model.predict(X_val_use)
        metrics['mae_val'] = mean_absolute_error(y_val, y_val_pred)
        metrics['rmse_val'] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        metrics['r2_oos_val'] = calculate_oos_r2(y_val.values, y_val_pred, y_train.values)
    else:
        metrics['mae_val'] = np.nan
        metrics['rmse_val'] = np.nan
        metrics['r2_oos_val'] = np.nan

    # 4. Test Metrics
    if len(X_test_use) > 0:
        y_test_pred = model.predict(X_test_use)
        metrics['mae_test'] = mean_absolute_error(y_test, y_test_pred)
        metrics['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics['r2_oos_test'] = calculate_oos_r2(y_test.values, y_test_pred, y_train.values)
    else:
        metrics['mae_test'] = np.nan
        metrics['rmse_test'] = np.nan
        metrics['r2_oos_test'] = np.nan

    return metrics


# =============================================
# Code Task 4: Store Metrics and Sample Sizes
# =============================================

def evaluate_all_models_across_horizons(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Dict[str, str]
) -> pd.DataFrame:
    """Evaluate all models across all horizons with split tracking."""
    logger.info("=" * 80)
    logger.info("Unified Model Evaluation (Train/Val/Test Protocol)")
    logger.info("=" * 80)
    
    models = get_models()
    results = []
    
    for horizon_name, target_col in horizons.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Horizon: {horizon_name} (target: {target_col})")
        logger.info(f"{'='*80}")
        
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not found, skipping.")
            continue
        
        try:
            # 1. Split
            X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
                df, feature_cols, target_col
            )
            
            # Fill NaN (Simple median fill based on train)
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            medians = X_train[numeric_cols].median()
            
            X_train = X_train.fillna(medians)
            if len(X_val) > 0: X_val = X_val.fillna(medians)
            if len(X_test) > 0: X_test = X_test.fillna(medians)
            
            # 2. Evaluate Models
            for model_name, model in models.items():
                logger.info(f"  Training {model_name}...")
                
                try:
                    m = evaluate_model(
                        model, X_train, y_train, X_val, y_val, X_test, y_test, model_name
                    )
                    
                    results.append({
                        'model': model_name,
                        'horizon': horizon_name,
                        # Validation (Selection)
                        'mae_val': m['mae_val'],
                        'rmse_val': m['rmse_val'],
                        'r2_oos_val': m['r2_oos_val'],
                        # Test (Reporting)
                        'mae_test': m['mae_test'],
                        'rmse_test': m['rmse_test'],
                        'r2_oos_test': m['r2_oos_test'],
                        # Metadata
                        'n_train': len(y_train),
                        'n_val': len(y_val),
                        'n_test': len(y_test)
                    })
                    
                    logger.info(f"    VAL  R²_OOS: {m['r2_oos_val']:.4f} | MAE: {m['mae_val']:.4f}")
                    if not np.isnan(m['r2_oos_test']):
                        logger.info(f"    TEST R²_OOS: {m['r2_oos_test']:.4f} | MAE: {m['mae_test']:.4f}")
                    
                except Exception as e:
                    logger.error(f"    Error training {model_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {horizon_name}: {e}")
            continue
    
    return pd.DataFrame(results)


# =============================================
# Code Task 5: Champion Selection (Validation Only)
# =============================================

def identify_champion_models(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select champions based ONLY on Validation R²_OOS.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Champion Model Identification")
    logger.info("(Selected on Validation R²_OOS; Evaluated on Test)")
    logger.info("=" * 80)
    
    horizons = sorted(results_df['horizon'].unique())
    champion_rows = []
    
    for horizon in horizons:
        horizon_df = results_df[results_df['horizon'] == horizon].copy()
        if len(horizon_df) == 0: continue
        
        # Filter out models with NaN validation metrics
        valid_df = horizon_df.dropna(subset=['r2_oos_val'])
        
        if len(valid_df) == 0:
            logger.warning(f"No valid validation results for {horizon}")
            continue

        # Sort by Validation R2 (descending)
        valid_df = valid_df.sort_values(['r2_oos_val', 'mae_val'], ascending=[False, True])
        
        champion = valid_df.iloc[0]
        champion_rows.append(champion)
        
        logger.info(f"\n{horizon}:")
        logger.info(f"  🏆 Champion: {champion['model']:15s}")
        logger.info(f"     VAL  R²_OOS: {champion['r2_oos_val']:8.4f} (Selection Metric)")
        logger.info(f"     TEST R²_OOS: {champion['r2_oos_test']:8.4f} (Ex Post Eval)")
        logger.info(f"     Samples: Train={champion['n_train']}, Val={champion['n_val']}, Test={champion['n_test']}")
        
        logger.info(f"\n  Leaderboard (Validation R²):")
        for _, row in valid_df.iterrows():
            mark = "🏆" if row['model'] == champion['model'] else "  "
            logger.info(f"    {mark} {row['model']:15s} | Val R²: {row['r2_oos_val']:8.4f} | Test R²: {row['r2_oos_test']:8.4f}")

    return pd.DataFrame(champion_rows)


# =============================================
# Code Task 7: Figures (Validation Only)
# =============================================

def plot_unified_results(results_df: pd.DataFrame, output_dir: Path):
    """
    Create visualization of Validation results.
    Does NOT plot Test metrics to avoid confusion.
    """
    logger.info("\nGenerating validation plots...")
    
    # Filter for Plot A (Nonlinear/Advanced)
    main_models = ['RandomForest', 'XGBoost', 'NeuralNetwork']
    
    # 1. Main Plot: Validation R2 (Nonlinear Models)
    plt.figure(figsize=(10, 6))
    subset = results_df[results_df['model'].isin(main_models)].copy()
    
    if not subset.empty:
        sns.barplot(data=subset, x='horizon', y='r2_oos_val', hue='model', palette='viridis')
        plt.title('Validation R² (Out-of-Sample) by Horizon\n(Nonlinear Models)', fontsize=14, fontweight='bold')
        plt.ylabel('Validation R²_OOS', fontsize=12)
        plt.xlabel('Horizon', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        
        path = output_dir / 'unified_val_r2_oos_nonlinear.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {path}")

    # 2. Appendix Plot: Validation R2 (All Models, Clipped)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=results_df, x='horizon', y='r2_oos_val', hue='model', palette='tab10')
    
    # Dynamic clipping for readability if linear models explode
    y_min = results_df['r2_oos_val'].min()
    if y_min < -5:
        plt.ylim(max(y_min, -5.0), 1.0) # Clip at -5
        plt.title('Validation R² (Out-of-Sample) - All Models\n(Clipped at -5.0)', fontsize=14)
    else:
        plt.title('Validation R² (Out-of-Sample) - All Models', fontsize=14)
        
    plt.ylabel('Validation R²_OOS', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    path = output_dir / 'unified_val_r2_oos_all_clipped.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Unified model evaluation protocol')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to extended features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Directories
    output_dir = Path(args.output_dir)
    results_dir = output_dir / 'results'
    plots_dir = output_dir / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load
    df, X, feature_cols = load_and_prepare_data(args.features_csv)
    
    horizons = {
        '1Y': 'ret_1y',
        '3Y': 'ret_3y',
        '5Y': 'ret_5y',
        '10Y': 'ret_10y'
    }
    
    # 2. Evaluate
    results_df = evaluate_all_models_across_horizons(df, feature_cols, horizons)
    
    # 3. Save Raw Results
    results_df.to_csv(results_dir / 'unified_model_comparison_raw.csv', index=False)
    
    # 4. Identify Champions
    champion_df = identify_champion_models(results_df)
    
    # 5. Create Publication Tables (Code Task 6)
    
    # Table A: Validation R2 Matrix (Evidence)
    val_r2_matrix = results_df.pivot(index='model', columns='horizon', values='r2_oos_val')
    # Reorder columns
    cols = [c for c in ['1Y', '3Y', '5Y', '10Y'] if c in val_r2_matrix.columns]
    val_r2_matrix = val_r2_matrix[cols]
    val_r2_matrix.to_csv(results_dir / 'table_val_r2_oos_matrix.csv')
    
    # Table B: Validation MAE Matrix
    val_mae_matrix = results_df.pivot(index='model', columns='horizon', values='mae_val')
    val_mae_matrix = val_mae_matrix[cols]
    val_mae_matrix.to_csv(results_dir / 'table_val_mae_matrix.csv')
    
    # Table C: Champion Summary (Test Evaluation)
    cols_summary = ['horizon', 'model', 'r2_oos_val', 'r2_oos_test', 'n_train', 'n_val', 'n_test']
    summary_export = champion_df[cols_summary].copy()
    summary_export.columns = ['Horizon', 'Champion_Model', 'Val_R2', 'Test_R2', 'N_Train', 'N_Val', 'N_Test']
    summary_export.to_csv(results_dir / 'table_champion_summary.csv', index=False)
    
    logger.info("\n" + "="*80)
    logger.info("FINAL CHAMPION SUMMARY (Selected on Validation)")
    logger.info("="*80)
    print(summary_export.to_string(index=False))
    
    # 6. Plots
    plot_unified_results(results_df, plots_dir)
    
    logger.info(f"\nOutputs saved to {output_dir}")

if __name__ == '__main__':
    main()
