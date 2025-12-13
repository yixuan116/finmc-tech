"""
AMD Robustness Check Pipeline
----------------------------
Purpose: Verify model robustness on AMD data (Cross-firm validation).
Note: AMD results are NOT used for champion selection (only NVDA is).

Usage:
    python -m scripts.robustness_check_amd --features-csv data/processed/amd_features_extended_v2.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.three_category_feature_importance import (
        load_features, create_target_variables, prepare_features
    )
except ImportError:
    # Fallback if running from scripts directory
    sys.path.insert(0, str(Path(__file__).parent))
    from three_category_feature_importance import (
        load_features, create_target_variables, prepare_features
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================
# 1. Configuration & Models
# =============================================

def get_models() -> Dict[str, Any]:
    """Return model configurations (Identical to unified_model_evaluation.py)."""
    return {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000),
        'RandomForest': RandomForestRegressor(
            n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            tree_method="hist", objective="reg:squarederror", random_state=42
        ),
        'NeuralNetwork': MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            alpha=0.01, learning_rate='adaptive', max_iter=500,
            random_state=42, early_stopping=True, validation_fraction=0.1
        ),
    }

def get_split_dates(horizon: str) -> Dict[str, str]:
    """
    Define temporal splits per horizon to ensure validation targets exist.
    
    Logic:
    - 1Y: Standard (Val 2021-2022)
    - 3Y: Shift back 2 years (Val 2019-2020)
    - 5Y: Shift back 4 years (Val 2017-2018)
    - 10Y: Shift back 8 years (Val 2013-2014)
    """
    splits = {
        '1Y': {
            'train_end': '2020-12-31',
            'val_start': '2021-01-01', 'val_end': '2022-12-31',
            'test_start': '2023-01-01'
        },
        '3Y': {
            'train_end': '2018-12-31',
            'val_start': '2019-01-01', 'val_end': '2020-12-31',
            'test_start': '2021-01-01'
        },
        '5Y': {
            'train_end': '2016-12-31',
            'val_start': '2017-01-01', 'val_end': '2018-12-31',
            'test_start': '2019-01-01'
        },
        '10Y': {
            'train_end': '2012-12-31',
            'val_start': '2013-01-01', 'val_end': '2014-12-31',
            'test_start': '2015-01-01'
        }
    }
    return splits.get(horizon)

# =============================================
# 2. Evaluation Logic
# =============================================

def train_and_validate(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    target_col: str, 
    horizon: str
) -> Dict[str, Dict[str, float]]:
    """Train on Train set, Evaluate on Validation set."""
    
    dates = get_split_dates(horizon)
    if not dates:
        logger.warning(f"Unknown horizon {horizon}")
        return {}

    # 1. Split Data
    train_mask = df.index <= dates['train_end']
    val_mask = (df.index >= dates['val_start']) & (df.index <= dates['val_end'])
    
    # Ensure target availability
    has_target = df[target_col].notna()
    
    X_train = df.loc[train_mask & has_target, feature_cols]
    y_train = df.loc[train_mask & has_target, target_col]
    
    X_val = df.loc[val_mask & has_target, feature_cols]
    y_val = df.loc[val_mask & has_target, target_col]
    
    # Impute missing values (median from train)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    if len(X_val) > 0:
        X_val = X_val.fillna(medians)
    
    logger.info(f"  Horizon {horizon}: Train={len(X_train)}, Val={len(X_val)}")
    
    if len(X_val) == 0:
        logger.warning(f"  No validation samples for {horizon}")
        return {}

    results = {}
    models = get_models()
    
    for name, model in models.items():
        try:
            # Scale for NN
            if name == 'NeuralNetwork':
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train)
                X_val_fold = scaler.transform(X_val)
            else:
                X_train_fold = X_train
                X_val_fold = X_val
                
            # Train
            model.fit(X_train_fold, y_train)
            
            # Validate
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {
                'r2': r2,
                'n_val': len(y_val)
            }
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            results[name] = {'r2': np.nan, 'n_val': 0}
            
    return results

# =============================================
# 3. Main Pipeline
# =============================================

def main():
    parser = argparse.ArgumentParser(description='AMD Robustness Check')
    parser.add_argument('--features-csv', type=str, 
                        default='data/processed/amd_features_extended_v2.csv',
                        help='Path to AMD features CSV')
    args = parser.parse_args()
    
    # Setup Paths
    output_dir = Path("outputs/robustness/amd")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    if not Path(args.features_csv).exists():
        logger.error(f"File not found: {args.features_csv}")
        # Try fallback
        fallback = 'data/processed/amd_features_extended.csv'
        if Path(fallback).exists():
            logger.info(f"Using fallback: {fallback}")
            args.features_csv = fallback
        else:
            return

    df = load_features(Path(args.features_csv))
    
    # Create Targets
    horizons = {'1Y': 4, '3Y': 12, '5Y': 20, '10Y': 40}
    df = create_target_variables(df, horizons)
    
    # Prepare Features
    X_clean, feature_cols, _ = prepare_features(df)
    
    # Re-attach features/targets to main df with index
    # (Align index)
    common_idx = df.index.intersection(X_clean.index)
    df = df.loc[common_idx]
    for col in feature_cols:
        df[col] = X_clean.loc[common_idx, col]

    # 2. Run Robustness Check
    all_results = {}
    
    for h_name, h_quarters in horizons.items():
        target = f'ret_{h_name}'
        if target not in df.columns: continue
        
        logger.info(f"Processing {h_name}...")
        res = train_and_validate(df, feature_cols, target, h_name)
        all_results[h_name] = res
        
    # 3. Format Outputs
    # R2 Matrix
    models = get_models().keys()
    r2_matrix = pd.DataFrame(index=list(models), columns=horizons.keys())
    n_matrix = pd.DataFrame(index=list(models), columns=horizons.keys())
    
    for h in horizons.keys():
        if h in all_results:
            for m in models:
                if m in all_results[h]:
                    r2_matrix.loc[m, h] = all_results[h][m]['r2']
                    n_matrix.loc[m, h] = all_results[h][m]['n_val']
    
    # Save CSVs
    r2_path = output_dir / "validation_r2_matrix_amd.csv"
    n_path = output_dir / "validation_n_matrix_amd.csv"
    
    r2_matrix.to_csv(r2_path)
    n_matrix.to_csv(n_path)
    
    logger.info(f"Saved R2 matrix: {r2_path}")
    logger.info(f"Saved N matrix: {n_path}")
    
    # Save Markdown Table
    md_path = output_dir / "table_4_2_amd_validation_r2.md"
    with open(md_path, "w") as f:
        f.write("### Table 4.2: AMD Robustness Check (Validation R²)\n\n")
        f.write(r2_matrix.to_markdown())
        f.write("\n\n*Note: Validation periods vary by horizon to ensure data availability.*\n")
    
    logger.info(f"Saved Markdown table: {md_path}")
    
    # Print Result
    print("\nAMD Robustness Check Results (Validation R²):")
    print(r2_matrix.to_string())

if __name__ == "__main__":
    main()
