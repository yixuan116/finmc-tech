"""
Random Forest model for return prediction.

Provides baseline RF model with fit, evaluate, save/load, and visualization.
"""

import sys
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from finmc_tech.config import Settings, cfg, get_logger

logger = get_logger(__name__)


def fit_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: Optional[int] = None,
) -> RandomForestRegressor:
    """
    Fit Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed. If None, uses cfg.RANDOM_STATE
    
    Returns:
        Fitted RandomForestRegressor model
    """
    if random_state is None:
        random_state = cfg.RANDOM_STATE
    
    logger.info("Fitting Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=random_state,
    )
    
    model.fit(X_train, y_train)
    
    logger.info(f"  ✓ Fitted RF model with {len(X_train)} samples")
    
    return model


def evaluate_rf(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Evaluate Random Forest model on test set.
    
    Args:
        model: Fitted RandomForestRegressor
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dict with metrics (R2, MAE, RMSE) and y_pred as pd.Series with same index
    """
    logger.info("Evaluating Random Forest model...")
    
    y_pred = model.predict(X_test)
    
    # Convert to Series with same index as y_test
    y_pred_series = pd.Series(y_pred, index=y_test.index, name="y_pred")
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "y_pred": y_pred_series,
    }
    
    logger.info(f"  Test R²: {metrics['R2']:.4f}")
    logger.info(f"  Test RMSE: {metrics['RMSE']:.4f}")
    logger.info(f"  Test MAE: {metrics['MAE']:.4f}")
    
    return metrics


def save_model(
    model: RandomForestRegressor,
    path: Path,
) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Fitted RandomForestRegressor
        path: Path to save model file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, path)
    logger.info(f"  ✓ Saved model to {path}")


def load_model(
    path: Path,
) -> RandomForestRegressor:
    """
    Load model using joblib.
    
    Args:
        path: Path to model file
    
    Returns:
        Loaded RandomForestRegressor model
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    model = joblib.load(path)
    logger.info(f"  ✓ Loaded model from {path}")
    
    return model


def plot_feature_importance(
    model: RandomForestRegressor,
    feature_names: List[str],
    outpath: Path,
    top_n: Optional[int] = None,
) -> None:
    """
    Plot feature importance as bar chart.
    
    Args:
        model: Fitted RandomForestRegressor
        feature_names: List of feature names
        outpath: Path to save plot
        top_n: Number of top features to show. If None, shows all
    """
    if len(feature_names) != len(model.feature_importances_):
        raise ValueError(
            f"Feature names length ({len(feature_names)}) must match "
            f"feature importances length ({len(model.feature_importances_)})"
        )
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    # Select top N if specified
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
    
    ax.barh(
        range(len(importance_df)),
        importance_df["importance"],
        align="center",
    )
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"  ✓ Saved feature importance plot to {outpath}")


def train_rf_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Settings,
    feature_names: Optional[List[str]] = None,
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Train Random Forest model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    config : Config
        Configuration object
    feature_names : Optional[List[str]]
        Feature names for importance analysis
    
    Returns
    -------
    Tuple[RandomForestRegressor, Dict]
        Trained model and training metrics
    """
    print("Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    
    # Training metrics
    y_pred_train = model.predict(X_train)
    train_metrics = {
        "r2": r2_score(y_train, y_pred_train),
        "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "mae": mean_absolute_error(y_train, y_pred_train),
    }
    
    # Feature importance
    if feature_names:
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        train_metrics["feature_importance"] = feature_importance
    
    print(f"  ✓ Training R²: {train_metrics['r2']:.4f}")
    print(f"  ✓ Training RMSE: {train_metrics['rmse']:.4f}")
    
    return model, train_metrics


def evaluate_rf_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Evaluate Random Forest model on test set (backward compatibility).
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns
    -------
    Dict
        Test metrics
    """
    # Use new evaluate_rf function
    metrics = evaluate_rf(model, X_test, y_test)
    
    # Convert to old format for backward compatibility
    return {
        "r2": metrics["R2"],
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "predictions": {
            "y_test": y_test.tolist(),
            "y_pred": metrics["y_pred"].tolist(),
        },
    }

