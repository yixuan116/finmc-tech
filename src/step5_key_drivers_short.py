#!/usr/bin/env python3
"""
Step 5: Short-Horizon Key Drivers Analysis for NVDA Monthly Return Forecasting.

This script extracts key drivers from the Champion RandomForest model using:
- MDI (Mean Decrease in Impurity) importance
- Permutation importance (≥100 repeats)
- SHAP values (TreeExplainer)

Generates:
- Driver ranking tables (Top-K for each method)
- Aggregate ranking (z-score normalized, averaged)
- Driver persistence analysis (rolling/expanding window)
- Comprehensive visualizations (SHAP, PDP/ICE, dependence plots)
- README markdown summary
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/processed/NVDA_revenue_features.csv"
DEFAULT_OUTPUT_DIR = "results/step5"
DEFAULT_TOP_K = 20
DEFAULT_PERM_REPEATS = 200
DEFAULT_CV_FOLDS = 5
DEFAULT_SHAP_SAMPLE = 2000
RANDOM_STATE = 42

# Champion RF hyperparameters (from Step 4)
CHAMPION_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------


def load_data(data_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Load and preprocess data from CSV."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Auto-detect date column
    date_col = None
    for col in ["px_date", "date", "Date", "period_end", "timestamp"]:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("No date column found. Expected: px_date, date, period_end, or timestamp")
    
    # Parse dates and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    
    # Auto-detect target column
    target_col = None
    for col in ["future_12m_return", "return_next_month", "target", "y"]:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("No target column found. Expected: future_12m_return, return_next_month, target, or y")
    
    # Select numeric features only
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.ffill().bfill()
    
    # Drop identifier columns
    drop_cols = {"fy", "fp", "form", "tag_used", "ticker"}
    existing_drop = [col for col in drop_cols if col in numeric_df.columns]
    if existing_drop:
        numeric_df = numeric_df.drop(columns=existing_drop)
    
    # Drop data leakage features (future information)
    leakage_features = [
        col for col in numeric_df.columns
        if "future_12m" in col.lower() and col != target_col
    ]
    if leakage_features:
        print(f"⚠ Excluding data leakage features: {leakage_features}")
        numeric_df = numeric_df.drop(columns=leakage_features)
    
    # Extract target and features
    y = numeric_df[target_col].copy()
    X = numeric_df.drop(columns=[target_col])
    
    print(f"✓ Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"  Date range: {X.index.min()} to {X.index.max()}")
    print(f"  Target: {target_col}")
    
    return X, y, X.index


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-series split (same logic as Step 4)."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    
    idx = X.index
    train_mask = idx < train_end_ts + pd.Timedelta(days=1)
    val_mask = (idx >= train_end_ts + pd.Timedelta(days=1)) & (idx <= val_end_ts)
    test_mask = idx > val_end_ts
    
    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_val = y.loc[val_mask]
    y_test = y.loc[test_mask]
    
    print(f"\nTime-series split:")
    print(f"  Train: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
    print(f"  Val: {len(X_val)} samples ({X_val.index.min() if len(X_val) > 0 else 'N/A'} to {X_val.index.max() if len(X_val) > 0 else 'N/A'})")
    print(f"  Test: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")
    
    # Combine train + val for training (as in Step 4)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    return X_train_full, X_test, y_train_full, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_scaling: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """Scale features if needed (same as Step 4)."""
    if not use_scaling:
        return X_train, X_test, None
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    print("✓ Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------


def train_champion_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict] = None,
) -> RandomForestRegressor:
    """Train Champion RF with specified hyperparameters."""
    if params is None:
        params = CHAMPION_RF_PARAMS
    
    print(f"\nTraining Champion RF with params: {params}")
    model = RandomForestRegressor(**params)
    model.fit(X_train.values, y_train.values)
    print("✓ Model trained")
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X.values)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Importance Computation
# ---------------------------------------------------------------------------


def compute_mdi_importance(
    model: RandomForestRegressor,
    feature_names: List[str],
) -> pd.Series:
    """Compute MDI (Gini) importance."""
    mdi = pd.Series(model.feature_importances_, index=feature_names)
    return mdi.sort_values(ascending=False)


def compute_permutation_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 200,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series]:
    """Compute permutation importance on test set."""
    print(f"Computing permutation importance (n_repeats={n_repeats})...")
    perm_result = permutation_importance(
        model,
        X_test.values,
        y_test.values,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="r2",
    )
    
    perm_mean = pd.Series(perm_result.importances_mean, index=X_test.columns)
    perm_std = pd.Series(perm_result.importances_std, index=X_test.columns)
    
    return perm_mean.sort_values(ascending=False), perm_std


def compute_shap_values(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    max_samples: int = 2000,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, pd.Series]:
    """Compute SHAP values using TreeExplainer."""
    print(f"Computing SHAP values (max_samples={max_samples})...")
    
    # Sample if too large
    if len(X_test) > max_samples:
        X_sample = X_test.sample(max_samples, random_state=random_state)
    else:
        X_sample = X_test
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample.values)
    
    # Mean absolute SHAP
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)
    
    return shap_values, mean_abs_shap


# ---------------------------------------------------------------------------
# Driver Aggregation & Stability
# ---------------------------------------------------------------------------


def compute_aggregate_ranking(
    mdi: pd.Series,
    perm_mean: pd.Series,
    shap_mean: pd.Series,
) -> pd.DataFrame:
    """Compute aggregate ranking using z-score normalization."""
    # Get all features
    all_features = set(mdi.index) | set(perm_mean.index) | set(shap_mean.index)
    
    # Normalize each to z-score
    def zscore_normalize(series: pd.Series) -> pd.Series:
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mean_val) / std_val
    
    mdi_z = zscore_normalize(mdi)
    perm_z = zscore_normalize(perm_mean)
    shap_z = zscore_normalize(shap_mean)
    
    # Average z-scores
    aggregate_scores = []
    for feature in all_features:
        score = (
            mdi_z.get(feature, 0.0) +
            perm_z.get(feature, 0.0) +
            shap_z.get(feature, 0.0)
        ) / 3.0
        aggregate_scores.append({
            "feature": feature,
            "mdi_z": mdi_z.get(feature, 0.0),
            "perm_z": perm_z.get(feature, 0.0),
            "shap_z": shap_z.get(feature, 0.0),
            "aggregate_score": score,
        })
    
    df = pd.DataFrame(aggregate_scores)
    df = df.sort_values("aggregate_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    return df


def compute_driver_persistence(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    cv_folds: int = 5,
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Compute driver persistence using rolling/expanding window."""
    if params is None:
        params = CHAMPION_RF_PARAMS
    
    print(f"\nComputing driver persistence (cv_folds={cv_folds})...")
    
    n_samples = len(X_train)
    fold_size = n_samples // (cv_folds + 1)
    
    top10_counts = {feature: 0 for feature in feature_names}
    
    for fold in range(cv_folds):
        # Expanding window: use all data up to fold boundary
        train_end_idx = (fold + 1) * fold_size
        if train_end_idx >= n_samples:
            train_end_idx = n_samples - 1
        
        X_fold = X_train.iloc[:train_end_idx]
        y_fold = y_train.iloc[:train_end_idx]
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_fold.values, y_fold.values)
        
        # Get top 10 features
        mdi = pd.Series(model.feature_importances_, index=feature_names)
        top10 = mdi.nlargest(10).index.tolist()
        
        for feature in top10:
            top10_counts[feature] += 1
        
        print(f"  Fold {fold + 1}/{cv_folds}: {len(X_fold)} samples, top10 computed")
    
    # Create persistence table
    persistence_df = pd.DataFrame([
        {
            "feature": feature,
            "appearances_in_top10": count,
            "persistence_rate": count / cv_folds,
        }
        for feature, count in top10_counts.items()
    ])
    persistence_df = persistence_df.sort_values("appearances_in_top10", ascending=False)
    
    return persistence_df


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------


def plot_mdi_topk(mdi: pd.Series, output_path: Path, top_k: int = 20) -> None:
    """Plot MDI importance (Top-K)."""
    top_features = mdi.head(top_k)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features.values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel("MDI Importance", fontsize=12)
    ax.set_title(f"Top {top_k} Features by MDI Importance", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_perm_topk(
    perm_mean: pd.Series,
    perm_std: pd.Series,
    output_path: Path,
    top_k: int = 20,
) -> None:
    """Plot Permutation importance (Top-K)."""
    top_features = perm_mean.head(top_k)
    top_std = perm_std.loc[top_features.index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features.values, xerr=top_std.values, color='coral', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel("Permutation Importance (Mean ± Std)", fontsize=12)
    ax.set_title(f"Top {top_k} Features by Permutation Importance", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_shap_bar(shap_mean: pd.Series, output_path: Path, top_k: int = 20) -> None:
    """Plot SHAP importance (bar chart)."""
    top_features = shap_mean.head(top_k)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features.values, color='purple')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP|", fontsize=12)
    ax.set_title(f"Top {top_k} Features by SHAP Importance", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
) -> None:
    """Plot SHAP beeswarm plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample.values,
        feature_names=X_sample.columns,
        plot_type="dot",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot SHAP dependence plot for a single feature."""
    try:
        feature_idx = X_sample.columns.get_loc(feature_name)
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_sample.values,
            feature_names=X_sample.columns,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not generate SHAP dependence plot for {feature_name}: {e}")


def plot_pdp_ice(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot Partial Dependence Plot (PDP) and ICE curves."""
    try:
        # Get feature quantiles for axis range
        feature_values = X_test[feature_name].values
        q5, q95 = np.percentile(feature_values, [5, 95])
        
        # Check if feature has sufficient variation
        if np.std(feature_values) < 1e-10:
            print(f"⚠ Warning: Feature {feature_name} has very low variance, PDP may be flat")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use sklearn's PartialDependenceDisplay
        display = PartialDependenceDisplay.from_estimator(
            model,
            X_test.values,
            features=[feature_name],
            feature_names=X_test.columns.tolist(),
            kind='both',  # Both PDP and ICE
            ax=ax,
            ice_lines_kw={'alpha': 0.3, 'linewidth': 0.5},
            pd_line_kw={'color': 'red', 'linewidth': 2},
        )
        
        ax.set_title(f"PDP/ICE: {feature_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Partial Dependence", fontsize=12)
        ax.grid(alpha=0.3)
        
        # Set axis range to quantiles
        ax.set_xlim(q5, q95)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not generate PDP/ICE for {feature_name}: {e}")


# ---------------------------------------------------------------------------
# README Generation
# ---------------------------------------------------------------------------


def generate_readme(
    output_dir: Path,
    test_metrics: Dict[str, float],
    top10_drivers: pd.DataFrame,
    persistence_df: pd.DataFrame,
) -> None:
    """Generate README markdown summary."""
    readme_path = output_dir / "README_step5_short.md"
    
    content = f"""# Step 5: Short-Horizon Key Drivers Analysis

## Baseline Metrics (Test Set)

| Metric | Value |
|--------|-------|
| R² | {test_metrics['R2']:.4f} |
| RMSE | {test_metrics['RMSE']:.4f} |
| MAE | {test_metrics['MAE']:.4f} |

## Top 10 Key Drivers (Aggregate Ranking)

| Rank | Feature | Aggregate Score | MDI Z | Perm Z | SHAP Z |
|------|---------|-----------------|-------|--------|--------|
"""
    
    for idx, row in top10_drivers.head(10).iterrows():
        content += f"| {int(row['rank'])} | {row['feature']} | {row['aggregate_score']:.4f} | {row['mdi_z']:.4f} | {row['perm_z']:.4f} | {row['shap_z']:.4f} |\n"
    
    content += f"""

## Driver Persistence (Top 10 Most Stable)

| Feature | Appearances in Top-10 | Persistence Rate |
|---------|----------------------|------------------|
"""
    
    for idx, row in persistence_df.head(10).iterrows():
        content += f"| {row['feature']} | {int(row['appearances_in_top10'])} | {row['persistence_rate']:.2%} |\n"
    
    content += """

## Key Interpretations

"""
    
    # Generate interpretations based on top drivers
    top3_features = top10_drivers.head(3)['feature'].tolist()
    
    # Check for TNX
    tnx_features = [f for f in top10_drivers['feature'] if 'tnx' in f.lower()]
    if tnx_features:
        content += f"- **TNX (Interest Rate) Effects**: {', '.join(tnx_features[:2])} appear in top drivers. Higher rates compress NVDA valuation, consistent with discount rate effects.\n\n"
    
    # Check for VIX interactions
    vix_interactions = [f for f in top10_drivers['feature'] if 'vix' in f.lower() and 'ix_' in f]
    if vix_interactions:
        content += f"- **VIX × Price/Revenue Interactions**: {', '.join(vix_interactions[:2])} dominate short-horizon variance. Market volatility interacts with firm fundamentals in non-linear ways.\n\n"
    
    # Check for revenue features
    revenue_features = [f for f in top10_drivers['feature'] if 'rev' in f.lower()]
    if revenue_features:
        content += f"- **Revenue Signals**: {', '.join(revenue_features[:2])} contribute to predictions, though with longer lag than price momentum.\n\n"
    
    content += """
## Output Files

- `mdi_importance.csv`: MDI importance for all features
- `permutation_importance.csv`: Permutation importance (mean ± std)
- `shap_importance.csv`: Mean absolute SHAP values
- `key_drivers_top10.csv`: Aggregate ranking (Top 10)
- `driver_persistence.csv`: Persistence analysis across CV folds
- `mdi_top20.png`: MDI importance visualization
- `perm_top20.png`: Permutation importance visualization
- `shap_bar.png`: SHAP importance (bar chart)
- `shap_beeswarm.png`: SHAP beeswarm plot
- `shap_dependence_<feature>.png`: SHAP dependence plots for top 3 drivers
- `pdp_<feature>.png`: PDP/ICE plots for top 5 drivers
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Saved: {readme_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Short-Horizon Key Drivers Analysis"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to data CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-K drivers to analyze (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--perm_repeats",
        type=int,
        default=DEFAULT_PERM_REPEATS,
        help=f"Permutation importance repeats (default: {DEFAULT_PERM_REPEATS})",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help=f"CV folds for persistence analysis (default: {DEFAULT_CV_FOLDS})",
    )
    parser.add_argument(
        "--shap_sample",
        type=int,
        default=DEFAULT_SHAP_SAMPLE,
        help=f"Max samples for SHAP (default: {DEFAULT_SHAP_SAMPLE})",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_path = Path(args.data)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 5: SHORT-HORIZON KEY DRIVERS ANALYSIS")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    X, y, date_index = load_data(data_path)
    
    # Time-series split
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    
    # Scale features (same as Step 4)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, use_scaling=True)
    
    # Train Champion RF
    model = train_champion_rf(X_train_scaled, y_train)
    
    # Evaluate baseline metrics
    test_metrics = evaluate_model(model, X_test_scaled, y_test)
    print(f"\nBaseline Metrics (Test Set):")
    print(f"  R²: {test_metrics['R2']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE: {test_metrics['MAE']:.4f}")
    
    # Compute importance metrics
    print("\n" + "-" * 80)
    print("COMPUTING IMPORTANCE METRICS")
    print("-" * 80)
    
    feature_names = X_train_scaled.columns.tolist()
    
    # MDI importance
    print("\n1. Computing MDI importance...")
    mdi = compute_mdi_importance(model, feature_names)
    mdi_df = pd.DataFrame({
        "feature": mdi.index,
        "mdi_importance": mdi.values,
    })
    mdi_df.to_csv(output_dir / "mdi_importance.csv", index=False)
    print(f"✓ Saved: {output_dir / 'mdi_importance.csv'}")
    
    # Permutation importance
    print("\n2. Computing permutation importance...")
    perm_mean, perm_std = compute_permutation_importance(
        model,
        X_test_scaled,
        y_test,
        n_repeats=args.perm_repeats,
    )
    perm_df = pd.DataFrame({
        "feature": perm_mean.index,
        "perm_importance_mean": perm_mean.values,
        "perm_importance_std": perm_std.values,
    })
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    print(f"✓ Saved: {output_dir / 'permutation_importance.csv'}")
    
    # SHAP values
    print("\n3. Computing SHAP values...")
    shap_values, shap_mean = compute_shap_values(
        model,
        X_test_scaled,
        max_samples=args.shap_sample,
    )
    shap_df = pd.DataFrame({
        "feature": shap_mean.index,
        "mean_abs_shap": shap_mean.values,
    })
    shap_df.to_csv(output_dir / "shap_importance.csv", index=False)
    print(f"✓ Saved: {output_dir / 'shap_importance.csv'}")
    
    # Aggregate ranking
    print("\n4. Computing aggregate ranking...")
    aggregate_df = compute_aggregate_ranking(mdi, perm_mean, shap_mean)
    top10_df = aggregate_df.head(10)
    top10_df.to_csv(output_dir / "key_drivers_top10.csv", index=False)
    print(f"✓ Saved: {output_dir / 'key_drivers_top10.csv'}")
    print("\nTop 10 Key Drivers:")
    for idx, row in top10_df.iterrows():
        print(f"  {int(row['rank']):2d}. {row['feature']:40s} (score: {row['aggregate_score']:.4f})")
    
    # Driver persistence
    print("\n5. Computing driver persistence...")
    persistence_df = compute_driver_persistence(
        X_train_scaled,
        y_train,
        feature_names,
        cv_folds=args.cv_folds,
    )
    persistence_df.to_csv(output_dir / "driver_persistence.csv", index=False)
    print(f"✓ Saved: {output_dir / 'driver_persistence.csv'}")
    
    # Generate visualizations
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    # MDI plot
    plot_mdi_topk(mdi, output_dir / "mdi_top20.png", top_k=args.topk)
    
    # Permutation plot
    plot_perm_topk(perm_mean, perm_std, output_dir / "perm_top20.png", top_k=args.topk)
    
    # SHAP plots
    plot_shap_bar(shap_mean, output_dir / "shap_bar.png", top_k=args.topk)
    
    if len(X_test_scaled) > args.shap_sample:
        X_sample = X_test_scaled.sample(args.shap_sample, random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model)
        shap_values_sample = explainer.shap_values(X_sample.values)
    else:
        X_sample = X_test_scaled
        shap_values_sample = shap_values
    
    plot_shap_beeswarm(shap_values_sample, X_sample, output_dir / "shap_beeswarm.png", max_display=args.topk)
    
    # SHAP dependence plots for top 3 drivers
    print("\n6. Generating SHAP dependence plots for top 3 drivers...")
    top3_features = top10_df.head(3)['feature'].tolist()
    for feature in top3_features:
        if feature in X_sample.columns:
            plot_shap_dependence(
                shap_values_sample,
                X_sample,
                feature,
                output_dir / f"shap_dependence_{feature.replace('/', '_').replace(' ', '_')}.png",
            )
    
    # PDP/ICE plots for top 5 drivers
    print("\n7. Generating PDP/ICE plots for top 5 drivers...")
    top5_features = top10_df.head(5)['feature'].tolist()
    # Also include TNX features if not already in top 5
    tnx_features = [f for f in feature_names if 'tnx' in f.lower() and f not in top5_features]
    if tnx_features:
        top5_features.extend(tnx_features[:2])  # Add up to 2 TNX features
    
    for feature in top5_features[:5]:  # Limit to 5 plots
        if feature in X_test_scaled.columns:
            plot_pdp_ice(
                model,
                X_test_scaled,
                feature,
                output_dir / f"pdp_{feature.replace('/', '_').replace(' ', '_')}.png",
            )
    
    # Generate README
    print("\n8. Generating README...")
    generate_readme(output_dir, test_metrics, top10_df, persistence_df)
    
    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - Tables: 5 CSV files")
    print(f"  - Plots: {len(list(output_dir.glob('*.png')))} PNG files")
    print(f"  - Summary: README_step5_short.md")
    print("=" * 80)


if __name__ == "__main__":
    main()

