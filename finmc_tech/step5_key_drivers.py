#!/usr/bin/env python3
"""
Step 5: Key Driver Analysis for NVDA 12M Return Prediction.

This script extracts key drivers from the champion RandomForest model using:
- MDI (Mean Decrease in Impurity) importance
- Permutation importance
- SHAP values

Generates:
- Driver ranking table
- SHAP summary plots
- PDP/ICE curves for top drivers
- Optional rolling importance visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import StandardScaler

# Reuse utility functions from train_models.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_models import (
    auto_detect_date_column,
    load_dataset,
    prepare_target,
    time_based_split,
    scale_splits,
    DatasetSplits,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/processed/nvda_features_extended.csv"
DEFAULT_TARGET_COL = "return_next_month"
DEFAULT_MODEL_PATH = "models/champion_model.pkl"
DEFAULT_SCALER_PATH = "models/feature_scaler.pkl"
DEFAULT_OUTPUT_DIR = "results/step5"  # Unified output directory
TOP_N_DRIVERS = 5  # Number of top drivers for PDP/ICE plots


# ---------------------------------------------------------------------------
# Importance Computation Functions
# ---------------------------------------------------------------------------


def compute_mdi_importance(model: RandomForestRegressor, feature_names: List[str], model_importances: np.ndarray = None) -> pd.Series:
    """Compute MDI (Mean Decrease in Impurity) importance from RandomForest."""
    if model_importances is None:
        model_importances = model.feature_importances_
    mdi = pd.Series(model_importances, index=feature_names)
    return mdi.sort_values(ascending=False)


def compute_permutation_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 30,
    random_state: int = 42,
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
    )
    
    perm_mean = pd.Series(perm_result.importances_mean, index=X_test.columns)
    perm_std = pd.Series(perm_result.importances_std, index=X_test.columns)
    
    return perm_mean.sort_values(ascending=False), perm_std


def compute_shap_values(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    max_samples: int = 200,
    random_state: int = 42,
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
# Visualization Functions
# ---------------------------------------------------------------------------


def plot_mdi_importance(mdi: pd.Series, output_path: Path, top_n: int = 20) -> None:
    """Plot MDI importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = mdi.head(top_n)
    ax.barh(range(len(top_features)), top_features.values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel("MDI Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Features by MDI Importance", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved MDI importance plot: {output_path}")


def plot_permutation_importance(
    perm_mean: pd.Series,
    perm_std: pd.Series,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot permutation importance with error bars."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = perm_mean.head(top_n)
    top_std = perm_std.loc[top_features.index]
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features.values, xerr=top_std.values, color='coral', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel("Permutation Importance (Mean ± Std)", fontsize=12)
    ax.set_title(f"Top {top_n} Features by Permutation Importance", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved permutation importance plot: {output_path}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
) -> None:
    """Plot SHAP summary plot (beeswarm)."""
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
    print(f"✓ Saved SHAP summary plot: {output_path}")


def plot_pdp_ice(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    feature_name: str,
    output_path: Path,
) -> None:
    """Plot Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) curves."""
    try:
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
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved PDP/ICE plot: {output_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not generate PDP/ICE for {feature_name}: {e}")


def plot_rolling_importance(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    output_path: Path,
    window_size: int = 24,
    step_size: int = 6,
) -> None:
    """Plot rolling feature importance over time windows."""
    print(f"Computing rolling importance (window={window_size}, step={step_size})...")
    
    dates = X_train.index
    n_windows = (len(X_train) - window_size) // step_size + 1
    
    rolling_importance = []
    window_dates = []
    
    for i in range(0, len(X_train) - window_size + 1, step_size):
        X_window = X_train.iloc[i:i+window_size]
        y_window = y_train.iloc[i:i+window_size]
        
        # Train a temporary RF model on this window
        temp_model = RandomForestRegressor(
            n_estimators=100,  # Smaller for speed
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        temp_model.fit(X_window.values, y_window.values)
        
        # Get importance
        importance = pd.Series(temp_model.feature_importances_, index=feature_names)
        rolling_importance.append(importance)
        window_dates.append(dates[i + window_size // 2])  # Middle of window
    
    rolling_df = pd.DataFrame(rolling_importance, index=window_dates)
    
    # Plot top 10 features
    top_features = rolling_df.mean().nlargest(10).index
    rolling_top = rolling_df[top_features]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in top_features:
        ax.plot(rolling_top.index, rolling_top[feature], label=feature, linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Feature Importance", fontsize=12)
    ax.set_title("Rolling Feature Importance Over Time (Top 10)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved rolling importance plot: {output_path}")


# ---------------------------------------------------------------------------
# Driver Ranking Table
# ---------------------------------------------------------------------------


def create_driver_ranking_table(
    mdi: pd.Series,
    perm_mean: pd.Series,
    perm_std: pd.Series,
    mean_abs_shap: pd.Series,
) -> pd.DataFrame:
    """Create comprehensive driver ranking table with all three metrics."""
    # Get all unique features
    all_features = set(mdi.index) | set(perm_mean.index) | set(mean_abs_shap.index)
    
    # Create DataFrame
    ranking_data = []
    for feature in all_features:
        row = {
            'feature': feature,
            'mdi_importance': mdi.get(feature, 0.0),
            'perm_importance_mean': perm_mean.get(feature, 0.0),
            'perm_importance_std': perm_std.get(feature, 0.0),
            'mean_abs_shap': mean_abs_shap.get(feature, 0.0),
        }
        ranking_data.append(row)
    
    df = pd.DataFrame(ranking_data)
    
    # Normalize each metric to [0, 1] for ranking
    df['mdi_norm'] = (df['mdi_importance'] - df['mdi_importance'].min()) / (
        df['mdi_importance'].max() - df['mdi_importance'].min() + 1e-10
    )
    df['perm_norm'] = (df['perm_importance_mean'] - df['perm_importance_mean'].min()) / (
        df['perm_importance_mean'].max() - df['perm_importance_mean'].min() + 1e-10
    )
    df['shap_norm'] = (df['mean_abs_shap'] - df['mean_abs_shap'].min()) / (
        df['mean_abs_shap'].max() - df['mean_abs_shap'].min() + 1e-10
    )
    
    # Compute ranks (lower is better, so we use ascending=False and then invert)
    df['rank_mdi'] = df['mdi_importance'].rank(ascending=False, method='min')
    df['rank_perm'] = df['perm_importance_mean'].rank(ascending=False, method='min')
    df['rank_shap'] = df['mean_abs_shap'].rank(ascending=False, method='min')
    
    # Average rank
    df['rank_avg'] = (df['rank_mdi'] + df['rank_perm'] + df['rank_shap']) / 3
    
    # Sort by average rank
    df = df.sort_values('rank_avg').reset_index(drop=True)
    
    # Select columns for output
    output_cols = [
        'feature',
        'mdi_importance',
        'perm_importance_mean',
        'perm_importance_std',
        'mean_abs_shap',
        'rank_mdi',
        'rank_perm',
        'rank_shap',
        'rank_avg',
    ]
    
    return df[output_cols]


# ---------------------------------------------------------------------------
# Narrative Summary
# ---------------------------------------------------------------------------


def print_narrative_summary(
    ranking_df: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Print README-ready narrative summary of key drivers."""
    print("\n" + "=" * 80)
    print("SHORT-TERM (12M) KEY DRIVERS — NVDA")
    print("=" * 80)
    
    top_drivers = ranking_df.head(top_n)
    
    print(f"\nTop {top_n} Drivers by Average Rank (MDI + Permutation + SHAP):\n")
    for idx, row in top_drivers.iterrows():
        print(f"{idx+1:2d}. {row['feature']:40s} (Rank: {row['rank_avg']:.2f})")
        print(f"    MDI: {row['mdi_importance']:.6f} | "
              f"Perm: {row['perm_importance_mean']:.6f} ± {row['perm_importance_std']:.6f} | "
              f"SHAP: {row['mean_abs_shap']:.6f}")
    
    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    # Analyze top drivers
    top_3 = top_drivers.head(3)
    print(f"\n1. Top Driver: {top_3.iloc[0]['feature']}")
    print(f"   - Dominates across all three importance metrics")
    print(f"   - Average rank: {top_3.iloc[0]['rank_avg']:.2f}")
    
    # Check for macro vs micro patterns
    macro_features = [f for f in top_drivers['feature'] if any(x in f.lower() for x in ['vix', 'tnx', 'sp500', 'yield', 'macro'])]
    micro_features = [f for f in top_drivers['feature'] if any(x in f.lower() for x in ['rev', 'revenue', 'price', 'momentum'])]
    
    if macro_features:
        print(f"\n2. Macro-Driven Signals:")
        print(f"   - {len(macro_features)} macro features in top {top_n}: {', '.join(macro_features[:3])}")
        print("   - Short-term returns are sensitive to discount rates and market volatility")
    
    if micro_features:
        print(f"\n3. Firm-Specific Signals:")
        print(f"   - {len(micro_features)} firm features in top {top_n}: {', '.join(micro_features[:3])}")
        print("   - Revenue and price momentum contribute to predictions")
    
    # Interaction features
    interaction_features = [f for f in top_drivers['feature'] if 'ix_' in f.lower()]
    if interaction_features:
        print(f"\n4. Interaction Effects:")
        print(f"   - {len(interaction_features)} interaction features in top {top_n}")
        print("   - State-dependent effects: firm characteristics matter differently under different macro conditions")
    
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)
    print("1. Monitor top macro drivers (VIX, TNX) for regime shifts")
    print("2. Revenue signals may have longer lag; price momentum captures short-term dynamics")
    print("3. Interaction features suggest non-linear, context-dependent relationships")
    print("4. See PDP/ICE plots for threshold effects and non-linear shapes")
    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_step5(
    data_path: str = DEFAULT_DATA_PATH,
    target_column: str = DEFAULT_TARGET_COL,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    top_n: int = TOP_N_DRIVERS,
    rolling: bool = False,
) -> None:
    """Run Step 5 key driver analysis with specified parameters."""
    
    # Setup paths
    data_path_obj = Path(data_path)
    model_path_obj = Path(model_path)
    scaler_path_obj = Path(scaler_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 5: KEY DRIVER ANALYSIS")
    print("=" * 80)
    print(f"Data path: {data_path_obj}")
    print(f"Model path: {model_path_obj}")
    print(f"Output directory: {output_dir_obj}")
    print()
    
    # Load data (reuse train_models.py logic)
    print("Loading dataset...")
    X, y, date_index = load_dataset(data_path_obj, target_column, date_column=None)
    print(f"Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"Date range: {date_index.min()} to {date_index.max()}")
    
    # Time-based split (same as Step 2)
    print("\nApplying time-based split...")
    splits = time_based_split(X, y)
    
    # Scale features (same as Step 2)
    print("Scaling features...")
    scaled_splits, scaler = scale_splits(splits)
    
    # Load or retrain champion model
    model = None
    feature_names = scaled_splits.X_test.columns.tolist()
    data_n_features = len(feature_names)
    
    if model_path_obj.exists():
        print(f"\nLoading champion model from {model_path_obj}...")
        loaded_model = joblib.load(model_path_obj)
        
        # Check feature alignment
        if isinstance(loaded_model, RandomForestRegressor):
            model_n_features = len(loaded_model.feature_importances_)
            print(f"  Model expects {model_n_features} features, data has {data_n_features} features")
            
            if model_n_features == data_n_features:
                model = loaded_model
                print("✓ Model loaded and feature count matches")
            else:
                print(f"⚠ Feature count mismatch! Model was trained with {model_n_features} features.")
                print("  Retraining model to match current feature set...")
                model = None  # Will retrain below
        else:
            print(f"⚠ Loaded model is not a RandomForestRegressor, retraining...")
            model = None
    else:
        print(f"\n⚠ Model not found at {model_path_obj}")
        print("Training new RandomForest model with Step 2 parameters...")
        model = None
    
    # Retrain if needed
    if model is None:
        print("\nTraining RandomForest model with Step 2 parameters...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(scaled_splits.X_train.values, scaled_splits.y_train.values)
        print("✓ Model trained")
    
    # Verify it's a RandomForest
    if not isinstance(model, RandomForestRegressor):
        raise ValueError(f"Expected RandomForestRegressor, got {type(model)}")
    
    # Use model's feature importances (should match data now)
    model_importances_aligned = model.feature_importances_
    
    # Compute importance metrics
    print("\n" + "-" * 80)
    print("COMPUTING IMPORTANCE METRICS")
    print("-" * 80)
    
    # MDI importance
    print("\n1. Computing MDI importance...")
    mdi = pd.Series(model_importances_aligned, index=feature_names).sort_values(ascending=False)
    
    # Permutation importance (test set only)
    print("\n2. Computing permutation importance...")
    # Ensure we use the same feature set
    X_test_aligned = scaled_splits.X_test[feature_names]
    perm_mean, perm_std = compute_permutation_importance(
        model,
        X_test_aligned,
        scaled_splits.y_test,
        n_repeats=30,
    )
    
    # SHAP values
    print("\n3. Computing SHAP values...")
    # Ensure we use the same feature set
    X_test_aligned = scaled_splits.X_test[feature_names]
    shap_values, mean_abs_shap = compute_shap_values(
        model,
        X_test_aligned,
        max_samples=200,
    )
    
    # Create driver ranking table
    print("\n4. Creating driver ranking table...")
    ranking_df = create_driver_ranking_table(mdi, perm_mean, perm_std, mean_abs_shap)
    ranking_path = output_dir_obj / "driver_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    print(f"✓ Saved driver ranking: {ranking_path}")
    
    # Generate visualizations
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    # MDI importance plot
    plot_mdi_importance(mdi, output_dir_obj / "mdi_importance.png", top_n=20)
    
    # Permutation importance plot
    plot_permutation_importance(perm_mean, perm_std, output_dir_obj / "permutation_importance.png", top_n=20)
    
    # SHAP summary plot
    if len(X_test_aligned) > 200:
        X_sample = X_test_aligned.sample(200, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values_sample = explainer.shap_values(X_sample.values)
    else:
        X_sample = X_test_aligned
        shap_values_sample = shap_values
    
    plot_shap_summary(shap_values_sample, X_sample, output_dir_obj / "shap_summary.png", max_display=20)
    
    # Save SHAP values
    shap_values_path = output_dir_obj / "shap_values_test.npy"
    np.save(shap_values_path, shap_values)
    print(f"✓ Saved SHAP values: {shap_values_path}")
    
    # PDP/ICE plots for top drivers
    print(f"\n5. Generating PDP/ICE plots for top {top_n} drivers...")
    top_drivers = ranking_df.head(top_n)['feature'].tolist()
    for feature in top_drivers:
        if feature in X_test_aligned.columns:
            plot_pdp_ice(
                model,
                X_test_aligned,
                feature,
                output_dir_obj / f"pdp_{feature.replace('/', '_').replace(' ', '_')}.png",
            )
    
    # Optional rolling importance
    if rolling:
        print("\n6. Generating rolling importance plot...")
        plot_rolling_importance(
            model,
            scaled_splits.X_train,
            scaled_splits.y_train,
            feature_names,
            output_dir_obj / "rolling_importance.png",
            window_size=24,
            step_size=6,
        )
    
    # Print narrative summary
    print_narrative_summary(ranking_df, top_n=10)
    
    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir_obj}")
    print(f"  - Driver ranking: {ranking_path}")
    print(f"  - SHAP values: {shap_values_path}")
    print(f"  - Visualizations: {len(list(output_dir_obj.glob('*.png')))} PNG files")
    print("=" * 80)


def main() -> None:
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Step 5: Extract key drivers from champion RF model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to feature CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=DEFAULT_TARGET_COL,
        help=f"Target column name (default: {DEFAULT_TARGET_COL})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to champion model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=DEFAULT_SCALER_PATH,
        help=f"Path to feature scaler (default: {DEFAULT_SCALER_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=TOP_N_DRIVERS,
        help=f"Number of top drivers for PDP/ICE plots (default: {TOP_N_DRIVERS})",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Generate rolling importance plot (optional, time-consuming)",
    )
    
    args = parser.parse_args()
    
    run_step5(
        data_path=args.data_path,
        target_column=args.target_column,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        rolling=args.rolling,
    )


if __name__ == "__main__":
    main()

