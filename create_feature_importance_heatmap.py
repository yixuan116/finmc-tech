"""
Create a feature importance heatmap across multiple models.

Similar to Gu-Kelly-Xiu (2020) Figure 5, showing feature importance
across different models (Linear, Ridge, RF, XGB, NN) as a heatmap.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import shap

# Model imports
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def load_models_and_scaler(models_dir: Path) -> tuple:
    """Load all trained models and scaler."""
    scaler = joblib.load(models_dir / "feature_scaler.pkl")
    champion_name = (models_dir.parent / "results" / "champion_model_name.txt").read_text().strip()
    
    # Load champion model (RF)
    champion = joblib.load(models_dir / "champion_model.pkl")
    
    # We need to retrain other models or load them if saved
    # For now, we'll compute importance from the champion and use SHAP for others
    return scaler, champion, champion_name


def compute_feature_importance_linear(model, X_train, y_train, feature_names):
    """Compute feature importance for linear models using coefficient magnitudes."""
    model.fit(X_train, y_train)
    coef = model.coef_
    importance = np.abs(coef)
    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = importance / importance.max()
    return pd.Series(importance, index=feature_names)


def compute_feature_importance_rf(model, feature_names):
    """Compute feature importance for Random Forest."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()
        return pd.Series(importance, index=feature_names)
    return pd.Series(0, index=feature_names)


def compute_feature_importance_xgb(model, feature_names):
    """Compute feature importance for XGBoost."""
    if hasattr(model, 'feature_importances_'):
        importance = model.get_booster().get_score(importance_type='gain')
        # Convert to array matching feature_names
        importance_array = np.array([importance.get(f'f{i}', 0) for i in range(len(feature_names))])
        # Normalize to [0, 1]
        if importance_array.max() > 0:
            importance_array = importance_array / importance_array.max()
        return pd.Series(importance_array, index=feature_names)
    return pd.Series(0, index=feature_names)


def compute_feature_importance_shap(model, X_sample, feature_names, model_type='tree'):
    """Compute SHAP-based feature importance."""
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Average absolute SHAP values as importance
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        importance = np.abs(shap_values).mean(axis=0)
        # Normalize to [0, 1]
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return pd.Series(importance, index=feature_names)
    except Exception as e:
        print(f"Warning: SHAP computation failed for {model_type}: {e}")
        return pd.Series(0, index=feature_names)


def compute_feature_importance_nn(model, X_train, y_train, feature_names):
    """Compute feature importance for neural network using permutation importance approximation."""
    # Use a simple gradient-based approximation
    # Or use SHAP
    try:
        X_sample = X_train.iloc[:min(100, len(X_train))]
        return compute_feature_importance_shap(model, X_sample, feature_names, model_type='kernel')
    except:
        # Fallback: use random (not ideal but better than nothing)
        return pd.Series(0, index=feature_names)


def create_importance_heatmap(
    data_path: str,
    models_dir: str = "models",
    output_dir: str = "results",
    top_n_features: int = 50,
    figsize: tuple = (12, 16),
):
    """
    Create feature importance heatmap across multiple models.
    
    Args:
        data_path: Path to feature CSV file
        models_dir: Directory containing saved models
        output_dir: Directory to save output
        top_n_features: Number of top features to display
        figsize: Figure size (width, height)
    """
    # Load data (same logic as train_models.py)
    print("Loading data...")
    df = pd.read_csv(data_path, index_col='px_date', parse_dates=True)
    
    # Get feature columns (exclude target, metadata, and data leakage features)
    target_col = 'future_12m_return'
    metadata_cols = ['period_end', 'fy', 'fp', 'form', 'tag_used', 'ticker']
    # CRITICAL: Exclude data leakage features (future information)
    leakage_features = ['future_12m_price', 'future_12m_logprice']
    feature_cols = [col for col in df.columns 
                   if col not in [target_col] + metadata_cols + leakage_features]
    
    if any(col in df.columns for col in leakage_features):
        print(f"⚠ Excluding data leakage features: {[f for f in leakage_features if f in df.columns]}")
    
    # Load scaler first to get correct feature order
    models_path = Path(models_dir)
    scaler = joblib.load(models_path / "feature_scaler.pkl")
    # Get ORIGINAL scaler features (may include leakage features from old training)
    scaler_features_original = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else feature_cols
    
    # CRITICAL: Filter out data leakage features for actual model training
    # But keep original scaler_features for scaler.transform() which requires all features
    scaler_features_for_training = [f for f in scaler_features_original if f not in leakage_features]
    
    # Use only legitimate features that are in scaler (after filtering leakage)
    feature_cols = [f for f in feature_cols if f in scaler_features_for_training]
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Drop rows where either X or y is missing
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"After cleaning: {len(X)} samples, {len(feature_cols)} legitimate features")
    
    # Time-based split (same as training)
    train_mask = X.index < '2021-01-01'
    val_mask = (X.index >= '2021-01-01') & (X.index < '2023-01-01')
    test_mask = X.index >= '2023-01-01'
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    # CRITICAL: Align with ORIGINAL scaler features (scaler.transform() needs all 63 features)
    # Add leakage features as zeros for scaler compatibility, then drop them after scaling
    X_train_aligned = X_train.reindex(columns=scaler_features_original, fill_value=0)
    X_test_aligned = X_test.reindex(columns=scaler_features_original, fill_value=0)
    
    # Scale data (scaler expects all original features including leakage)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_aligned),
        index=X_train_aligned.index,
        columns=scaler_features_original
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_aligned),
        index=X_test_aligned.index,
        columns=scaler_features_original
    )
    
    # CRITICAL: Remove data leakage features AFTER scaling (prevent data leakage)
    X_train_scaled = X_train_scaled.drop(columns=leakage_features, errors='ignore')
    X_test_scaled = X_test_scaled.drop(columns=leakage_features, errors='ignore')
    
    # Final feature list for model training (excludes leakage)
    feature_cols = [f for f in scaler_features_original if f not in leakage_features]
    
    print(f"Data loaded: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples")
    print(f"Features: {len(feature_cols)}")
    
    # Initialize models (same as training)
    models = {
        "Linear": LinearRegression(),
        "Ridge": RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0]),
        "RF": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
        "XGB": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42,
        ),
        "NN": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
        ),
    }
    
    # Train models and compute importance
    print("\nTraining models and computing feature importance...")
    importance_dict = {}
    
    for name, model in models.items():
        print(f"  Processing {name}...")
        
        if name in ["Linear", "Ridge"]:
            # Linear models: use coefficient magnitudes
            model.fit(X_train_scaled, y_train)
            coef = np.abs(model.coef_)
            if coef.max() > 0:
                importance = coef / coef.max()
            else:
                importance = coef
            importance_dict[name] = pd.Series(importance, index=feature_cols)
        
        elif name == "RF":
            # Random Forest: use feature_importances_
            model.fit(X_train_scaled, y_train)
            importance = model.feature_importances_
            if importance.max() > 0:
                importance = importance / importance.max()
            importance_dict[name] = pd.Series(importance, index=feature_cols)
        
        elif name == "XGB":
            # XGBoost: use feature_importances_ (gain)
            model.fit(X_train_scaled, y_train)
            importance = model.feature_importances_
            if importance.max() > 0:
                importance = importance / importance.max()
            importance_dict[name] = pd.Series(importance, index=feature_cols)
        
        elif name == "NN":
            # Neural Network: use SHAP
            model.fit(X_train_scaled, y_train)
            # Use a sample for SHAP (faster)
            X_sample = X_train_scaled.iloc[:min(100, len(X_train_scaled))]
            try:
                explainer = shap.KernelExplainer(model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
                importance = np.abs(shap_values).mean(axis=0)
                if importance.max() > 0:
                    importance = importance / importance.max()
                importance_dict[name] = pd.Series(importance, index=feature_cols)
            except Exception as e:
                print(f"    Warning: SHAP failed for NN, using zero importance: {e}")
                importance_dict[name] = pd.Series(0, index=feature_cols)
    
    # Create DataFrame
    importance_df = pd.DataFrame(importance_dict)
    
    # Select top N features by average importance across models
    avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    top_features = avg_importance.head(top_n_features).index.tolist()
    importance_df_top = importance_df.loc[top_features]
    
    print(f"\nSelected top {len(top_features)} features for visualization")
    
    # Create heatmap
    print("Creating heatmap...")
    plt.figure(figsize=figsize)
    
    # Use blue colormap similar to the paper
    sns.heatmap(
        importance_df_top,
        cmap='Blues',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Normalized Feature Importance'},
        linewidths=0.5,
        linecolor='gray',
    )
    
    plt.title('Feature Importance Across Models\n(Similar to Gu-Kelly-Xiu 2020 Figure 5)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "feature_importance_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Heatmap saved to: {output_file}")
    
    # Also save the importance DataFrame
    csv_file = output_path / "feature_importance_heatmap.csv"
    importance_df.to_csv(csv_file)
    print(f"✓ Importance data saved to: {csv_file}")
    
    # Save top features only
    csv_top_file = output_path / "feature_importance_heatmap_top.csv"
    importance_df_top.to_csv(csv_top_file)
    print(f"✓ Top features importance saved to: {csv_top_file}")
    
    # Create big single heatmap (primary visualization)
    plot_feature_importance_heatmap_big(
        importance_df=importance_df,
        output_dir=output_dir,
        figsize=(8, max(12, len(importance_df) * 0.15)),
    )
    
    # Create facet-style heatmap (optional, secondary visualization)
    importance_df_ordered = create_facet_heatmap(
        importance_df=importance_df,
        output_dir=output_dir,
        figsize=(10, 20),
    )
    
    plt.close()
    
    return importance_df, importance_df_top


def plot_feature_importance_heatmap_big(
    importance_df: pd.DataFrame,
    output_dir: str = "results",
    figsize: tuple = (8, 16),
) -> None:
    """
    Create a single large heatmap: rows = features, columns = models.
    
    This is the primary visualization for feature importance comparison.
    Each model column is normalized to [0, 1] independently for readability.
    No cross-model averaging or weighting is applied.
    
    Args:
        importance_df: DataFrame with features as rows, models as columns
        output_dir: Directory to save output
        figsize: Figure size (width, height)
    """
    print("\nCreating big single heatmap (primary visualization)...")
    
    # Ensure all features have values for all models (fill missing with 0)
    importance_df = importance_df.fillna(0)
    
    # Normalize each model column independently to [0, 1]
    # This ensures readability while preserving each model's own importance distribution
    importance_df_normalized = importance_df.copy()
    for col in importance_df_normalized.columns:
        col_max = importance_df_normalized[col].max()
        if col_max > 0:
            importance_df_normalized[col] = importance_df_normalized[col] / col_max
        else:
            importance_df_normalized[col] = 0
    
    # Sort features by average importance across models (for better visualization)
    avg_importance = importance_df_normalized.mean(axis=1).sort_values(ascending=False)
    feature_order = avg_importance.index.tolist()
    importance_df_sorted = importance_df_normalized.loc[feature_order]
    
    # Get dimensions
    n_features = len(importance_df_sorted)
    n_models = len(importance_df_sorted.columns)
    model_names = importance_df_sorted.columns.tolist()
    
    print(f"  Features: {n_features}, Models: {n_models}")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Each model column normalized independently to [0, 1]")
    
    # Create the big heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn heatmap with better styling
    sns.heatmap(
        importance_df_sorted,
        cmap='Blues',
        ax=ax,
        annot=False,  # Don't annotate cells (too many features)
        fmt='.2f',
        cbar_kws={'label': 'Normalized Feature Importance (per model)'},
        linewidths=0.3,
        linecolor='gray',
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=1,
    )
    
    # Set labels and title
    ax.set_title(
        'Feature Importance Across Models\n(Each Model Normalized Independently to [0, 1])',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    
    # Rotate feature names for better readability
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=0, fontsize=10, fontweight='bold')
    
    # Set y-axis labels (feature names)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_order, rotation=0, fontsize=7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "feature_importance_heatmap_big.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Big heatmap saved to: {output_file}")
    print(f"  Dimensions: {n_features} features × {n_models} models")
    
    plt.close()
    
    return importance_df_sorted


def create_facet_heatmap(
    importance_df: pd.DataFrame,
    output_dir: str = "results",
    figsize: tuple = (10, 20),
) -> None:
    """
    Create a facet-style heatmap with one panel per model.
    
    This provides a fair, transparent, side-by-side comparison of all models
    without any cross-model averaging or normalization.
    
    Args:
        importance_df: DataFrame with features as rows, models as columns
        output_dir: Directory to save output
        figsize: Figure size (width, height)
    """
    print("\nCreating facet-style heatmap (one panel per model)...")
    
    # Ensure all features have values for all models (fill missing with 0)
    importance_df = importance_df.fillna(0)
    
    # Determine global feature order (sort by max importance across any model)
    max_importance_per_feature = importance_df.max(axis=1)
    feature_order = max_importance_per_feature.sort_values(ascending=False).index.tolist()
    
    # Reindex DataFrame to use global feature order
    importance_df_ordered = importance_df.loc[feature_order]
    
    # Get model names (columns)
    model_names = importance_df_ordered.columns.tolist()
    n_models = len(model_names)
    n_features = len(importance_df_ordered)
    
    print(f"  Features: {n_features}, Models: {n_models}")
    print(f"  Feature order: sorted by max importance across all models")
    
    # Determine global color scale (use max across all models for fair comparison)
    global_max = importance_df_ordered.max().max()
    global_min = importance_df_ordered.min().min()
    
    # Create facet plot: n_models rows × 1 column
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharey=True)
    
    # Handle case where n_models = 1 (single subplot)
    if n_models == 1:
        axes = [axes]
    
    # Plot each model in its own panel
    for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
        # Extract importance values for this model (single column)
        model_importance = importance_df_ordered[model_name].values
        
        # Reshape to 2D for heatmap (features × 1 model)
        # We'll create a 2D array with shape (n_features, 1)
        heatmap_data = model_importance.reshape(-1, 1)
        
        # Create heatmap for this model using seaborn for better appearance
        sns.heatmap(
            heatmap_data,
            cmap='Blues',
            ax=ax,
            cbar=False,  # We'll add a shared colorbar later
            vmin=global_min,
            vmax=global_max,
            linewidths=0.5,
            linecolor='gray',
            xticklabels=False,
            yticklabels=False,
        )
        
        # Set labels and title
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('', fontsize=10)
        
        # Set y-axis: show feature names (only for first subplot or all if not too many)
        if idx == 0 or n_features <= 30:
            ax.set_yticks(np.arange(n_features) + 0.5)
            ax.set_yticklabels(feature_order, fontsize=6, rotation=0)
        else:
            ax.set_yticks([])
        
        # Set x-axis: single tick for model name
        ax.set_xticks([0.5])
        ax.set_xticklabels([model_name], fontsize=10)
    
    # Add shared colorbar on the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    
    # Create a mappable for colorbar using the data range
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=plt.cm.Blues, norm=Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Feature Importance', fontsize=10, rotation=270, labelpad=20)
    
    # Overall title
    fig.suptitle(
        'Feature Importance: Facet Comparison Across Models\n(No Cross-Model Averaging)',
        fontsize=14,
        fontweight='bold',
        y=0.995,
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 0.98])
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "feature_importance_facet.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Facet heatmap saved to: {output_file}")
    print(f"  Dimensions: {n_features} features × {n_models} models")
    print(f"  Models: {', '.join(model_names)}")
    
    plt.close()
    
    return importance_df_ordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create feature importance heatmap across multiple models"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/nvda_features_extended.csv",
        help="Path to feature CSV file",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory containing saved models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top features to display",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,16",
        help="Figure size as 'width,height'",
    )
    
    args = parser.parse_args()
    
    figsize = tuple(map(int, args.figsize.split(',')))
    
    create_importance_heatmap(
        data_path=args.data_path,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        top_n_features=args.top_n,
        figsize=figsize,
    )

