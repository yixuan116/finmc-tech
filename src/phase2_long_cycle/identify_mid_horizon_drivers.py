"""
Mid-Horizon Driver Identification for NVDA 3-Year Forward Log Return.

This script trains a RandomForest model to identify key drivers of 
NVDA's 3-year forward log return (y_log_12q).

Step 5-6 in the pipeline: DRIVER DISCOVERY
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data."""
    logger.info(f"Loading training data from {data_path}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.sort_index().reset_index()
    
    logger.info(f"Loaded {len(df)} samples")
    return df


def prepare_features_and_target(df: pd.DataFrame, target: str = "y_log_12q"):
    """
    Prepare features and target for modeling.
    
    Args:
        df: Training DataFrame
        target: Target variable name
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    logger.info(f"Preparing features and target: {target}")
    
    # Drop rows with missing target
    df = df.dropna(subset=[target]).copy()
    logger.info(f"Samples after dropping missing target: {len(df)}")
    
    # Columns NOT to use as features
    exclude_cols = {
        "period_end_date",
        "price_q",
        "y_log_4q",
        "y_log_8q",
        "y_log_12q",
        "y_log_16q",
        "y_log_20q",
    }
    
    # Get feature columns (exclude target and other columns)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Select only numeric features
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target]
    
    logger.info(f"Feature count: {len(X.columns)}")
    logger.info(f"Target: {target}")
    logger.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    return X, y, X.columns.tolist()


def train_model(X_train, y_train, n_estimators=600, random_state=42):
    """
    Train RandomForest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Trained model
    """
    logger.info(f"Training RandomForest with {n_estimators} trees")
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=-1,
        verbose=0
    )
    
    rf.fit(X_train, y_train)
    
    return rf


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance.
    
    Returns:
        train_r2, test_r2
    """
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    logger.info(f"Train R²: {train_r2:.4f}")
    logger.info(f"Test  R²: {test_r2:.4f}")
    
    return train_r2, test_r2


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract and sort feature importance.
    
    Returns:
        DataFrame with feature importance
    """
    importances = model.feature_importances_
    
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    return fi


def plot_feature_importance(fi, output_path: Path, top_n=20):
    """
    Plot feature importance.
    
    Args:
        fi: DataFrame with feature importance
        top_n: Number of top features to plot
        output_path: Path to save plot
    """
    logger.info(f"Plotting top {top_n} feature importance")
    
    top = fi.head(top_n)
    
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"][::-1], top["importance"][::-1], color='steelblue')
    plt.title("Mid-Horizon Feature Importance (3-Year Log Return)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Importance (RandomForest Gain)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Importance plot saved to: {output_path}")


def identify_drivers_by_horizon(
    data_path: Path,
    output_dir: Path = Path("images"),
    test_size: float = 0.2,
    n_estimators: int = 600
):
    """
    Identify key drivers for all horizons (1-5 years) and create heatmap.
    
    Args:
        data_path: Path to training CSV
        output_dir: Directory to save output
        test_size: Test set size fraction
        n_estimators: Number of trees in RandomForest
    """
    logger.info("=" * 60)
    logger.info("Multi-Horizon Driver Identification (1-5 Years)")
    logger.info("=" * 60)
    
    # Load data
    df = load_training_data(data_path)
    
    # Target variables for different horizons
    targets = {
        'y_log_4q': '1 year',
        'y_log_8q': '2 years',
        'y_log_12q': '3 years',
        'y_log_16q': '4 years',
        'y_log_20q': '5 years'
    }
    
    # Store importance for each horizon
    importance_dict = {}
    results_summary = []
    
    for target, horizon_name in targets.items():
        if target not in df.columns:
            logger.warning(f"Target {target} not found, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for {horizon_name} ({target})")
        logger.info(f"{'='*60}")
        
        # Prepare features and target
        X, y, feature_names = prepare_features_and_target(df, target)
        
        if len(X) < 10:
            logger.warning(f"Not enough samples for {target}, skipping")
            continue
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train model
        model = train_model(X_train, y_train, n_estimators=n_estimators)
        
        # Evaluate
        train_r2, test_r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Feature importance
        fi = get_feature_importance(model, feature_names, top_n=len(feature_names))
        importance_dict[horizon_name] = fi.set_index('feature')['importance']
        
        results_summary.append({
            'horizon': horizon_name,
            'target': target,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_samples': len(X)
        })
        
        logger.info(f"Top 3 features for {horizon_name}:")
        for i, row in fi.head(3).iterrows():
            logger.info(f"  {i+1}. {row['feature']:30s}: {row['importance']:.4f}")
    
    # Create importance DataFrame for heatmap
    importance_df = pd.DataFrame(importance_dict)
    importance_df = importance_df.fillna(0)  # Fill missing with 0
    
    # Sort by average importance across all horizons
    importance_df['avg_importance'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    importance_df = importance_df.drop('avg_importance', axis=1)
    
    logger.info(f"\n{'='*60}")
    logger.info("Summary of Model Performance:")
    logger.info(f"{'='*60}")
    for result in results_summary:
        logger.info(f"{result['horizon']:10s} ({result['target']:12s}): "
                   f"Train R²={result['train_r2']:.4f}, "
                   f"Test R²={result['test_r2']:.4f}, "
                   f"N={result['n_samples']}")
    
    # Plot heatmap
    plot_importance_heatmap(importance_df, output_dir)
    
    logger.info("=" * 60)
    logger.info("Multi-horizon driver identification complete!")
    logger.info("=" * 60)
    
    return importance_df, results_summary


def plot_importance_heatmap(importance_df: pd.DataFrame, output_dir: Path):
    """
    Plot feature importance heatmap across all horizons.
    
    Args:
        importance_df: DataFrame with features as rows, horizons as columns
        output_dir: Directory to save plot
    """
    logger.info("Creating feature importance heatmap")
    
    import seaborn as sns
    
    # Select top features (by average importance)
    top_n = min(30, len(importance_df))
    top_features = importance_df.head(top_n)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(12, len(top_features) * 0.4)))
    
    sns.heatmap(
        top_features,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Feature Importance'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title('Feature Importance by Horizon (1-5 Years)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.tight_layout()
    
    output_path = output_dir / "feature_importance_by_horizon_heatmap.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Importance heatmap saved to: {output_path}")


def identify_drivers(
    data_path: Path,
    target: str = "y_log_12q",
    output_dir: Path = Path("images"),
    top_n: int = 20,
    test_size: float = 0.2,
    n_estimators: int = 600
):
    """
    Main function: identify key drivers for mid-horizon return.
    
    Args:
        data_path: Path to training CSV
        target: Target variable name
        output_dir: Directory to save output
        top_n: Number of top features to display
        test_size: Test set size fraction
        n_estimators: Number of trees in RandomForest
    """
    logger.info("=" * 60)
    logger.info("Mid-Horizon Driver Identification")
    logger.info("=" * 60)
    
    # Load data
    df = load_training_data(data_path)
    
    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(df, target)
    
    logger.info(f"\nTotal samples: {len(df)}")
    logger.info(f"Feature count: {len(X.columns)}")
    
    # Train/test split (no shuffle to preserve temporal order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=42
    )
    
    logger.info(f"\nTrain samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, n_estimators=n_estimators)
    
    # Evaluate
    train_r2, test_r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Feature importance
    fi = get_feature_importance(model, feature_names, top_n=top_n)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Top {top_n} Drivers for NVDA 3-Year Return:")
    logger.info(f"{'='*60}")
    for i, row in fi.head(top_n).iterrows():
        logger.info(f"{i+1:2d}. {row['feature']:30s}: {row['importance']:.4f}")
    
    # Plot
    output_path = output_dir / "mid_horizon_feature_importance_y_log_12q.png"
    plot_feature_importance(fi, output_path, top_n=top_n)
    
    logger.info("=" * 60)
    logger.info("Driver identification complete!")
    logger.info("=" * 60)
    
    return fi, model, train_r2, test_r2


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Identify key drivers for NVDA forward log returns"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/processed/nvda_long_cycle_train.csv'),
        help='Path to training CSV'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target variable name (if None, analyze all horizons)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('images'),
        help='Output directory for plots'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top features to display'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=600,
        help='Number of trees in RandomForest'
    )
    parser.add_argument(
        '--all-horizons',
        action='store_true',
        help='Analyze all horizons (1-5 years) and create heatmap'
    )
    
    args = parser.parse_args()
    
    if args.all_horizons or args.target is None:
        # Analyze all horizons
        identify_drivers_by_horizon(
            data_path=args.data,
            output_dir=args.output,
            n_estimators=args.n_estimators
        )
    else:
        # Single target analysis
        identify_drivers(
            data_path=args.data,
            target=args.target,
            output_dir=args.output,
            top_n=args.top_n,
            n_estimators=args.n_estimators
        )


if __name__ == '__main__':
    main()

