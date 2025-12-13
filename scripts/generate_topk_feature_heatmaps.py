"""
Generate Clean Top-K Feature × Horizon Heatmaps (RF + XGB)

This script produces professional heatmaps showing top K features across horizons.
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.three_category_feature_importance import (
    load_features, create_target_variables, prepare_features,
    train_rf_models, train_xgb_models, classify_feature
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================
# (1) Build Top-K Feature List
# =============================================

def compute_overall_importance(
    importance_dict: Dict[str, pd.DataFrame],
    horizons: List[str]
) -> Dict[str, float]:
    """
    Compute overall importance score for each feature across all horizons.
    
    Args:
        importance_dict: Dictionary mapping horizon to DataFrame with feature importance
        horizons: List of horizon names
        
    Returns:
        Dictionary mapping feature name to overall importance score
    """
    overall_score = {}
    
    for horizon in horizons:
        if horizon not in importance_dict:
            continue
        
        df = importance_dict[horizon]
        for _, row in df.iterrows():
            feature = row['feature']
            importance = row['importance_pct']
            
            if feature not in overall_score:
                overall_score[feature] = 0.0
            
            overall_score[feature] += importance
    
    return overall_score


def get_top_k_features(
    overall_importance: Dict[str, float],
    top_k: int = 20
) -> List[str]:
    """
    Get top K features by overall importance.
    
    Args:
        overall_importance: Dictionary mapping feature to overall importance
        top_k: Number of top features to return
        
    Returns:
        List of top K feature names
    """
    sorted_features = sorted(overall_importance.items(), key=lambda x: -x[1])
    top_features = [f[0] for f in sorted_features[:top_k]]
    
    return top_features


# =============================================
# (2) Construct K × 4 Matrix
# =============================================

def build_feature_horizon_matrix(
    importance_dict: Dict[str, pd.DataFrame],
    top_features: List[str],
    horizons: List[str]
) -> pd.DataFrame:
    """
    Build a matrix of shape (K × 4) with features as rows and horizons as columns.
    
    Args:
        importance_dict: Dictionary mapping horizon to DataFrame with feature importance
        top_features: List of top K feature names
        horizons: List of horizon names
        
    Returns:
        DataFrame with features as index and horizons as columns
    """
    # Initialize matrix
    matrix_data = {}
    
    for horizon in horizons:
        matrix_data[horizon] = []
        
        if horizon not in importance_dict:
            # Fill with zeros if horizon not available
            matrix_data[horizon] = [0.0] * len(top_features)
            continue
        
        df = importance_dict[horizon]
        # Create a lookup dictionary for this horizon
        importance_lookup = dict(zip(df['feature'], df['importance_pct']))
        
        for feature in top_features:
            importance = importance_lookup.get(feature, 0.0)
            matrix_data[horizon].append(importance)
    
    # Create DataFrame
    matrix_df = pd.DataFrame(matrix_data, index=top_features)
    
    # Fill NaN with 0
    matrix_df = matrix_df.fillna(0.0)
    
    return matrix_df


# =============================================
# (3) Plot Heatmap - Professional Paper-Quality Style
# =============================================

def plot_topk_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: Path,
    category_map: Dict[str, str],
    top_k: int = 20,
) -> None:
    """
    Plot a professional, publication-quality heatmap showing top K features across horizons.
    
    Features are grouped by category (Firm / Macro / Interaction) and sorted by
    overall importance within each category.
    
    Args:
        df: DataFrame with features as rows and horizons as columns
            - index = feature names
            - columns = ["1y", "3y", "5y", "10y"] (will be converted to ["1Y", "3Y", "5Y", "10Y"])
            - values = importance in percent
        title: Plot title
        save_path: Path to save the figure
        category_map: Dictionary mapping feature name to category ("Firm", "Macro", "Interaction")
        top_k: Number of top features to display
    """
    logger.info(f"Plotting professional heatmap: {save_path.name}")
    
    # (1) Restrict to top K features by overall importance
    overall = df.sum(axis=1)
    top_features = overall.sort_values(ascending=False).head(top_k).index
    df_top = df.loc[top_features].copy()
    
    # (2) Build metadata DataFrame with category and overall score
    meta = pd.DataFrame({
        "feature": df_top.index,
        "category": [category_map.get(f, "Firm") for f in df_top.index],
        "overall": df_top.sum(axis=1),
    })
    
    # (3) Define category order and sort
    cat_order = ["Firm", "Macro", "Interaction"]
    meta["cat_rank"] = meta["category"].map(
        {c: i for i, c in enumerate(cat_order)}
    )
    meta = meta.sort_values(["cat_rank", "overall"], ascending=[True, False])
    
    # Reorder df_top according to sorted metadata
    df_top = df_top.loc[meta["feature"]]
    
    # Normalize column names to uppercase for display
    df_top.columns = [col.upper() for col in df_top.columns]
    
    # (4) Prepare annotations (suppress very small values to reduce visual noise)
    annot_df = df_top.copy()
    annot_df = annot_df.applymap(lambda x: f"{x:.1f}" if x >= 0.3 else "")
    
    # Print data to console as fallback (User Request)
    print("\n" + "="*80)
    print(f"TOP-{top_k} FEATURES DATA TABLE (Copy to Excel/PPT)")
    print("="*80)
    print(df_top.to_string())
    print("="*80 + "\n")

    # (5) Create the plot
    plt.figure(figsize=(10, 10))  # Square and compact
    
    # Use consistent color scale per model
    vmax = df_top.values.max()
    
    ax = sns.heatmap(
        df_top,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        annot=annot_df,
        fmt="",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Importance (%)"},
        linewidths=0.4,
        linecolor="white",
    )
    
    # (6) Axes formatting
    # Add timestamp to title to verify freshness
    import datetime
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"{title} (New V2 Data - {ts_str})", fontsize=14, pad=12, fontweight='bold')
    
    ax.set_xlabel("Horizon", fontsize=11, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=11, fontweight='bold')
    
    # X ticks: 1Y, 3Y, 5Y, 10Y without rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    # Y ticks: feature names, left aligned
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9, ha='right')
    
    # (7) Visual category encoding - color ytick labels by category
    colors = {
        "Firm": "#1f77b4",        # blue
        "Macro": "#d62728",       # red
        "Interaction": "#2ca02c",  # green
    }
    
    for label in ax.get_yticklabels():
        feat = label.get_text()
        cat = category_map.get(feat, "Firm")
        label.set_color(colors.get(cat, "black"))
        label.set_fontweight('normal')
    
    # (8) Tight layout for publication-style spacing
    plt.tight_layout()
    
    # (9) Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Attempt 1: Save to original path
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save to {save_path}: {e}")

    # Attempt 2: Force save to CWD (Project Root)
    try:
        root_filename = f"ROOT_{save_path.name}"
        root_path = Path.cwd() / root_filename
        plt.savefig(root_path, dpi=300, bbox_inches='tight')
        print(f"\n[URGENT] Plot saved to project root: {root_path}")
    except Exception as e:
        logger.error(f"Failed to save to root {root_path}: {e}")

    plt.close()


# =============================================
# (4) Save Matrices to CSV
# =============================================

def save_matrix_csv(
    matrix_df: pd.DataFrame,
    model_name: str,
    top_k: int,
    output_dir: Path
):
    """Save the feature-horizon matrix to CSV."""
    output_path = output_dir / f'{model_name.lower()}_top{top_k}_matrix.csv'
    matrix_df.to_csv(output_path)
    logger.info(f"Saved: {output_path}")


# =============================================
# (5) Print Summary
# =============================================

def print_summary(
    rf_matrix_df: pd.DataFrame,
    xgb_matrix_df: pd.DataFrame,
    rf_top_features: List[str],
    xgb_top_features: List[str],
    rf_importance_dict: Dict[str, pd.DataFrame],
    xgb_importance_dict: Dict[str, pd.DataFrame],
    classifications: Dict[str, str],
    horizons: List[str]
):
    """Print summary statistics and insights."""
    logger.info("\n" + "=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    
    # (1) Top-K lists
    logger.info("\n1. Top-K Feature Lists:")
    logger.info(f"\n   RF Model (Top {len(rf_top_features)}):")
    for idx, feature in enumerate(rf_top_features, 1):
        category = classifications.get(feature, 'Firm')
        logger.info(f"     {idx:2d}. {feature:40s} ({category})")
    
    logger.info(f"\n   XGB Model (Top {len(xgb_top_features)}):")
    for idx, feature in enumerate(xgb_top_features, 1):
        category = classifications.get(feature, 'Firm')
        logger.info(f"     {idx:2d}. {feature:40s} ({category})")
    
    # (2) Features stable across 3+ horizons
    logger.info("\n2. Features Stable Across 3+ Horizons:")
    
    for model_name, matrix_df, importance_dict in [
        ("RF", rf_matrix_df, rf_importance_dict),
        ("XGB", xgb_matrix_df, xgb_importance_dict)
    ]:
        stable_features = []
        
        for feature in matrix_df.index:
            # Count how many horizons this feature has non-zero importance
            non_zero_count = (matrix_df.loc[feature] > 0.1).sum()
            if non_zero_count >= 3:
                max_importance = matrix_df.loc[feature].max()
                stable_features.append((feature, non_zero_count, max_importance))
        
        if stable_features:
            logger.info(f"\n   {model_name} Model:")
            for feature, count, max_imp in sorted(stable_features, key=lambda x: -x[2]):
                category = classifications.get(feature, 'Firm')
                logger.info(f"     {feature:40s}: {count} horizons, max {max_imp:.2f}% ({category})")
        else:
            logger.info(f"\n   {model_name} Model: No features appear in 3+ horizons")
    
    # (3) Category distribution by horizon
    logger.info("\n3. Category Distribution by Horizon (Top-K):")
    
    for model_name, matrix_df, importance_dict in [
        ("RF", rf_matrix_df, rf_importance_dict),
        ("XGB", xgb_matrix_df, xgb_importance_dict)
    ]:
        logger.info(f"\n   {model_name} Model:")
        
        for horizon in horizons:
            if horizon not in importance_dict:
                continue
            
            # Get top features for this horizon from the matrix
            horizon_importance = matrix_df[horizon].sort_values(ascending=False)
            
            # Count categories
            cat_counts = {'Firm': 0, 'Macro': 0, 'Interaction': 0}
            cat_importance = {'Firm': 0.0, 'Macro': 0.0, 'Interaction': 0.0}
            
            for feature in horizon_importance.index:
                if horizon_importance[feature] > 0.1:  # Only count significant features
                    category = classifications.get(feature, 'Firm')
                    cat_counts[category] += 1
                    cat_importance[category] += horizon_importance[feature]
            
            total_importance = sum(cat_importance.values())
            if total_importance > 0:
                logger.info(f"     {horizon.upper()}:")
                logger.info(f"       Count: Firm={cat_counts['Firm']}, Macro={cat_counts['Macro']}, Interaction={cat_counts['Interaction']}")
                logger.info(f"       Importance: Firm={cat_importance['Firm']:.1f}%, Macro={cat_importance['Macro']:.1f}%, Interaction={cat_importance['Interaction']:.1f}%")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Generate Top-K feature heatmaps')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to extended features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance',
                       help='Base output directory')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top features to include')
    
    args = parser.parse_args()
    
    # Create output directories
    # Use output_dir directly if provided, don't create sub-plots folder unless needed
    output_dir = Path(args.output_dir)
    rankings_dir = output_dir.parent / 'rankings' # Put rankings alongside plots
    
    # If output_dir ends in 'plots', use it directly. Otherwise create 'plots' subfolder
    if output_dir.name == 'plots':
        plots_dir = output_dir
    else:
        plots_dir = output_dir / 'plots'
        
    rankings_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Generate Clean Top-K Feature × Horizon Heatmaps")
    logger.info("=" * 80)
    
    horizons = ['1y', '3y', '5y', '10y']
    
    # Load data and train models
    logger.info("\n[Step 1] Loading data and computing importance...")
    df = load_features(Path(args.features_csv))
    
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
    
    # Train models
    logger.info("\n[Step 2] Training models...")
    rf_importance = train_rf_models(df, feature_cols, horizon_quarters)
    xgb_importance = train_xgb_models(df, feature_cols, horizon_quarters)
    
    # Add categories to importance DataFrames
    for horizon in rf_importance.keys():
        rf_importance[horizon]['category'] = rf_importance[horizon]['feature'].apply(
            lambda f: classifications.get(f, 'Firm')
        )
        xgb_importance[horizon]['category'] = xgb_importance[horizon]['feature'].apply(
            lambda f: classifications.get(f, 'Firm')
        )
    
    # (1) Build Top-K feature lists
    logger.info("\n[Step 3] Building Top-K feature lists...")
    rf_overall = compute_overall_importance(rf_importance, horizons)
    xgb_overall = compute_overall_importance(xgb_importance, horizons)
    
    rf_top_features = get_top_k_features(rf_overall, top_k=args.top_k)
    xgb_top_features = get_top_k_features(xgb_overall, top_k=args.top_k)
    
    logger.info(f"RF Top-{args.top_k} features selected")
    logger.info(f"XGB Top-{args.top_k} features selected")
    
    # (2) Construct matrices
    logger.info("\n[Step 4] Constructing feature-horizon matrices...")
    rf_matrix = build_feature_horizon_matrix(rf_importance, rf_top_features, horizons)
    xgb_matrix = build_feature_horizon_matrix(xgb_importance, xgb_top_features, horizons)
    
    # (3) Plot heatmaps with professional styling
    logger.info("\n[Step 5] Plotting professional heatmaps...")
    plot_topk_heatmap(
        rf_matrix,
        title=f"RF Top-{args.top_k} Feature Importance Across Horizons",
        save_path=plots_dir / f"rf_top{args.top_k}_feature_matrix_paper.png",
        category_map=classifications,
        top_k=args.top_k,
    )
    
    plot_topk_heatmap(
        xgb_matrix,
        title=f"XGB Top-{args.top_k} Feature Importance Across Horizons",
        save_path=plots_dir / f"xgb_top{args.top_k}_feature_matrix_paper.png",
        category_map=classifications,
        top_k=args.top_k,
    )
    
    # (4) Save matrices to CSV
    logger.info("\n[Step 6] Saving matrices to CSV...")
    save_matrix_csv(rf_matrix, "RF", args.top_k, rankings_dir)
    save_matrix_csv(xgb_matrix, "XGB", args.top_k, rankings_dir)
    
    # (5) Print summary
    logger.info("\n[Step 7] Computing summary statistics...")
    print_summary(
        rf_matrix, xgb_matrix,
        rf_top_features, xgb_top_features,
        rf_importance, xgb_importance,
        classifications, horizons
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All steps completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

