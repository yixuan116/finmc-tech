"""
Generate Combined Feature Ranking Table (1Y / 3Y / 5Y / 10Y)

This script creates a single consolidated ranking table showing top N features
per horizon for Random Forest and XGBoost models.
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
# (1) Construct Combined Table for RF
# =============================================

def create_combined_table(
    importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 15,
    model_name: str = "RF"
) -> pd.DataFrame:
    """
    Create a single combined ranking table.
    
    Columns:
        Rank, 1Y Feature, 1Y Cat, 1Y Importance,
        3Y Feature, 3Y Cat, 3Y Importance,
        5Y Feature, 5Y Cat, 5Y Importance,
        10Y Feature, 10Y Cat, 10Y Importance
    """
    logger.info(f"Creating combined table for {model_name}")
    
    horizons = ['1y', '3y', '5y', '10y']
    
    # Initialize result DataFrame
    columns = ['Rank']
    for horizon in horizons:
        columns.extend([f'{horizon.upper()} Feature', f'{horizon.upper()} Cat', f'{horizon.upper()} Importance'])
    
    result_df = pd.DataFrame(index=range(1, top_n + 1), columns=columns)
    result_df['Rank'] = range(1, top_n + 1)
    
    # Fill in data for each horizon
    for horizon in horizons:
        if horizon not in importance_dict:
            logger.warning(f"Horizon {horizon} not found, skipping")
            continue
        
        df = importance_dict[horizon].head(top_n).reset_index(drop=True)
        
        feature_col = f'{horizon.upper()} Feature'
        cat_col = f'{horizon.upper()} Cat'
        imp_col = f'{horizon.upper()} Importance'
        
        # Pad with NaN if fewer than top_n features
        n_features = len(df)
        if n_features < top_n:
            # Extend with NaN
            feature_vals = list(df['feature'].values) + [np.nan] * (top_n - n_features)
            cat_vals = list(df['category'].values) + [np.nan] * (top_n - n_features)
            imp_vals = list(df['importance_pct'].values) + [np.nan] * (top_n - n_features)
        else:
            feature_vals = df['feature'].values
            cat_vals = df['category'].values
            imp_vals = df['importance_pct'].values
        
        result_df[feature_col] = feature_vals[:top_n]
        result_df[cat_col] = cat_vals[:top_n]
        result_df[imp_col] = imp_vals[:top_n]
    
    return result_df


# =============================================
# Print Markdown Table
# =============================================

def print_markdown_table(df: pd.DataFrame, model_name: str):
    """Print DataFrame as Markdown table."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{model_name} Combined Top 15 Features by Horizon")
    logger.info("=" * 80)
    
    # Create markdown string
    md_lines = []
    
    # Header
    header = "| " + " | ".join(df.columns) + " |"
    md_lines.append(header)
    
    # Separator
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    md_lines.append(separator)
    
    # Rows
    for idx, row in df.iterrows():
        row_str = "| " + " | ".join([
            f"{val:.2f}" if isinstance(val, (int, float)) and pd.notna(val) else str(val) if pd.notna(val) else ""
            for val in row.values
        ]) + " |"
        md_lines.append(row_str)
    
    md_text = "\n".join(md_lines)
    
    # Print to console
    print("\n" + md_text + "\n")
    
    return md_text


# =============================================
# (3) Optional Visualization
# =============================================

def create_matrix_visualization(
    importance_dict: Dict[str, pd.DataFrame],
    model_name: str,
    top_n: int = 15,
    output_dir: Path = None
):
    """Create a matrix heatmap showing top features across horizons."""
    logger.info(f"Creating matrix visualization for {model_name}")
    
    horizons = ['1y', '3y', '5y', '10y']
    
    # Create matrix: rows = rank, columns = horizons
    matrix = np.full((top_n, len(horizons)), np.nan)
    feature_names = {}
    
    for col_idx, horizon in enumerate(horizons):
        if horizon not in importance_dict:
            continue
        
        df = importance_dict[horizon].head(top_n).reset_index(drop=True)
        
        for row_idx in range(min(top_n, len(df))):
            matrix[row_idx, col_idx] = df.iloc[row_idx]['importance_pct']
            feature_names[(row_idx, col_idx)] = df.iloc[row_idx]['feature']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get max value for normalization
    max_val = np.nanmax(matrix) if np.any(~np.isnan(matrix)) else 1
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val)
    
    # Add annotations
    for row_idx in range(top_n):
        for col_idx in range(len(horizons)):
            if not np.isnan(matrix[row_idx, col_idx]):
                feature = feature_names.get((row_idx, col_idx), "")
                # Shorten feature name
                short_name = feature[:10] + "..." if len(feature) > 10 else feature
                imp_val = matrix[row_idx, col_idx]
                
                # Choose text color based on background
                text_color = 'white' if imp_val > max_val * 0.5 else 'black'
                
                ax.text(col_idx, row_idx, f"{short_name}\n{imp_val:.1f}%",
                       ha='center', va='center', fontsize=6, color=text_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.3))
    
    # Set labels
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([h.upper() for h in horizons], fontsize=12)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"Rank {i+1}" for i in range(top_n)], fontsize=10)
    ax.set_xlabel('Horizon', fontsize=12)
    ax.set_ylabel('Rank', fontsize=12)
    ax.set_title(f'{model_name} Top {top_n} Features Across Horizons', fontsize=14, pad=20)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Importance (%)')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / f'{model_name.lower()}_combined_matrix.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


# =============================================
# (4) Summary Statistics
# =============================================

def compute_summary_statistics(
    rf_importance_dict: Dict[str, pd.DataFrame],
    xgb_importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 15
):
    """Compute and print summary statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    
    horizons = ['1y', '3y', '5y', '10y']
    
    # (1) Horizon-wise agreement between RF/XGB
    logger.info("\n1. Horizon-wise Agreement (Top 15 Overlap %):")
    for horizon in horizons:
        if horizon in rf_importance_dict and horizon in xgb_importance_dict:
            rf_top = set(rf_importance_dict[horizon].head(top_n)['feature'].values)
            xgb_top = set(xgb_importance_dict[horizon].head(top_n)['feature'].values)
            overlap = len(rf_top & xgb_top)
            overlap_pct = overlap / top_n * 100
            logger.info(f"   {horizon.upper()}: {overlap}/{top_n} features overlap ({overlap_pct:.1f}%)")
    
    # (2) Drivers that appear consistently across 3+ horizons
    logger.info("\n2. Features Appearing in 3+ Horizons:")
    
    # RF
    rf_horizon_features = {}
    for horizon in horizons:
        if horizon in rf_importance_dict:
            rf_horizon_features[horizon] = set(rf_importance_dict[horizon].head(top_n)['feature'].values)
    
    rf_consensus = {}
    for feature in set().union(*rf_horizon_features.values()):
        count = sum(1 for h in horizons if h in rf_horizon_features and feature in rf_horizon_features[h])
        if count >= 3:
            rf_consensus[feature] = count
    
    if rf_consensus:
        logger.info("   RF Model:")
        for feature, count in sorted(rf_consensus.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"     {feature}: appears in {count} horizons")
    else:
        logger.info("   RF Model: No features appear in 3+ horizons")
    
    # XGB
    xgb_horizon_features = {}
    for horizon in horizons:
        if horizon in xgb_importance_dict:
            xgb_horizon_features[horizon] = set(xgb_importance_dict[horizon].head(top_n)['feature'].values)
    
    xgb_consensus = {}
    for feature in set().union(*xgb_horizon_features.values()):
        count = sum(1 for h in horizons if h in xgb_horizon_features and feature in xgb_horizon_features[h])
        if count >= 3:
            xgb_consensus[feature] = count
    
    if xgb_consensus:
        logger.info("   XGB Model:")
        for feature, count in sorted(xgb_consensus.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"     {feature}: appears in {count} horizons")
    else:
        logger.info("   XGB Model: No features appear in 3+ horizons")
    
    # (3) Category distribution patterns
    logger.info("\n3. Category Distribution in Top 15 (by Horizon):")
    
    for model_name, importance_dict in [("RF", rf_importance_dict), ("XGB", xgb_importance_dict)]:
        logger.info(f"\n   {model_name} Model:")
        for horizon in horizons:
            if horizon in importance_dict:
                df = importance_dict[horizon].head(top_n)
                cat_counts = df['category'].value_counts().to_dict()
                logger.info(f"     {horizon.upper()}: {cat_counts}")


# =============================================
# Main Function
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Generate combined feature ranking table')
    parser.add_argument('--features-csv', type=str,
                       default='data/processed/nvda_features_extended_v2.csv',
                       help='Path to extended features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/feature_importance',
                       help='Base output directory')
    parser.add_argument('--top-n', type=int, default=15,
                       help='Number of top features to include')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    rankings_dir = output_dir / 'rankings'
    plots_dir = output_dir / 'plots'
    rankings_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Generate Combined Feature Ranking Table")
    logger.info("=" * 80)
    
    # Load data and train models
    logger.info("\n[Step 1] Loading data and computing importance...")
    df = load_features(Path(args.features_csv))
    
    horizons = {
        '1y': 4,
        '3y': 12,
        '5y': 20,
        '10y': 40
    }
    
    df = create_target_variables(df, horizons)
    X, feature_cols, classifications = prepare_features(df)
    
    # Add feature columns back to df
    for col in feature_cols:
        if col not in df.columns:
            df[col] = X[col]
    
    # Train models
    logger.info("\n[Step 2] Training models...")
    rf_importance = train_rf_models(df, feature_cols, horizons)
    xgb_importance = train_xgb_models(df, feature_cols, horizons)
    
    # Add categories to importance DataFrames
    for horizon in rf_importance.keys():
        rf_importance[horizon]['category'] = rf_importance[horizon]['feature'].apply(
            lambda f: classifications.get(f, 'Firm')
        )
        xgb_importance[horizon]['category'] = xgb_importance[horizon]['feature'].apply(
            lambda f: classifications.get(f, 'Firm')
        )
    
    # (1) Create combined table for RF
    logger.info("\n[Step 3] Creating combined table for RF...")
    rf_combined = create_combined_table(rf_importance, top_n=args.top_n, model_name="RF")
    rf_path = rankings_dir / 'rf_combined_top15.csv'
    rf_combined.to_csv(rf_path, index=False)
    logger.info(f"Saved: {rf_path}")
    
    # Print Markdown
    rf_md = print_markdown_table(rf_combined, "RF")
    
    # (2) Create combined table for XGB
    logger.info("\n[Step 4] Creating combined table for XGB...")
    xgb_combined = create_combined_table(xgb_importance, top_n=args.top_n, model_name="XGB")
    xgb_path = rankings_dir / 'xgb_combined_top15.csv'
    xgb_combined.to_csv(xgb_path, index=False)
    logger.info(f"Saved: {xgb_path}")
    
    # Print Markdown
    xgb_md = print_markdown_table(xgb_combined, "XGB")
    
    # (3) Optional visualization
    logger.info("\n[Step 5] Creating matrix visualizations...")
    create_matrix_visualization(rf_importance, "RF", top_n=args.top_n, output_dir=plots_dir)
    create_matrix_visualization(xgb_importance, "XGB", top_n=args.top_n, output_dir=plots_dir)
    
    # (4) Summary statistics
    logger.info("\n[Step 6] Computing summary statistics...")
    compute_summary_statistics(rf_importance, xgb_importance, top_n=args.top_n)
    
    # Save Markdown files
    with open(rankings_dir / 'rf_combined_top15.md', 'w') as f:
        f.write("# RF Combined Top 15 Features by Horizon\n\n")
        f.write(rf_md)
    
    with open(rankings_dir / 'xgb_combined_top15.md', 'w') as f:
        f.write("# XGB Combined Top 15 Features by Horizon\n\n")
        f.write(xgb_md)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ All steps completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

