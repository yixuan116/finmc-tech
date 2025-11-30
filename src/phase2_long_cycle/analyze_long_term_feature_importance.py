"""
Analyze feature importance for long-term (5-10 years) predictions, with focus on FCF features.

This script:
1. Performs feature importance analysis for 5-year, 7-year, and 10-year targets
2. Focuses on FCF feature importance changes in long-term predictions
3. Generates comparison reports
"""

import argparse
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


def prepare_features_and_target(df: pd.DataFrame, target: str):
    """Prepare features and target variables"""
    exclude_cols = {
        "period_end_date", "price_q",
        "y_log_4q", "y_log_8q", "y_log_12q", "y_log_16q", "y_log_20q",
        "y_log_24q", "y_log_28q", "y_log_32q", "y_log_36q", "y_log_40q"
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target]
    
    # Remove samples with missing target values
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y, X.columns.tolist()


def train_and_analyze(
    df: pd.DataFrame,
    target: str,
    n_estimators: int = 600,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Train model and analyze feature importance"""
    logger.info(f"Analyzing target: {target}")
    
    X, y, feature_names = prepare_features_and_target(df, target)
    
    if len(X) < 10:
        logger.warning(f"Insufficient samples ({len(X)}), skipping {target}")
        return None
    
    logger.info(f"  Number of samples: {len(X)}")
    logger.info(f"  Number of features: {len(feature_names)}")
    logger.info(f"  Target statistics: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    logger.info(f"  Train R²: {train_r2:.4f}")
    logger.info(f"  Test  R²: {test_r2:.4f}")
    
    # Get feature importance
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    return {
        "target": target,
        "n_samples": len(X),
        "train_r2": train_r2,
        "test_r2": test_r2,
        "feature_importance": fi,
        "model": rf
    }


def analyze_fcf_importance(results: dict) -> pd.DataFrame:
    """Analyze FCF feature importance"""
    fi = results["feature_importance"]
    
    # Filter FCF-related features
    fcf_features = fi[fi['feature'].str.contains('fcf', case=False, na=False)].copy()
    
    if len(fcf_features) == 0:
        return pd.DataFrame()
    
    # Add rank information
    fcf_features['rank'] = fcf_features['feature'].apply(
        lambda x: fi.index[fi['feature'] == x].tolist()[0] + 1
    )
    
    # Calculate total importance
    fcf_total = fcf_features['importance'].sum()
    all_total = fi['importance'].sum()
    fcf_pct = fcf_total / all_total * 100
    
    fcf_features['fcf_total'] = fcf_total
    fcf_features['fcf_pct_of_total'] = fcf_pct
    fcf_features['target'] = results["target"]
    
    return fcf_features


def create_comparison_report(all_results: list, output_dir: Path):
    """Create comparison report"""
    logger.info("Creating comparison report")
    
    # Collect FCF feature importance from all results
    fcf_summary = []
    for result in all_results:
        fcf_data = analyze_fcf_importance(result)
        if len(fcf_data) > 0:
            fcf_summary.append(fcf_data)
    
    if len(fcf_summary) == 0:
        logger.warning("No FCF feature data found")
        return
    
    fcf_df = pd.concat(fcf_summary, ignore_index=True)
    
    # Create summary table
    summary_rows = []
    for result in all_results:
        target = result["target"]
        quarters = int(target.split('_')[-1].replace('q', ''))
        years = quarters / 4
        
        fcf_data = analyze_fcf_importance(result)
        fcf_total = fcf_data['fcf_total'].iloc[0] if len(fcf_data) > 0 else 0
        fcf_pct = fcf_data['fcf_pct_of_total'].iloc[0] if len(fcf_data) > 0 else 0
        
        # Find top-ranked FCF feature
        top_fcf = fcf_data.sort_values('rank').iloc[0] if len(fcf_data) > 0 else None
        
        summary_rows.append({
            'target': target,
            'years': years,
            'n_samples': result['n_samples'],
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'fcf_total_importance': fcf_total,
            'fcf_pct_of_total': fcf_pct,
            'top_fcf_feature': top_fcf['feature'] if top_fcf is not None else None,
            'top_fcf_rank': top_fcf['rank'] if top_fcf is not None else None,
            'top_fcf_importance': top_fcf['importance'] if top_fcf is not None else None,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary table
    summary_path = output_dir / "fcf_importance_by_horizon.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary table saved to: {summary_path}")
    
    # Print report
    print("\n" + "=" * 80)
    print("FCF Feature Importance - Long-Term Prediction Comparison Report")
    print("=" * 80)
    
    for _, row in summary_df.iterrows():
        print(f"\n{row['target']} ({row['years']:.1f}-year prediction):")
        print(f"  Number of samples: {row['n_samples']}")
        print(f"  Model performance: Train R²={row['train_r2']:.4f}, Test R²={row['test_r2']:.4f}")
        print(f"  FCF total importance: {row['fcf_total_importance']:.4f} ({row['fcf_pct_of_total']:.2f}% of total)")
        if row['top_fcf_feature']:
            print(f"  Top FCF feature: {row['top_fcf_feature']} (rank {row['top_fcf_rank']}, importance {row['top_fcf_importance']:.4f})")
    
    # Create visualization
    create_fcf_importance_plot(summary_df, output_dir)
    
    return summary_df


def create_fcf_importance_plot(summary_df: pd.DataFrame, output_dir: Path):
    """Create FCF importance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FCF Feature Importance Across Long-Term Horizons', fontsize=16, fontweight='bold')
    
    # 1. FCF total importance vs prediction horizon
    ax1 = axes[0, 0]
    ax1.plot(summary_df['years'], summary_df['fcf_total_importance'], 
             marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Prediction Horizon (Years)', fontsize=11)
    ax1.set_ylabel('FCF Total Importance', fontsize=11)
    ax1.set_title('FCF Total Importance vs Horizon', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(summary_df['years'])
    
    # 2. FCF percentage of total importance
    ax2 = axes[0, 1]
    ax2.plot(summary_df['years'], summary_df['fcf_pct_of_total'], 
             marker='s', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Prediction Horizon (Years)', fontsize=11)
    ax2.set_ylabel('FCF % of Total Importance', fontsize=11)
    ax2.set_title('FCF % of Total Importance vs Horizon', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(summary_df['years'])
    
    # 3. Top FCF feature rank
    ax3 = axes[1, 0]
    ax3.plot(summary_df['years'], summary_df['top_fcf_rank'], 
             marker='^', linewidth=2, markersize=8, color='green', linestyle='--')
    ax3.set_xlabel('Prediction Horizon (Years)', fontsize=11)
    ax3.set_ylabel('Top FCF Feature Rank', fontsize=11)
    ax3.set_title('Top FCF Feature Rank vs Horizon', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_xticks(summary_df['years'])
    ax3.invert_yaxis()  # Lower rank is better, so invert y-axis
    
    # 4. Model performance
    ax4 = axes[1, 1]
    ax4.plot(summary_df['years'], summary_df['train_r2'], 
             marker='o', label='Train R²', linewidth=2, markersize=8)
    ax4.plot(summary_df['years'], summary_df['test_r2'], 
             marker='s', label='Test R²', linewidth=2, markersize=8)
    ax4.set_xlabel('Prediction Horizon (Years)', fontsize=11)
    ax4.set_ylabel('R² Score', fontsize=11)
    ax4.set_title('Model Performance vs Horizon', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xticks(summary_df['years'])
    
    plt.tight_layout()
    
    plot_path = output_dir / "fcf_importance_long_term_analysis.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization plot saved to: {plot_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze feature importance for long-term (5-10 years) predictions"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('outputs/data/training/training_data_extended_10y.csv'),
        help='Path to training CSV with extended targets'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/feature_importance/data/long_term'),
        help='Output directory'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['y_log_20q', 'y_log_28q', 'y_log_40q'],  # 5-year, 7-year, 10-year
        help='Target variables to analyze'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=600,
        help='Number of trees in RandomForest'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Analyze each target variable
    all_results = []
    for target in args.targets:
        if target not in df.columns:
            logger.warning(f"Target variable {target} does not exist, skipping")
            continue
        
        result = train_and_analyze(
            df, target,
            n_estimators=args.n_estimators
        )
        
        if result is not None:
            all_results.append(result)
            
            # Save detailed feature importance
            fi_path = args.output / f"feature_importance_{target}.csv"
            result["feature_importance"].to_csv(fi_path, index=False)
            logger.info(f"  Feature importance saved to: {fi_path}")
    
    # Create comparison report
    if len(all_results) > 0:
        summary_df = create_comparison_report(all_results, args.output)
        
        logger.info("=" * 80)
        logger.info("Long-term Feature Importance analysis completed!")
        logger.info("=" * 80)
    else:
        logger.error("No target variables were successfully analyzed")


if __name__ == '__main__':
    main()

