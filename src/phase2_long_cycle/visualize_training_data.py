"""
Visualize training data: prices, targets, and features.

Creates plots to visualize:
1. Price time series
2. Target variable distributions
3. Target variable time series
4. Feature-target relationships
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data CSV."""
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def plot_price_time_series(df: pd.DataFrame, output_dir: Path):
    """Plot price time series."""
    logger.info("Plotting price time series")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df.index, df['price_q'], linewidth=2, color='#1f77b4')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_title('NVDA Quarterly Price Time Series', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    plt.tight_layout()
    output_path = output_dir / 'price_time_series.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_target_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot distributions of target variables."""
    logger.info("Plotting target distributions")
    
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    target_cols = [col for col in target_cols if col in df.columns]
    
    n_cols = len(target_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(target_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=20, alpha=0.7, color=f'C{i}', edgecolor='black')
        axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
        axes[i].set_xlabel(f'{col} (log return)', fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'{col}\n(n={len(data)}, μ={data.mean():.3f}, σ={data.std():.3f})', fontsize=11)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Target Variable Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'target_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_target_time_series(df: pd.DataFrame, output_dir: Path):
    """Plot target variables over time."""
    logger.info("Plotting target time series")
    
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    target_cols = [col for col in target_cols if col in df.columns]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for col in target_cols:
        data = df[col].dropna()
        ax.plot(data.index, data.values, marker='o', markersize=4, linewidth=1.5, 
               label=col, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Log Return', fontsize=12)
    ax.set_title('Target Variables Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    output_path = output_dir / 'target_time_series.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_feature_target_correlations(df: pd.DataFrame, output_dir: Path):
    """Plot correlations between features and targets."""
    logger.info("Plotting feature-target correlations")
    
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    target_cols = [col for col in target_cols if col in df.columns]
    feature_cols = [col for col in df.columns 
                   if not col.startswith('y_') and col != 'price_q']
    
    # Calculate correlations
    corr_data = []
    for target in target_cols:
        for feature in feature_cols:
            corr = df[[target, feature]].corr().iloc[0, 1]
            if not np.isnan(corr):
                corr_data.append({
                    'target': target,
                    'feature': feature,
                    'correlation': corr
                })
    
    corr_df = pd.DataFrame(corr_data)
    
    # Pivot for heatmap
    corr_pivot = corr_df.pivot(index='feature', columns='target', values='correlation')
    
    # Sort columns by horizon (4q, 8q, 12q, 16q, 20q)
    col_order = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    col_order = [col for col in col_order if col in corr_pivot.columns]
    corr_pivot = corr_pivot[col_order]
    
    # Sort rows by absolute correlation with longest available horizon
    longest_horizon = col_order[-1] if col_order else 'y_log_12q'
    if longest_horizon in corr_pivot.columns:
        corr_pivot = corr_pivot.reindex(
            corr_pivot[longest_horizon].abs().sort_values(ascending=False).index
        )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, max(10, len(corr_pivot) * 0.3)))
    
    sns.heatmap(
        corr_pivot,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Feature-Target Correlations (1年 → 5年)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Variable (Horizon)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    # Update x-axis labels to show horizon in years
    x_labels = []
    for col in corr_pivot.columns:
        horizon_q = int(col.split('_')[-1].replace('q', ''))
        horizon_years = horizon_q // 4
        x_labels.append(f'{col}\n({horizon_years}yr)')
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    
    plt.tight_layout()
    output_path = output_dir / 'feature_target_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_key_features_vs_targets(df: pd.DataFrame, output_dir: Path):
    """Plot key features vs targets."""
    logger.info("Plotting key features vs targets")
    
    # Select key features
    key_features = ['rev_ttm', 'gross_margin_pct', 'rev_yoy', 'rev_cagr_3y', 'net_margin_pct']
    key_features = [f for f in key_features if f in df.columns]
    target = 'y_log_12q'
    
    if target not in df.columns:
        logger.warning(f"Target {target} not found, skipping")
        return
    
    n_features = len(key_features)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(key_features):
        if i >= len(axes):
            break
        
        data = df[[feature, target]].dropna()
        
        axes[i].scatter(data[feature], data[target], alpha=0.6, s=50)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel(target, fontsize=10)
        axes[i].set_title(f'{feature} vs {target}', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation
        corr = data[feature].corr(data[target])
        axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for i in range(len(key_features), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Key Features vs Target (y_log_12q)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'key_features_vs_targets.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def create_summary_plot(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive summary plot."""
    logger.info("Creating summary plot")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Price time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['price_q'], linewidth=2, color='#1f77b4')
    ax1.set_title('NVDA Quarterly Price', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Target time series
    ax2 = fig.add_subplot(gs[1, :])
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    target_cols = [col for col in target_cols if col in df.columns]
    for col in target_cols:
        data = df[col].dropna()
        ax2.plot(data.index, data.values, marker='o', markersize=3, 
                linewidth=1.5, label=col, alpha=0.7)
    ax2.set_title('Target Variables Over Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Log Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Target distributions
    ax3 = fig.add_subplot(gs[2, 0])
    target_cols = ['y_log_4q', 'y_log_8q', 'y_log_12q', 'y_log_16q', 'y_log_20q']
    target_cols = [col for col in target_cols if col in df.columns]
    for col in target_cols:
        data = df[col].dropna()
        ax3.hist(data, bins=15, alpha=0.5, label=col)
    ax3.set_title('Target Distributions', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Log Return')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Revenue vs Target
    ax4 = fig.add_subplot(gs[2, 1])
    data = df[['rev_ttm', 'y_log_12q']].dropna()
    ax4.scatter(data['rev_ttm'], data['y_log_12q'], alpha=0.6, s=50)
    ax4.set_title('Revenue TTM vs y_log_12q', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Revenue TTM')
    ax4.set_ylabel('y_log_12q')
    corr = data['rev_ttm'].corr(data['y_log_12q'])
    ax4.text(0.05, 0.95, f'Corr: {corr:.3f}', 
            transform=ax4.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('NVDA Long-Cycle Training Data Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'summary_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def visualize_training_data(
    data_path: Path,
    output_dir: Path,
    simple: bool = True
):
    """
    Create visualizations for training data.
    
    Args:
        data_path: Path to training CSV
        output_dir: Directory to save plots
        simple: If True, only create summary plot. If False, create all plots.
    """
    logger.info("=" * 60)
    logger.info("Creating visualizations for training data")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_training_data(data_path)
    
    if simple:
        # Only create comprehensive summary plot
        create_summary_plot(df, output_dir)
        logger.info("Created summary plot only (use --all for all plots)")
    else:
        # Create all plots
        plot_price_time_series(df, output_dir)
        plot_target_distributions(df, output_dir)
        plot_target_time_series(df, output_dir)
        plot_feature_target_correlations(df, output_dir)
        plot_key_features_vs_targets(df, output_dir)
        create_summary_plot(df, output_dir)
    
    logger.info("=" * 60)
    logger.info("Visualization complete!")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize training data"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/processed/nvda_long_cycle_train.csv'),
        help='Path to training CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/figs/phase2_long_cycle'),
        help='Directory to save plots'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all plots (default: only summary plot)'
    )
    
    args = parser.parse_args()
    
    visualize_training_data(args.data, args.output, simple=not args.all)


if __name__ == '__main__':
    main()

