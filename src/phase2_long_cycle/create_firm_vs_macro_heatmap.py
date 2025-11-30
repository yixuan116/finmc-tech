#!/usr/bin/env python3
"""
Create Firm vs Macro Feature Importance Comparison Heatmap

This script generates the heatmap showing the importance percentage of
Firm Level, Macro, and Other features across different prediction horizons.

The heatmap is saved to:
    outputs/feature_importance/plots/long_term/firm_vs_macro_importance_comparison.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_firm_vs_macro_heatmap():
    """Create the firm vs macro importance comparison heatmap."""
    
    # Load summary data
    summary_path = Path("outputs/feature_importance/data/long_term/summary/firm_vs_macro_importance_summary.csv")
    
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        print("Please run the firm vs macro analysis first.")
        return
    
    df = pd.read_csv(summary_path)
    
    # Prepare heatmap data - only include categories with non-zero values
    heatmap_dict = {
        'Firm Level': df['firm_level_pct'].values,
        'Macro': df['macro_pct'].values
    }
    
    # Only include "Other" if it has any non-zero values
    if df['other_pct'].sum() > 0.01:  # Threshold: > 0.01% total
        heatmap_dict['Other'] = df['other_pct'].values
    
    heatmap_data = pd.DataFrame(
        heatmap_dict,
        index=[f"{int(y)}y" for y in df['years']]
    )
    
    # Transpose so categories are rows and horizons are columns
    heatmap_data = heatmap_data.T
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.6)))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Importance %'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title('Feature Importance % by Category Across Horizons', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Prediction Horizon', fontsize=12)
    ax.set_ylabel('Feature Category', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("outputs/feature_importance/plots/long_term/firm_vs_macro_importance_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved to: {output_path}")
    print(f"\nData summary:")
    print(f"  - Firm Level importance: {df['firm_level_pct'].mean():.1f}% (average)")
    print(f"  - Macro importance: {df['macro_pct'].mean():.1f}% (average)")
    if df['other_pct'].sum() > 0.01:
        print(f"  - Other importance: {df['other_pct'].mean():.1f}% (average)")
    else:
        print(f"  - Other importance: 0.0% (excluded from heatmap)")

if __name__ == '__main__':
    create_firm_vs_macro_heatmap()

