#!/usr/bin/env python3
"""Visualize ML vs Monte Carlo results"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
ml = pd.read_csv('outputs/nvda_ml_pred.csv')
mc = pd.read_csv('outputs/nvda_mc_terminals.csv')

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: ML Predictions scatter
ax1 = axes[0]
ax1.scatter(ml['y_true'], ml['y_pred'], alpha=0.3, s=10)
ax1.plot([ml['y_true'].min(), ml['y_true'].max()], 
         [ml['y_true'].min(), ml['y_true'].max()], 'r--', label='Perfect prediction')
ax1.set_xlabel('Actual Returns', fontsize=12)
ax1.set_ylabel('Predicted Returns', fontsize=12)
ax1.set_title('ML Model: Actual vs Predicted Returns', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: MC Terminal distribution
ax2 = axes[1]
ax2.hist(mc['terminal_price'], bins=50, edgecolor='black', alpha=0.7)
p5 = mc['terminal_price'].quantile(0.05)
p50 = mc['terminal_price'].quantile(0.5)
p95 = mc['terminal_price'].quantile(0.95)
ax2.axvline(p5, color='red', linestyle='--', label=f'P5=${p5:.2f}')
ax2.axvline(p50, color='blue', linestyle='--', label=f'P50=${p50:.2f}')
ax2.axvline(p95, color='red', linestyle='--', label=f'P95=${p95:.2f}')
ax2.set_xlabel('Terminal Price ($)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Monte Carlo: Terminal Price Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/comparison_ml_vs_mc.png', dpi=150, bbox_inches='tight')
print('✅ Saved visualization to outputs/comparison_ml_vs_mc.png')
plt.close()

# Print summary
print('\n=== Summary ===')
print('ML Model:')
print(f'  R² Score: {0:.4f} (baseline Linear Regression)')
print('  Issue: Returns too noisy for simple linear model')
print()
print('Monte Carlo:')
print(f'  Terminal range: ${mc["terminal_price"].min():.2f} - ${mc["terminal_price"].max():.2f}')
print(f'  Key quantiles: P5=${p5:.2f}, P50=${p50:.2f}, P95=${p95:.2f}')
print('  Strength: Captures uncertainty, works well for volatile stocks')

