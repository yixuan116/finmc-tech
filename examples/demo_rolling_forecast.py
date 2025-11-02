#!/usr/bin/env python3
"""
Rolling Time-Series Forecast for NVDA (2018-2025)
- 3 training window setups: Expanding-Long / Sliding-5y / Sliding-3y
- Year-by-year MC simulations with P5/P50/P95/VaR/CVaR
- Comparative visualizations: sign_acc and bandwidth trends
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import json

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("✓ Numba available - HPC backend enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba not available - using serial backend")

# Paths
CSV_PATH = "examples/NVDA_data_2010_2025.csv"

# ============================================================================
# 1. Load & Preprocess Data
# ============================================================================
print("\n" + "="*70)
print("1. Loading NVDA data (2010-2025)")
print("="*70)

df = pd.read_csv(CSV_PATH)
df = df[df["ticker"] == "NVDA"].copy()
# Convert date column to datetime (handling tz-aware dates)
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df["mu_rolling"] = df["log_returns"].rolling(30).mean()
df["sigma_rolling"] = df["log_returns"].rolling(30).std()
df = df.dropna(subset=["mu_rolling", "sigma_rolling"])

print(f"✓ Loaded {len(df)} days of data")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# 2. Rolling Forecast: Year-by-Year Predictions
# ============================================================================
print("\n" + "="*70)
print("2. Rolling Forecast with 3 window setups")
print("="*70)

df['year'] = df['date'].dt.year
test_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
train_windows = {
    'Expanding-Long': lambda df, test_year: df[df['year'] < test_year],
    'Sliding-5y': lambda df, test_year: df[(df['year'] >= test_year - 5) & (df['year'] < test_year)],
    'Sliding-3y': lambda df, test_year: df[(df['year'] >= test_year - 3) & (df['year'] < test_year)]
}

WINDOW = 30
results_forecast = []

for test_year in test_years:
    test_df = df[df['year'] == test_year]
    if len(test_df) < 30:
        continue
    
    for setup_name, get_train_window in train_windows.items():
        train_df = get_train_window(df, test_year)
        if len(train_df) < WINDOW * 2:
            continue
        
        train_log_rets = train_df['log_returns'].values
        test_log_rets = test_df['log_returns'].values
        
        X_train, y_train = [], []
        for i in range(len(train_log_rets) - WINDOW):
            X_train.append(train_log_rets[i:i+WINDOW])
            y_train.append(train_log_rets[i+WINDOW])
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        X_test, y_test = [], []
        for i in range(len(test_log_rets) - WINDOW):
            X_test.append(test_log_rets[i:i+WINDOW])
            y_test.append(test_log_rets[i+WINDOW])
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        if len(y_test) == 0:
            continue
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        sign_acc = np.mean(np.sign(y_test) == np.sign(y_pred))
        ic = np.corrcoef(y_test, y_pred)[0, 1]
        
        results_forecast.append({
            'year': test_year,
            'setup': setup_name,
            'R²': r2,
            'MAE': mae,
            'sign_acc': sign_acc,
            'IC': ic,
            'n_samples': len(y_test)
        })
        
        print(f"  {test_year}×{setup_name}: R²={r2:.4f}, sign_acc={sign_acc:.3f}")

results_df = pd.DataFrame(results_forecast)
print(f"\n✓ Generated {len(results_df)} forecast results")

# ============================================================================
# 3. Monte Carlo: Year-by-Year Simulations
# ============================================================================
print("\n" + "="*70)
print("3. Monte Carlo simulations for each year")
print("="*70)

def simulate_paths_serial(mu, sigma, s0, steps=252, n_paths=5000, dt=1/252):
    np.random.seed(42)
    paths = np.zeros((n_paths, steps))
    for i in range(n_paths):
        s = s0
        for j in range(steps):
            z = np.random.normal()
            s *= np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            paths[i, j] = s
    return paths

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def simulate_paths_numba(mu, sigma, s0, steps=252, n_paths=5000, dt=1/252):
        out = np.empty((n_paths, steps))
        for i in prange(n_paths):
            s = s0
            for j in range(steps):
                z = np.random.normal()
                s *= np.exp((mu - 0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*z)
                out[i, j] = s
        return out
else:
    simulate_paths_numba = simulate_paths_serial

mc_func = simulate_paths_numba
results_mc = []

for test_year in test_years:
    year_df = df[df['year'] == test_year]
    if len(year_df) < 10:
        continue
    
    first_day = year_df.iloc[0]
    S0 = first_day['close']
    MU = first_day['mu_rolling']
    SIGMA = first_day['sigma_rolling']
    
    if np.isnan(MU) or np.isnan(SIGMA):
        continue
    
    paths = mc_func(MU, SIGMA, S0, steps=252, n_paths=5000, dt=1/252)
    terminals = paths[:, -1]
    
    p5, p50, p95 = np.percentile(terminals, [5, 50, 95])
    var_5 = np.percentile(terminals, 5)
    cvar_5 = terminals[terminals <= var_5].mean()
    bandwidth = (p95 - p5) / p50
    
    results_mc.append({
        'year': test_year,
        'S0': S0,
        'MU_annualized': MU * 252,
        'SIGMA_annualized': SIGMA * np.sqrt(252),
        'P5': p5,
        'P50': p50,
        'P95': p95,
        'VaR_5': var_5,
        'CVaR_5': cvar_5,
        'bandwidth': bandwidth
    })
    
    print(f"  {test_year}: P50=${p50:.2f}, bandwidth={bandwidth:.4f}")

results_mc_df = pd.DataFrame(results_mc)
print(f"\n✓ Generated {len(results_mc_df)} MC results")

# ============================================================================
# 4. Create Visualizations
# ============================================================================
print("\n" + "="*70)
print("4. Generating visualizations")
print("="*70)

# Get actual year-end prices for comparison
actual_prices = []
for year in results_mc_df['year']:
    year_df = df[df['year'] == year]
    if len(year_df) > 0:
        actual_prices.append(year_df['close'].iloc[-1])  # Last trading day of year
    else:
        actual_prices.append(np.nan)

results_mc_df['actual_price'] = actual_prices
results_mc_df['prediction_error'] = (results_mc_df['P50'] - results_mc_df['actual_price']) / results_mc_df['actual_price'] * 100

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# Plot 1: Sign accuracy by setup
ax1 = plt.subplot(3, 2, 1)
for setup in results_df['setup'].unique():
    subset = results_df[results_df['setup'] == setup]
    ax1.plot(subset['year'], subset['sign_acc'], marker='o', label=setup, linewidth=2, markersize=6)
ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Sign Accuracy')
ax1.set_title('Model Stability: Sign Accuracy Over Years')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim([0.4, 0.6])

# Plot 2: Bandwidth (uncertainty)
ax2 = plt.subplot(3, 2, 2)
ax2.plot(results_mc_df['year'], results_mc_df['bandwidth'], marker='s', color='red', linewidth=2, markersize=8)
ax2.fill_between(results_mc_df['year'], 0, results_mc_df['bandwidth'], alpha=0.2, color='red')
ax2.set_xlabel('Year')
ax2.set_ylabel('Bandwidth = (P95-P5)/P50')
ax2.set_title('Market Uncertainty: Bandwidth Over Years')
ax2.grid(alpha=0.3)

# Plot 3: Actual vs Predicted (MC P50) Prices
ax3 = plt.subplot(3, 2, 3)
ax3.plot(results_mc_df['year'], results_mc_df['actual_price'], marker='o', label='Actual', linewidth=3, markersize=10, color='black')
ax3.plot(results_mc_df['year'], results_mc_df['P50'], marker='s', label='MC Prediction (P50)', linewidth=2, markersize=8, color='blue')
ax3.fill_between(results_mc_df['year'], results_mc_df['P5'], results_mc_df['P95'], alpha=0.2, color='blue', label='90% CI')
ax3.set_xlabel('Year')
ax3.set_ylabel('Price ($)')
ax3.set_title('Actual vs Predicted Stock Prices')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Prediction Error (%)
ax4 = plt.subplot(3, 2, 4)
colors = ['green' if abs(x) < 20 else 'orange' if abs(x) < 50 else 'red' for x in results_mc_df['prediction_error']]
ax4.bar(results_mc_df['year'], results_mc_df['prediction_error'], color=colors, alpha=0.7)
ax4.axhline(0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Year')
ax4.set_ylabel('Prediction Error (%)')
ax4.set_title('MC Forecast Accuracy: (P50 - Actual) / Actual × 100%')
ax4.grid(alpha=0.3, axis='y')

# Plot 5: R² by setup
ax5 = plt.subplot(3, 2, 5)
for setup in results_df['setup'].unique():
    subset = results_df[results_df['setup'] == setup]
    ax5.plot(subset['year'], subset['R²'], marker='o', label=setup, linewidth=2, markersize=6)
ax5.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Zero')
ax5.set_xlabel('Year')
ax5.set_ylabel('R² Score')
ax5.set_title('ML Model Performance: R² Over Years')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Summary table
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')
summary_text = "ML Model: Linear Regression with 30-day rolling windows\n\n"
summary_text += "Training Setups:\n"
summary_text += "  • Expanding-Long: All historical data before test year\n"
summary_text += "  • Sliding-5y: 5-year rolling window\n"
summary_text += "  • Sliding-3y: 3-year rolling window\n\n"
summary_text += "Key Findings:\n"
mean_sign_acc = results_df['sign_acc'].mean()
mean_r2 = results_df['R²'].mean()
max_error = results_mc_df['prediction_error'].abs().max()
summary_text += f"  • Mean sign accuracy: {mean_sign_acc:.3f}\n"
summary_text += f"  • Mean R²: {mean_r2:.4f}\n"
summary_text += f"  • Max MC error: {max_error:.1f}%\n"
summary_text += f"  • Best year: {results_df.loc[results_df['sign_acc'].idxmax(), 'year']}\n"
ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig("outputs/comparison_yearly.png", dpi=150, bbox_inches='tight')
print("✓ Saved: outputs/comparison_yearly.png")
plt.close()

# ============================================================================
# 5. Save Results
# ============================================================================
results_df.to_csv("outputs/results_forecast.csv", index=False)
results_mc_df.to_csv("outputs/results_mc.csv", index=False)
results_all = results_df.merge(results_mc_df[['year', 'bandwidth']], on='year', how='outer')
results_all.to_csv("outputs/results_all.csv", index=False)

print("✓ Saved: outputs/results_forecast.csv")
print("✓ Saved: outputs/results_mc.csv")
print("✓ Saved: outputs/results_all.csv")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Rolling forecast: {len(results_df)} year×setup combinations")
print(f"✓ Monte Carlo simulations: {len(results_mc_df)} years")
print(f"✓ Visualization: outputs/comparison_yearly.png")
print("="*70)
print("\n✅ Done! All results saved to outputs/")
