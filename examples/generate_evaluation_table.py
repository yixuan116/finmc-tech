#!/usr/bin/env python3
"""
Generate comprehensive evaluation table for rolling forecast
"""
import pandas as pd
import numpy as np

# Load results
mc_df = pd.read_csv("outputs/results_mc.csv")
forecast_df = pd.read_csv("outputs/results_forecast.csv")

# Load actual data to compute year-over-year direction
df = pd.read_csv("examples/NVDA_data_2010_2025.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df = df[df['ticker'] == 'NVDA'].copy()
df['year'] = df['date'].dt.year

# Compute year-over-year direction (up or down)
year_directions = {}
for year in mc_df['year']:
    year_df = df[df['year'] == year]
    if len(year_df) > 0:
        actual_return = (year_df['close'].iloc[-1] / year_df['close'].iloc[0] - 1) * 100
        year_directions[year] = '↑' if actual_return > 0 else '↓'
    else:
        year_directions[year] = '?'

# Get ML sign predictions (using Expanding-Long setup as primary)
ml_directions = {}
for year in forecast_df['year'].unique():
    year_forecast = forecast_df[forecast_df['year'] == year]
    expanding = year_forecast[year_forecast['setup'] == 'Expanding-Long']
    if len(expanding) > 0:
        # If we have sign_acc, we can infer direction prediction
        # For now, use MU sign as proxy
        year_mc = mc_df[mc_df['year'] == year]
        if len(year_mc) > 0:
            mu = year_mc['MU_annualized'].iloc[0]
            ml_directions[year] = '↑' if mu > 0 else '↓'

# Build evaluation table
eval_table = []
for year in mc_df['year']:
    row = mc_df[mc_df['year'] == year].iloc[0]
    
    # Get actual year data to check coverage
    year_df = df[df['year'] == year]
    if len(year_df) > 0:
        actual_close = year_df['close'].iloc[-1]
        
        # Check if actual price is within P5-P95 range
        in_range = '✅' if row['P5'] <= actual_close <= row['P95'] else '❌'
        
        eval_table.append({
            'Year': int(year),
            'Actual Close ($)': f"{actual_close:.2f}",
            'Predicted (P50)': f"{row['P50']:.2f}",
            '±90% Band': f"[{row['P5']:.1f}, {row['P95']:.1f}]",
            '% Error': f"{row['prediction_error']:.1f}%",
            'Sign Pred': ml_directions.get(year, '?'),
            'Actual Dir': year_directions.get(year, '?'),
            'Coverage': in_range
        })

eval_df = pd.DataFrame(eval_table)

# Print formatted table
print("\n" + "="*120)
print("📊 NVDA Rolling Forecast Evaluation Table (2018-2025)")
print("="*120)
print(eval_df.to_string(index=False))

# Save to CSV
eval_df.to_csv("outputs/evaluation_table.csv", index=False)
print(f"\n✓ Saved: outputs/evaluation_table.csv")

# Summary statistics
print("\n" + "="*120)
print("Summary Statistics")
print("="*120)
print(f"✓ Average |% Error|: {abs(eval_df['% Error'].str.rstrip('%').astype(float)).mean():.1f}%")
print(f"✓ Coverage Rate: {(eval_df['Coverage'] == '✅').sum()}/{len(eval_df)} = {(eval_df['Coverage'] == '✅').mean()*100:.1f}%")
print(f"✓ Best Year (lowest error): {eval_df.loc[abs(eval_df['% Error'].str.rstrip('%').astype(float)).idxmin(), 'Year']}")
print(f"✓ Worst Year (highest error): {eval_df.loc[abs(eval_df['% Error'].str.rstrip('%').astype(float)).idxmax(), 'Year']}")
print("="*120 + "\n")
