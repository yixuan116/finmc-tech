#!/usr/bin/env python3
"""
Prepare NVDA stock price data aggregated by year for PPT presentation.

This script:
1. Loads NVDA daily stock data
2. Aggregates data by year (average, year-end, high, low, returns, volatility)
3. Creates comprehensive visualizations
4. Saves data and plots for PPT use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Try to find NVDA data file
DATA_PATHS = [
    "data/raw/nvda_yf_daily_1999_2025.csv",
    "data/raw/nvda_crsp_daily_1999_2025.csv",
    "data/processed/nvda_daily_wrds_1999_2025.csv",
    "examples/NVDA_data_2010_2025.csv",
]

def find_nvda_data():
    """Find and load NVDA data file."""
    for path in DATA_PATHS:
        if Path(path).exists():
            print(f"✓ Found data file: {path}")
            return path
    
    # If no file found, try to fetch from Yahoo Finance
    print("⚠ No local data file found. Attempting to fetch from Yahoo Finance...")
    try:
        from src.data.fetch import fetch_stock_data
        data = fetch_stock_data("NVDA", start="1999-01-01")
        data['ticker'] = 'NVDA'
        # Save for future use
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        data.to_csv("data/raw/nvda_yf_daily_1999_2025.csv", index=False)
        print("✓ Fetched and saved data to data/raw/nvda_yf_daily_1999_2025.csv")
        return "data/raw/nvda_yf_daily_1999_2025.csv"
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        raise

def load_and_prepare_data(data_path):
    """Load and prepare NVDA data."""
    print(f"\n{'='*70}")
    print("Loading NVDA Data")
    print(f"{'='*70}")
    
    df = pd.read_csv(data_path)
    
    # Handle different column name formats
    date_col = None
    for col in ['date', 'Date', 'DATE']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("No date column found in data")
    
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.sort_values(date_col)
    
    # Filter NVDA if ticker column exists
    if 'ticker' in df.columns:
        df = df[df['ticker'] == 'NVDA'].copy()
    
    # Handle different price column names
    price_col = None
    for col in ['close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError("No price column found in data")
    
    # Calculate returns if not present
    if 'log_returns' not in df.columns and 'returns' not in df.columns:
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        df['returns'] = df[price_col].pct_change()
    
    # Add year column
    df['year'] = df[date_col].dt.year
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"  Year range: {df['year'].min()} to {df['year'].max()}")
    
    return df, date_col, price_col

def aggregate_by_year(df, date_col, price_col):
    """Aggregate data by year."""
    print(f"\n{'='*70}")
    print("Aggregating Data by Year")
    print(f"{'='*70}")
    
    yearly_data = []
    
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year].copy()
        
        if len(year_df) == 0:
            continue
        
        # Price statistics
        year_start_price = year_df[price_col].iloc[0]
        year_end_price = year_df[price_col].iloc[-1]
        year_avg_price = year_df[price_col].mean()
        year_high = year_df[price_col].max()
        year_low = year_df[price_col].min()
        
        # Returns
        year_return = (year_end_price / year_start_price - 1) * 100  # Percentage
        year_log_return = np.log(year_end_price / year_start_price) * 100
        
        # Volatility (annualized from daily)
        if 'log_returns' in year_df.columns:
            daily_vol = year_df['log_returns'].std()
            trading_days = len(year_df)
            annualized_vol = daily_vol * np.sqrt(trading_days) * 100  # Percentage
        else:
            annualized_vol = np.nan
        
        # Volume
        if 'volume' in year_df.columns or 'Volume' in year_df.columns:
            vol_col = 'volume' if 'volume' in year_df.columns else 'Volume'
            year_avg_volume = year_df[vol_col].mean()
            year_total_volume = year_df[vol_col].sum()
        else:
            year_avg_volume = np.nan
            year_total_volume = np.nan
        
        # Trading days
        trading_days = len(year_df)
        
        yearly_data.append({
            'Year': year,
            'Year_Start_Price': year_start_price,
            'Year_End_Price': year_end_price,
            'Year_Avg_Price': year_avg_price,
            'Year_High': year_high,
            'Year_Low': year_low,
            'Year_Return_Pct': year_return,
            'Year_Log_Return_Pct': year_log_return,
            'Annualized_Volatility_Pct': annualized_vol,
            'Avg_Daily_Volume': year_avg_volume,
            'Total_Volume': year_total_volume,
            'Trading_Days': trading_days,
        })
    
    yearly_df = pd.DataFrame(yearly_data)
    
    # Calculate cumulative return
    yearly_df['Cumulative_Return_Pct'] = ((yearly_df['Year_End_Price'] / yearly_df['Year_Start_Price'].iloc[0]) - 1) * 100
    
    print(f"✓ Aggregated data for {len(yearly_df)} years")
    print(f"\nYearly Summary:")
    print(yearly_df[['Year', 'Year_End_Price', 'Year_Return_Pct', 'Annualized_Volatility_Pct']].to_string(index=False))
    
    return yearly_df

def create_visualizations(yearly_df, output_dir):
    """Create comprehensive visualizations."""
    print(f"\n{'='*70}")
    print("Creating Visualizations")
    print(f"{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Year-End Price Trend (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(yearly_df['Year'], yearly_df['Year_End_Price'], 
             marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Year-End Price')
    ax1.fill_between(yearly_df['Year'], yearly_df['Year_Low'], yearly_df['Year_High'], 
                     alpha=0.2, color='gray', label='Year Range (High-Low)')
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title('NVDA Stock Price: Year-End Values & Range', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # 2. Yearly Returns (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['green' if x > 0 else 'red' for x in yearly_df['Year_Return_Pct']]
    ax2.bar(yearly_df['Year'], yearly_df['Year_Return_Pct'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('NVDA Yearly Returns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative Return (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(yearly_df['Year'], yearly_df['Cumulative_Return_Pct'], 
             marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax3.set_title('NVDA Cumulative Return (from First Year)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Average Price Trend (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(yearly_df['Year'], yearly_df['Year_Avg_Price'], 
             marker='s', linewidth=2, markersize=7, color='#ff7f0e', label='Average Price')
    ax4.plot(yearly_df['Year'], yearly_df['Year_Start_Price'], 
             marker='^', linewidth=1.5, markersize=6, color='#9467bd', alpha=0.7, label='Year-Start Price')
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax4.set_title('NVDA Average & Year-Start Prices', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Volatility Trend (Middle Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(yearly_df['Year'], yearly_df['Annualized_Volatility_Pct'], 
             marker='D', linewidth=2, markersize=7, color='#d62728')
    ax5.fill_between(yearly_df['Year'], 0, yearly_df['Annualized_Volatility_Pct'], 
                     alpha=0.3, color='#d62728')
    ax5.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
    ax5.set_title('NVDA Annualized Volatility', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. High-Low Range (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    range_pct = ((yearly_df['Year_High'] - yearly_df['Year_Low']) / yearly_df['Year_Avg_Price']) * 100
    ax6.bar(yearly_df['Year'], range_pct, color='#8c564b', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax6.set_ylabel('High-Low Range (%)', fontsize=12, fontweight='bold')
    ax6.set_title('NVDA Yearly Price Range (as % of Avg)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Volume Trend (Bottom Left)
    ax7 = fig.add_subplot(gs[2, 0])
    if not yearly_df['Avg_Daily_Volume'].isna().all():
        ax7.plot(yearly_df['Year'], yearly_df['Avg_Daily_Volume'] / 1e6, 
                 marker='o', linewidth=2, markersize=6, color='#7f7f7f')
        ax7.set_ylabel('Avg Daily Volume (Millions)', fontsize=12, fontweight='bold')
        ax7.set_title('NVDA Average Daily Trading Volume', fontsize=14, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Volume data not available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('NVDA Average Daily Trading Volume', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Return vs Volatility Scatter (Bottom Middle)
    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(yearly_df['Annualized_Volatility_Pct'], yearly_df['Year_Return_Pct'],
                         c=yearly_df['Year'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    ax8.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax8.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax8.set_xlabel('Volatility (%)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax8.set_title('NVDA Return vs Volatility', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8, label='Year')
    
    # 9. Summary Statistics Table (Bottom Right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate key statistics
    total_years = len(yearly_df)
    avg_return = yearly_df['Year_Return_Pct'].mean()
    avg_vol = yearly_df['Annualized_Volatility_Pct'].mean()
    best_year = yearly_df.loc[yearly_df['Year_Return_Pct'].idxmax()]
    worst_year = yearly_df.loc[yearly_df['Year_Return_Pct'].idxmin()]
    current_price = yearly_df['Year_End_Price'].iloc[-1]
    first_price = yearly_df['Year_Start_Price'].iloc[0]
    total_return = ((current_price / first_price) - 1) * 100
    
    summary_text = f"""
NVDA Stock Analysis Summary
({yearly_df['Year'].min()}-{yearly_df['Year'].max()})

Key Statistics:
• Total Years: {total_years}
• Average Annual Return: {avg_return:.2f}%
• Average Volatility: {avg_vol:.2f}%
• Total Return: {total_return:.1f}%

Best Year: {int(best_year['Year'])}
  Return: {best_year['Year_Return_Pct']:.1f}%
  End Price: ${best_year['Year_End_Price']:.2f}

Worst Year: {int(worst_year['Year'])}
  Return: {worst_year['Year_Return_Pct']:.1f}%
  End Price: ${worst_year['Year_End_Price']:.2f}

Current Status:
  Year-End Price: ${current_price:.2f}
  """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 10. Full Price History (Bottom Row, Spanning 3 columns)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.plot(yearly_df['Year'], yearly_df['Year_End_Price'], 
             marker='o', linewidth=3, markersize=10, color='#1f77b4', label='Year-End Price', zorder=3)
    ax10.fill_between(yearly_df['Year'], yearly_df['Year_Low'], yearly_df['Year_High'], 
                      alpha=0.15, color='gray', label='Year Range', zorder=1)
    ax10.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax10.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
    ax10.set_title('NVDA Stock Price: Complete Yearly History (1999-2025)', fontsize=16, fontweight='bold')
    ax10.legend(fontsize=11)
    ax10.grid(True, alpha=0.3, zorder=0)
    ax10.set_yscale('log')
    
    # Add value labels on the line
    for idx, row in yearly_df.iterrows():
        if idx % 3 == 0 or idx == len(yearly_df) - 1:  # Label every 3rd year and last year
            ax10.annotate(f'${row["Year_End_Price"]:.1f}', 
                         (row['Year'], row['Year_End_Price']),
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.suptitle('NVDA Stock Analysis: Yearly Aggregated Data (1999-2025)', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / 'nvda_yearly_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved comprehensive visualization: {output_path}")
    plt.close()
    
    # Create a simpler single-page summary for PPT
    create_ppt_summary(yearly_df, output_dir)
    
    return output_path

def create_ppt_summary(yearly_df, output_dir):
    """Create a simple summary chart for PPT."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Price Trend
    ax = axes[0, 0]
    ax.plot(yearly_df['Year'], yearly_df['Year_End_Price'], 
           marker='o', linewidth=3, markersize=8, color='#1f77b4')
    ax.fill_between(yearly_df['Year'], yearly_df['Year_Low'], yearly_df['Year_High'], 
                   alpha=0.2, color='gray')
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax.set_title('NVDA Stock Price: Year-End Values', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Returns
    ax = axes[0, 1]
    colors = ['green' if x > 0 else 'red' for x in yearly_df['Year_Return_Pct']]
    ax.bar(yearly_df['Year'], yearly_df['Year_Return_Pct'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return (%)', fontsize=14, fontweight='bold')
    ax.set_title('NVDA Yearly Returns', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Volatility
    ax = axes[1, 0]
    ax.plot(yearly_df['Year'], yearly_df['Annualized_Volatility_Pct'], 
           marker='D', linewidth=2.5, markersize=8, color='#d62728')
    ax.fill_between(yearly_df['Year'], 0, yearly_df['Annualized_Volatility_Pct'], 
                    alpha=0.3, color='#d62728')
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Volatility (%)', fontsize=14, fontweight='bold')
    ax.set_title('NVDA Annualized Volatility', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create a formatted table
    summary_data = yearly_df[['Year', 'Year_End_Price', 'Year_Return_Pct', 'Annualized_Volatility_Pct']].copy()
    summary_data['Year_End_Price'] = summary_data['Year_End_Price'].apply(lambda x: f'${x:.2f}')
    summary_data['Year_Return_Pct'] = summary_data['Year_Return_Pct'].apply(lambda x: f'{x:.1f}%')
    summary_data['Annualized_Volatility_Pct'] = summary_data['Annualized_Volatility_Pct'].apply(lambda x: f'{x:.1f}%')
    summary_data.columns = ['Year', 'End Price', 'Return', 'Volatility']
    
    table = ax.table(cellText=summary_data.values, colLabels=summary_data.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('NVDA Yearly Summary Table', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('NVDA Stock Analysis: Yearly Summary (1999-2025)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'nvda_yearly_summary_ppt.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PPT summary: {output_path}")
    plt.close()

def main():
    """Main function."""
    print("\n" + "="*70)
    print("NVDA Yearly Data Preparation for PPT")
    print("="*70)
    
    # Find and load data
    data_path = find_nvda_data()
    df, date_col, price_col = load_and_prepare_data(data_path)
    
    # Aggregate by year
    yearly_df = aggregate_by_year(df, date_col, price_col)
    
    # Create output directory
    output_dir = Path("outputs/ppt_materials")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save yearly data to CSV
    csv_path = output_dir / 'nvda_yearly_data.csv'
    yearly_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved yearly data to CSV: {csv_path}")
    
    # Create visualizations
    create_visualizations(yearly_df, output_dir)
    
    print(f"\n{'='*70}")
    print("✓ Complete! All files saved to:", output_dir)
    print(f"{'='*70}\n")
    
    return yearly_df

if __name__ == "__main__":
    yearly_df = main()

