#!/usr/bin/env python3
"""
Plot NVDA daily stock price from 2009 to latest, with year-end labels.
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
plt.rcParams['figure.figsize'] = (16, 8)
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
        data = fetch_stock_data("NVDA", start="2009-01-01")
        data['ticker'] = 'NVDA'
        # Save for future use
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        data.to_csv("data/raw/nvda_yf_daily_1999_2025.csv", index=False)
        print("✓ Fetched and saved data")
        return "data/raw/nvda_yf_daily_1999_2025.csv"
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        raise

def load_and_prepare_data(data_path):
    """Load and prepare NVDA data from 2009."""
    print(f"\n{'='*70}")
    print("Loading NVDA Daily Data (2009 onwards)")
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
    
    # Filter from 2009 onwards
    df = df[df[date_col] >= '2009-01-01'].copy()
    
    # Handle different price column names
    price_col = None
    for col in ['close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError("No price column found in data")
    
    # Add year column
    df['year'] = df[date_col].dt.year
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"  Year range: {df['year'].min()} to {df['year'].max()}")
    
    return df, date_col, price_col

def get_year_end_prices(df, date_col, price_col):
    """Get year-end prices for labeling."""
    year_end_prices = []
    
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year].copy()
        if len(year_df) == 0:
            continue
        
        # Get the last trading day of the year
        last_date = year_df[date_col].max()
        last_price = year_df[year_df[date_col] == last_date][price_col].iloc[0]
        
        year_end_prices.append({
            'year': year,
            'date': last_date,
            'price': last_price
        })
    
    return pd.DataFrame(year_end_prices)

def create_daily_plot(df, date_col, price_col, year_end_df, output_dir):
    """Create daily price plot with year-end labels."""
    print(f"\n{'='*70}")
    print("Creating Daily Price Plot with Year-End Labels")
    print(f"{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plot daily prices
    ax.plot(df[date_col], df[price_col], 
           linewidth=1.5, color='#1f77b4', alpha=0.8, label='Daily Close Price')
    
    # Add year-end markers and labels
    for idx, row in year_end_df.iterrows():
        year = int(row['year'])
        date = row['date']
        price = row['price']
        
        # Plot marker
        ax.plot(date, price, marker='o', markersize=10, 
               color='red', markeredgecolor='white', markeredgewidth=2, zorder=5)
        
        # Add label with price only (no year, no box, no arrow)
        label_text = f"${price:.2f}"
        ax.annotate(label_text, 
                   xy=(date, price),
                   xytext=(10, 50),  # Offset from point (increased vertical offset to avoid overlap)
                   textcoords='offset points',
                   fontsize=16,  # Larger font size
                   fontweight='bold',
                   color='black',  # Text color
                   ha='left',
                   zorder=6)
    
    # Add vertical lines for year boundaries (optional, subtle)
    for year in sorted(df['year'].unique())[1:]:  # Skip first year
        year_start = pd.Timestamp(f'{year}-01-01')
        if year_start >= df[date_col].min():
            ax.axvline(year_start, color='gray', linestyle='--', 
                      linewidth=0.5, alpha=0.3, zorder=1)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax.set_title('NVDA Stock Price: Daily Close (2009 - Latest)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Use log scale for better visualization
    ax.set_yscale('log')
    
    # Format x-axis dates
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'nvda_daily_2009_with_labels.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved daily plot: {output_path}")
    plt.close()
    
    # Also create a version with linear scale
    create_linear_scale_plot(df, date_col, price_col, year_end_df, output_dir)
    
    return output_path

def create_linear_scale_plot(df, date_col, price_col, year_end_df, output_dir):
    """Create daily price plot with linear scale."""
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plot daily prices
    ax.plot(df[date_col], df[price_col], 
           linewidth=1.5, color='#1f77b4', alpha=0.8, label='Daily Close Price')
    
    # Add year-end markers and labels
    for idx, row in year_end_df.iterrows():
        year = int(row['year'])
        date = row['date']
        price = row['price']
        
        # Plot marker
        ax.plot(date, price, marker='o', markersize=10, 
               color='red', markeredgecolor='white', markeredgewidth=2, zorder=5)
        
        # Add label with year and price
        label_text = f"{year}\n${price:.2f}"
        ax.annotate(label_text, 
                   xy=(date, price),
                   xytext=(10, 20),
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='yellow', 
                            edgecolor='black',
                            alpha=0.8),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.3',
                                 color='black',
                                 lw=1.5),
                   ha='left',
                   zorder=6)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax.set_title('NVDA Stock Price: Daily Close (2009 - Latest) - Linear Scale', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Linear scale (not log)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'nvda_daily_2009_linear_scale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved linear scale plot: {output_path}")
    plt.close()

def main():
    """Main function."""
    print("\n" + "="*70)
    print("NVDA Daily Price Plot (2009 - Latest) with Year-End Labels")
    print("="*70)
    
    # Find and load data
    data_path = find_nvda_data()
    df, date_col, price_col = load_and_prepare_data(data_path)
    
    # Get year-end prices
    year_end_df = get_year_end_prices(df, date_col, price_col)
    
    print(f"\nYear-End Prices:")
    print(year_end_df.to_string(index=False))
    
    # Create output directory
    output_dir = Path("outputs/ppt_materials")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_daily_plot(df, date_col, price_col, year_end_df, output_dir)
    
    # Save year-end data
    csv_path = output_dir / 'nvda_year_end_prices_2009.csv'
    year_end_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved year-end prices to CSV: {csv_path}")
    
    print(f"\n{'='*70}")
    print("✓ Complete! All files saved to:", output_dir)
    print(f"{'='*70}\n")
    
    return df, year_end_df

if __name__ == "__main__":
    df, year_end_df = main()

