"""
Simple script to view stock data structure and statistics.
Focus only on data exploration, no simulations.
"""

import sys
sys.path.append("../")

from src.data.fetch import fetch_stock_data, compute_statistics


def main():
    print("=" * 70)
    print("NVDA Stock Data Viewer")
    print("=" * 70)
    
    # Fetch data
    print("\nFetching NVIDIA data from Yahoo Finance (2010-2025)...")
    try:
        data = fetch_stock_data("NVDA")
        print(f"✓ Successfully fetched {len(data)} data points")
        print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Show data structure
    print("\n" + "=" * 70)
    print("Data Structure:")
    print("=" * 70)
    print(f"Shape: {data.shape}")
    print(f"\nColumns: {list(data.columns)}")
    
    # Show first few rows
    print("\n" + "=" * 70)
    print("First 5 rows:")
    print("=" * 70)
    print(data.head().to_string())
    
    # Show last few rows
    print("\n" + "=" * 70)
    print("Last 5 rows:")
    print("=" * 70)
    print(data.tail().to_string())
    
    # Show statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print("=" * 70)
    stats = compute_statistics(data)
    print(f"  Annualized Return: {stats['annualized_return']:.2%}")
    print(f"  Volatility: {stats['volatility']:.2%}")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Min Daily Return: {stats['min_return']:.2%}")
    print(f"  Max Daily Return: {stats['max_return']:.2%}")
    print(f"  Skewness: {stats['skewness']:.2f}")
    print(f"  Kurtosis: {stats['kurtosis']:.2f}")
    
    # Show price evolution by year
    print("\n" + "=" * 70)
    print("Average Price by Year:")
    print("=" * 70)
    for year in [2010, 2015, 2020, 2021, 2022, 2023, 2024]:
        year_data = data[data['date'].dt.year == year]
        if not year_data.empty:
            avg_price = year_data['close'].mean()
            min_price = year_data['close'].min()
            max_price = year_data['close'].max()
            print(f"  {year}: Avg=${avg_price:.2f}, Min=${min_price:.2f}, Max=${max_price:.2f}")
    
    # Show data info
    print("\n" + "=" * 70)
    print("Data Info:")
    print("=" * 70)
    print(data.info())
    
    print("\n" + "=" * 70)
    print("✓ Data viewing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

