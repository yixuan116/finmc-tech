"""
Export stock data to CSV file.
"""

import sys
sys.path.append("../")

from src.data.fetch import fetch_stock_data


def main():
    print("=" * 70)
    print("Export NVDA Data to CSV")
    print("=" * 70)
    
    # Fetch data
    print("\nFetching NVIDIA data from Yahoo Finance (2010-2025)...")
    try:
        data = fetch_stock_data("NVDA")
        print(f"✓ Successfully fetched {len(data)} data points")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Add ticker column
    data['ticker'] = 'NVDA'
    
    # Reorder columns to put ticker first
    cols = ['ticker'] + [col for col in data.columns if col != 'ticker']
    data = data[cols]
    
    # Export to CSV
    output_file = "NVDA_data_2010_2025.csv"
    data.to_csv(output_file, index=False)
    print(f"\n✓ Data exported to: {output_file}")
    print(f"  Rows: {len(data)}")
    print(f"  Columns: {list(data.columns)}")
    
    print("\n" + "=" * 70)
    print("✓ Export complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

