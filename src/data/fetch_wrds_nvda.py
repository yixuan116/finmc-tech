"""Fetch NVIDIA financial data from WRDS Compustat database."""

import os
import sys
from pathlib import Path

import pandas as pd

try:
    import wrds
except ImportError:
    wrds = None


def fetch_wrds_nvda_funda(
    start_date: str = "2010-01-01",
    end_date: str = "2025-10-31",
    ticker: str = "NVDA",
    output_path: str = None,
) -> pd.DataFrame:
    """
    Fetch NVIDIA financial data from WRDS Compustat funda table.

    Parameters
    ----------
    start_date : str, default "2010-01-01"
        Start date in YYYY-MM-DD format
    end_date : str, default "2025-10-31"
        End date in YYYY-MM-DD format
    ticker : str, default "NVDA"
        Stock ticker symbol
    output_path : str, optional
        Path to save CSV file. If None, uses outputs/NVDA_funda_2010_2025.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with Compustat funda data for NVIDIA

    Raises
    ------
    ImportError
        If wrds library is not installed
    ConnectionError
        If WRDS credentials are not configured
    """
    if wrds is None:
        raise ImportError(
            "wrds library is not installed. "
            "Install it with: pip install wrds"
        )

    # Set default output path if not provided
    if output_path is None:
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / "outputs" / "NVDA_funda_2010_2025.csv"

    print("Connecting to WRDS...")
    try:
        db = wrds.Connection()
    except Exception as e:
        error_msg = (
            f"Failed to connect to WRDS. "
            f"This may be due to missing WRDS credentials.\n"
            f"Please ensure you have:\n"
            f"  1. Installed the wrds library: pip install wrds\n"
            f"  2. Configured your WRDS credentials (typically by running "
            f"'wrds.Connection()' interactively and entering your username/password)\n"
            f"  3. Set up your WRDS account and IP whitelist if required\n"
            f"Error details: {str(e)}"
        )
        print(error_msg)
        sys.exit(1)

    print(f"Querying Compustat funda table for {ticker}...")
    query = f"""
        SELECT gvkey, tic, datadate, revt, cogs, xrd, capx, ni, at, lt, che
        FROM comp.funda
        WHERE tic = '{ticker}'
        AND datadate >= '{start_date}'
        AND datadate <= '{end_date}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND consol = 'C'
        AND popsrc = 'D'
        ORDER BY datadate
    """

    try:
        df = db.raw_sql(query, date_cols=["datadate"])
        print(f"Retrieved {len(df)} rows from WRDS")
    except Exception as e:
        db.close()
        raise ConnectionError(f"Failed to execute query: {str(e)}")

    db.close()
    print("Disconnected from WRDS")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

    return df


def main():
    """Main function to fetch and save NVIDIA funda data."""
    try:
        df = fetch_wrds_nvda_funda()
        print(f"\nSuccessfully fetched {len(df)} records")
        print(f"Columns: {', '.join(df.columns)}")
        if len(df) > 0:
            print(f"Date range: {df['datadate'].min()} to {df['datadate'].max()}")
    except (ImportError, ConnectionError) as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

