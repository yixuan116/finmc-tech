"""
Fetch NVDA data from WRDS (CRSP daily prices + Compustat quarterly fundamentals).

This script extends the NVDA dataset back to IPO (~1999) using WRDS:
- CRSP daily stock prices (crsp.dsf)
- Compustat quarterly fundamentals (comp.fundq)

Saves clean CSVs to data/raw/ and data/processed/.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import wrds
except ImportError:
    wrds = None

from finmc_tech.config import get_logger

logger = get_logger(__name__)


def _check_wrds_available() -> None:
    """Check if WRDS library is available."""
    if wrds is None:
        raise ImportError(
            "wrds library is not installed. "
            "Install it with: pip install wrds"
        )


def _read_pgpass_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Read credentials from .pgpass file if it exists.
    
    Returns:
        Tuple of (username, password) or (None, None) if not found
    """
    pgpass_file = Path.home() / ".pgpass"
    if not pgpass_file.exists():
        return None, None
    
    try:
        # .pgpass format: hostname:port:database:username:password
        with open(pgpass_file, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split(':')
                if len(parts) >= 5:
                    # Extract username and password
                    username = parts[3] if len(parts) > 3 else None
                    password = parts[4] if len(parts) > 4 else None
                    return username, password
    except Exception:
        pass
    
    return None, None


def _create_wrds_connection() -> object:
    """
    Create WRDS connection with proper error handling.
    
    Tries to connect using:
    1. Environment variables (WRDS_USERNAME, WRDS_PASSWORD)
    2. Credentials from .pgpass file
    3. Config file (~/.wrdsrc)
    4. Default Connection() behavior
    
    Returns:
        WRDS connection object
    
    Raises:
        ConnectionError: If connection fails or credentials are missing
    """
    _check_wrds_available()
    
    # Check for environment variables
    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")
    
    # If not in environment, try reading from .pgpass
    if not (username and password):
        pgpass_user, pgpass_pass = _read_pgpass_credentials()
        if pgpass_user and pgpass_pass:
            username = pgpass_user
            password = pgpass_pass
    
    # Check for config file
    config_file = Path.home() / ".wrdsrc"
    has_config = config_file.exists()
    
    logger.info("Connecting to WRDS...")
    
    try:
        # Try connecting with username/password if available
        if username and password:
            logger.info("Using WRDS credentials from environment variables or .pgpass file")
            conn = wrds.Connection(wrds_username=username, wrds_password=password)
        elif has_config:
            logger.info("Using WRDS credentials from config file (~/.wrdsrc)")
            conn = wrds.Connection()
        else:
            # Try default connection (may prompt interactively)
            logger.info("Attempting WRDS connection (checking for saved credentials)...")
            conn = wrds.Connection()
        
        return conn
    
    except EOFError:
        # This happens when Connection() tries to prompt for input in non-interactive mode
        raise ConnectionError(
            "WRDS connection failed: Interactive authentication not available.\n"
            "Please configure WRDS credentials using one of these methods:\n"
            "  1. Set environment variables:\n"
            "     export WRDS_USERNAME='your_username'\n"
            "     export WRDS_PASSWORD='your_password'\n"
            "  2. Create ~/.wrdsrc config file with your credentials\n"
            "  3. Run 'wrds.Connection()' interactively once to save credentials\n"
            "\n"
            "For more info, see: https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/"
        )
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "password" in error_msg.lower() or "PAM authentication failed" in error_msg:
            raise ConnectionError(
                f"WRDS 认证失败 (Authentication Failed): {error_msg}\n\n"
                "可能的原因：\n"
                "  1. 用户名或密码错误\n"
                "  2. WRDS 账户未激活或需要设置\n"
                "  3. IP 地址未在白名单中（需要访问 https://wrds-www.wharton.upenn.edu/ 登录并设置）\n"
                "  4. 账户需要特殊权限\n\n"
                "请检查：\n"
                "  - 确认用户名和密码正确\n"
                "  - 访问 https://wrds-www.wharton.upenn.edu/ 确认账户状态\n"
                "  - 在 WRDS 网站设置 IP 白名单（如果需要）\n"
                "  - 如果问题持续，请联系 WRDS 支持：support@wrds.wharton.upenn.edu"
            )
        else:
            raise ConnectionError(
                f"连接 WRDS 失败: {error_msg}\n\n"
                "请确保：\n"
                "  1. WRDS 账户已激活\n"
                "  2. IP 地址已添加到白名单（如果需要）\n"
                "  3. 凭证配置正确\n"
                "  4. 网络连接正常"
            )


def fetch_crsp_prices(
    ticker: str = "NVDA",
    start_date: str = "1999-01-01",
    end_date: Optional[str] = None,
    conn: Optional[object] = None,
) -> pd.DataFrame:
    """
    Fetch NVDA daily stock prices from WRDS CRSP (crsp.dsf).
    
    Args:
        ticker: Stock ticker symbol (default: "NVDA")
        start_date: Start date in YYYY-MM-DD format (default: "1999-01-01")
        end_date: End date in YYYY-MM-DD format. If None, fetches to latest.
        conn: WRDS connection object. If None, creates a new connection.
    
    Returns:
        DataFrame with columns: date, permno, ticker, adj_close, ret, vol, shrout
    
    Raises:
        ImportError: If wrds library is not installed
        ConnectionError: If WRDS connection fails
    """
    _check_wrds_available()
    
    # Create connection if not provided
    close_conn = False
    if conn is None:
        logger.info("Connecting to WRDS...")
        try:
            conn = wrds.Connection()
            close_conn = True
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to WRDS. "
                f"This may be due to missing WRDS credentials.\n"
                f"Please ensure you have:\n"
                f"  1. Installed the wrds library: pip install wrds\n"
                f"  2. Configured your WRDS credentials\n"
                f"  3. Set up your WRDS account and IP whitelist if required\n"
                f"Error details: {str(e)}"
            )
    
    # First, get permno(s) for the ticker from crsp.stocknames
    logger.info(f"Looking up permno for ticker {ticker}...")
    name_query = f"""
        SELECT DISTINCT permno, ticker, nameenddt
        FROM crsp.stocknames
        WHERE ticker = '{ticker}'
        ORDER BY nameenddt DESC
    """
    
    try:
        name_df = conn.raw_sql(name_query)
        if name_df.empty:
            raise ValueError(f"No permno found for ticker {ticker}")
        
        # Get all permnos (ticker might have changed permno over time)
        permnos = name_df['permno'].unique().tolist()
        permno_list = ','.join(map(str, permnos))
        logger.info(f"Found {len(permnos)} permno(s) for {ticker}: {permno_list}")
    except Exception as e:
        if close_conn:
            conn.close()
        raise ConnectionError(f"Failed to lookup permno for {ticker}: {str(e)}")
    
    # Build query using permno
    date_filter = f"AND date >= '{start_date}'"
    if end_date:
        date_filter += f" AND date <= '{end_date}'"
    
    query = f"""
        SELECT 
            date,
            permno,
            prc,
            vol,
            ret,
            shrout
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
        {date_filter}
        ORDER BY date, permno
    """
    
    logger.info(f"Querying CRSP daily stock file (crsp.dsf) for {ticker}...")
    try:
        df = conn.raw_sql(query, date_cols=["date"])
        logger.info(f"Retrieved {len(df)} rows from CRSP")
    except Exception as e:
        if close_conn:
            conn.close()
        raise ConnectionError(f"Failed to execute CRSP query: {str(e)}")
    
    if close_conn:
        conn.close()
        logger.info("Disconnected from WRDS")
    
    if df.empty:
        logger.warning(f"No CRSP data found for {ticker}")
        return df
    
    # Clean and process data
    # Fix price sign: CRSP sometimes stores prices as negative for certain flags
    df["prc"] = df["prc"].abs()
    
    # Rename prc to adj_close (using prc as proxy for adjusted close)
    df = df.rename(columns={"prc": "adj_close"})
    
    # Add ticker column (all rows are for the same ticker)
    df["ticker"] = ticker
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Select and order columns
    columns = ["date", "permno", "ticker", "adj_close", "ret", "vol", "shrout"]
    df = df[columns]
    
    logger.info(f"Processed CRSP data: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def fetch_compustat_fundamentals(
    ticker: str = "NVDA",
    start_date: str = "1999-01-01",
    end_date: Optional[str] = None,
    conn: Optional[object] = None,
) -> pd.DataFrame:
    """
    Fetch NVDA quarterly fundamentals from WRDS Compustat (comp.fundq).
    
    Args:
        ticker: Stock ticker symbol (default: "NVDA")
        start_date: Start date in YYYY-MM-DD format (default: "1999-01-01")
        end_date: End date in YYYY-MM-DD format. If None, fetches to latest.
        conn: WRDS connection object. If None, creates a new connection.
    
    Returns:
        DataFrame with quarterly fundamentals
    
    Raises:
        ImportError: If wrds library is not installed
        ConnectionError: If WRDS connection fails
    """
    _check_wrds_available()
    
    # Create connection if not provided
    close_conn = False
    if conn is None:
        conn = _create_wrds_connection()
        close_conn = True
    
    # Build query
    # Extract year from start_date for fyearq filter
    start_year = pd.to_datetime(start_date).year
    date_filter = f"AND datadate >= '{start_date}'"
    if end_date:
        date_filter += f" AND datadate <= '{end_date}'"
    
    query = f"""
        SELECT 
            datadate,
            tic,
            gvkey,
            revtq AS revt,
            cogsq AS cogs,
            xsgaq AS xsga,
            niq,
            atq,
            saleq
        FROM comp.fundq
        WHERE tic = '{ticker}'
        AND datafmt = 'STD'
        AND consol = 'C'
        AND indfmt = 'INDL'
        AND fyearq >= {start_year}
        {date_filter}
        ORDER BY datadate
    """
    
    logger.info(f"Querying Compustat quarterly file (comp.fundq) for {ticker}...")
    try:
        df = conn.raw_sql(query, date_cols=["datadate"])
        logger.info(f"Retrieved {len(df)} rows from Compustat")
    except Exception as e:
        error_msg = str(e)
        # If permission denied, try using comp.funda (annual data) as fallback
        if "permission denied" in error_msg.lower() or "InsufficientPrivilege" in error_msg:
            logger.warning(
                f"Permission denied for comp.fundq. "
                f"This may require additional WRDS subscription or permissions.\n"
                f"Trying comp.funda (annual data) as fallback..."
            )
            # Fallback to annual data
            fallback_query = f"""
                SELECT 
                    datadate,
                    tic,
                    gvkey,
                    revt,
                    cogs,
                    xsga,
                    ni,
                    at,
                    sale
                FROM comp.funda
                WHERE tic = '{ticker}'
                AND datafmt = 'STD'
                AND consol = 'C'
                AND indfmt = 'INDL'
                AND fyear >= {start_year}
                {date_filter}
                ORDER BY datadate
            """
            try:
                df = conn.raw_sql(fallback_query, date_cols=["datadate"])
                logger.info(f"Retrieved {len(df)} rows from Compustat annual data (comp.funda)")
                logger.warning(
                    "Note: Using annual data instead of quarterly. "
                    "For quarterly data, contact WRDS support to enable comp.fundq access."
                )
            except Exception as e2:
                if close_conn:
                    conn.close()
                raise ConnectionError(
                    f"Failed to execute Compustat query (both fundq and funda): {str(e2)}\n"
                    f"Original error: {str(e)}"
                )
        else:
            if close_conn:
                conn.close()
            raise ConnectionError(f"Failed to execute Compustat query: {str(e)}")
    
    if close_conn:
        conn.close()
        logger.info("Disconnected from WRDS")
    
    if df.empty:
        logger.warning(f"No Compustat data found for {ticker}")
        return df
    
    # Clean and process data
    # Rename datadate to period_end
    df = df.rename(columns={"datadate": "period_end"})
    
    # Ensure period_end is datetime
    df["period_end"] = pd.to_datetime(df["period_end"])
    
    # Sort by period_end
    df = df.sort_values("period_end").reset_index(drop=True)
    
    # Rename revenue column
    df = df.rename(columns={"revt": "revenue"})
    
    # Add simple quarterly transforms
    # Year-over-year revenue change
    df["rev_yoy"] = df["revenue"].pct_change(periods=4)
    
    # Quarter-on-quarter revenue change
    df["rev_qoq"] = df["revenue"].pct_change(periods=1)
    
    logger.info(
        f"Processed Compustat data: {len(df)} rows, "
        f"date range: {df['period_end'].min()} to {df['period_end'].max()}"
    )
    
    return df


def merge_daily_with_fundamentals(
    crsp_df: pd.DataFrame,
    fundq_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge CRSP daily prices with Compustat quarterly fundamentals using forward-fill.
    
    For each daily row, uses the latest quarterly fundamental data that is <= current date.
    
    Args:
        crsp_df: DataFrame with daily CRSP prices (must have 'date' column)
        fundq_df: DataFrame with quarterly fundamentals (must have 'period_end' column)
    
    Returns:
        Merged DataFrame with daily frequency, forward-filled fundamentals
    """
    if crsp_df.empty:
        logger.warning("CRSP DataFrame is empty, returning empty DataFrame")
        return pd.DataFrame()
    
    if fundq_df.empty:
        logger.warning("Compustat DataFrame is empty, returning CRSP data only")
        return crsp_df.copy()
    
    # Ensure date columns are datetime
    crsp_df = crsp_df.copy()
    fundq_df = fundq_df.copy()
    crsp_df["date"] = pd.to_datetime(crsp_df["date"])
    fundq_df["period_end"] = pd.to_datetime(fundq_df["period_end"])
    
    # Sort both DataFrames
    crsp_df = crsp_df.sort_values("date").reset_index(drop=True)
    fundq_df = fundq_df.sort_values("period_end").reset_index(drop=True)
    
    # Use merge_asof to forward-fill quarterly data to daily
    # This aligns each daily row with the latest quarter-end <= that date
    merged_df = pd.merge_asof(
        crsp_df,
        fundq_df,
        left_on="date",
        right_on="period_end",
        direction="backward",  # Use latest quarter-end <= current date
    )
    
    # Drop period_end column (we already have date)
    if "period_end" in merged_df.columns:
        merged_df = merged_df.drop(columns=["period_end"])
    
    logger.info(
        f"Merged daily data: {len(merged_df)} rows, "
        f"date range: {merged_df['date'].min()} to {merged_df['date'].max()}"
    )
    
    return merged_df


def main() -> None:
    """
    Main function to fetch NVDA data from WRDS and save CSVs.
    
    Fetches:
    1. CRSP daily prices -> data/raw/nvda_crsp_daily_1999_2025.csv
    2. Compustat quarterly fundamentals -> data/raw/nvda_compustat_fundq_1999_2025.csv
    3. Merged daily dataset -> data/processed/nvda_daily_wrds_1999_2025.csv
    """
    ticker = "NVDA"
    start_date = "1999-01-01"
    end_date = None  # Fetch to latest
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Create directories if they don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    crsp_output = raw_dir / "nvda_crsp_daily_1999_2025.csv"
    fundq_output = raw_dir / "nvda_compustat_fundq_1999_2025.csv"
    merged_output = processed_dir / "nvda_daily_wrds_1999_2025.csv"
    
    logger.info("=" * 70)
    logger.info("Fetching NVDA data from WRDS (CRSP + Compustat)")
    logger.info("=" * 70)
    
    conn = None
    try:
        # Connect to WRDS once
        conn = _create_wrds_connection()
        
        # 1. Fetch CRSP daily prices
        logger.info("\n[Step 1] Fetching CRSP daily prices...")
        crsp_df = fetch_crsp_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            conn=conn,
        )
        
        if crsp_df.empty:
            logger.error("No CRSP data retrieved. Exiting.")
            return
        
        # Save CRSP data
        crsp_df.to_csv(crsp_output, index=False)
        logger.info(f"✓ Saved CRSP data to {crsp_output}")
        logger.info(f"  Rows: {len(crsp_df)}")
        logger.info(f"  Date range: {crsp_df['date'].min().date()} to {crsp_df['date'].max().date()}")
        logger.info(f"  Columns: {list(crsp_df.columns)}")
        logger.info(f"\n  Preview:\n{crsp_df.head(3).to_string()}")
        
        # 2. Fetch Compustat quarterly fundamentals
        logger.info("\n[Step 2] Fetching Compustat quarterly fundamentals...")
        fundq_df = pd.DataFrame()  # Initialize empty DataFrame
        try:
            fundq_df = fetch_compustat_fundamentals(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                conn=conn,
            )
        except ConnectionError as e:
            error_msg = str(e)
            if "permission denied" in error_msg.lower() or "InsufficientPrivilege" in error_msg:
                logger.warning(
                    "\n⚠️  Compustat data access denied. This requires additional WRDS subscription.\n"
                    "   Your account may need Compustat access enabled.\n"
                    "   Contact WRDS support: support@wrds.wharton.upenn.edu\n"
                    "   Continuing with CRSP data only..."
                )
            else:
                logger.warning(f"Failed to fetch Compustat data: {error_msg}\nContinuing with CRSP data only...")
        except Exception as e:
            logger.warning(f"Unexpected error fetching Compustat data: {str(e)}\nContinuing with CRSP data only...")
        
        if fundq_df.empty:
            logger.warning("No Compustat data retrieved. Saving CRSP data only.")
        else:
            # Save Compustat data
            fundq_df.to_csv(fundq_output, index=False)
            logger.info(f"✓ Saved Compustat data to {fundq_output}")
            logger.info(f"  Rows: {len(fundq_df)}")
            logger.info(f"  Date range: {fundq_df['period_end'].min().date()} to {fundq_df['period_end'].max().date()}")
            logger.info(f"  Columns: {list(fundq_df.columns)}")
            logger.info(f"\n  Preview:\n{fundq_df.head(3).to_string()}")
        
        # 3. Merge daily prices with quarterly fundamentals
        logger.info("\n[Step 3] Merging daily prices with quarterly fundamentals...")
        merged_df = merge_daily_with_fundamentals(crsp_df, fundq_df)
        
        if not merged_df.empty:
            # Save merged data
            merged_df.to_csv(merged_output, index=False)
            logger.info(f"✓ Saved merged data to {merged_output}")
            logger.info(f"  Rows: {len(merged_df)}")
            logger.info(f"  Date range: {merged_df['date'].min().date()} to {merged_df['date'].max().date()}")
            logger.info(f"  Columns: {list(merged_df.columns)}")
            logger.info(f"\n  Preview:\n{merged_df.head(3).to_string()}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ All data fetched and saved successfully!")
        logger.info("=" * 70)
        
    except ImportError as e:
        logger.error(f"\n✗ Import error: {e}")
        logger.error("Please install wrds: pip install wrds")
    except ConnectionError as e:
        logger.error(f"\n✗ Connection error: {e}")
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
    finally:
        # Ensure connection is closed
        if conn is not None:
            try:
                conn.close()
                logger.info("\nDisconnected from WRDS")
            except Exception:
                pass


if __name__ == "__main__":
    main()

