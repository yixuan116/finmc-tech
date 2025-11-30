"""
Build long-horizon (3-5 year) firm-level features for NVIDIA.

This module processes quarterly NVDA firm fundamentals and creates engineered
features suitable for long-cycle modeling (LSTM/XGBoost with 3-5 year horizons).

Features:
---------
Base Columns:
- rev: Revenue
- log_rev: Log of revenue
- rev_ttm: Trailing 12-month revenue (4-quarter rolling sum)
- gross_margin_pct: Gross margin as percentage (0-1)
- operating_margin_pct: Operating income / revenue
- net_margin_pct: Net income / revenue
- net_income_ttm: Trailing 12-month net income
- eps_ttm: Trailing 12-month EPS
- net_margin_ttm: TTM net margin

Growth Features:
- rev_yoy: Year-over-year revenue growth
- rev_yoy_ttm: TTM revenue YoY growth
- eps_yoy: Year-over-year EPS growth
- rev_cagr_2y: 2-year revenue CAGR
- rev_cagr_3y: 3-year revenue CAGR
- eps_cagr_2y: 2-year EPS CAGR (if available)
- eps_cagr_3y: 3-year EPS CAGR (if available)
- rev_yoy_accel: Revenue YoY acceleration
- rev_cagr_2y_chg: Change in 2-year CAGR
- eps_yoy_accel: EPS YoY acceleration

Stability Features:
- gross_margin_yoy_change: YoY change in gross margin
- operating_margin_yoy_change: YoY change in operating margin
- net_margin_yoy_change: YoY change in net margin
- gross_margin_trend_8q: 8-quarter rolling mean of gross margin
- gross_margin_vol_8q: 8-quarter rolling std of gross margin
- rev_yoy_vol_8q: 8-quarter rolling std of revenue YoY
- eps_yoy_vol_8q: 8-quarter rolling std of EPS YoY (if available)

Guidance Features (if available):
- rev_surprise: (actual - guidance) / guidance
- eps_surprise: (actual - guidance) / guidance
- rev_surprise_rolling_4q: 4-quarter rolling mean of revenue surprise
- eps_surprise_rolling_4q: 4-quarter rolling mean of EPS surprise
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fundamentals(path: Path) -> pd.DataFrame:
    """
    Load quarterly fundamentals from JSON file.
    
    Args:
        path: Path to master JSON file
        
    Returns:
        DataFrame with one row per quarter
    """
    logger.info(f"Loading fundamentals from {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    if 'NVDA_Firm_Fundamentals' not in data:
        raise ValueError("JSON must have 'NVDA_Firm_Fundamentals' key")
    
    quarters = data['NVDA_Firm_Fundamentals']
    logger.info(f"Loaded {len(quarters)} quarters")
    
    # Flatten nested structure
    rows = []
    for q in quarters:
        row = {
            'fiscal_year': q.get('fiscal_year'),
            'fiscal_quarter': q.get('fiscal_quarter'),
            'period_end': q.get('period_end'),
            'report_date': q.get('report_date'),
        }
        
        # Extract financials
        fin = q.get('financials', {})
        row['revenue'] = fin.get('revenue')
        row['gross_margin_gaap'] = fin.get('gross_margin_gaap')
        row['gross_margin_nongaap'] = fin.get('gross_margin_nongaap')
        row['operating_expenses_gaap'] = fin.get('operating_expenses_gaap')
        row['operating_expenses_nongaap'] = fin.get('operating_expenses_nongaap')
        row['operating_income_gaap'] = fin.get('operating_income_gaap')
        row['operating_income_nongaap'] = fin.get('operating_income_nongaap')
        row['net_income_gaap'] = fin.get('net_income_gaap')
        row['net_income_nongaap'] = fin.get('net_income_nongaap')
        row['eps_gaap'] = fin.get('eps_gaap')
        row['eps_nongaap'] = fin.get('eps_nongaap')
        
        # Extract cash flow data
        cash_flows = fin.get('cash_flows', {})
        row['operating_cash_flow'] = cash_flows.get('operating_cash_flow')
        row['investing_cash_flow'] = cash_flows.get('investing_cash_flow')
        row['financing_cash_flow'] = cash_flows.get('financing_cash_flow')
        row['capital_expenditures'] = fin.get('capital_expenditures')
        row['free_cash_flow'] = fin.get('free_cash_flow')
        row['free_cash_flow_margin'] = fin.get('free_cash_flow_margin')
        row['fcf_conversion'] = fin.get('fcf_conversion')
        row['stock_based_compensation'] = cash_flows.get('stock_based_compensation')
        row['depreciation_amortization'] = cash_flows.get('depreciation_amortization')
        row['cash_balance'] = cash_flows.get('cash_balance')
        
        # Extract guidance if available
        guidance = q.get('guidance', {})
        if guidance:
            row['guidance_revenue'] = guidance.get('next_q_revenue')
            # Note: guidance format may vary, adapt as needed
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validate core fields
    if df['revenue'].isna().all():
        raise ValueError("No revenue data found")
    
    return df


def infer_period_end_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer period_end_date from available date fields.
    
    Uses period_end if available, falls back to report_date.
    """
    logger.info("Inferring period_end_date")
    
    # Use period_end if available, otherwise report_date
    df['period_end_date'] = df['period_end'].fillna(df['report_date'])
    
    # Convert to datetime
    df['period_end_date'] = pd.to_datetime(df['period_end_date'], errors='coerce')
    
    # Check for missing dates
    missing = df['period_end_date'].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} quarters have missing period_end_date")
    
    # Sort by date
    df = df.sort_values('period_end_date').reset_index(drop=True)
    
    # Set as index
    df = df.set_index('period_end_date')
    
    return df


def normalize_gross_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize gross margin to percentage (0-1 range).
    
    Handles multiple formats:
    - Already in 0-1 range (decimal percentage, e.g. 0.4464 = 44.64%)
    - In 0-100 range (percentage points, e.g. 65.4 = 65.4%)
    - Absolute dollar values (if values are very large compared to typical margins)
    
    Note: Data may have mixed formats, so we normalize per-row.
    """
    logger.info("Normalizing gross margin")
    
    # Use GAAP gross margin, fall back to non-GAAP if needed
    gm = df['gross_margin_gaap'].fillna(df['gross_margin_nongaap'])
    revenue = df['revenue']
    
    non_null = gm.dropna()
    
    if len(non_null) == 0:
        df['gross_margin_pct'] = np.nan
        df['gross_margin_abs'] = np.nan
        logger.warning("No gross margin data available")
        return df
    
    # Normalize per-row to handle mixed formats
    # Strategy: if value > 1, assume percentage points (0-100), divide by 100
    # Otherwise, assume already in 0-1 range
    gm_pct = gm.copy()
    mask_over_one = gm > 1.0
    if mask_over_one.any():
        gm_pct[mask_over_one] = gm[mask_over_one] / 100.0
        logger.info(f"Converted {mask_over_one.sum()} values from percentage points to decimal")
    
    # Check if any values are likely absolute dollars (very large compared to revenue)
    # This is rare, but handle it if needed
    if revenue.notna().any():
        sample_revenue = revenue[revenue.notna()].median()
        # If gross margin is > 50% of revenue, it's likely in dollars
        mask_dollars = (gm > sample_revenue * 0.5) & revenue.notna()
        if mask_dollars.any():
            gm_pct[mask_dollars] = gm[mask_dollars] / revenue[mask_dollars]
            logger.info(f"Converted {mask_dollars.sum()} values from absolute dollars to percentage")
    
    df['gross_margin_pct'] = gm_pct
    df['gross_margin_abs'] = gm_pct * revenue
    df['gross_margin_abs'] = df['gross_margin_abs'].replace([np.inf, -np.inf], np.nan)
    
    # Validate: gross margin should be between 0 and 1
    invalid = (df['gross_margin_pct'] < 0) | (df['gross_margin_pct'] > 1)
    if invalid.any():
        logger.warning(f"{invalid.sum()} gross margin values outside [0, 1] range, clipping")
        df.loc[invalid, 'gross_margin_pct'] = df.loc[invalid, 'gross_margin_pct'].clip(0, 1)
    
    return df


def add_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add base level features: revenue, margins, TTM metrics.
    """
    logger.info("Adding base level features")
    
    # Revenue features
    df['rev'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['log_rev'] = np.log(df['rev'].clip(lower=1))  # Avoid log(0)
    
    # TTM revenue (4-quarter rolling sum)
    df['rev_ttm'] = df['rev'].rolling(window=4, min_periods=1).sum()
    
    # Operating income (use if available, otherwise calculate)
    op_income = df['operating_income_gaap'].fillna(df['operating_income_nongaap'])
    
    # If not available, calculate from revenue and expenses
    if op_income.isna().all():
        op_exp = df['operating_expenses_gaap'].fillna(df['operating_expenses_nongaap'])
        op_income = df['rev'] - op_exp
        logger.info("Calculated operating_income from revenue - expenses")
    
    # Margins
    df['operating_margin_pct'] = op_income / df['rev']
    df['operating_margin_pct'] = df['operating_margin_pct'].replace([np.inf, -np.inf], np.nan)
    
    # Net margin
    net_income = df['net_income_gaap'].fillna(df['net_income_nongaap'])
    df['net_margin_pct'] = net_income / df['rev']
    df['net_margin_pct'] = df['net_margin_pct'].replace([np.inf, -np.inf], np.nan)
    
    # TTM metrics
    df['net_income_ttm'] = net_income.rolling(window=4, min_periods=1).sum()
    df['eps_ttm'] = df['eps_gaap'].fillna(df['eps_nongaap']).rolling(window=4, min_periods=1).sum()
    df['net_margin_ttm'] = df['net_income_ttm'] / df['rev_ttm']
    df['net_margin_ttm'] = df['net_margin_ttm'].replace([np.inf, -np.inf], np.nan)
    
    return df


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add growth and acceleration features: YoY, CAGR, acceleration.
    """
    logger.info("Adding growth features")
    
    # Year-over-year growth (4-quarter lag)
    df['rev_yoy'] = df['rev'] / df['rev'].shift(4) - 1
    df['rev_yoy_ttm'] = df['rev_ttm'] / df['rev_ttm'].shift(4) - 1
    
    # EPS YoY
    eps = df['eps_gaap'].fillna(df['eps_nongaap'])
    if eps.notna().sum() > 0:
        df['eps_yoy'] = eps / eps.shift(4) - 1
    else:
        df['eps_yoy'] = np.nan
        logger.warning("No EPS data available, skipping EPS features")
    
    # Multi-year CAGRs
    # 2-year CAGR (8-quarter lag)
    df['rev_cagr_2y'] = (df['rev_ttm'] / df['rev_ttm'].shift(8)) ** (1/2) - 1
    df['rev_cagr_2y'] = df['rev_cagr_2y'].replace([np.inf, -np.inf], np.nan)
    
    # 3-year CAGR (12-quarter lag)
    df['rev_cagr_3y'] = (df['rev_ttm'] / df['rev_ttm'].shift(12)) ** (1/3) - 1
    df['rev_cagr_3y'] = df['rev_cagr_3y'].replace([np.inf, -np.inf], np.nan)
    
    # EPS CAGRs (if EPS available)
    if eps.notna().sum() > 4:
        eps_ttm = eps.rolling(window=4, min_periods=1).sum()
        df['eps_cagr_2y'] = (eps_ttm / eps_ttm.shift(8)) ** (1/2) - 1
        df['eps_cagr_2y'] = df['eps_cagr_2y'].replace([np.inf, -np.inf], np.nan)
        
        df['eps_cagr_3y'] = (eps_ttm / eps_ttm.shift(12)) ** (1/3) - 1
        df['eps_cagr_3y'] = df['eps_cagr_3y'].replace([np.inf, -np.inf], np.nan)
    else:
        df['eps_cagr_2y'] = np.nan
        df['eps_cagr_3y'] = np.nan
    
    # Acceleration (change in growth)
    df['rev_yoy_accel'] = df['rev_yoy'] - df['rev_yoy'].shift(4)
    df['rev_cagr_2y_chg'] = df['rev_cagr_2y'] - df['rev_cagr_2y'].shift(4)
    
    if eps.notna().sum() > 0:
        df['eps_yoy_accel'] = df['eps_yoy'] - df['eps_yoy'].shift(4)
    else:
        df['eps_yoy_accel'] = np.nan
    
    return df


def add_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add margin trends and volatility features.
    """
    logger.info("Adding stability features")
    
    # Year-over-year margin changes
    df['gross_margin_yoy_change'] = df['gross_margin_pct'] - df['gross_margin_pct'].shift(4)
    df['operating_margin_yoy_change'] = df['operating_margin_pct'] - df['operating_margin_pct'].shift(4)
    df['net_margin_yoy_change'] = df['net_margin_pct'] - df['net_margin_pct'].shift(4)
    
    # Rolling trends and volatility (8-quarter window)
    df['gross_margin_trend_8q'] = df['gross_margin_pct'].rolling(window=8, min_periods=1).mean()
    df['gross_margin_vol_8q'] = df['gross_margin_pct'].rolling(window=8, min_periods=2).std()
    
    df['rev_yoy_vol_8q'] = df['rev_yoy'].rolling(window=8, min_periods=2).std()
    
    # EPS volatility (if available)
    if df['eps_yoy'].notna().sum() > 0:
        df['eps_yoy_vol_8q'] = df['eps_yoy'].rolling(window=8, min_periods=2).std()
    else:
        df['eps_yoy_vol_8q'] = np.nan
    
    return df


def add_guidance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add guidance vs actual surprise features (if guidance data exists).
    """
    logger.info("Checking for guidance data")
    
    # Check if guidance exists
    has_guidance = 'guidance_revenue' in df.columns and df['guidance_revenue'].notna().sum() > 0
    
    if not has_guidance:
        logger.warning("No guidance data found, skipping guidance features")
        return df
    
    logger.info("Adding guidance features")
    
    # Revenue surprise
    df['rev_surprise'] = (df['rev'] - df['guidance_revenue']) / df['guidance_revenue']
    df['rev_surprise'] = df['rev_surprise'].replace([np.inf, -np.inf], np.nan)
    
    # Rolling surprise metrics
    df['rev_surprise_rolling_4q'] = df['rev_surprise'].rolling(window=4, min_periods=1).mean()
    
    # EPS guidance (if available)
    if 'guidance_eps' in df.columns and df['guidance_eps'].notna().sum() > 0:
        eps = df['eps_gaap'].fillna(df['eps_nongaap'])
        df['eps_surprise'] = (eps - df['guidance_eps']) / df['guidance_eps']
        df['eps_surprise'] = df['eps_surprise'].replace([np.inf, -np.inf], np.nan)
        df['eps_surprise_rolling_4q'] = df['eps_surprise'].rolling(window=4, min_periods=1).mean()
    else:
        df['eps_surprise'] = np.nan
        df['eps_surprise_rolling_4q'] = np.nan
    
    return df


def add_cash_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cash flow related features.
    
    Features:
    - OCF (Operating Cash Flow) levels and ratios
    - FCF (Free Cash Flow) levels and ratios
    - CapEx intensity and trends
    - Cash balance metrics
    - Cash flow growth rates
    """
    logger.info("Adding cash flow features")
    
    # Convert to numeric
    df['ocf'] = pd.to_numeric(df['operating_cash_flow'], errors='coerce')
    df['icf'] = pd.to_numeric(df['investing_cash_flow'], errors='coerce')
    df['fcf'] = pd.to_numeric(df['financing_cash_flow'], errors='coerce')
    df['fcf_free'] = pd.to_numeric(df['free_cash_flow'], errors='coerce')
    df['capex'] = pd.to_numeric(df['capital_expenditures'], errors='coerce')
    df['cash_bal'] = pd.to_numeric(df['cash_balance'], errors='coerce')
    
    # Use computed FCF if available, otherwise calculate
    if df['fcf_free'].notna().sum() > 0:
        df['fcf'] = df['fcf_free'].fillna(df['ocf'] - df['capex'].abs())
    else:
        # Calculate FCF = OCF - |CapEx| (CapEx is negative, so we use abs)
        df['fcf'] = df['ocf'] - df['capex'].abs()
    
    # OCF TTM (trailing 12 months)
    df['ocf_ttm'] = df['ocf'].rolling(window=4, min_periods=1).sum()
    
    # FCF TTM
    df['fcf_ttm'] = df['fcf'].rolling(window=4, min_periods=1).sum()
    
    # OCF margin (OCF / Revenue)
    revenue = pd.to_numeric(df['revenue'], errors='coerce')
    df['ocf_margin'] = df['ocf'] / revenue
    df['ocf_margin'] = df['ocf_margin'].replace([np.inf, -np.inf], np.nan)
    
    # FCF margin (FCF / Revenue) - use pre-computed if available
    if 'free_cash_flow_margin' in df.columns:
        df['fcf_margin'] = pd.to_numeric(df['free_cash_flow_margin'], errors='coerce')
    else:
        df['fcf_margin'] = df['fcf'] / revenue
        df['fcf_margin'] = df['fcf_margin'].replace([np.inf, -np.inf], np.nan)
    
    # FCF conversion (FCF / Net Income) - use pre-computed if available
    if 'fcf_conversion' in df.columns:
        df['fcf_conversion'] = pd.to_numeric(df['fcf_conversion'], errors='coerce')
    else:
        net_income = pd.to_numeric(df['net_income_gaap'], errors='coerce')
        df['fcf_conversion'] = df['fcf'] / net_income
        df['fcf_conversion'] = df['fcf_conversion'].replace([np.inf, -np.inf], np.nan)
    
    # CapEx intensity (CapEx / Revenue)
    df['capex_intensity'] = df['capex'].abs() / revenue
    df['capex_intensity'] = df['capex_intensity'].replace([np.inf, -np.inf], np.nan)
    
    # Cash to revenue ratio
    df['cash_to_revenue'] = df['cash_bal'] / revenue
    df['cash_to_revenue'] = df['cash_to_revenue'].replace([np.inf, -np.inf], np.nan)
    
    # Year-over-year growth rates
    df['ocf_yoy'] = df['ocf'].pct_change(periods=4) * 100
    df['fcf_yoy'] = df['fcf'].pct_change(periods=4) * 100
    df['capex_yoy'] = df['capex'].pct_change(periods=4) * 100
    
    # Replace inf values
    for col in ['ocf_yoy', 'fcf_yoy', 'capex_yoy']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"OCF available: {df['ocf'].notna().sum()}/{len(df)}")
    logger.info(f"FCF available: {df['fcf'].notna().sum()}/{len(df)}")
    logger.info(f"CapEx available: {df['capex'].notna().sum()}/{len(df)}")
    
    return df


def build_long_horizon_features(
    input_path: Path,
    output_path: Path,
    drop_all_nan_cols: bool = True
) -> pd.DataFrame:
    """
    Build all long-horizon features from fundamentals JSON.
    
    Args:
        input_path: Path to master fundamentals JSON
        output_path: Path to output CSV
        drop_all_nan_cols: Whether to drop columns that are all NaN
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("=" * 60)
    logger.info("Building long-horizon features for NVIDIA")
    logger.info("=" * 60)
    
    # Load data
    df = load_fundamentals(input_path)
    
    # Process dates
    df = infer_period_end_date(df)
    
    # Normalize gross margin
    df = normalize_gross_margin(df)
    
    # Add features in sequence
    df = add_level_features(df)
    df = add_growth_features(df)
    df = add_stability_features(df)
    df = add_guidance_features(df)
    df = add_cash_flow_features(df)  # Add cash flow features
    
    # Select feature columns (exclude raw input columns)
    feature_cols = [
        'rev', 'log_rev', 'rev_ttm',
        'gross_margin_pct', 'gross_margin_abs',
        'operating_margin_pct', 'net_margin_pct',
        'net_income_ttm', 'eps_ttm', 'net_margin_ttm',
        'rev_yoy', 'rev_yoy_ttm', 'eps_yoy',
        'rev_cagr_2y', 'rev_cagr_3y', 'eps_cagr_2y', 'eps_cagr_3y',
        'rev_yoy_accel', 'rev_cagr_2y_chg', 'eps_yoy_accel',
        'gross_margin_yoy_change', 'operating_margin_yoy_change', 'net_margin_yoy_change',
        'gross_margin_trend_8q', 'gross_margin_vol_8q',
        'rev_yoy_vol_8q', 'eps_yoy_vol_8q',
        # Cash flow features
        'ocf', 'ocf_ttm', 'ocf_margin', 'ocf_yoy',
        'fcf', 'fcf_ttm', 'fcf_margin', 'fcf_yoy', 'fcf_conversion',
        'capex', 'capex_intensity', 'capex_yoy',
        'cash_balance', 'cash_to_revenue',
    ]
    
    # Add guidance features if they exist
    if 'rev_surprise' in df.columns:
        feature_cols.extend([
            'rev_surprise', 'eps_surprise',
            'rev_surprise_rolling_4q', 'eps_surprise_rolling_4q'
        ])
    
    # Select only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    df_features = df[available_cols].copy()
    
    # Drop columns that are all NaN
    if drop_all_nan_cols:
        before = len(df_features.columns)
        df_features = df_features.dropna(axis=1, how='all')
        after = len(df_features.columns)
        if before > after:
            logger.info(f"Dropped {before - after} columns that were all NaN")
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Feature engineering complete")
    logger.info("=" * 60)
    logger.info(f"Number of quarters: {len(df_features)}")
    logger.info(f"Number of features: {len(df_features.columns)}")
    logger.info(f"Date range: {df_features.index.min()} to {df_features.index.max()}")
    logger.info(f"\nOutput saved to: {output_path}")
    
    logger.info("\nFirst 3 rows:")
    logger.info(f"\n{df_features.head(3)}")
    logger.info("\nLast 3 rows:")
    logger.info(f"\n{df_features.tail(3)}")
    
    logger.info("\nFeature columns:")
    for col in df_features.columns:
        non_null = df_features[col].notna().sum()
        pct = 100 * non_null / len(df_features)
        logger.info(f"  {col}: {non_null}/{len(df_features)} ({pct:.1f}% non-null)")
    
    return df_features


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build long-horizon firm-level features for NVIDIA"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/processed/nvda_firm_fundamentals_master.json'),
        help='Path to input master JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/nvda_long_horizon_firm_features.csv'),
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    # Check if input exists, try alternative location
    if not args.input.exists():
        alt_path = Path('outputs/data/fundamentals/nvda_firm_fundamentals_master.json')
        if alt_path.exists():
            logger.info(f"Input not found at {args.input}, using {alt_path}")
            args.input = alt_path
        else:
            raise FileNotFoundError(f"Input file not found: {args.input}")
    
    build_long_horizon_features(args.input, args.output)


if __name__ == '__main__':
    main()

