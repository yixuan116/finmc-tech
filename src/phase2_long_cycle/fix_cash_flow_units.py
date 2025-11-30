"""
Fix unit errors in extracted cash flow data.

Some PDFs use "thousands" instead of "millions", causing values to be
multiplied by 1000 instead of 1,000,000. This script:
1. Detects the unit used in each PDF
2. Fixes values that were incorrectly converted
3. Saves corrected data
"""

import pandas as pd
import pdfplumber
import re
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_unit_in_pdf(pdf_path: Path) -> str:
    """
    Detect the unit used in PDF (thousands, millions, or billions).
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Unit string: 'thousands', 'millions', or 'billions'
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check first few pages for unit indicator
            for i in range(min(10, len(pdf.pages))):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    text_lower = text.lower()
                    if 'in thousands' in text_lower or '(in thousands)' in text_lower:
                        return 'thousands'
                    elif 'in millions' in text_lower or '(in millions)' in text_lower:
                        return 'millions'
                    elif 'in billions' in text_lower or '(in billions)' in text_lower:
                        return 'billions'
    except Exception as e:
        logger.warning(f"Error detecting unit in {pdf_path.name}: {e}")
    
    # Default to millions (most common)
    return 'millions'


def fix_value_by_unit(value: float, detected_unit: str, assumed_unit: str = 'millions') -> float:
    """
    Fix a value based on detected unit vs assumed unit.
    
    Args:
        value: Current value (in actual dollars)
        detected_unit: Unit detected in PDF
        assumed_unit: Unit that was assumed during extraction (default: millions)
        
    Returns:
        Corrected value in actual dollars
    """
    if detected_unit == assumed_unit:
        return value  # No correction needed
    
    # Conversion factors to actual dollars
    unit_factors = {
        'thousands': 1e3,
        'millions': 1e6,
        'billions': 1e9
    }
    
    # Convert back to raw value
    raw_value = value / unit_factors[assumed_unit]
    
    # Convert to correct unit
    corrected_value = raw_value * unit_factors[detected_unit]
    
    return corrected_value


def main():
    """Main fixing function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix unit errors in cash flow data"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path("outputs/data/cash_flow/cash_flow_values_extracted.csv"),
        help='Input CSV file with extracted data'
    )
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        default=Path("NVDA files"),
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs/data/cash_flow/cash_flow_values_extracted_fixed.csv"),
        help='Output CSV file with corrected data'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Fixing Unit Errors in Cash Flow Data")
    logger.info("=" * 80)
    
    # Read extracted data
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records from {args.input}\n")
    
    # Identify files that need fixing
    # Files with OCF > 100 billion are likely in thousands
    large_ocf = df[df['operating_cash_flow'] > 1e11]
    logger.info(f"Found {len(large_ocf)} files with potentially incorrect units")
    
    # Also check missing OCF files
    missing_ocf = df[df['operating_cash_flow'].isna()]
    logger.info(f"Found {len(missing_ocf)} files with missing OCF\n")
    
    files_to_fix = pd.concat([large_ocf, missing_ocf]).drop_duplicates(subset=['file'])
    
    # Fix each file
    fixed_count = 0
    for idx, row in files_to_fix.iterrows():
        pdf_name = row['file']
        pdf_path = args.pdf_dir / pdf_name
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        
        logger.info(f"Processing {pdf_name}...")
        
        # Detect unit
        detected_unit = detect_unit_in_pdf(pdf_path)
        logger.info(f"  Detected unit: {detected_unit}")
        
        # Fix values if unit is different from assumed (millions)
        if detected_unit != 'millions':
            logger.info(f"  Fixing values (was assuming millions, actually {detected_unit})")
            
            # Fix all cash flow related columns
            cf_columns = [
                'operating_cash_flow',
                'investing_cash_flow',
                'financing_cash_flow',
                'capital_expenditures',
                'free_cash_flow',
                'stock_based_compensation',
                'depreciation_amortization',
                'cash_balance'
            ]
            
            for col in cf_columns:
                if col in df.columns and pd.notna(df.loc[idx, col]):
                    old_value = df.loc[idx, col]
                    new_value = fix_value_by_unit(old_value, detected_unit, 'millions')
                    df.loc[idx, col] = new_value
                    logger.info(f"    {col}: ${old_value:,.0f} -> ${new_value:,.0f}")
            
            fixed_count += 1
        else:
            logger.info(f"  Unit is correct (millions), no fix needed")
    
    # Save corrected data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {len(files_to_fix)}")
    logger.info(f"Files fixed: {fixed_count}")
    logger.info(f"Corrected data saved to: {args.output}")
    
    # Show before/after comparison
    logger.info("\n" + "=" * 80)
    logger.info("Before/After Comparison (sample)")
    logger.info("=" * 80)
    
    if len(large_ocf) > 0:
        sample = large_ocf.head(3)
        for idx, row in sample.iterrows():
            pdf_name = row['file']
            old_ocf = pd.read_csv(args.input).loc[idx, 'operating_cash_flow']
            new_ocf = df.loc[idx, 'operating_cash_flow']
            logger.info(f"{pdf_name}:")
            logger.info(f"  Before: ${old_ocf:,.0f}")
            logger.info(f"  After:  ${new_ocf:,.0f}")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

