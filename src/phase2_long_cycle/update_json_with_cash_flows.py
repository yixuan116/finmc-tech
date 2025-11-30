"""
Update JSON file with extracted cash flow data from CSV.

This script:
1. Reads extracted cash flow data from CSV
2. Matches PDF filenames to JSON records via source_files
3. Updates financials.cash_flows in JSON
4. Saves updated JSON
"""

import json
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def match_pdf_to_json_record(pdf_name: str, json_record: Dict) -> bool:
    """
    Check if PDF filename matches a JSON record.
    
    Args:
        pdf_name: PDF filename (e.g., "Q1 26 10-Q.pdf")
        json_record: JSON record with source_files
        
    Returns:
        True if matches, False otherwise
    """
    if 'source_files' not in json_record:
        return False
    
    source_files = json_record['source_files']
    
    # Check ten_q and ten_k fields
    ten_q = source_files.get('ten_q', '')
    ten_k = source_files.get('ten_k', '')
    
    # Direct match
    if pdf_name == ten_q or pdf_name == ten_k:
        return True
    
    # Partial match (handle variations in naming)
    # Remove common variations
    pdf_clean = pdf_name.lower().replace(' ', '').replace('-', '').replace('_', '')
    ten_q_clean = ten_q.lower().replace(' ', '').replace('-', '').replace('_', '') if ten_q else ''
    ten_k_clean = ten_k.lower().replace(' ', '').replace('-', '').replace('_', '') if ten_k else ''
    
    if pdf_clean and (pdf_clean in ten_q_clean or ten_q_clean in pdf_clean):
        return True
    if pdf_clean and (pdf_clean in ten_k_clean or ten_k_clean in pdf_clean):
        return True
    
    return False


def match_by_year_quarter(pdf_name: str, json_record: Dict) -> bool:
    """
    Match PDF to JSON record by extracting year and quarter from filename.
    
    Args:
        pdf_name: PDF filename (e.g., "Q1 26 10-Q.pdf" or "Q3 26 10-Q.pdf")
        json_record: JSON record with fiscal_year and fiscal_quarter
        
    Returns:
        True if year and quarter match
    """
    import re
    
    fiscal_year = json_record.get('fiscal_year')
    fiscal_quarter = json_record.get('fiscal_quarter')
    
    if not fiscal_year or not fiscal_quarter:
        return False
    
    # Extract year and quarter from PDF filename
    # Patterns: "Q1 26", "Q2 26", "Q3 26", "Q4 26", "Q1 09", etc.
    year_pattern = r'(\d{2})'
    quarter_pattern = r'Q([1-4])'
    
    # Try to extract year (last 2 digits)
    year_match = re.search(year_pattern, pdf_name)
    quarter_match = re.search(quarter_pattern, pdf_name, re.IGNORECASE)
    
    if year_match and quarter_match:
        pdf_year_2digit = int(year_match.group(1))
        pdf_quarter = quarter_match.group(1)
        
        # Convert 2-digit year to 4-digit (assume 2000-2099)
        if pdf_year_2digit < 50:
            pdf_year = 2000 + pdf_year_2digit
        else:
            pdf_year = 1900 + pdf_year_2digit
        
        # Match
        if pdf_year == fiscal_year and pdf_quarter == fiscal_quarter:
            return True
    
    return False


def update_json_with_cash_flows(
    csv_path: Path = Path("outputs/data/cash_flow/cash_flow_values_extracted.csv"),
    json_path: Path = Path("outputs/data/fundamentals/nvda_firm_fundamentals_master.json"),
    output_path: Optional[Path] = None
) -> None:
    """
    Update JSON file with cash flow data from CSV.
    
    Args:
        csv_path: Path to CSV with extracted cash flow data
        json_path: Path to JSON file to update
        output_path: Optional output path (default: overwrite json_path)
    """
    logger.info("=" * 80)
    logger.info("Updating JSON with Cash Flow Data")
    logger.info("=" * 80)
    
    # Read CSV
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from CSV\n")
    
    # Read JSON
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data['NVDA_Firm_Fundamentals'])} records from JSON\n")
    
    # Track which CSV rows have been used
    used_csv_indices = set()
    
    # Update each record
    updated_count = 0
    matched_count = 0
    
    for record in data['NVDA_Firm_Fundamentals']:
        # Try to match with CSV records
        matched = False
        fiscal_year = record.get('fiscal_year')
        fiscal_quarter = record.get('fiscal_quarter')
        
        for idx, csv_row in df.iterrows():
            if idx in used_csv_indices:
                continue  # Skip already used rows
            
            pdf_name = csv_row['file']
            
            # Try filename match first (most reliable)
            if match_pdf_to_json_record(pdf_name, record):
                matched = True
                matched_count += 1
                used_csv_indices.add(idx)
                
                # Initialize financials.cash_flows if needed
                if 'financials' not in record:
                    record['financials'] = {}
                
                if 'cash_flows' not in record['financials']:
                    record['financials']['cash_flows'] = {}
                
                # Update cash flow data
                cash_flow_fields = {
                    'operating_cash_flow': 'operating_cash_flow',
                    'investing_cash_flow': 'investing_cash_flow',
                    'financing_cash_flow': 'financing_cash_flow',
                    'capital_expenditures': 'capital_expenditures',
                    'free_cash_flow': 'free_cash_flow',
                    'stock_based_compensation': 'stock_based_compensation',
                    'depreciation_amortization': 'depreciation_amortization',
                    'cash_balance': 'cash_balance'
                }
                
                updated_fields = []
                for csv_col, json_key in cash_flow_fields.items():
                    if csv_col in csv_row and pd.notna(csv_row[csv_col]):
                        # Convert to int if it's a float
                        value = csv_row[csv_col]
                        if isinstance(value, float):
                            value = int(value)
                        record['financials']['cash_flows'][json_key] = value
                        updated_fields.append(json_key)
                
                if updated_fields:
                    updated_count += 1
                    fy = record.get('fiscal_year', '?')
                    q = record.get('fiscal_quarter', '?')
                    logger.info(f"FY{fy} {q}: Updated {len(updated_fields)} fields (file: {pdf_name})")
                
                break  # Found match, move to next record
        
        # If no filename match, try year/quarter match
        if not matched:
            for idx, csv_row in df.iterrows():
                if idx in used_csv_indices:
                    continue
                
                pdf_name = csv_row['file']
                if match_by_year_quarter(pdf_name, record):
                    matched = True
                    matched_count += 1
                    used_csv_indices.add(idx)
                    
                    # Initialize financials.cash_flows if needed
                    if 'financials' not in record:
                        record['financials'] = {}
                    
                    if 'cash_flows' not in record['financials']:
                        record['financials']['cash_flows'] = {}
                    
                    # Update cash flow data
                    cash_flow_fields = {
                        'operating_cash_flow': 'operating_cash_flow',
                        'investing_cash_flow': 'investing_cash_flow',
                        'financing_cash_flow': 'financing_cash_flow',
                        'capital_expenditures': 'capital_expenditures',
                        'free_cash_flow': 'free_cash_flow',
                        'stock_based_compensation': 'stock_based_compensation',
                        'depreciation_amortization': 'depreciation_amortization',
                        'cash_balance': 'cash_balance'
                    }
                    
                    updated_fields = []
                    for csv_col, json_key in cash_flow_fields.items():
                        if csv_col in csv_row and pd.notna(csv_row[csv_col]):
                            value = csv_row[csv_col]
                            if isinstance(value, float):
                                value = int(value)
                            record['financials']['cash_flows'][json_key] = value
                            updated_fields.append(json_key)
                    
                    if updated_fields:
                        updated_count += 1
                        fy = record.get('fiscal_year', '?')
                        q = record.get('fiscal_quarter', '?')
                        logger.info(f"FY{fy} {q}: Updated {len(updated_fields)} fields (file: {pdf_name}, matched by year/quarter)")
                    
                    break
    
    # Save updated JSON
    output_file = output_path or json_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"CSV records: {len(df)}")
    logger.info(f"JSON records: {len(data['NVDA_Firm_Fundamentals'])}")
    logger.info(f"Matched records: {matched_count}")
    logger.info(f"Updated records: {updated_count}")
    logger.info(f"Updated JSON saved to: {output_file}")
    logger.info("=" * 80)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update JSON file with cash flow data from CSV"
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path("outputs/data/cash_flow/cash_flow_values_extracted.csv"),
        help='CSV file with extracted cash flow data'
    )
    parser.add_argument(
        '--json',
        type=Path,
        default=Path("outputs/data/fundamentals/nvda_firm_fundamentals_master.json"),
        help='JSON file to update'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output JSON file (default: overwrite input JSON)'
    )
    
    args = parser.parse_args()
    
    update_json_with_cash_flows(args.csv, args.json, args.output)


if __name__ == '__main__':
    main()

