"""
Batch extract cash flow values from all NVIDIA 10-Q/10-K PDF files.

This script:
1. Reads the scan results CSV to get list of files with cash flow fields
2. Extracts actual cash flow values from each PDF
3. Saves results to CSV
4. Optionally updates JSON file
"""

from pathlib import Path
import pandas as pd
import json
import logging
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_cash_flow_from_pdf import (
    extract_cash_flow_from_pdf,
    find_cash_flow_page
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_capex_from_text(text: str) -> Optional[int]:
    """
    Extract capital expenditures from text.
    
    Args:
        text: Text content from PDF
        
    Returns:
        CapEx value in actual dollars, or None if not found
    """
    import re
    
    # First, try to find the line containing property and equipment
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in [
            'purchases related to property and equipment',
            'purchases of property and equipment',
            'purchase of property and equipment',
            'capital expenditures'
        ]):
            # Extract numbers from this line
            # Format: "Purchases related to property and equipment and intangible assets (1,227) (369)"
            # We want the first number in parentheses (current period, negative)
            paren_matches = re.findall(r'\(([\d,]+)\)', line)
            if paren_matches:
                try:
                    # Take the first number in parentheses (current period)
                    value_str = paren_matches[0].replace(',', '')
                    value = -float(value_str)  # Negative because in parentheses
                    
                    # Detect unit from context
                    unit = 'millions'
                    if 'in thousands' in text.lower():
                        unit = 'thousands'
                    elif 'in millions' in text.lower():
                        unit = 'millions'
                    
                    # Convert to actual dollars
                    if unit == 'thousands':
                        final_value = int(value * 1e3)
                    else:  # millions
                        final_value = int(value * 1e6)
                    
                    return final_value
                except (ValueError, IndexError):
                    continue
            
            # Fallback: try to find any numbers in the line
            all_numbers = re.findall(r'([\d,]+)', line)
            if all_numbers:
                try:
                    # Usually the first number after the text
                    value_str = all_numbers[0].replace(',', '')
                    value = float(value_str)
                    
                    # Check if negative (in parentheses or has minus sign)
                    if '(' in line and ')' in line:
                        value = -abs(value)
                    
                    unit = 'millions'
                    if 'in thousands' in text.lower():
                        unit = 'thousands'
                    
                    if unit == 'thousands':
                        final_value = int(value * 1e3)
                    else:
                        final_value = int(value * 1e6)
                    
                    return final_value
                except (ValueError, IndexError):
                    continue
    
    # Fallback: try regex patterns
    patterns = [
        r'purchases\s+related\s+to\s+property\s+and\s+equipment[^:]*[:\s]+([\(\)\d,\.\-\s]+)',
        r'purchases\s+of\s+property\s+and\s+equipment[^:]*[:\s]+([\(\)\d,\.\-\s]+)',
        r'purchase\s+of\s+property\s+and\s+equipment[^:]*[:\s]+([\(\)\d,\.\-\s]+)',
        r'capital\s+expenditures[:\s]+([\(\)\d,\.\-\s]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            value_str = match.group(1).strip()
            value_str = value_str.replace('$', '').replace(',', '').strip()
            
            is_negative = False
            if value_str.startswith('(') and value_str.endswith(')'):
                is_negative = True
                value_str = value_str[1:-1].strip()
            
            numbers = re.findall(r'[\d\.]+', value_str)
            if not numbers:
                continue
            
            try:
                value = float(numbers[0])
                if is_negative:
                    value = -value
                
                # Detect unit
                unit = 'millions'
                if 'in thousands' in text.lower():
                    unit = 'thousands'
                
                if unit == 'thousands':
                    final_value = int(value * 1e3)
                else:
                    final_value = int(value * 1e6)
                
                return final_value
            except (ValueError, IndexError):
                continue
    
    return None


def extract_sbc_from_text(text: str) -> Optional[int]:
    """Extract stock-based compensation from text."""
    import re
    
    patterns = [
        r'stock[-\s]based\s+compensation[:\s]+([\(\)\d,\.\-\s]+)',
        r'share[-\s]based\s+compensation[:\s]+([\(\)\d,\.\-\s]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            value_str = match.group(1).strip()
            value_str = value_str.replace('$', '').replace(',', '').strip()
            
            is_negative = False
            if value_str.startswith('(') and value_str.endswith(')'):
                is_negative = True
                value_str = value_str[1:-1].strip()
            
            numbers = re.findall(r'[\d\.]+', value_str)
            if not numbers:
                continue
            
            try:
                value = float(numbers[0])
                if is_negative:
                    value = -value
                
                # PDF typically in millions
                final_value = int(value * 1e6)
                return final_value
            except (ValueError, IndexError):
                continue
    
    return None


def extract_da_from_text(text: str) -> Optional[int]:
    """Extract depreciation and amortization from text."""
    import re
    
    patterns = [
        r'depreciation\s+and\s+amortization[:\s]+([\(\)\d,\.\-\s]+)',
        r'depreciation[:\s]+([\(\)\d,\.\-\s]+)',
        r'amortization[:\s]+([\(\)\d,\.\-\s]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            value_str = match.group(1).strip()
            value_str = value_str.replace('$', '').replace(',', '').strip()
            
            is_negative = False
            if value_str.startswith('(') and value_str.endswith(')'):
                is_negative = True
                value_str = value_str[1:-1].strip()
            
            numbers = re.findall(r'[\d\.]+', value_str)
            if not numbers:
                continue
            
            try:
                value = float(numbers[0])
                if is_negative:
                    value = -value
                
                # PDF typically in millions
                final_value = int(value * 1e6)
                return final_value
            except (ValueError, IndexError):
                continue
    
    return None


def extract_cash_balance_from_text(text: str) -> Optional[int]:
    """Extract cash and cash equivalents balance from text."""
    import re
    
    patterns = [
        r'cash\s+and\s+cash\s+equivalents\s+at\s+end\s+of\s+period[:\s]+([\(\)\d,\.\-\s]+)',
        r'cash\s+and\s+cash\s+equivalents[:\s]+([\(\)\d,\.\-\s]+)',
        r'cash\s+&\s+cash\s+equivalents[:\s]+([\(\)\d,\.\-\s]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            value_str = match.group(1).strip()
            value_str = value_str.replace('$', '').replace(',', '').strip()
            
            is_negative = False
            if value_str.startswith('(') and value_str.endswith(')'):
                is_negative = True
                value_str = value_str[1:-1].strip()
            
            numbers = re.findall(r'[\d\.]+', value_str)
            if not numbers:
                continue
            
            try:
                value = float(numbers[0])
                if is_negative:
                    value = -value
                
                # PDF typically in millions
                final_value = int(value * 1e6)
                return final_value
            except (ValueError, IndexError):
                continue
    
    return None


def extract_all_cash_flow_fields(pdf_path: Path) -> Dict[str, Optional[int]]:
    """
    Extract all cash flow related fields from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with all extracted values
    """
    import pdfplumber
    
    result = {
        'operating_cash_flow': None,
        'investing_cash_flow': None,
        'financing_cash_flow': None,
        'capital_expenditures': None,
        'free_cash_flow': None,
        'stock_based_compensation': None,
        'depreciation_amortization': None,
        'cash_balance': None,
    }
    
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return result
    
    try:
        # Find cash flow page
        page_num = find_cash_flow_page(pdf_path)
        if page_num is None:
            logger.warning(f"Could not find cash flow page in {pdf_path.name}")
            return result
        
        # Extract text from cash flow page and surrounding pages
        with pdfplumber.open(pdf_path) as pdf:
            start_page = max(0, page_num - 1)
            end_page = min(len(pdf.pages), page_num + 3)
            
            text_parts = []
            for i in range(start_page, end_page):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            full_text = '\n'.join(text_parts)
        
        # Extract main cash flow values
        main_cf = extract_cash_flow_from_pdf(pdf_path)
        result['operating_cash_flow'] = main_cf.get('operating_cash_flow')
        result['investing_cash_flow'] = main_cf.get('investing_cash_flow')
        result['financing_cash_flow'] = main_cf.get('financing_cash_flow')
        result['free_cash_flow'] = main_cf.get('free_cash_flow')
        
        # Extract additional fields
        result['capital_expenditures'] = extract_capex_from_text(full_text)
        result['stock_based_compensation'] = extract_sbc_from_text(full_text)
        result['depreciation_amortization'] = extract_da_from_text(full_text)
        result['cash_balance'] = extract_cash_balance_from_text(full_text)
        
        # Calculate free cash flow if not found but we have OCF and CapEx
        if result['free_cash_flow'] is None:
            if result['operating_cash_flow'] is not None and result['capital_expenditures'] is not None:
                result['free_cash_flow'] = result['operating_cash_flow'] + result['capital_expenditures']
                # CapEx is typically negative, so we add it (subtract absolute value)
        
    except Exception as e:
        logger.error(f"Error extracting from {pdf_path.name}: {e}")
    
    return result


def main():
    """Main batch extraction function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch extract cash flow values from all PDF files"
    )
    parser.add_argument(
        '--scan-results',
        type=Path,
        default=Path("outputs/data/cash_flow/cash_flow_field_scan_results.csv"),
        help='CSV file with scan results'
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
        default=Path("outputs/data/cash_flow/cash_flow_values_extracted.csv"),
        help='Output CSV file'
    )
    parser.add_argument(
        '--json',
        type=Path,
        default=None,
        help='Optional: JSON file to update'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Batch Extracting Cash Flow Values")
    logger.info("=" * 80)
    
    # Read scan results
    if not args.scan_results.exists():
        logger.error(f"Scan results file not found: {args.scan_results}")
        return
    
    scan_df = pd.read_csv(args.scan_results)
    logger.info(f"Found {len(scan_df)} files to process\n")
    
    # Extract values
    results = []
    for i, row in scan_df.iterrows():
        pdf_name = row['file']
        pdf_path = args.pdf_dir / pdf_name
        
        logger.info(f"[{i+1}/{len(scan_df)}] Processing {pdf_name}...")
        
        extracted = extract_all_cash_flow_fields(pdf_path)
        
        result = {
            'file': pdf_name,
            **extracted
        }
        
        results.append(result)
        
        # Log what was found
        found = [k for k, v in extracted.items() if v is not None]
        if found:
            logger.info(f"  ✓ Extracted: {', '.join(found)}")
        else:
            logger.warning(f"  ✗ No values extracted")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"\n✓ Results saved to: {args.output}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Extraction Summary")
    logger.info("=" * 80)
    
    for col in df.columns:
        if col != 'file':
            count = df[col].notna().sum()
            pct = count / len(df) * 100
            logger.info(f"  {col:30}: {count:3}/{len(df)} ({pct:5.1f}%)")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

