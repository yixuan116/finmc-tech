"""
Extract cash flow data from NVDA 10-Q and 10-K PDF files.

This script searches for Statement of Cash Flows in PDF files and extracts:
- Operating Cash Flow (CFO)
- Investing Cash Flow (CFI)
- Financing Cash Flow (CFF)
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict

import pdfplumber
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_cash_flow_page(pdf_path: Path) -> Optional[int]:
    """
    Find the page number containing Cash Flow Statement.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Page number (0-indexed) or None if not found
    """
    logger.info(f"Searching for Cash Flow Statement in {pdf_path.name}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_lower = text.lower()
                # Look for cash flow statement indicators
                if any(keyword in text_lower for keyword in [
                    'statement of cash flows',
                    'cash flow statement',
                    'cash flows from operating',
                    'cash provided by operating'
                ]):
                    logger.info(f"Found Cash Flow Statement on page {i+1}")
                    return i
    
    logger.warning("Cash Flow Statement not found")
    return None


def extract_cash_flow_values(text: str) -> Dict[str, Optional[float]]:
    """
    Extract cash flow values from text using regex patterns.
    
    Args:
        text: Text content from PDF
        
    Returns:
        Dictionary with cash flow values
    """
    result = {
        'operating_cash_flow': None,
        'investing_cash_flow': None,
        'financing_cash_flow': None,
        'free_cash_flow': None
    }
    
    # Patterns to match cash flow values
    # PDF format: "Net cash provided by operating activities 27,414 15,345"
    # We need to extract the first number (current quarter)
    patterns = {
        'operating': [
            r'net\s+cash\s+provided\s+by\s+operating\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'net\s+cash\s+from\s+operating\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'cash\s+provided\s+by\s+operating\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'operating\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
        ],
        'investing': [
            r'net\s+cash\s+(?:used\s+in|provided\s+by)\s+investing\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'cash\s+(?:used\s+in|provided\s+by)\s+investing\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'investing\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
        ],
        'financing': [
            r'net\s+cash\s+(?:used\s+in|provided\s+by)\s+financing\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'cash\s+(?:used\s+in|provided\s+by)\s+financing\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'financing\s+activities\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
        ],
        'free': [
            r'free\s+cash\s+flow\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
            r'fcf\s+([\$]?\s*[\(\)\d,\.\-\s]+)',
        ]
    }
    
    # Map flow types to result keys
    key_map = {
        'operating': 'operating_cash_flow',
        'investing': 'investing_cash_flow',
        'financing': 'financing_cash_flow',
        'free': 'free_cash_flow'
    }
    
    # Try to extract values
    for flow_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                value_str = match.group(1).strip()
                # Remove $ sign and clean the value string
                value_str = value_str.replace('$', '').replace(',', '').strip()
                
                # Handle parentheses notation for negative numbers: (5,216) -> -5216
                is_negative = False
                if value_str.startswith('(') and value_str.endswith(')'):
                    is_negative = True
                    value_str = value_str[1:-1].strip()
                
                # Extract the first number (current quarter value)
                # Format might be: "27,414 15,345" - we want the first one
                numbers = re.findall(r'[\d\.]+', value_str)
                if not numbers:
                    continue
                
                try:
                    # Take the first number (current quarter)
                    value = float(numbers[0])
                    if is_negative:
                        value = -value
                    
                    # PDF says "(In millions)", so convert to actual dollars
                    final_value = int(value * 1e6)
                    
                    result[key_map[flow_type]] = final_value
                    logger.info(f"Extracted {flow_type}: {final_value:,} (from '{match.group(0)[:80]}...')")
                    break
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse value '{value_str}': {e}")
                    continue
            if result[key_map[flow_type]] is not None:
                break
    
    return result


def extract_cash_flow_from_pdf(pdf_path: Path) -> Dict[str, Optional[float]]:
    """
    Extract cash flow data from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with cash flow values
    """
    logger.info(f"Extracting cash flow from {pdf_path.name}")
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return {
            'operating_cash_flow': None,
            'investing_cash_flow': None,
            'financing_cash_flow': None,
            'free_cash_flow': None
        }
    
    # Find the page with cash flow statement
    page_num = find_cash_flow_page(pdf_path)
    
    if page_num is None:
        logger.warning("Could not find Cash Flow Statement page")
        return {
            'operating_cash_flow': None,
            'investing_cash_flow': None,
            'financing_cash_flow': None,
            'free_cash_flow': None
        }
    
    # Extract text from the cash flow page and surrounding pages
    with pdfplumber.open(pdf_path) as pdf:
        # Get text from the cash flow page and a few pages around it
        start_page = max(0, page_num - 1)
        end_page = min(len(pdf.pages), page_num + 3)
        
        text_parts = []
        for i in range(start_page, end_page):
            page = pdf.pages[i]
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        full_text = '\n'.join(text_parts)
    
    # Extract values
    cash_flow_data = extract_cash_flow_values(full_text)
    
    return cash_flow_data


def update_json_with_cash_flows(
    json_path: Path = Path("outputs/data/fundamentals/nvda_firm_fundamentals_master.json"),
    pdf_dir: Path = Path("NVDA files")
) -> None:
    """
    Extract cash flow data and update JSON file directly.
    
    Args:
        json_path: Path to JSON file to update
        pdf_dir: Directory containing PDF files
    """
    logger.info("=" * 60)
    logger.info("Extracting Cash Flow Data for FY2026 and Updating JSON")
    logger.info("=" * 60)
    
    # Load existing JSON
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # FY2026 PDF files
    pdf_files = {
        'Q1': pdf_dir / "Q1 26 10-Q.pdf",
        'Q2': pdf_dir / "Q2 26 10-Q.pdf",
        'Q3': pdf_dir / "Q3 26 10-Q.pdf",
    }
    
    # Update each quarter
    updated_count = 0
    for quarter, pdf_path in pdf_files.items():
        logger.info(f"\nProcessing FY2026 {quarter}...")
        
        # Find the corresponding quarter in JSON
        quarter_data = None
        for q in data['NVDA_Firm_Fundamentals']:
            if q['fiscal_year'] == 2026 and q['fiscal_quarter'] == quarter:
                quarter_data = q
                break
        
        if not quarter_data:
            logger.warning(f"FY2026 {quarter} not found in JSON, skipping...")
            continue
        
        # Extract cash flow data
        cash_flow_data = extract_cash_flow_from_pdf(pdf_path)
        
        # Initialize financials.cash_flows if it doesn't exist
        if 'financials' not in quarter_data:
            quarter_data['financials'] = {}
        
        if 'cash_flows' not in quarter_data['financials']:
            quarter_data['financials']['cash_flows'] = {}
        
        # Update cash flow data
        for key, value in cash_flow_data.items():
            if value is not None:
                quarter_data['financials']['cash_flows'][key] = value
                logger.info(f"  Updated {key}: ${value:,}")
            else:
                logger.info(f"  {key}: Not found (skipping)")
        
        updated_count += 1
    
    # Save updated JSON
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Updating JSON file: {json_path}")
    logger.info(f"{'=' * 60}")
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Successfully updated {updated_count} quarters in JSON file")
    logger.info(f"✓ JSON file saved to: {json_path}")


def extract_fy2026_cash_flows(pdf_dir: Path = Path("NVDA files")) -> pd.DataFrame:
    """
    Extract cash flow data for FY2026 quarters (legacy function for CSV output).
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        DataFrame with extracted cash flow data
    """
    logger.info("=" * 60)
    logger.info("Extracting Cash Flow Data for FY2026")
    logger.info("=" * 60)
    
    # FY2026 PDF files
    pdf_files = {
        'Q1': pdf_dir / "Q1 26 10-Q.pdf",
        'Q2': pdf_dir / "Q2 26 10-Q.pdf",
        'Q3': pdf_dir / "Q3 26 10-Q.pdf",
    }
    
    results = []
    
    for quarter, pdf_path in pdf_files.items():
        logger.info(f"\nProcessing FY2026 {quarter}...")
        
        cash_flow_data = extract_cash_flow_from_pdf(pdf_path)
        
        result = {
            'fiscal_year': 2026,
            'fiscal_quarter': quarter,
            'pdf_file': pdf_path.name,
            **cash_flow_data
        }
        
        results.append(result)
        
        # Print summary
        logger.info(f"FY2026 {quarter} Results:")
        for key, value in cash_flow_data.items():
            if value is not None:
                logger.info(f"  {key}: ${value:,}")
            else:
                logger.info(f"  {key}: Not found")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"\n{df.to_string()}")
    
    return df


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract cash flow data from NVDA PDF files and update JSON"
    )
    parser.add_argument(
        '--pdf-dir',
        type=Path,
        default=Path("NVDA files"),
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--json',
        type=Path,
        default=Path("outputs/data/fundamentals/nvda_firm_fundamentals_master.json"),
        help='JSON file to update'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=None,
        help='Optional: Also save to CSV file'
    )
    
    args = parser.parse_args()
    
    # Update JSON file directly
    update_json_with_cash_flows(args.json, args.pdf_dir)
    
    # Optionally save to CSV
    if args.csv:
        df = extract_fy2026_cash_flows(args.pdf_dir)
        df.to_csv(args.csv, index=False)
        logger.info(f"\n✓ Also saved to CSV: {args.csv}")


if __name__ == '__main__':
    main()

