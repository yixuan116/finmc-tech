"""
Scan all NVIDIA 10-K / 10-Q filings and detect which cash-flow-related fields exist in each PDF.

Goal:
    1. Iterate through all PDF filings in NVDA files/.
    2. Extract full text from each PDF.
    3. Search for the presence of important cash-flow fields.
    4. For each PDF, print which fields appear (True/False).
    5. Produce a summary table showing field availability across all filings.
    6. Produce a count of how many PDFs contain each field.

Output:
    Printed summary in the terminal and saved to CSV.

This script does NOT extract values yet — it only detects which fields
exist in the filings, so we know what to extract in the next step.
"""

from pathlib import Path
import pdfplumber
import re
import pandas as pd
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FILINGS_DIR = Path("NVDA files")

# Keywords to search (case-insensitive)
KEYWORDS = {
    "ocf": [
        "net cash provided by operating activities",
        "net cash used in operating activities",
        "cash provided by operating activities",
        "cash used in operating activities",
        "cash flows from operating activities",
    ],
    "invest_cf": [
        "net cash used in investing activities",
        "net cash provided by investing activities",
        "cash used in investing activities",
        "cash provided by investing activities",
        "cash flows from investing activities",
    ],
    "finance_cf": [
        "net cash used in financing activities",
        "net cash provided by financing activities",
        "cash used in financing activities",
        "cash provided by financing activities",
        "cash flows from financing activities",
    ],
    "capex": [
        "purchases of property and equipment",
        "purchase of property and equipment",
        "capital expenditures",
        "capital expenditure",
        "capex",
    ],
    "fcf": [
        "free cash flow",
        "free cashflow",
    ],
    "sbc": [
        "stock-based compensation",
        "stock based compensation",
        "share-based compensation",
    ],
    "da": [
        "depreciation and amortization",
        "depreciation",
        "amortization",
    ],
    "cash_balance": [
        "cash and cash equivalents",
        "cash & cash equivalents",
    ],
    "change_in_cash": [
        "change in cash and cash equivalents",
        "change in cash",
        "net increase in cash",
        "net decrease in cash",
    ],
}

records = []


def scan_pdf(pdf_path: Path) -> Dict[str, bool]:
    """
    Scan a PDF file for cash flow related keywords.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with field detection results
    """
    detected = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from all pages
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = "\n".join(text_parts).lower()
        
        # Check for each field
        for field, kws in KEYWORDS.items():
            detected[field] = any(kw in full_text for kw in kws)
        
        detected["file"] = pdf_path.name
        detected["file_size_kb"] = pdf_path.stat().st_size / 1024
        detected["num_pages"] = len(pdf.pages)
        
        return detected
        
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return {
            "file": pdf_path.name,
            "error": str(e),
            **{field: False for field in KEYWORDS.keys()}
        }


def main():
    """Main scanning function."""
    logger.info("=" * 80)
    logger.info("Scanning PDF filings for Cash Flow Fields")
    logger.info("=" * 80)
    logger.info(f"Scanning directory: {FILINGS_DIR.resolve()}\n")
    
    # Get all PDF files, but prioritize 10-Q and 10-K files
    all_pdf_files = list(FILINGS_DIR.glob("*.pdf"))
    
    # Filter to only 10-Q and 10-K files (skip press releases, transcripts, etc.)
    # These typically have "10-Q", "10K", "10-K" in filename
    pdf_files = [
        f for f in all_pdf_files 
        if any(keyword in f.name.upper() for keyword in ['10-Q', '10K', '10-K', '10Q'])
    ]
    
    logger.info(f"Found {len(all_pdf_files)} total PDF files")
    logger.info(f"Filtering to {len(pdf_files)} 10-Q/10-K files\n")
    
    # Scan each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Scanning {pdf_path.name}...")
        
        detected = scan_pdf(pdf_path)
        records.append(detected)
        
        # Print detected fields
        detected_fields = [k for k, v in detected.items() if v and k not in ['file', 'file_size_kb', 'num_pages', 'error']]
        if detected_fields:
            logger.info(f"  ✓ Found: {', '.join(detected_fields)}")
        else:
            logger.info(f"  ✗ No cash flow fields detected")
    
    # Create summary DataFrame
    df = pd.DataFrame(records)
    
    # Reorder columns
    field_cols = list(KEYWORDS.keys())
    other_cols = ['file', 'file_size_kb', 'num_pages', 'error']
    col_order = ['file'] + field_cols + [c for c in other_cols if c != 'file']
    df = df[[c for c in col_order if c in df.columns]]
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    print("\n" + df.to_string(index=False))
    
    # Count how many filings contain each field
    logger.info("\n" + "=" * 80)
    logger.info("FIELD APPEARANCE COUNT")
    logger.info("=" * 80)
    
    field_counts = {}
    for field in KEYWORDS.keys():
        if field in df.columns:
            count = df[field].sum()
            total = len(df[df[field].notna()])
            field_counts[field] = {
                'count': count,
                'total': total,
                'percentage': (count / total * 100) if total > 0 else 0
            }
    
    count_df = pd.DataFrame(field_counts).T
    count_df.columns = ['Files with Field', 'Total Files', 'Percentage']
    count_df = count_df.sort_values('Files with Field', ascending=False)
    print("\n" + count_df.to_string())
    
    # Save to CSV
    output_csv = Path("outputs/data/cash_flow/cash_flow_field_scan_results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"\n✓ Results saved to: {output_csv}")
    
    # Save summary counts
    summary_csv = Path("outputs/data/cash_flow/cash_flow_field_summary.csv")
    count_df.to_csv(summary_csv)
    logger.info(f"✓ Summary saved to: {summary_csv}")
    
    # Print recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    high_coverage = count_df[count_df['Percentage'] >= 80]
    if len(high_coverage) > 0:
        logger.info("\nFields with high coverage (>=80%):")
        for field in high_coverage.index:
            logger.info(f"  ✓ {field}: {high_coverage.loc[field, 'Percentage']:.1f}%")
    
    low_coverage = count_df[count_df['Percentage'] < 50]
    if len(low_coverage) > 0:
        logger.info("\nFields with low coverage (<50%):")
        for field in low_coverage.index:
            logger.info(f"  ⚠ {field}: {low_coverage.loc[field, 'Percentage']:.1f}%")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()

