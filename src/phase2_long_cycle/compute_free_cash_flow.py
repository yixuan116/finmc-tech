"""
Compute Free Cash Flow and related metrics for NVIDIA fundamentals data.

This script:
1. Loads the master JSON file with NVIDIA fundamentals
2. Computes Free Cash Flow (FCF) = Operating Cash Flow - Capital Expenditures
3. Computes FCF margin = FCF / Revenue
4. Computes FCF conversion = FCF / Net Income
5. Writes results back to JSON (with backup)

Usage:
    python -m src.phase2_long_cycle.compute_free_cash_flow
    python -m src.phase2_long_cycle.compute_free_cash_flow --input-json outputs/data/fundamentals/nvda_firm_fundamentals_master.json
    python -m src.phase2_long_cycle.compute_free_cash_flow --input-json outputs/data/fundamentals/nvda_firm_fundamentals_master.json --output-json outputs/data/fundamentals/nvda_firm_fundamentals_master.json
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_float(value: Any) -> Optional[float]:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        
    Returns:
        Float value or None if conversion fails
    """
    if value is None:
        return None
    
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Remove commas and other formatting
            cleaned = value.replace(',', '').strip()
            if cleaned:
                return float(cleaned)
        return None
    except (ValueError, TypeError, AttributeError):
        logger.warning(f"Could not convert value to float: {value}")
        return None


def compute_fcf_metrics(
    cfo: Optional[float],
    capex_outflow: Optional[float],
    revenue: Optional[float],
    net_income: Optional[float]
) -> Dict[str, Optional[float]]:
    """
    Compute Free Cash Flow and related metrics.
    
    Args:
        cfo: Operating Cash Flow
        capex_outflow: Capital Expenditures (as positive outflow)
        revenue: Revenue
        net_income: Net Income (GAAP)
        
    Returns:
        Dictionary with computed metrics
    """
    result = {
        'free_cash_flow': None,
        'free_cash_flow_margin': None,
        'fcf_conversion': None
    }
    
    # Compute FCF
    if cfo is not None and capex_outflow is not None:
        result['free_cash_flow'] = cfo - capex_outflow
    else:
        return result
    
    fcf = result['free_cash_flow']
    
    # Compute FCF margin
    if fcf is not None and revenue is not None and revenue != 0:
        result['free_cash_flow_margin'] = fcf / revenue
    else:
        result['free_cash_flow_margin'] = None
    
    # Compute FCF conversion
    if fcf is not None and net_income is not None and net_income != 0:
        result['fcf_conversion'] = fcf / net_income
    else:
        result['fcf_conversion'] = None
    
    return result


def process_period(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single period record and compute FCF metrics.
    
    Args:
        record: JSON record for a fiscal period
        
    Returns:
        Dictionary with processing results
    """
    fiscal_year = record.get('fiscal_year')
    fiscal_quarter = record.get('fiscal_quarter', '?')
    
    # Extract financials
    financials = record.get('financials', {})
    
    # Extract required values
    revenue = safe_float(financials.get('revenue'))
    net_income = safe_float(financials.get('net_income_gaap'))
    
    # Extract CFO from cash_flows
    cash_flows = financials.get('cash_flows', {})
    cfo = safe_float(cash_flows.get('operating_cash_flow'))
    
    # Extract CapEx (try both names)
    capex_primary = safe_float(financials.get('capex'))
    capex_fallback = safe_float(financials.get('capital_expenditures'))
    
    # Decide which CapEx to use
    if capex_primary is not None:
        capex_raw = capex_primary
    elif capex_fallback is not None:
        capex_raw = capex_fallback
    else:
        capex_raw = None
    
    # Normalize CapEx sign (we want positive outflow)
    if capex_raw is not None:
        capex_outflow = abs(float(capex_raw))
    else:
        capex_outflow = None
    
    # Compute FCF metrics
    fcf_metrics = compute_fcf_metrics(cfo, capex_outflow, revenue, net_income)
    
    # Update financials with computed values
    if 'free_cash_flow' in fcf_metrics and fcf_metrics['free_cash_flow'] is not None:
        financials['free_cash_flow'] = fcf_metrics['free_cash_flow']
    else:
        financials['free_cash_flow'] = None
    
    if 'free_cash_flow_margin' in fcf_metrics and fcf_metrics['free_cash_flow_margin'] is not None:
        financials['free_cash_flow_margin'] = fcf_metrics['free_cash_flow_margin']
    else:
        financials['free_cash_flow_margin'] = None
    
    if 'fcf_conversion' in fcf_metrics and fcf_metrics['fcf_conversion'] is not None:
        financials['fcf_conversion'] = fcf_metrics['fcf_conversion']
    else:
        financials['fcf_conversion'] = None
    
    # Return processing result for summary
    return {
        'fiscal_year': fiscal_year,
        'fiscal_quarter': fiscal_quarter,
        'cfo_available': cfo is not None,
        'capex_available': capex_outflow is not None,
        'fcf_computed': fcf_metrics['free_cash_flow'] is not None,
        'fcf_value': fcf_metrics['free_cash_flow']
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute Free Cash Flow metrics for NVIDIA fundamentals"
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        default=Path("outputs/data/fundamentals/nvda_firm_fundamentals_master.json"),
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Output JSON file path (default: same as input, overwrite after backup)'
    )
    
    args = parser.parse_args()
    
    input_path = args.input_json
    output_path = args.output_json or input_path
    
    logger.info("=" * 80)
    logger.info("Computing Free Cash Flow Metrics")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}\n")
    
    # Load JSON
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return
    
    if 'NVDA_Firm_Fundamentals' not in data:
        logger.error("JSON structure invalid: missing 'NVDA_Firm_Fundamentals'")
        return
    
    records = data['NVDA_Firm_Fundamentals']
    logger.info(f"Loaded {len(records)} periods\n")
    
    # Process each period
    results = []
    for record in records:
        result = process_period(record)
        results.append(result)
    
    # Create backup before writing
    if output_path.exists() and output_path == input_path:
        backup_path = Path(str(output_path) + '.bak')
        logger.info(f"Creating backup: {backup_path}")
        shutil.copy2(output_path, backup_path)
        logger.info(f"✓ Backup created\n")
    
    # Write updated JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Updated JSON saved to: {output_path}\n")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    # Summary table
    print("\nFiscalYear Quarter | CFO_available | CapEx_available | FCF_computed")
    print("-" * 80)
    
    for r in results[:10]:  # Show first 10
        fy = r['fiscal_year']
        q = r['fiscal_quarter']
        cfo = "yes" if r['cfo_available'] else "no"
        capex = "yes" if r['capex_available'] else "no"
        fcf = "yes" if r['fcf_computed'] else "no"
        print(f"{fy:9} {q:7} | {cfo:13} | {capex:15} | {fcf:12}")
    
    if len(results) > 10:
        print(f"... ({len(results) - 10} more periods)")
    
    # Aggregate counts
    total = len(results)
    fcf_computed = sum(1 for r in results if r['fcf_computed'])
    missing_cfo = sum(1 for r in results if not r['cfo_available'])
    missing_capex = sum(1 for r in results if not r['capex_available'])
    
    print("\n" + "=" * 80)
    print("Aggregate Statistics:")
    print("=" * 80)
    print(f"Total periods:     {total}")
    print(f"FCF computed:      {fcf_computed} ({fcf_computed/total*100:.1f}%)")
    print(f"Missing CFO:       {missing_cfo}")
    print(f"Missing CapEx:     {missing_capex}")
    print("=" * 80)
    
    # Show sample FCF values
    fcf_values = [r['fcf_value'] for r in results if r['fcf_value'] is not None]
    if fcf_values:
        print(f"\nSample FCF values:")
        print(f"  Min: ${min(fcf_values):,.0f}")
        print(f"  Max: ${max(fcf_values):,.0f}")
        print(f"  Avg: ${sum(fcf_values)/len(fcf_values):,.0f}")
    
    logger.info("\n✓ Processing complete!")


if __name__ == '__main__':
    main()

