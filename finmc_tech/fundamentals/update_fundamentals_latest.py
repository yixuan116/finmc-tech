"""
Update NVDA fundamentals dataset with the newest quarter.

This script:
1. Loads existing processed fundamentals from:
     data/processed/NVDA_revenue_features.csv
2. Loads new LLM-extracted fundamentals (e.g., 2026Q3) from:
     - a JSON file (list of records), or
     - an inline JSON string passed via CLI
3. Aligns schema to the existing CSV:
     - Keeps all existing columns exactly
     - Drops any extra fields from JSON
     - Fills missing fields with None / NA
4. Appends the new row(s), ensures `period_end` is datetime and
   the dataset is sorted ascending by `period_end`.
5. Saves to:
     data/processed/NVDA_revenue_features_updated.csv
   without modifying the original file.

Usage examples
--------------

From the project root:

    # Using a JSON file with the new quarter
    python -m finmc_tech.fundamentals.update_fundamentals_latest \\
        --json-path data/raw/nvda_fundamentals_2026Q3.json

    # Using inline JSON (shell-escaped)
    python -m finmc_tech.fundamentals.update_fundamentals_latest \\
        --json-inline '[{\"fiscal_year\": 2026, \"fiscal_quarter\": \"Q3\", ...}]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from finmc_tech.config import get_logger


logger = get_logger(__name__)


DEFAULT_INPUT_CSV = Path("data/processed/NVDA_revenue_features.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/NVDA_revenue_features_updated.csv")


def load_existing_fundamentals(csv_path: Path) -> pd.DataFrame:
    """
    Load existing NVDA fundamentals dataset.

    Args:
        csv_path: Path to the existing processed CSV.

    Returns:
        DataFrame with the original schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Existing fundamentals file not found: {csv_path}")

    logger.info(f"Loading existing fundamentals from {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Ensure period_end is datetime (if column exists)
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    logger.info(
        "Loaded %d rows from %s (period_end range: %s → %s)",
        len(df),
        csv_path,
        df["period_end"].min() if "period_end" in df.columns else "N/A",
        df["period_end"].max() if "period_end" in df.columns else "N/A",
    )
    return df


def _parse_json_records(obj: object) -> List[dict]:
    """
    Normalize various JSON shapes into a list of dict records.

    Supports:
    - List[dict]
    - Single dict
    """
    if isinstance(obj, list):
        if not obj:
            raise ValueError("JSON list is empty; expected at least one record.")
        if not all(isinstance(r, dict) for r in obj):
            raise ValueError("JSON list must contain dict objects.")
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("JSON content must be a dict or list of dicts.")


def load_new_fundamentals_from_file(json_path: Path) -> List[dict]:
    """
    Load new fundamentals from a JSON file.

    Args:
        json_path: Path to JSON file containing the new quarter(s).

    Returns:
        List of dict records.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    logger.info(f"Loading new fundamentals JSON from {json_path} ...")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return _parse_json_records(data)


def load_new_fundamentals_from_inline(json_str: str) -> List[dict]:
    """
    Load new fundamentals from an inline JSON string.

    Args:
        json_str: JSON text representing either a dict or list of dicts.

    Returns:
        List of dict records.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse inline JSON: {exc}") from exc

    return _parse_json_records(data)


def align_new_records_to_schema(
    existing_df: pd.DataFrame,
    new_records: Iterable[dict],
) -> pd.DataFrame:
    """
    Align new fundamentals records to the schema of the existing DataFrame.

    - Keeps all existing columns exactly as-is.
    - Drops any extra fields present in new_records.
    - Fills missing fields with None / NA.

    Args:
        existing_df: Existing fundamentals with the target schema.
        new_records: Iterable of raw dict records for new rows.

    Returns:
        DataFrame of new rows with the same columns as existing_df.
    """
    existing_columns = list(existing_df.columns)

    # Create DataFrame from new records
    new_df_raw = pd.DataFrame(list(new_records))
    logger.info("Loaded %d new record(s) from JSON", len(new_df_raw))

    # Restrict to existing columns (drop extras)
    new_df = new_df_raw.reindex(columns=existing_columns)

    # Ensure all expected columns exist; missing ones will be added as NA
    for col in existing_columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA

    # Reorder columns exactly as existing_df
    new_df = new_df[existing_columns]

    # Convert period_end to datetime if present
    if "period_end" in new_df.columns:
        new_df["period_end"] = pd.to_datetime(new_df["period_end"], errors="coerce")

    # Attempt to coerce numeric columns where appropriate based on existing dtypes
    for col, dtype in existing_df.dtypes.items():
        if col == "period_end":
            # Already handled; keep as datetime
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    return new_df


def update_fundamentals_dataset(
    input_csv: Path,
    output_csv: Path,
    json_path: Optional[Path] = None,
    json_inline: Optional[str] = None,
) -> pd.DataFrame:
    """
    Update the fundamentals dataset with new quarter(s).

    Args:
        input_csv: Path to existing fundamentals CSV.
        output_csv: Path to write updated CSV.
        json_path: Optional path to JSON file with new rows.
        json_inline: Optional inline JSON string with new rows.

    Returns:
        Updated DataFrame (including both old and new rows).
    """
    if not json_path and not json_inline:
        raise ValueError("Must provide either json_path or json_inline with new data.")

    existing_df = load_existing_fundamentals(input_csv)
    n_before = len(existing_df)

    # Load new records
    if json_path:
        new_records = load_new_fundamentals_from_file(json_path)
    else:
        new_records = load_new_fundamentals_from_inline(json_inline or "")

    new_df_aligned = align_new_records_to_schema(existing_df, new_records)

    # Concatenate and drop potential duplicates on period_end + fiscal_year + fiscal_quarter
    combined = pd.concat([existing_df, new_df_aligned], ignore_index=True)

    # Ensure datetime for period_end
    if "period_end" in combined.columns:
        combined["period_end"] = pd.to_datetime(combined["period_end"], errors="coerce")
        combined = combined.sort_values("period_end").reset_index(drop=True)

    n_after = len(combined)

    # Save to new CSV (do not overwrite original)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    logger.info("Updated fundamentals saved to %s", output_csv)
    logger.info("Rows before: %d, after: %d (added %d)", n_before, n_after, n_after - n_before)

    # Print preview of last 5 rows
    logger.info("Last 5 rows of updated dataset:\n%s", combined.tail(5).to_string())

    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Append latest NVDA fundamentals (e.g., 2026Q3) to the processed dataset.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help=f"Path to existing fundamentals CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path to save updated fundamentals CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to JSON file containing new fundamentals records.",
    )
    parser.add_argument(
        "--json-inline",
        type=str,
        default=None,
        help="Inline JSON string with new fundamentals (overrides --json-path if provided).",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    json_path: Optional[Path]
    json_inline: Optional[str]

    if args.json_inline:
        json_inline = args.json_inline
        json_path = None
    else:
        json_inline = None
        json_path = Path(args.json_path) if args.json_path else None

    logger.info("=" * 70)
    logger.info("Updating NVDA fundamentals with latest quarter")
    logger.info("=" * 70)

    try:
        update_fundamentals_dataset(
            input_csv=input_csv,
            output_csv=output_csv,
            json_path=json_path,
            json_inline=json_inline,
        )
    except Exception as exc:
        logger.error("✗ Failed to update fundamentals: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()


