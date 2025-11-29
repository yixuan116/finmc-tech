"""
Refresh NVDA firm-level revenue and derived features to the latest quarter.

This script reuses the original SEC XBRL data source and feature pipeline:

- Raw revenue data via `src.data.fetch_sec_revenue.fetch_revenues_panel`
- Revenue + feature engineering via `src.data.create_nvda_revenue_features`

It:
1. Loads existing raw and processed files:
     - data/raw/NVDA_revenue.csv
     - data/processed/NVDA_revenue_features.csv
2. Fetches the latest quarterly revenue data for NVDA from the SEC XBRL API.
3. Extends the datasets **only with new quarters after the last existing one**.
4. Writes updated versions to:
     - data/raw/NVDA_revenue_latest.csv
     - data/processed/NVDA_revenue_features_latest.csv
5. Preserves the original CSVs (no overwrite).

Run from project root:

    python -m finmc_tech.fundamentals.refresh_nvda_revenue
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from finmc_tech.config import get_logger


logger = get_logger(__name__)


# Add project root to sys.path so we can import legacy src.* modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_sec_revenue import fetch_revenues_panel  # type: ignore  # noqa: E402
from src.data.create_nvda_revenue_features import (  # type: ignore  # noqa: E402
    add_price_data,
    clean_and_deduplicate,
    create_forward_return_target,
    create_revenue_features,
    load_revenue_data,
)
from src.data.create_extended_features import (  # type: ignore  # noqa: E402
    create_all_extended_features,
)


RAW_EXISTING_PATH = PROJECT_ROOT / "data" / "raw" / "NVDA_revenue.csv"
FEAT_EXISTING_PATH = PROJECT_ROOT / "data" / "processed" / "NVDA_revenue_features.csv"

RAW_LATEST_PATH = PROJECT_ROOT / "data" / "raw" / "NVDA_revenue_latest.csv"
FEAT_LATEST_PATH = PROJECT_ROOT / "data" / "processed" / "NVDA_revenue_features_latest.csv"

PANEL_LATEST_PATH = PROJECT_ROOT / "outputs" / "revenues_panel_nvda_latest.csv"


def _load_existing_raw() -> pd.DataFrame:
    """Load existing raw NVDA revenue data."""
    if not RAW_EXISTING_PATH.exists():
        raise FileNotFoundError(f"Existing raw revenue file not found: {RAW_EXISTING_PATH}")

    df = pd.read_csv(RAW_EXISTING_PATH)
    if "period_end" not in df.columns:
        raise ValueError(f"'period_end' column not found in {RAW_EXISTING_PATH}")

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df


def _load_existing_features() -> pd.DataFrame:
    """Load existing processed NVDA revenue features."""
    if not FEAT_EXISTING_PATH.exists():
        raise FileNotFoundError(
            f"Existing processed revenue features file not found: {FEAT_EXISTING_PATH}"
        )

    df = pd.read_csv(FEAT_EXISTING_PATH)
    if "period_end" not in df.columns:
        raise ValueError(f"'period_end' column not found in {FEAT_EXISTING_PATH}")

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df


def _align_to_schema(
    base_df: pd.DataFrame,
    new_df: pd.DataFrame,
    rename_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Align a new DataFrame to the schema (columns + dtypes) of a base DataFrame.

    - Optionally renames columns in new_df using rename_map.
    - Keeps only columns present in base_df.
    - Adds any missing columns as NA.
    - Orders columns to match base_df.
    - Attempts to coerce numeric columns to match base dtypes.
    """
    aligned = new_df.copy()

    if rename_map:
        aligned = aligned.rename(columns=rename_map)

    base_cols = list(base_df.columns)

    # Restrict to base columns; missing ones will be added below
    aligned = aligned.reindex(columns=base_cols)

    # Ensure all columns exist
    for col in base_cols:
        if col not in aligned.columns:
            aligned[col] = pd.NA

    # Match dtypes where reasonable
    for col, dtype in base_df.dtypes.items():
        if col not in aligned.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            aligned[col] = pd.to_datetime(aligned[col], errors="coerce")
        elif pd.api.types.is_numeric_dtype(dtype):
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        # For object/string, leave as-is

    # Reorder exactly like base_df
    aligned = aligned[base_cols]
    return aligned


def refresh_nvda_revenue() -> None:
    """
    Refresh NVDA revenue and feature datasets to include the latest quarter.

    Steps:
    1. Load existing raw & feature CSVs to determine last `period_end`.
    2. Fetch full NVDA revenue history from SEC XBRL via fetch_revenues_panel.
    3. Standardize & deduplicate via load_revenue_data + clean_and_deduplicate.
    4. Compute price, revenue features, macro/extended features, forward returns.
    5. Append **only new quarters** (after last existing period_end) to:
         - NVDA_revenue_latest.csv
         - NVDA_revenue_features_latest.csv
    """
    logger.info("=" * 70)
    logger.info("Refreshing NVDA revenue and features from SEC XBRL")
    logger.info("=" * 70)

    # Load existing datasets
    raw_existing = _load_existing_raw()
    feat_existing = _load_existing_features()

    last_period_end_raw = raw_existing["period_end"].max()
    last_period_end_feat = feat_existing["period_end"].max()
    last_period_end = max(last_period_end_raw, last_period_end_feat)

    logger.info(
        "Existing data up to period_end: %s (raw) / %s (features)",
        last_period_end_raw.date(),
        last_period_end_feat.date(),
    )

    # 1. Fetch latest revenue panel for NVDA from SEC XBRL
    logger.info("Fetching latest revenue panel for NVDA from SEC XBRL...")
    panel_df = fetch_revenues_panel(["NVDA"], output_path=str(PANEL_LATEST_PATH))

    logger.info(
        "Fetched %d raw SEC records for NVDA (end_date range: %s → %s)",
        len(panel_df),
        panel_df["end_date"].min(),
        panel_df["end_date"].max(),
    )

    # 2. Standardize and deduplicate (same logic as original pipeline)
    revenue_df = load_revenue_data(input_file=str(PANEL_LATEST_PATH))
    revenue_df = clean_and_deduplicate(revenue_df)

    logger.info(
        "After cleaning / deduplication: %d records (end_date range: %s → %s)",
        len(revenue_df),
        revenue_df["end_date"].min().date(),
        revenue_df["end_date"].max().date(),
    )

    # 3. Identify new quarters (end_date > last_period_end)
    revenue_df["end_date"] = pd.to_datetime(revenue_df["end_date"], errors="coerce")
    new_revenue = revenue_df[revenue_df["end_date"] > last_period_end].copy()

    if new_revenue.empty:
        logger.info("No new quarters found after %s. Nothing to update.", last_period_end.date())
        # Still write *_latest files as copies for reproducibility
        raw_existing.to_csv(RAW_LATEST_PATH, index=False)
        feat_existing.to_csv(FEAT_LATEST_PATH, index=False)
        logger.info("Copied existing files to *_latest without changes.")
        return

    logger.info("Found %d new quarter(s) after %s.", len(new_revenue), last_period_end.date())

    # 4. Build latest raw revenue CSV
    # Map standardized columns back to NVDA_revenue.csv schema
    # NVDA_revenue.csv columns: period_end,revenue,fy,fp,form,tag_used,ticker
    rename_map_raw = {
        "end_date": "period_end",
        "period": "fp",
        "tag": "tag_used",
    }
    # Align all cleaned revenue records (not just new) to schema,
    # then keep only rows strictly newer than last_period_end to append.
    aligned_revenue_all = _align_to_schema(
        base_df=raw_existing,
        new_df=revenue_df,
        rename_map=rename_map_raw,
    )

    aligned_revenue_all["period_end"] = pd.to_datetime(
        aligned_revenue_all["period_end"], errors="coerce"
    )
    new_raw_rows = aligned_revenue_all[aligned_revenue_all["period_end"] > last_period_end].copy()

    # Combine with existing raw data; keep original rows for overlapping periods
    raw_latest = pd.concat([raw_existing, new_raw_rows], ignore_index=True)
    raw_latest["period_end"] = pd.to_datetime(raw_latest["period_end"], errors="coerce")
    raw_latest = (
        raw_latest.sort_values("period_end")
        .drop_duplicates(subset=["ticker", "period_end", "fp"], keep="first")
        .reset_index(drop=True)
    )

    # 5. Build latest feature dataset
    # Compute price + revenue features + extended features + forward returns
    logger.info("Computing updated revenue features (including new quarters)...")
    feat_full = add_price_data(revenue_df, ticker="NVDA")
    feat_full = create_revenue_features(feat_full)
    feat_full = create_all_extended_features(feat_full, ticker="NVDA")
    feat_full = create_forward_return_target(feat_full, trading_days=252)

    # Rename columns to match existing features schema where needed
    rename_map_feat = {
        "end_date": "period_end",
        "period": "fp",
        "tag": "tag_used",
    }
    feat_full_renamed = feat_full.rename(columns=rename_map_feat)
    if "period_end" not in feat_full_renamed.columns:
        raise ValueError("Processed features are missing 'period_end' column after renaming.")

    feat_full_renamed["period_end"] = pd.to_datetime(
        feat_full_renamed["period_end"], errors="coerce"
    )

    # Keep only new periods > last_period_end
    new_feat_rows = feat_full_renamed[feat_full_renamed["period_end"] > last_period_end].copy()
    if new_feat_rows.empty:
        logger.warning(
            "No new feature rows found after %s, although new revenue rows exist.",
            last_period_end.date(),
        )

    # Align new feature rows to existing schema
    new_feat_aligned = _align_to_schema(
        base_df=feat_existing,
        new_df=new_feat_rows,
        rename_map=None,
    )

    # Combine with existing features; keep originals for any overlapping periods
    feat_latest = pd.concat([feat_existing, new_feat_aligned], ignore_index=True)
    feat_latest["period_end"] = pd.to_datetime(feat_latest["period_end"], errors="coerce")
    feat_latest = (
        feat_latest.sort_values("period_end")
        .drop_duplicates(subset=["ticker", "period_end"], keep="first")
        .reset_index(drop=True)
    )

    # 6. Save latest CSVs (do not overwrite originals)
    RAW_LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEAT_LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_latest.to_csv(RAW_LATEST_PATH, index=False)
    feat_latest.to_csv(FEAT_LATEST_PATH, index=False)

    # 7. Summary
    logger.info("-" * 70)
    logger.info("Refresh complete.")
    logger.info(
        "Raw rows: %d → %d (added %d)",
        len(raw_existing),
        len(raw_latest),
        len(raw_latest) - len(raw_existing),
    )
    logger.info(
        "Feature rows: %d → %d (added %d)",
        len(feat_existing),
        len(feat_latest),
        len(feat_latest) - len(feat_existing),
    )

    last5_raw = raw_latest.sort_values("period_end").tail(5)
    logger.info("Last 5 raw rows:\n%s", last5_raw.to_string(index=False))

    # Pick key columns for preview: period_end, revenue, rev_qoq, rev_yoy
    preview_cols = [c for c in ["period_end", "revenue", "rev_qoq", "rev_yoy"] if c in feat_latest.columns]
    last5_feat = feat_latest.sort_values("period_end").tail(5)[preview_cols]
    logger.info("Last 5 feature rows (key fields):\n%s", last5_feat.to_string(index=False))


def main() -> None:
    """CLI entrypoint."""
    try:
        refresh_nvda_revenue()
    except Exception as exc:
        logger.error("✗ Error while refreshing NVDA revenue: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()


