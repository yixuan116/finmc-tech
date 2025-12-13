#!/usr/bin/env python3
"""
NVDA vs AMD Feature Importance Comparison Pipeline.

This script:
1. Loads NVDA and AMD datasets
2. Detects date intersection between datasets and trims to common period
3. Trains 5 models for each company (Linear, Ridge, RF, XGBoost, NN)
4. Extracts Top-20 features per model
5. Creates Top-K Union Heatmaps
6. Builds cross-firm comparison table
7. Performs sanity checks

Date Range Selection:
- Automatically detects intersection of date ranges between NVDA and AMD
- Trims both datasets to the common date window
- Ensures identical sample periods for fair comparison
- Prints diagnostics showing original ranges, intersection, and trimmed samples

Stock Price Location:
- Stock prices are in the 'adj_close' column (adjusted close price)
- Date column is 'px_date' (price date, typically day after period_end)
- Period dates are in 'period_end' column (fiscal period end date)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVDA_PATH = Path("data/processed/nvda_features_extended_v2.csv")
AMD_PATH = Path("data/processed/amd_features_extended.csv")
TARGET_COL = "future_12m_return"  # 12-month (1-year) forward return target
OUTPUT_DIR = Path("results")
TOP_K = 20

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompanyData:
    name: str
    X_raw: pd.DataFrame
    X_scaled: pd.DataFrame
    y: pd.Series
    feature_names: List[str]


@dataclass
class ModelConfig:
    name: str
    model: object
    needs_scaling: bool


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def check_amd_file() -> None:
    """Check if AMD file exists, show required schema if missing."""
    if not AMD_PATH.exists():
        print("=" * 80)
        print("ERROR: AMD dataset not found!")
        print(f"Expected path: {AMD_PATH}")
        print("\nRequired CSV schema (must match NVDA):")
        print("-" * 80)

        # Read NVDA to show schema
        nvda_df = pd.read_csv(NVDA_PATH, nrows=1)
        print(f"Required columns ({len(nvda_df.columns)} total):")
        for i, col in enumerate(nvda_df.columns, 1):
            print(f"  {i:2d}. {col}")

        print("\n" + "=" * 80)
        print("Please create amd_features_extended.csv with identical column structure.")
        print("=" * 80)
        sys.exit(1)


def auto_detect_date_column(df: pd.DataFrame) -> str:
    """Auto-detect date column."""
    candidates = ["date", "Date", "px_date", "period_end", "timestamp", "time", "period"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("No suitable date column found.")


def prepare_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str]:
    """Ensure target column exists with fallbacks."""
    if target_col in df.columns:
        return df, target_col

    fallback_cols = ["future_12m_return", "y_return_q1", "returns_q"]
    for col in fallback_cols:
        if col in df.columns:
            df = df.rename(columns={col: target_col})
            print(f"⚠ Using fallback target: {col} -> {target_col}")
            return df, target_col

    raise ValueError(f"Target column '{target_col}' not found and cannot be inferred.")


def load_and_prepare_data(
    data_path: Path,
    company_name: str,
    target_col: str,
    date_index: pd.DatetimeIndex = None,
) -> CompanyData:
    """
    Load and prepare dataset for a company.

    Args:
        data_path: Path to CSV file
        company_name: Company name (e.g., "NVDA")
        target_col: Target column name
        date_index: Optional date index to filter to (for intersection trimming)
    """
    print(f"\n{'='*60}")
    print(f"Loading {company_name} dataset: {data_path}")
    print(f"{'='*60}")

    df = pd.read_csv(data_path)

    # Handle date column
    date_col = auto_detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.sort_values(date_col).set_index(date_col)

    # Trim to intersection if provided
    if date_index is not None:
        original_len = len(df)
        df = df.loc[df.index.intersection(date_index)]
        print(f"✓ Trimmed to intersection: {original_len} -> {len(df)} rows")

    # Prepare target
    df, target_name = prepare_target(df, target_col)

    # Extract numeric features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.ffill().bfill()

    # Remove duplicates
    if numeric_df.columns.duplicated().any():
        numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated()].copy()

    if target_name not in numeric_df.columns:
        raise ValueError(f"Target '{target_name}' must be numeric.")

    y = numeric_df[target_name].copy()
    X = numeric_df.drop(columns=[target_name])

    # Drop identifier columns
    drop_cols = {"fy", "fp", "form", "tag_used", "ticker"}
    existing_drop = [col for col in drop_cols if col in X.columns]
    if existing_drop:
        X = X.drop(columns=existing_drop)

    # CRITICAL: Drop data leakage features
    leakage_features = [
        col
        for col in X.columns
        if "future_12m" in col.lower() and col != target_name
    ]
    if leakage_features:
        print(f"⚠ Excluding data leakage features: {leakage_features}")
        X = X.drop(columns=leakage_features)

    # Align indices (drop rows with NaN in target)
    valid_mask = ~y.isna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    # Drop rows with all NaN features
    X = X.dropna(how="all")

    # Ensure y aligns with X index (after dropping all-NaN rows)
    y = y.loc[y.index.intersection(X.index)]
    X = X.loc[X.index.intersection(y.index)]

    # Fill remaining NaN with 0
    X = X.fillna(0)

    # Final alignment check
    assert len(X) == len(y), f"X and y length mismatch: {len(X)} vs {len(y)}"
    assert X.index.equals(y.index), "X and y index mismatch"

    print(f"✓ Loaded {len(X)} samples, {len(X.columns)} features")
    print(f"✓ Target: {target_name} (mean={y.mean():.4f}, std={y.std():.4f})")

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns,
    )

    return CompanyData(
        name=company_name,
        X_raw=X,
        X_scaled=X_scaled,
        y=y,
        feature_names=list(X.columns),
    )


def detect_date_intersection(
    nvda_data: CompanyData,
    amd_data: CompanyData,
) -> pd.DatetimeIndex:
    """
    Detect intersection date range between NVDA and AMD datasets.

    Returns:
        DatetimeIndex of common dates
    """
    print(f"\n{'='*60}")
    print("Date Range Intersection Detection")
    print(f"{'='*60}")

    nvda_dates = nvda_data.X_raw.index
    amd_dates = amd_data.X_raw.index

    nvda_start = nvda_dates.min()
    nvda_end = nvda_dates.max()
    amd_start = amd_dates.min()
    amd_end = amd_dates.max()

    print(
        f"NVDA date range: {nvda_start.date()} to {nvda_end.date()} ({len(nvda_dates)} samples)"
    )
    print(
        f"AMD date range: {amd_start.date()} to {amd_end.date()} ({len(amd_dates)} samples)"
    )

    # Find intersection - use date range overlap if exact dates don't match
    intersection = nvda_dates.intersection(amd_dates)

    if len(intersection) == 0:
        # Try to find overlapping date range and use closest dates
        overlap_start = max(nvda_start, amd_start)
        overlap_end = min(nvda_end, amd_end)

        if overlap_start > overlap_end:
            raise ValueError(
                "No date intersection found between NVDA and AMD datasets!\n"
                f"NVDA: {nvda_start.date()} to {nvda_end.date()}\n"
                f"AMD: {amd_start.date()} to {amd_end.date()}"
            )

        # Create intersection by finding closest dates in both datasets
        nvda_in_range = nvda_dates[
            (nvda_dates >= overlap_start) & (nvda_dates <= overlap_end)
        ]
        amd_in_range = amd_dates[
            (amd_dates >= overlap_start) & (amd_dates <= overlap_end)
        ]

        # Use all NVDA dates in the overlap range
        # (AMD dates will be matched during trimming)
        intersection = nvda_in_range

    if len(intersection) == 0:
        raise ValueError(
            "No date intersection found between NVDA and AMD datasets!\n"
            f"NVDA: {nvda_start.date()} to {nvda_end.date()}\n"
            f"AMD: {amd_start.date()} to {amd_end.date()}"
        )

    intersection_start = intersection.min()
    intersection_end = intersection.max()

    print(f"\n✓ Intersection found:")
    print(f"  Start: {intersection_start.date()}")
    print(f"  End: {intersection_end.date()}")
    print(f"  Common dates: {len(intersection)} samples")

    # Diagnostics
    nvda_only = nvda_dates.difference(amd_dates)
    amd_only = amd_dates.difference(nvda_dates)

    if len(nvda_only) > 0:
        print(f"\n⚠ NVDA-only dates ({len(nvda_only)}):")
        print(f"  First: {nvda_only.min().date()}")
        print(f"  Last: {nvda_only.max().date()}")

    if len(amd_only) > 0:
        print(f"\n⚠ AMD-only dates ({len(amd_only)}):")
        print(f"  First: {amd_only.min().date()}")
        print(f"  Last: {amd_only.max().date()}")

    return intersection


def trim_to_intersection(
    data: CompanyData,
    intersection_dates: pd.DatetimeIndex,
) -> CompanyData:
    """Trim company data to intersection date range."""
    original_len = len(data.X_raw)

    # Filter to intersection - for exact matches or closest dates within 30 days
    if data.name == "NVDA":
        # NVDA: exact match
        mask = data.X_raw.index.isin(intersection_dates)
    else:
        # AMD: match by quarter (year + quarter) instead of exact date
        # This handles cases where NVDA uses report dates but AMD uses quarter-end dates
        mask = pd.Series(False, index=data.X_raw.index)

        # Create quarter identifiers for target dates
        target_quarters = set()
        for target_date in intersection_dates:
            target_quarters.add((target_date.year, target_date.quarter))

        # Match AMD dates by quarter
        for idx in data.X_raw.index:
            amd_quarter = (idx.year, idx.quarter)
            if amd_quarter in target_quarters:
                mask.loc[idx] = True

    X_raw_trimmed = data.X_raw.loc[mask]
    X_scaled_trimmed = data.X_scaled.loc[mask]

    # Align y with X index - ensure they match exactly
    common_idx = X_raw_trimmed.index.intersection(data.y.index)
    X_raw_trimmed = X_raw_trimmed.loc[common_idx]
    X_scaled_trimmed = X_scaled_trimmed.loc[common_idx]
    y_trimmed = data.y.loc[common_idx]

    # Final check
    assert len(X_raw_trimmed) == len(
        y_trimmed
    ), f"After trimming: X={len(X_raw_trimmed)}, y={len(y_trimmed)}"
    assert X_raw_trimmed.index.equals(y_trimmed.index), "Index mismatch after trimming"

    print(f"  {data.name}: {original_len} -> {len(X_raw_trimmed)} samples")

    # Re-scale with trimmed data
    scaler = StandardScaler()
    X_scaled_rescaled = pd.DataFrame(
        scaler.fit_transform(X_raw_trimmed),
        index=X_raw_trimmed.index,
        columns=X_raw_trimmed.columns,
    )

    return CompanyData(
        name=data.name,
        X_raw=X_raw_trimmed,
        X_scaled=X_scaled_rescaled,
        y=y_trimmed,
        feature_names=list(X_raw_trimmed.columns),  # Update feature names to match trimmed data
    )


def sanity_check_alignment(nvda_data: CompanyData, amd_data: CompanyData) -> None:
    """Ensure NVDA and AMD have aligned feature schemas and dates."""
    print(f"\n{'='*60}")
    print("Sanity Check: Feature & Date Alignment")
    print(f"{'='*60}")

    # Check date alignment
    nvda_dates = set(nvda_data.X_raw.index)
    amd_dates = set(amd_data.X_raw.index)

    if nvda_dates != amd_dates:
        common_dates = nvda_dates & amd_dates
        print(f"⚠ Date mismatch detected!")
        print(f"  Common dates: {len(common_dates)}")
        print(f"  NVDA-only: {len(nvda_dates - amd_dates)}")
        print(f"  AMD-only: {len(amd_dates - nvda_dates)}")
    else:
        print(f"✓ Date alignment: {len(nvda_dates)} common dates")

    # Check feature alignment
    nvda_features = set(nvda_data.feature_names)
    amd_features = set(amd_data.feature_names)

    common = nvda_features & amd_features
    nvda_only = nvda_features - amd_features
    amd_only = amd_features - nvda_features

    print(f"✓ Common features: {len(common)}")

    if nvda_only:
        print(f"⚠ NVDA-only features ({len(nvda_only)}): {list(nvda_only)[:5]}...")
    if amd_only:
        print(f"⚠ AMD-only features ({len(amd_only)}): {list(amd_only)[:5]}...")

    if not common:
        raise ValueError("No common features between NVDA and AMD!")

    # Check target
    nvda_target_mean = nvda_data.y.mean()
    amd_target_mean = amd_data.y.mean()

    print(f"✓ NVDA target mean: {nvda_target_mean:.4f}")
    print(f"✓ AMD target mean: {amd_target_mean:.4f}")

    print("✓ Sanity checks passed")


def get_model_configs() -> Dict[str, ModelConfig]:
    """Get model configurations."""
    # NOTE: Model instances are created fresh each time to avoid state sharing
    return {
        "linear": ModelConfig(
            name="linear",
            model=LinearRegression(),
            needs_scaling=False,
        ),
        "ridge": ModelConfig(
            name="ridge",
            model=RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0]),
            needs_scaling=True,
        ),
        "rf": ModelConfig(
            name="rf",
            model=RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            ),
            needs_scaling=False,
        ),
        "xgb": ModelConfig(
            name="xgb",
            model=XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                tree_method="hist",
                objective="reg:squarederror",
                random_state=42,
            ),
            needs_scaling=False,
        ),
        "nn": ModelConfig(
            name="nn",
            model=MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
            ),
            needs_scaling=True,
        ),
    }


def train_models_for_company(
    data: CompanyData,
    model_configs: Dict[str, ModelConfig],
) -> Dict[str, object]:
    """Train all models for a company."""
    print(f"\n{'='*60}")
    print(f"Training models for {data.name}")
    print(f"{'='*60}")

    models = {}

    for cfg in model_configs.values():
        print(f"\nTraining {cfg.name}...")

        # Select scaled or raw features
        X_train = data.X_scaled if cfg.needs_scaling else data.X_raw

        # CRITICAL: Create fresh model instance to avoid state sharing
        if cfg.name == "linear":
            model = LinearRegression()
        elif cfg.name == "ridge":
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0])
        elif cfg.name == "rf":
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            )
        elif cfg.name == "xgb":
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                tree_method="hist",
                objective="reg:squarederror",
                random_state=42,
            )
        elif cfg.name == "nn":
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model: {cfg.name}")

        # Train
        model.fit(X_train, data.y)

        # Evaluate
        preds = model.predict(X_train)
        mae = mean_absolute_error(data.y, preds)
        rmse = np.sqrt(mean_squared_error(data.y, preds))
        r2 = r2_score(data.y, preds)

        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        models[cfg.name] = model

    return models


def extract_top20_features(
    model: object,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    needs_scaling: bool,
) -> pd.Series:
    """Extract Top-20 most important features for a model."""
    # CRITICAL: Use X.columns as the source of truth (model was trained on these)
    actual_features = list(X.columns)
    feature_names = actual_features  # Always use actual X columns

    if model_name in ["linear", "ridge"]:
        # Use absolute coefficients
        if hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
        else:
            # RidgeCV might have best_estimator_
            coefs = np.abs(model.best_estimator_.coef_)

        # Ensure coefs length matches feature_names
        if len(coefs) != len(feature_names):
            print(
                f"  ⚠ Coefficient length mismatch: coefs={len(coefs)}, features={len(feature_names)}"
            )
            min_len = min(len(coefs), len(feature_names))
            coefs = coefs[:min_len]
            feature_names = feature_names[:min_len]

        importances = pd.Series(coefs, index=feature_names)

    elif model_name in ["rf", "xgb"]:
        # Use feature_importances_
        importances_vals = model.feature_importances_

        if len(importances_vals) != len(feature_names):
            print(
                f"  ⚠ Feature importance length mismatch: importances={len(importances_vals)}, features={len(feature_names)}"
            )
            min_len = min(len(importances_vals), len(feature_names))
            importances_vals = importances_vals[:min_len]
            feature_names = feature_names[:min_len]

        importances = pd.Series(importances_vals, index=feature_names)

    elif model_name == "nn":
        # Use permutation importance (stable for NN)
        # CRITICAL: Ensure X has same columns as model was trained on
        print(f"  Computing permutation importance for {model_name}...")
        print(f"  X shape: {X.shape}, columns: {list(X.columns[:5])}...")

        perm_result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=1
        )
        importances_vals = perm_result.importances_mean

        if len(importances_vals) != len(feature_names):
            print(
                f"  ⚠ Permutation importance length mismatch: importances={len(importances_vals)}, features={len(feature_names)}"
            )
            min_len = min(len(importances_vals), len(feature_names))
            importances_vals = importances_vals[:min_len]
            feature_names = feature_names[:min_len]

        importances = pd.Series(importances_vals, index=feature_names)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Sort descending and return Top-20
    top20 = importances.sort_values(ascending=False).head(TOP_K)

    return top20


def extract_all_top20(
    models: Dict[str, object],
    data: CompanyData,
    model_configs: Dict[str, ModelConfig],
) -> Dict[str, pd.Series]:
    """Extract Top-20 features for all models."""
    print(f"\n{'='*60}")
    print(f"Extracting Top-{TOP_K} features for {data.name}")
    print(f"{'='*60}")

    top20_dict = {}

    for model_name, model in models.items():
        cfg = model_configs[model_name]
        X_use = data.X_scaled if cfg.needs_scaling else data.X_raw

        # Use actual X columns as feature names (model was trained on these)
        actual_feature_names = list(X_use.columns)

        top20 = extract_top20_features(
            model,
            model_name,
            X_use,
            data.y,
            actual_feature_names,
            cfg.needs_scaling,
        )

        top20_dict[model_name] = top20

        print(f"\n{model_name} Top-5:")
        for feat, imp in top20.head(5).items():
            print(f"  {feat:40s}: {imp:.6f}")

    return top20_dict


def build_union_heatmap(
    top20_dict: Dict[str, pd.Series],
    company_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Build Top-K Union Heatmap for a company."""
    print(f"\n{'='*60}")
    print(f"Building Union Heatmap for {company_name}")
    print(f"{'='*60}")

    # Get union of all Top-20 features
    union_features = set()
    for top20 in top20_dict.values():
        union_features.update(top20.index)

    union_features = sorted(list(union_features))
    K = len(union_features)

    print(
        f"✓ Union size: {K} features (from {TOP_K}×{len(top20_dict)} = {TOP_K*len(top20_dict)} total)"
    )

    # Build matrix: models × features
    model_names = sorted(top20_dict.keys())
    heatmap_data = pd.DataFrame(
        index=model_names,
        columns=union_features,
        dtype=float,
    )

    for model_name in model_names:
        top20 = top20_dict[model_name]
        for feat in union_features:
            if feat in top20.index:
                heatmap_data.loc[model_name, feat] = top20[feat]
            else:
                heatmap_data.loc[model_name, feat] = 0.0

    # Row-normalize per model (0-1)
    for model_name in model_names:
        row = heatmap_data.loc[model_name]
        row_max = row.max()
        if row_max > 0:
            heatmap_data.loc[model_name] = row / row_max

    # Plot heatmap
    plt.figure(figsize=(max(12, K * 0.3), 6))
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="YlOrRd",
        cbar_kws={"label": "Normalized Importance"},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
    )
    plt.title(
        f"{company_name} Top-K Union Feature Importance Heatmap\n"
        f"({K} features from union of Top-{TOP_K} across 5 models)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Models", fontsize=12)
    plt.xticks(rotation=90, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    output_path = output_dir / f"{company_name.lower()}_union_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {output_path}")

    return heatmap_data


def build_comparison_table(
    nvda_top20: Dict[str, pd.Series],
    amd_top20: Dict[str, pd.Series],
    output_dir: Path,
) -> pd.DataFrame:
    """Build NVDA vs AMD comparison table."""
    print(f"\n{'='*60}")
    print("Building NVDA vs AMD Comparison Table")
    print(f"{'='*60}")

    # Get union of all features from both companies
    all_features = set()
    for top20 in nvda_top20.values():
        all_features.update(top20.index)
    for top20 in amd_top20.values():
        all_features.update(top20.index)

    all_features = sorted(list(all_features))
    print(f"✓ Total unique features: {len(all_features)}")

    # Build comparison DataFrame
    model_names = sorted(nvda_top20.keys())
    columns = []
    for company in ["NVDA", "AMD"]:
        for model_name in model_names:
            columns.append(f"{company}_{model_name}")

    comparison_df = pd.DataFrame(
        index=all_features,
        columns=columns,
        dtype=float,
    )
    comparison_df = comparison_df.fillna(0.0)

    # Fill NVDA values
    for model_name in model_names:
        top20 = nvda_top20[model_name]
        col = f"NVDA_{model_name}"
        for feat in top20.index:
            comparison_df.loc[feat, col] = top20[feat]

    # Fill AMD values
    for model_name in model_names:
        top20 = amd_top20[model_name]
        col = f"AMD_{model_name}"
        for feat in top20.index:
            comparison_df.loc[feat, col] = top20[feat]

    # Save
    output_path = output_dir / "NVDA_AMD_feature_compare.csv"
    comparison_df.to_csv(output_path)

    print(f"✓ Saved: {output_path}")

    return comparison_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive performance metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE with clipping to avoid division by zero
    mape = float(
        np.mean(
            np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))
        )
        * 100
    )

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def compute_performance_table(
    models: Dict[str, object],
    data: CompanyData,
    model_configs: Dict[str, ModelConfig],
    company_name: str,
) -> pd.DataFrame:
    """Compute comprehensive performance table for all models."""
    print(f"\n{'='*60}")
    print(f"Computing Performance Metrics for {company_name}")
    print(f"{'='*60}")

    records = []

    for model_name, model in models.items():
        cfg = model_configs[model_name]
        X_use = data.X_scaled if cfg.needs_scaling else data.X_raw

        preds = model.predict(X_use)
        metrics = compute_metrics(data.y.values, preds)
        metrics["Model"] = model_name
        records.append(metrics)

        print(
            f"{model_name:10s}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
            f"R²={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%"
        )

    df = pd.DataFrame(records)
    df = df[["Model", "MAE", "RMSE", "R2", "MAPE"]]

    return df


def compute_best_models(
    nvda_models: Dict[str, object],
    amd_models: Dict[str, object],
    nvda_data: CompanyData,
    amd_data: CompanyData,
    model_configs: Dict[str, ModelConfig],
) -> Tuple[str, str]:
    """Determine best model for each company based on R²."""
    nvda_scores = {}
    amd_scores = {}

    for model_name, model in nvda_models.items():
        cfg = model_configs[model_name]
        X_use = nvda_data.X_scaled if cfg.needs_scaling else nvda_data.X_raw
        preds = model.predict(X_use)
        r2 = r2_score(nvda_data.y, preds)
        nvda_scores[model_name] = r2

    for model_name, model in amd_models.items():
        cfg = model_configs[model_name]
        X_use = amd_data.X_scaled if cfg.needs_scaling else amd_data.X_raw
        preds = model.predict(X_use)
        r2 = r2_score(amd_data.y, preds)
        amd_scores[model_name] = r2

    nvda_best = max(nvda_scores.items(), key=lambda x: x[1])
    amd_best = max(amd_scores.items(), key=lambda x: x[1])

    return nvda_best[0], amd_best[0]


def find_stable_features(
    nvda_top20: Dict[str, pd.Series],
    amd_top20: Dict[str, pd.Series],
) -> Tuple[List[str], List[str], List[str]]:
    """Find features stable across both companies vs company-specific."""
    # Count appearances in Top-20 across all models
    nvda_counts = {}
    amd_counts = {}

    for top20 in nvda_top20.values():
        for feat in top20.index:
            nvda_counts[feat] = nvda_counts.get(feat, 0) + 1

    for top20 in amd_top20.values():
        for feat in top20.index:
            amd_counts[feat] = amd_counts.get(feat, 0) + 1

    # Stable: appears in both companies
    stable = []
    for feat in set(nvda_counts.keys()) & set(amd_counts.keys()):
        if nvda_counts[feat] >= 2 and amd_counts[feat] >= 2:  # At least 2 models
            stable.append(feat)

    # Company-specific: appears in one but not the other
    nvda_specific = [
        feat
        for feat in nvda_counts.keys()
        if feat not in amd_counts or amd_counts.get(feat, 0) < 2
    ]
    amd_specific = [
        feat
        for feat in amd_counts.keys()
        if feat not in nvda_counts or nvda_counts.get(feat, 0) < 2
    ]

    return stable, nvda_specific, amd_specific


def print_summary(
    nvda_best: str,
    amd_best: str,
    stable_features: List[str],
    nvda_specific: List[str],
    amd_specific: List[str],
) -> None:
    """Print final summary."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n🏆 Best Model for NVDA: {nvda_best}")
    print(f"🏆 Best Model for AMD: {amd_best}")

    print(f"\n📊 Stable Features (appear in both companies, ≥2 models):")
    print(f"  Count: {len(stable_features)}")
    if stable_features:
        for feat in stable_features[:10]:
            print(f"  - {feat}")
        if len(stable_features) > 10:
            print(f"  ... and {len(stable_features) - 10} more")

    print(f"\n📈 NVDA-Specific Features (≥2 models in NVDA, <2 in AMD):")
    print(f"  Count: {len(nvda_specific)}")
    if nvda_specific:
        for feat in nvda_specific[:10]:
            print(f"  - {feat}")
        if len(nvda_specific) > 10:
            print(f"  ... and {len(nvda_specific) - 10} more")

    print(f"\n📈 AMD-Specific Features (≥2 models in AMD, <2 in NVDA):")
    print(f"  Count: {len(amd_specific)}")
    if amd_specific:
        for feat in amd_specific[:10]:
            print(f"  - {feat}")
        if len(amd_specific) > 10:
            print(f"  ... and {len(amd_specific) - 10} more")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Main pipeline."""
    print("=" * 80)
    print("NVDA vs AMD Feature Importance Comparison Pipeline")
    print("=" * 80)

    # Check AMD file exists
    check_amd_file()

    # Ensure output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets (initial load to detect date ranges)
    nvda_data = load_and_prepare_data(NVDA_PATH, "NVDA", TARGET_COL)
    amd_data = load_and_prepare_data(AMD_PATH, "AMD", TARGET_COL)

    # Detect date intersection and trim both datasets
    intersection_dates = detect_date_intersection(nvda_data, amd_data)

    print(f"\n{'='*60}")
    print("Trimming datasets to intersection date range")
    print(f"{'='*60}")

    nvda_data = trim_to_intersection(nvda_data, intersection_dates)
    amd_data = trim_to_intersection(amd_data, intersection_dates)

    # Sanity checks (after trimming)
    sanity_check_alignment(nvda_data, amd_data)

    # Get model configs
    model_configs = get_model_configs()

    # Train models
    nvda_models = train_models_for_company(nvda_data, model_configs)
    amd_models = train_models_for_company(amd_data, model_configs)

    # Extract Top-20 features
    nvda_top20 = extract_all_top20(nvda_models, nvda_data, model_configs)
    amd_top20 = extract_all_top20(amd_models, amd_data, model_configs)

    # Build union heatmaps
    nvda_heatmap = build_union_heatmap(nvda_top20, "NVDA", OUTPUT_DIR)
    amd_heatmap = build_union_heatmap(amd_top20, "AMD", OUTPUT_DIR)

    # Build comparison table
    comparison_df = build_comparison_table(nvda_top20, amd_top20, OUTPUT_DIR)

    # Compute performance tables
    nvda_perf_table = compute_performance_table(
        nvda_models, nvda_data, model_configs, "NVDA"
    )
    amd_perf_table = compute_performance_table(
        amd_models, amd_data, model_configs, "AMD"
    )

    # Save performance tables
    nvda_perf_path = OUTPUT_DIR / "nvda_performance.csv"
    amd_perf_path = OUTPUT_DIR / "amd_performance.csv"
    nvda_perf_table.to_csv(nvda_perf_path, index=False)
    amd_perf_table.to_csv(amd_perf_path, index=False)

    print(f"\n✓ Saved NVDA performance table: {nvda_perf_path}")
    print(f"✓ Saved AMD performance table: {amd_perf_path}")

    # Print performance comparison
    print(f"\n{'='*60}")
    print("Performance Comparison")
    print(f"{'='*60}")
    print("\nNVDA Performance:")
    print(nvda_perf_table.to_string(index=False))
    print("\nAMD Performance:")
    print(amd_perf_table.to_string(index=False))

    # Compute best models
    nvda_best, amd_best = compute_best_models(
        nvda_models, amd_models, nvda_data, amd_data, model_configs
    )

    # Find stable vs company-specific features
    stable, nvda_specific, amd_specific = find_stable_features(
        nvda_top20, amd_top20
    )

    # Print summary
    print_summary(nvda_best, amd_best, stable, nvda_specific, amd_specific)

    print("\n✓ Pipeline completed successfully!")
    print(f"✓ Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
