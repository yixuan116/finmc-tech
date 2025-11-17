#!/usr/bin/env python3
"""
Step 2: Train Multiple Models for NVDA Monthly Return Prediction.

Pipeline:
1. Load engineered features (macro + micro + interactions)
2. Time-based split (train/val/test)
3. Train 5 models (Linear, Ridge, RF, XGB, MLP)
4. Evaluate (MAE, RMSE, R2, MAPE) on test set
5. Save predictions, metrics, plots, champion model, SHAP insights
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class ModelResult:
    name: str
    model: object
    predictions: np.ndarray
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NVDA monthly return models (Step 2).")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to engineered features CSV (e.g., data/processed/nvda_features_extended.csv)",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="return_next_month",
        help="Target column name (default: return_next_month)",
    )
    parser.add_argument(
        "--date_column",
        type=str,
        default=None,
        help="Optional explicit date column (if None, auto-detect).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to store outputs (metrics, plots, shap, etc.).",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to store serialized models.",
    )
    return parser.parse_args()


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def auto_detect_date_column(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col and user_col in df.columns:
        return user_col

    candidates = [
        "date",
        "Date",
        "px_date",
        "period_end",
        "timestamp",
        "time",
        "period",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("No suitable date column found. Please specify --date_column.")


def prepare_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str]:
    """Ensure target column exists. If missing, attempt fallbacks."""
    if target_col in df.columns:
        return df, target_col

    fallback_cols = ["future_12m_return", "y_return_q1", "returns_q"]
    for col in fallback_cols:
        if col in df.columns:
            df = df.rename(columns={col: target_col})
            print(f"⚠ Target column '{target_col}' not found. Using fallback '{col}'.")
            return df, target_col

    if "adj_close" in df.columns:
        df[target_col] = df["adj_close"].shift(-1) / df["adj_close"] - 1
        df = df.dropna(subset=[target_col])
        print(
            f"⚠ Target column '{target_col}' not found. "
            "Computed using next-period adj_close returns."
        )
        return df, target_col

    raise ValueError(
        f"Target column '{target_col}' not found and cannot be inferred."
    )


def load_dataset(
    data_path: Path, target_col: str, date_column: str | None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(data_path)

    date_col = auto_detect_date_column(df, date_column)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    df, target_name = prepare_target(df, target_col)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.ffill().bfill()

    if numeric_df.columns.duplicated().any():
        numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated()].copy()

    if target_name not in numeric_df.columns:
        raise ValueError(f"Target column '{target_name}' must be numeric after processing.")

    y = numeric_df[target_name].copy()
    X = numeric_df.drop(columns=[target_name])

    # Drop identifier-like columns if present
    drop_cols = {"fy", "fp", "form", "tag_used", "ticker"}
    existing_drop = [col for col in drop_cols if col in X.columns]
    if existing_drop:
        X = X.drop(columns=existing_drop)
    
    # CRITICAL: Drop data leakage features (future information)
    # These features contain information from the target variable
    leakage_features = [col for col in X.columns if 'future_12m' in col.lower() and col != target_name]
    if leakage_features:
        print(f"⚠ Excluding data leakage features: {leakage_features}")
        X = X.drop(columns=leakage_features)

    return X, y, df.index


def time_based_split(
    X: pd.DataFrame, y: pd.Series
) -> DatasetSplits:
    train_end = pd.Timestamp("2020-12-31")
    val_end = pd.Timestamp("2022-12-31")

    idx = X.index
    train_mask = idx < train_end + pd.Timedelta(days=1)
    val_mask = (idx >= train_end + pd.Timedelta(days=1)) & (idx <= val_end)
    test_mask = idx > val_end

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Time splits failed; check available date range in dataset.")

    splits = DatasetSplits(
        X_train=X.loc[train_mask],
        X_val=X.loc[val_mask],
        X_test=X.loc[test_mask],
        y_train=y.loc[train_mask],
        y_val=y.loc[val_mask],
        y_test=y.loc[test_mask],
    )
    print(
        f"Split sizes -> Train: {len(splits.X_train)}, "
        f"Val: {len(splits.X_val)}, Test: {len(splits.X_test)}"
    )
    return splits


def scale_splits(splits: DatasetSplits) -> Tuple[DatasetSplits, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(splits.X_train)

    scaled = DatasetSplits(
        X_train=pd.DataFrame(
            scaler.transform(splits.X_train),
            index=splits.X_train.index,
            columns=splits.X_train.columns,
        ),
        X_val=pd.DataFrame(
            scaler.transform(splits.X_val),
            index=splits.X_val.index,
            columns=splits.X_val.columns,
        ),
        X_test=pd.DataFrame(
            scaler.transform(splits.X_test),
            index=splits.X_test.index,
            columns=splits.X_test.columns,
        ),
        y_train=splits.y_train.copy(),
        y_val=splits.y_val.copy(),
        y_test=splits.y_test.copy(),
    )
    return scaled, scaler


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_models(
    splits: DatasetSplits,
) -> Dict[str, ModelResult]:
    model_defs = {
        "linear": LinearRegression(),
        "ridge": RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0]),
        "rf": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
        "xgb": XGBRegressor(
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
        "nn": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
        ),
    }

    results: Dict[str, ModelResult] = {}
    for name, model in model_defs.items():
        print(f"\nTraining model: {name}")
        model.fit(splits.X_train, splits.y_train)
        preds = model.predict(splits.X_test)
        metrics = compute_metrics(splits.y_test.values, preds)
        results[name] = ModelResult(
            name=name,
            model=model,
            predictions=preds,
            metrics=metrics,
        )
        print(f"  Metrics: {metrics}")
    return results


def plot_predictions(
    y_test: pd.Series,
    preds: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    for name, values in preds.items():
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
        plt.plot(y_test.index, values, label=f"Predicted ({name})", linewidth=2)
        plt.title(f"Predicted vs Actual Returns - {name}")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"pred_vs_actual_{name}.png", dpi=150)
        plt.close()


def save_metrics_table(results: Dict[str, ModelResult], output_path: Path) -> pd.DataFrame:
    records = []
    for name, result in results.items():
        row = {"Model": name}
        row.update(result.metrics)
        records.append(row)
    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.set_index("Model")
    metrics_df.to_csv(output_path)
    print(f"✓ Saved performance table: {output_path}")
    return metrics_df


def select_champion(
    results: Dict[str, ModelResult],
    models_dir: Path,
    results_dir: Path,
) -> ModelResult:
    best = max(results.values(), key=lambda res: res.metrics["R2"])
    joblib.dump(best.model, models_dir / "champion_model.pkl")
    (results_dir / "champion_model_name.txt").write_text(best.name)
    print(f"🏆 Champion model: {best.name} (R2={best.metrics['R2']:.4f})")
    return best


def compute_shap_outputs(
    model: object,
    model_name: str,
    X_train: pd.DataFrame,
    shap_dir: Path,
    max_samples: int = 200,
) -> None:
    if len(X_train) > max_samples:
        sample = X_train.sample(max_samples, random_state=42)
    else:
        sample = X_train

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    np.save(shap_dir / f"shap_{model_name}_values.npy", shap_values)

    importances = pd.DataFrame(
        {
            "feature": sample.columns,
            "importance": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(shap_dir / f"shap_{model_name}_importances.csv", index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(shap_dir / f"shap_{model_name}_summary.png", dpi=150)
    plt.close()

    print(f"✓ Saved SHAP outputs for {model_name}")


def summarize_outputs(results_dir: Path, models_dir: Path) -> None:
    print("\nOutput directory tree:")
    for base in [results_dir, models_dir]:
        if base.exists():
            print(f"\n{base}:")
            for path in sorted(base.rglob("*")):
                indent = " " * (len(path.relative_to(base).parts) * 2)
                if path.is_file():
                    print(f"{indent}- {path.name}")
                else:
                    print(f"{indent}{path.name}/")


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    results_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    shap_dir = results_dir / "shap"

    ensure_directories(results_dir, models_dir, shap_dir)

    print("=== STEP 2: MODEL TRAINING PIPELINE ===")
    print(f"Data path: {data_path}")

    X, y, _ = load_dataset(data_path, args.target_column, args.date_column)
    splits = time_based_split(X, y)
    scaled_splits, scaler = scale_splits(splits)

    results = train_models(scaled_splits)

    preds = {name: res.predictions for name, res in results.items()}
    plot_predictions(scaled_splits.y_test, preds, results_dir)

    metrics_path = results_dir / "performance_step2.csv"
    save_metrics_table(results, metrics_path)

    champion = select_champion(results, models_dir, results_dir)

    # Save scaler for reproducibility
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")

    # Save predictions table
    preds_df = pd.DataFrame(
        {"actual": scaled_splits.y_test}
    )
    for name, res in results.items():
        preds_df[f"pred_{name}"] = res.predictions
    preds_df.to_csv(results_dir / "predictions_test.csv")
    print(f"✓ Saved predictions: {results_dir / 'predictions_test.csv'}")

    # SHAP only for tree models
    tree_models = {name: res.model for name, res in results.items() if name in {"rf", "xgb"}}
    for name, model in tree_models.items():
        compute_shap_outputs(model, name, scaled_splits.X_train, shap_dir)

    summarize_outputs(results_dir, models_dir)

    print("\nRun command:")
    print(f"python train_models.py --data_path {data_path}")


if __name__ == "__main__":
    main()

