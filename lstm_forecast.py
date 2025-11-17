"""
LSTM Forecast for Monthly NVDA Returns (Phase 2).

Trains an LSTM model to forecast next-month returns using selected features
from Phase 1 (Random Forest feature importance). Supports TensorFlow/Keras
(primary) and PyTorch (fallback) backends.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Backend detection
BACKEND = None
HAS_TF = False
HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
    BACKEND = "tf"
except ImportError:
    pass

if not HAS_TF:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        HAS_TORCH = True
        BACKEND = "torch"
    except ImportError:
        pass


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if HAS_TF:
        tf.random.set_seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def read_csv_detect_date(path: str, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Read CSV and detect date column, set index to monthly date.
    
    Args:
        path: Path to CSV file
        date_col: Optional explicit date column name
    
    Returns:
        DataFrame with 'date' index (month-end frequency)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Detect date column
    if date_col is None:
        for col in ["date", "px_date", "period_end", "Date"]:
            if col in df.columns:
                date_col = col
                break
    
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    
    # Parse dates and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    
    # Resample to month-end if needed (if daily data)
    if len(df) > 0:
        freq = pd.infer_freq(df.index)
        if freq and "D" in freq:  # Daily frequency
            df = df.resample("M").last()
            print(f"  Resampled daily data to monthly (M) frequency")
    
    return df


def load_and_merge_data(
    prices_path: str,
    macro_path: Optional[str],
    firm_path: Optional[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load prices, macro, and firm CSVs, merge, and prepare X/y.
    
    Args:
        prices_path: Path to prices CSV
        macro_path: Path to macro CSV (optional)
        firm_path: Path to firm-aligned CSV (optional)
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        Merged DataFrame with Ret and features
    """
    print("=" * 70)
    print("Loading and Merging Data")
    print("=" * 70)
    
    # Load prices
    print(f"\n1. Loading prices from {prices_path}...")
    prices_df = read_csv_detect_date(prices_path)
    
    # Get price column
    price_col = None
    for col in ["adj_close", "close", "Close"]:
        if col in prices_df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"No price column found. Available: {list(prices_df.columns)}")
    
    # Compute monthly returns: Ret_t = (P_t / P_{t-1}) - 1
    prices_df["Ret"] = prices_df[price_col].pct_change()
    print(f"  ✓ Computed monthly returns from {price_col}")
    
    # Start with prices_df (includes Ret and any other feature columns)
    exclude_from_prices = {price_col, "date", "Date", "ticker", "fy", "fp", "form", "tag_used"}
    prices_feature_cols = [col for col in prices_df.columns if col not in exclude_from_prices]
    merged_df = prices_df[prices_feature_cols].copy()
    print(f"  ✓ Included {len(prices_feature_cols)} columns from prices file")
    
    # Merge macro data
    if macro_path and Path(macro_path).exists():
        print(f"\n2. Loading macro data from {macro_path}...")
        macro_df = read_csv_detect_date(macro_path)
        merged_df = merged_df.join(macro_df, how="inner")
        print(f"  ✓ Macro: {len(macro_df)} rows, {len(macro_df.columns)} columns")
    else:
        print(f"\n2. Skipping macro data (file not found or not provided)")
    
    # Merge firm data
    if firm_path and Path(firm_path).exists():
        print(f"\n3. Loading firm data from {firm_path}...")
        firm_df = read_csv_detect_date(firm_path)
        merged_df = merged_df.join(firm_df, how="inner")
        print(f"  ✓ Firm: {len(firm_df)} rows, {len(firm_df.columns)} columns")
    else:
        print(f"\n3. Skipping firm data (file not found or not provided)")
    
    print(f"\n4. Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Time slice
    if start_date or end_date:
        print(f"\n5. Applying time window filter...")
        original_len = len(merged_df)
        if start_date:
            start_dt = pd.to_datetime(start_date)
            merged_df = merged_df[merged_df.index >= start_dt]
            print(f"  ✓ Filtered from {start_date} onwards")
        if end_date:
            end_dt = pd.to_datetime(end_date)
            merged_df = merged_df[merged_df.index <= end_dt]
            print(f"  ✓ Filtered up to {end_date}")
        print(f"  ✓ Reduced from {original_len} to {len(merged_df)} rows")
    
    # Create target: y = Ret.shift(-1) (next-month return)
    merged_df["y"] = merged_df["Ret"].shift(-1)
    
    # Drop rows with NaN
    merged_df = merged_df.dropna()
    
    if len(merged_df) < 24:
        raise ValueError(
            f"Insufficient data after merging: {len(merged_df)} rows. "
            "Need at least 24 months."
        )
    
    print(f"\n6. Final merged dataset:")
    print(f"  ✓ Samples: {len(merged_df)}")
    print(f"  ✓ Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    return merged_df


def select_features(
    merged_df: pd.DataFrame,
    top_k: int = 15,
    features_json: Optional[str] = None,
    importance_csv: str = "results/rf_feature_importance.csv",
) -> List[str]:
    """
    Select features based on RF importance or explicit JSON list.
    
    Args:
        merged_df: Merged DataFrame
        top_k: Number of top features to select
        features_json: Optional path to JSON list of feature names
        importance_csv: Path to RF feature importance CSV
    
    Returns:
        List of selected feature names
    """
    print("\n" + "=" * 70)
    print("Feature Selection")
    print("=" * 70)
    
    # Exclude columns
    exclude_cols = {
        "Ret", "y", "adj_close", "close", "Close",
        "date", "Date", "px_date", "period_end",
        "ticker", "fy", "fp", "form", "tag_used",
    }
    exclude_cols.update([col for col in merged_df.columns if col.startswith("future_12m_")])
    
    available_cols = [col for col in merged_df.columns if col not in exclude_cols]
    
    # Try to convert to numeric
    numeric_cols = []
    for col in available_cols:
        try:
            test_series = pd.to_numeric(merged_df[col], errors="coerce")
            if not test_series.isna().all():
                numeric_cols.append(col)
        except:
            continue
    
    if features_json and Path(features_json).exists():
        print(f"\n1. Loading features from JSON: {features_json}")
        with open(features_json, "r") as f:
            selected = json.load(f)
        # Filter to available numeric columns
        selected = [f for f in selected if f in numeric_cols]
        print(f"  ✓ Selected {len(selected)} features from JSON")
    else:
        print(f"\n1. Loading Top-{top_k} features from RF importance...")
        if not Path(importance_csv).exists():
            print(f"  ⚠ Importance CSV not found: {importance_csv}")
            print(f"  ✓ Using all {len(numeric_cols)} available numeric features")
            selected = numeric_cols[:top_k] if len(numeric_cols) > top_k else numeric_cols
        else:
            importance_df = pd.read_csv(importance_csv)
            # Get top-k features
            top_features = importance_df.head(top_k)["feature"].tolist()
            # Filter to available numeric columns
            selected = [f for f in top_features if f in numeric_cols]
            print(f"  ✓ Selected {len(selected)} features from RF importance")
    
    print(f"\n2. Selected features ({len(selected)}):")
    for i, feat in enumerate(selected, 1):
        print(f"  {i:2d}. {feat}")
    
    return selected


def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = 12,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create sequences for LSTM training.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        lookback: Number of time steps to look back
        horizon: Prediction horizon (1 = next month)
    
    Returns:
        X_seq: Sequences array [N, lookback, F]
        y_seq: Target array [N]
        dates: Date index for sequences
    """
    X_seq_list = []
    y_seq_list = []
    dates_list = []
    
    for i in range(lookback, len(X) - horizon + 1):
        X_seq_list.append(X.iloc[i - lookback:i].values)
        y_seq_list.append(y.iloc[i + horizon - 1])
        dates_list.append(X.index[i + horizon - 1])
    
    X_seq = np.array(X_seq_list, dtype=np.float32)
    y_seq = np.array(y_seq_list, dtype=np.float32)
    dates = pd.DatetimeIndex(dates_list)
    
    return X_seq, y_seq, dates


def build_tf_model(input_shape: Tuple[int, int], lr: float = 1e-3) -> keras.Model:
    """Build TensorFlow/Keras LSTM model."""
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=False, input_shape=input_shape),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    
    return model


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM sequences."""
    
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X = torch.FloatTensor(X_seq)
        self.y = torch.FloatTensor(y_seq)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """PyTorch LSTM model."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        return out


def train_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    backend: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    model_dir: Path = Path("models"),
    n_jobs: int = 4,
) -> Tuple:
    """
    Train LSTM model (TF or Torch).
    
    Returns:
        Trained model, history (if TF), scaler (not used here, but for consistency)
    """
    print("\n" + "=" * 70)
    print(f"Training LSTM Model ({backend.upper()})")
    print("=" * 70)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if backend == "tf" or (backend == "auto" and HAS_TF):
        return train_tf_model(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            epochs, batch_size, lr, patience, model_dir,
        )
    elif backend == "torch" or (backend == "auto" and HAS_TORCH):
        return train_torch_model(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            epochs, batch_size, lr, patience, model_dir, n_jobs,
        )
    else:
        raise ValueError("No backend available. Install TensorFlow or PyTorch.")


def train_tf_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    model_dir: Path,
) -> Tuple:
    """Train TensorFlow/Keras model."""
    print(f"\nParameters:")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {lr}")
    print(f"  patience: {patience}")
    
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_tf_model(input_shape, lr)
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            str(model_dir / "lstm_tf.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    
    print(f"\nTraining on {len(X_train_seq)} sequences...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
    
    # Load best model
    model = keras.models.load_model(str(model_dir / "lstm_tf.keras"))
    
    print(f"\n✓ Model trained and saved to {model_dir / 'lstm_tf.keras'}")
    
    return model, history, None


def train_torch_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    model_dir: Path,
    n_jobs: int,
) -> Tuple:
    """Train PyTorch model."""
    print(f"\nParameters:")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {lr}")
    print(f"  patience: {patience}")
    
    input_size = X_train_seq.shape[2]
    model = LSTMModel(input_size, hidden_size=64)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Datasets and DataLoaders
    train_dataset = LSTMDataset(X_train_seq, y_train_seq)
    val_dataset = LSTMDataset(X_val_seq, y_val_seq)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, n_jobs),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, n_jobs),
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "val_loss": []}
    
    print(f"\nTraining on {len(X_train_seq)} sequences...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(model_dir / "lstm_torch.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(str(model_dir / "lstm_torch.pt")))
    
    print(f"\n✓ Model trained and saved to {model_dir / 'lstm_torch.pt'}")
    
    return model, history, None


def evaluate_model(
    model,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    backend: str,
) -> Tuple[np.ndarray, Dict]:
    """
    Evaluate model and return predictions and metrics.
    
    Returns:
        y_pred: Predictions array
        metrics: Dictionary with R2, MAE, RMSE
    """
    if backend == "tf" or (backend == "auto" and HAS_TF):
        y_pred = model.predict(X_seq, verbose=0).flatten()
    else:
        model.eval()
        with torch.no_grad():
            dataset = LSTMDataset(X_seq, y_seq)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            y_pred_list = []
            for X_batch, _ in loader:
                y_pred_batch = model(X_batch).squeeze().numpy()
                y_pred_list.append(y_pred_batch)
            y_pred = np.concatenate(y_pred_list)
    
    # Metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(y_seq, y_pred)
    mae = mean_absolute_error(y_seq, y_pred)
    rmse = np.sqrt(mean_squared_error(y_seq, y_pred))
    
    metrics = {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }
    
    return y_pred, metrics


def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    output_path: Path,
) -> None:
    """Plot predictions vs actual (scatter)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # y=x line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    
    ax.set_xlabel("Actual Return", fontsize=12)
    ax.set_ylabel("Predicted Return", fontsize=12)
    ax.set_title(f"LSTM Predictions vs Actual ({split.upper()})", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Plot calibration curve."""
    # Bin predictions
    bin_edges = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[:-1])
    
    bin_means_pred = []
    bin_means_true = []
    
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_pred.append(y_pred[mask].mean())
            bin_means_true.append(y_true[mask].mean())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(bin_means_pred, bin_means_true, s=100, alpha=0.7)
    
    # Perfect calibration line
    min_val = min(min(bin_means_pred), min(bin_means_true))
    max_val = max(max(bin_means_pred), max(bin_means_true))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect calibration")
    
    ax.set_xlabel("Mean Predicted Return (binned)", fontsize=12)
    ax.set_ylabel("Mean Actual Return (binned)", fontsize=12)
    ax.set_title("LSTM Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(
    dates: pd.DatetimeIndex,
    residuals: np.ndarray,
    output_path: Path,
) -> None:
    """Plot time-series residuals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, residuals, alpha=0.7, linewidth=1.5)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_title("LSTM Residuals Over Time (TEST)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_corr(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 12,
    output_path: Path = None,
) -> None:
    """Plot rolling correlation."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=dates)
    rolling_corr = df["y_true"].rolling(window=window).corr(df["y_pred"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, rolling_corr, linewidth=2, alpha=0.8)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"Rolling Correlation ({window}M)", fontsize=12)
    ax.set_title("LSTM Rolling Correlation: Predicted vs Actual (TEST)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def update_readme(
    metrics: Dict,
    readme_path: Path = Path("README.md"),
    plots_dir: Path = Path("plots"),
) -> None:
    """Update README with LSTM Phase 2 section."""
    print(f"\nUpdating README.md...")
    
    if not readme_path.exists():
        print(f"  ⚠ README.md not found, skipping update")
        return
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Prepare section
    section = f"""
### Phase 2 — LSTM Forecast

After identifying the most important features using Random Forest (Phase 1), we train a Long Short-Term Memory (LSTM) neural network to forecast next-month returns. LSTM is well-suited for time-series forecasting because it can capture long-term dependencies and temporal patterns in sequential data. By using only the Top-K features selected from Phase 1, we reduce noise and improve model generalization.

**Model Performance (TEST):**

| Metric | Value |
|--------|-------|
| R² | {metrics['test']['R2']:.4f} |
| MAE | {metrics['test']['MAE']:.4f} |
| RMSE | {metrics['test']['RMSE']:.4f} |

**Key Visualizations:**

![LSTM Predictions vs Actual]({plots_dir}/pred_vs_actual_return_lstm.png)

![LSTM Rolling Correlation]({plots_dir}/rolling_corr_lstm.png)

**HPC Design:**
Training uses vectorized NumPy kernels and multi-core input pipelines (tf.data prefetch or torch DataLoader). This step continues the pipeline's HPC design: parallel I/O and batch computation, reproducible with fixed seeds. The model architecture (LSTM → Dense layers) is optimized for both TensorFlow/Keras (primary) and PyTorch (fallback) backends, ensuring portability across different compute environments.
"""
    
    start_marker = "<!-- LSTM_PHASE2_START -->"
    end_marker = "<!-- LSTM_PHASE2_END -->"
    
    if start_marker in content and end_marker in content:
        import re
        pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
        replacement = start_marker + section + "\n" + end_marker
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print(f"  ✓ Replaced existing LSTM section")
    else:
        # Append at end
        content += "\n" + start_marker + section + "\n" + end_marker
        print(f"  ✓ Appended LSTM section at end")
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"  ✓ Updated README.md")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LSTM Forecast for Monthly NVDA Returns (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick default (auto backend, Top-15 from RF):
  python3 lstm_forecast.py --save-readme

  # Specify window + faster test:
  python3 lstm_forecast.py --start 2015-01-01 --end 2023-12-31 --lookback 12 --epochs 60 --batch-size 32 --save-readme

  # Force backend + custom Top-K:
  python3 lstm_forecast.py --backend tf --top-k 20 --epochs 80 --n-jobs 4 --save-readme

  # Use explicit feature list:
  python3 lstm_forecast.py --features-json configs/selected_features.json --epochs 80 --save-readme
        """,
    )
    
    parser.add_argument("--prices", type=str, default="data/prices/nvda_prices.csv")
    parser.add_argument("--macro", type=str, default="data/processed/macro.csv")
    parser.add_argument("--firm", type=str, default="data/processed/firm_aligned.csv")
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--features-json", type=str, default=None)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "tf", "torch"])
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-readme", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--model-dir", type=str, default="models")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Determine backend
    backend = args.backend
    if backend == "auto":
        if HAS_TF:
            backend = "tf"
            print("✓ Using TensorFlow/Keras backend")
        elif HAS_TORCH:
            backend = "torch"
            print("✓ Using PyTorch backend")
        else:
            print("✗ Error: Neither TensorFlow nor PyTorch is installed.")
            print("  Please install one: pip install tensorflow  or  pip install torch")
            return 1
    else:
        if backend == "tf" and not HAS_TF:
            print("✗ Error: TensorFlow not installed. Install with: pip install tensorflow")
            return 1
        if backend == "torch" and not HAS_TORCH:
            print("✗ Error: PyTorch not installed. Install with: pip install torch")
            return 1
    
    # Create directories
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and merge data
        merged_df = load_and_merge_data(
            args.prices,
            args.macro,
            args.firm,
            args.start,
            args.end,
        )
        
        # Select features
        selected_features = select_features(
            merged_df,
            args.top_k,
            args.features_json,
        )
        
        # Prepare X and y
        X = merged_df[selected_features].copy()
        y = merged_df["y"].copy()
        dates = X.index
        
        # Chronological split
        if args.train_end:
            train_end_dt = pd.to_datetime(args.train_end)
            train_mask = dates <= train_end_dt
        else:
            split_idx = int(len(X) * 0.8)
            train_mask = dates <= dates[split_idx]
        
        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        X_test = X[~train_mask].copy()
        y_test = y[~train_mask].copy()
        
        if len(X_train) < 12:
            raise ValueError(f"Insufficient training data: {len(X_train)} samples. Need at least 12.")
        if len(X_test) < 12:
            raise ValueError(f"Insufficient test data: {len(X_test)} samples. Need at least 12.")
        
        print(f"\n" + "=" * 70)
        print("Train/Test Split")
        print("=" * 70)
        print(f"  Train: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
        print(f"  Test: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns,
        )
        
        # Save scaler
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"\n✓ Saved scaler to {model_dir / 'scaler.pkl'}")
        
        # Create sequences
        print(f"\n" + "=" * 70)
        print("Creating Sequences")
        print("=" * 70)
        print(f"  Lookback: {args.lookback} months")
        print(f"  Horizon: {args.horizon} month(s)")
        
        X_train_seq, y_train_seq, dates_train_seq = create_sequences(
            X_train_scaled, y_train, args.lookback, args.horizon
        )
        X_test_seq, y_test_seq, dates_test_seq = create_sequences(
            X_test_scaled, y_test, args.lookback, args.horizon
        )
        
        print(f"  Train sequences: {len(X_train_seq)}")
        print(f"  Test sequences: {len(X_test_seq)}")
        
        # Split train into train/val
        val_size = max(12, int(len(X_train_seq) * 0.2))
        X_val_seq = X_train_seq[-val_size:]
        y_val_seq = y_train_seq[-val_size:]
        X_train_seq = X_train_seq[:-val_size]
        y_train_seq = y_train_seq[:-val_size]
        
        print(f"  Train sequences (after val split): {len(X_train_seq)}")
        print(f"  Val sequences: {len(X_val_seq)}")
        
        # Train model
        model, history, _ = train_model(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            backend, args.epochs, args.batch_size, args.lr, args.patience,
            model_dir, args.n_jobs,
        )
        
        # Evaluate
        print(f"\n" + "=" * 70)
        print("Evaluation")
        print("=" * 70)
        
        y_train_pred, metrics_train = evaluate_model(model, X_train_seq, y_train_seq, backend)
        y_val_pred, metrics_val = evaluate_model(model, X_val_seq, y_val_seq, backend)
        y_test_pred, metrics_test = evaluate_model(model, X_test_seq, y_test_seq, backend)
        
        print(f"\nTrain Metrics:")
        print(f"  R²: {metrics_train['R2']:.4f}")
        print(f"  MAE: {metrics_train['MAE']:.4f}")
        print(f"  RMSE: {metrics_train['RMSE']:.4f}")
        
        print(f"\nVal Metrics:")
        print(f"  R²: {metrics_val['R2']:.4f}")
        print(f"  MAE: {metrics_val['MAE']:.4f}")
        print(f"  RMSE: {metrics_val['RMSE']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  R²: {metrics_test['R2']:.4f}")
        print(f"  MAE: {metrics_test['MAE']:.4f}")
        print(f"  RMSE: {metrics_test['RMSE']:.4f}")
        
        # Save predictions
        preds_df = pd.DataFrame({
            "date": list(dates_train_seq) + list(dates_val_seq) + list(dates_test_seq),
            "y_true": np.concatenate([y_train_seq, y_val_seq, y_test_seq]),
            "y_pred": np.concatenate([y_train_pred, y_val_pred, y_test_pred]),
            "split": ["train"] * len(y_train_seq) + ["val"] * len(y_val_seq) + ["test"] * len(y_test_seq),
        })
        preds_df.to_csv(output_dir / "lstm_predictions.csv", index=False)
        print(f"\n✓ Saved predictions to {output_dir / 'lstm_predictions.csv'}")
        
        # Save metrics
        metrics_dict = {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        }
        with open(output_dir / "lstm_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✓ Saved metrics to {output_dir / 'lstm_metrics.json'}")
        
        # Compute μ/σ for Monte Carlo
        mu_hat = float(y_test_pred.mean())
        residuals_test = y_test_seq - y_test_pred
        sigma_hat = float(residuals_test.std())
        
        mu_sigma_dict = {
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "method": "mu from y_pred_test mean, sigma from residuals std",
        }
        with open(output_dir / "lstm_mu_sigma.json", "w") as f:
            json.dump(mu_sigma_dict, f, indent=2)
        print(f"✓ Saved μ/σ to {output_dir / 'lstm_mu_sigma.json'}")
        print(f"  μ_hat = {mu_hat:.6f}")
        print(f"  σ_hat = {sigma_hat:.6f}")
        
        # Plots
        print(f"\n" + "=" * 70)
        print("Generating Plots")
        print("=" * 70)
        
        # Test predictions vs actual
        plot_predictions(
            dates_test_seq, y_test_seq, y_test_pred, "test",
            plots_dir / "pred_vs_actual_return_lstm.png",
        )
        
        # Calibration
        plot_calibration(
            y_test_seq, y_test_pred,
            plots_dir / "calibration_return_lstm.png",
        )
        
        # Residuals
        residuals_test = y_test_seq - y_test_pred
        plot_residuals(
            dates_test_seq, residuals_test,
            plots_dir / "residuals_return_lstm.png",
        )
        
        # Rolling correlation
        plot_rolling_corr(
            dates_test_seq, y_test_seq, y_test_pred,
            window=12,
            output_path=plots_dir / "rolling_corr_lstm.png",
        )
        
        print(f"✓ Generated all plots")
        
        # Update README
        if args.save_readme:
            update_readme(metrics_dict, Path("README.md"), plots_dir)
        
        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

