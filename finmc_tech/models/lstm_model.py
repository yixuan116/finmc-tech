"""
LSTM model for return prediction (optional, requires keras/tensorflow).

Gate behind "dl" extra in requirements.
"""

# Guarded import with clear error message
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TF = True
except ImportError as e:
    HAS_TF = False
    _TF_IMPORT_ERROR = e

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

from finmc_tech.config import get_logger

logger = get_logger(__name__)


def build_lstm(
    input_dim: int,
    timesteps: int = 6,
    units: int = 64,
    dropout: float = 0.1,
) -> Sequential:
    """
    Build and compile LSTM model.
    
    Args:
        input_dim: Number of features per timestep
        timesteps: Number of timesteps in sequence
        units: Number of LSTM units
        dropout: Dropout rate
    
    Returns:
        Compiled Keras Sequential model
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow/Keras not installed. "
            "Install with: pip install tensorflow>=2.13.0 "
            "or pip install finmc-tech[dl]"
        ) from _TF_IMPORT_ERROR
    
    logger.info(f"Building LSTM model: input_dim={input_dim}, timesteps={timesteps}, units={units}")
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(timesteps, input_dim)),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    
    logger.info("  ✓ LSTM model compiled")
    return model


def to_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timesteps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series data to sequences for LSTM.
    
    Creates sliding windows of timesteps length, aligning y with last timestamp.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        timesteps: Number of timesteps in sequence
    
    Returns:
        X_seq: Sequence array (n_samples - timesteps + 1, timesteps, n_features)
        y_seq: Target array (n_samples - timesteps + 1,) aligned with last timestamp
    """
    n_samples, n_features = X.shape
    
    if len(y) != n_samples:
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    X_seq = []
    y_seq = []
    
    # Create sequences: for each position i, use [i-timesteps+1:i+1]
    # y_seq[i] corresponds to y[i] (last timestamp in sequence)
    for i in range(timesteps - 1, n_samples):
        X_seq.append(X[i - timesteps + 1:i + 1])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def fit_lstm(
    model: Sequential,
    X_seq_train: np.ndarray,
    y_seq_train: np.ndarray,
    X_seq_val: Optional[np.ndarray] = None,
    y_seq_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    out_best: Optional[Path] = None,
) -> Sequential:
    """
    Fit LSTM model with early stopping and optional best model saving.
    
    Args:
        model: Compiled Keras model
        X_seq_train: Training sequences (n_samples, timesteps, n_features)
        y_seq_train: Training targets (n_samples,)
        X_seq_val: Validation sequences (optional)
        y_seq_val: Validation targets (optional)
        epochs: Maximum number of epochs
        batch_size: Batch size
        patience: Early stopping patience
        out_best: Path to save best model (optional)
    
    Returns:
        Fitted model (best model if out_best provided)
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow/Keras not installed. "
            "Install with: pip install tensorflow>=2.13.0"
        ) from _TF_IMPORT_ERROR
    
    logger.info(f"Fitting LSTM: {len(X_seq_train)} train samples, epochs={epochs}, patience={patience}")
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss" if X_seq_val is not None else "loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stop)
    
    # Save best model
    if out_best is not None:
        out_best = Path(out_best)
        out_best.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            str(out_best),
            monitor="val_loss" if X_seq_val is not None else "loss",
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint)
    
    # Train
    validation_data = (X_seq_val, y_seq_val) if X_seq_val is not None else None
    
    history = model.fit(
        X_seq_train,
        y_seq_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Load best model if saved
    if out_best is not None and out_best.exists():
        logger.info(f"  Loading best model from {out_best}")
        from tensorflow.keras.models import load_model
        model = load_model(str(out_best))
    
    logger.info("  ✓ LSTM training complete")
    return model


def predict_lstm(
    model: Sequential,
    X_seq_test: np.ndarray,
) -> np.ndarray:
    """
    Predict using LSTM model.
    
    Returns predictions aligned with last timestamp of each sequence.
    
    Args:
        model: Fitted Keras model
        X_seq_test: Test sequences (n_samples, timesteps, n_features)
    
    Returns:
        Predictions (n_samples,) aligned with last timestamp
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow/Keras not installed. "
            "Install with: pip install tensorflow>=2.13.0"
        ) from _TF_IMPORT_ERROR
    
    yhat = model.predict(X_seq_test, verbose=0)
    
    # Flatten if needed (Dense(1) outputs shape (n_samples, 1))
    if yhat.ndim > 1:
        yhat = yhat.flatten()
    
    return yhat


# Backward compatibility functions
def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[Sequential, dict]:
    """Backward compatibility wrapper."""
    model = build_lstm(
        input_dim=X_train.shape[2],
        timesteps=X_train.shape[1],
        units=config.LSTM_UNITS,
        dropout=config.LSTM_DROPOUT,
    )
    
    model = fit_lstm(
        model,
        X_train,
        y_train,
        X_seq_val=X_val,
        y_seq_val=y_val,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
    )
    
    return model, {}


def prepare_lstm_data(
    X: pd.DataFrame,
    y: pd.Series,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward compatibility wrapper."""
    return to_sequences(X.values, y.values, sequence_length)

