#!/usr/bin/env python3
"""
Example: How to load and use the feature scaler for predictions.

This script demonstrates:
1. Loading the saved scaler
2. Loading the champion model
3. Preparing new data with the same features
4. Scaling and making predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def load_scaler_and_model():
    """Load the saved scaler and champion model."""
    models_dir = Path("models")
    
    # Load scaler
    scaler_path = models_dir / "feature_scaler.pkl"
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded scaler from: {scaler_path}")
    print(f"  Type: {type(scaler).__name__}")
    print(f"  Expected features: {len(scaler.mean_)}")
    
    # Load champion model
    model_path = models_dir / "champion_model.pkl"
    model = joblib.load(model_path)
    print(f"✓ Loaded champion model from: {model_path}")
    print(f"  Type: {type(model).__name__}")
    
    # Get feature names from scaler (if available)
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_
        print(f"  Feature names: {list(feature_names[:5])}... ({len(feature_names)} total)")
    else:
        print(f"  (Feature names not stored in scaler)")
    
    return scaler, model


def prepare_features_for_prediction(df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Prepare features from a DataFrame to match the scaler's expected format.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features (must have same columns as training data)
    scaler : StandardScaler
        Fitted scaler that knows the expected feature names
    
    Returns
    -------
    np.ndarray
        Scaled feature matrix ready for model prediction
    """
    # If scaler has feature names, use them to select columns
    if hasattr(scaler, 'feature_names_in_'):
        required_features = scaler.feature_names_in_
        
        # Check which features are missing
        missing = set(required_features) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required features: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Select features in the same order as training
        X = df[required_features].values
    else:
        # Fallback: use all numeric columns (must match training)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values
        
        if X.shape[1] != len(scaler.mean_):
            raise ValueError(
                f"Feature count mismatch: got {X.shape[1]}, expected {len(scaler.mean_)}"
            )
    
    # Handle NaN values (fill with 0 or forward-fill)
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled


def make_prediction(scaler, model, new_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data using the saved scaler and model.
    
    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler
    model : sklearn model
        Trained model
    new_data : pd.DataFrame
        New data with features
    
    Returns
    -------
    np.ndarray
        Predictions
    """
    # Prepare and scale features
    X_scaled = prepare_features_for_prediction(new_data, scaler)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions


if __name__ == "__main__":
    print("=" * 70)
    print("Example: Loading Scaler and Making Predictions")
    print("=" * 70)
    
    # 1. Load scaler and model
    scaler, model = load_scaler_and_model()
    
    # 2. Example: Load the same data format used for training
    data_path = Path("data/processed/nvda_features_extended_v2.csv")
    if data_path.exists():
        print(f"\n2. Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Set date index if available
        date_col = 'px_date' if 'px_date' in df.columns else 'period_end'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        
        # Use the most recent row as example
        print(f"   Data shape: {df.shape}")
        print(f"   Using most recent row for prediction example")
        
        # Get the last row (most recent data)
        latest_data = df.iloc[-1:].copy()
        
        # Make prediction
        try:
            pred = make_prediction(scaler, model, latest_data)
            print(f"\n✓ Prediction successful!")
            print(f"   Predicted return: {pred[0]:.4f}")
            print(f"   (This is the predicted next-period return)")
        except Exception as e:
            print(f"\n✗ Prediction failed: {e}")
            print(f"   This usually means feature columns don't match.")
            print(f"   Make sure your new data has the same features as training data.")
    else:
        print(f"\n2. Data file not found: {data_path}")
        print("   (This is just a demonstration)")
    
    print("\n" + "=" * 70)
    print("Usage Summary:")
    print("=" * 70)
    print("""
# In your prediction script:
from examples.load_scaler_example import load_scaler_and_model, make_prediction

# Load scaler and model
scaler, model = load_scaler_and_model()

# Prepare your new data (must have same features as training)
new_data = pd.DataFrame({...})  # Your features here

# Make prediction
prediction = make_prediction(scaler, model, new_data)
print(f"Predicted return: {prediction[0]}")
    """)

