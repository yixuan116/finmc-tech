"""Machine learning models for stock forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class ForecastingModel:
    """
    Wrapper class for forecasting models with cross-validation and evaluation.
    """
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize forecasting model.

        Parameters
        ----------
        model_type : str, default "random_forest"
            Type of model to use. Options: "random_forest", "gradient_boosting", "ridge", "lasso"
        **kwargs
            Additional model parameters
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **kwargs
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                **kwargs
            )
        elif model_type == "ridge":
            self.model = Ridge(alpha=1.0, **kwargs)
        elif model_type == "lasso":
            self.model = Lasso(alpha=1.0, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, scale_features: bool = True):
        """
        Train the model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        scale_features : bool, default True
            Whether to scale features
        """
        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame, scale_features: bool = True) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        scale_features : bool, default True
            Whether to scale features

        Returns
        -------
        np.ndarray
            Predictions
        """
        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        scale_features : bool, default True
            Whether to scale features

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics: MSE, MAE, RMSE, R2
        """
        y_pred = self.predict(X, scale_features=scale_features)
        
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        return {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }


def train_forecasting_model(
    features: pd.DataFrame,
    test_size: float = 0.2,
    model_type: str = "random_forest",
    random_seed: int = 42,
    **model_kwargs,
) -> ForecastingModel:
    """
    Train a forecasting model with train/test split.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with 'target' column
    test_size : float, default 0.2
        Proportion of data to use for testing
    model_type : str, default "random_forest"
        Type of model to train
    random_seed : int, default 42
        Random seed for reproducibility
    **model_kwargs
        Additional model parameters

    Returns
    -------
    ForecastingModel
        Trained model

    Examples
    --------
    >>> model = train_forecasting_model(features)
    >>> metrics = model.evaluate(X_test, y_test)
    """
    # Prepare data
    X = features.drop(columns=["target"])
    y = features["target"]
    
    # Time series split (sequential)
    n_train = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    
    # Train model
    model = ForecastingModel(model_type=model_type, **model_kwargs)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model


def predict_returns(
    model: ForecastingModel,
    features: pd.DataFrame,
    n_periods: int = 30,
) -> np.ndarray:
    """
    Predict future returns using trained model.

    Parameters
    ----------
    model : ForecastingModel
        Trained forecasting model
    features : pd.DataFrame
        Feature matrix
    n_periods : int, default 30
        Number of periods to forecast ahead

    Returns
    -------
    np.ndarray
        Predicted returns
    """
    # Remove target column if present
    X = features.drop(columns=["target"]) if "target" in features.columns else features
    
    # Get last period of data
    X_last = X.iloc[-1:].copy()
    
    # Make predictions
    predictions = []
    
    for _ in range(n_periods):
        pred = model.predict(X_last)
        predictions.append(pred[0])
    
    return np.array(predictions)


def compare_models(
    features: pd.DataFrame,
    model_types: Optional[list] = None,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix with 'target' column
    model_types : List[str], optional
        List of model types to compare
    test_size : float, default 0.2
        Proportion of data to use for testing

    Returns
    -------
    pd.DataFrame
        Comparison of model metrics
    """
    if model_types is None:
        model_types = ["random_forest", "gradient_boosting", "ridge", "lasso"]
    
    results = []
    
    X = features.drop(columns=["target"])
    y = features["target"]
    
    # Time series split
    n_train = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    
    for model_type in model_types:
        model = ForecastingModel(model_type=model_type)
        model.fit(X_train, y_train)
        
        test_metrics = model.evaluate(X_test, y_test)
        test_metrics["model_type"] = model_type
        
        results.append(test_metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[["model_type", "RMSE", "MAE", "R2"]]
    
    return comparison_df

