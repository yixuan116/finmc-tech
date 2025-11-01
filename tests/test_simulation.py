"""Tests for Monte Carlo simulation module."""

import pytest
import numpy as np
import pandas as pd
from src.simulation.monte_carlo import MonteCarloForecast, MonteCarloConfig
from src.simulation.uncertainty import UncertaintyAnalyzer


def create_sample_data():
    """Create sample stock data for testing."""
    import pandas as pd
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = np.random.normal(0.001, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "date": dates,
        "close": prices,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "volume": np.random.randint(1000000, 10000000, 252),
    })
    
    from src.data.fetch import compute_returns, compute_volatility
    data = compute_returns(data)
    data = compute_volatility(data)
    
    return data


def test_monte_carlo_config():
    """Test Monte Carlo configuration."""
    config = MonteCarloConfig(
        n_simulations=1000,
        days_ahead=30,
        confidence_level=0.95
    )
    
    assert config.n_simulations == 1000
    assert config.days_ahead == 30
    assert config.confidence_level == 0.95


def test_monte_carlo_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        MonteCarloConfig(n_simulations=0)
    
    with pytest.raises(ValueError):
        MonteCarloConfig(days_ahead=0)
    
    with pytest.raises(ValueError):
        MonteCarloConfig(confidence_level=1.5)


def test_monte_carlo_forecast():
    """Test Monte Carlo forecasting."""
    data = create_sample_data()
    
    forecast = MonteCarloForecast(
        n_simulations=1000,
        days_ahead=30,
        random_seed=42
    )
    
    results = forecast.run(data)
    
    assert "paths" in results
    assert "final_prices" in results
    assert "expected_return" in results
    assert "expected_price" in results
    assert "ci_lower" in results
    assert "ci_upper" in results
    
    assert len(results["paths"]) == 1000
    assert results["paths"].shape[1] == 31  # n_steps + 1
    assert results["ci_lower"] < results["ci_upper"]


def test_uncertainty_analyzer():
    """Test uncertainty analysis."""
    data = create_sample_data()
    
    forecast = MonteCarloForecast(n_simulations=1000, random_seed=42)
    results = forecast.run(data)
    
    analyzer = UncertaintyAnalyzer(results)
    
    # Test risk metrics
    risk_metrics = analyzer.compute_risk_metrics()
    assert "VaR_5pct" in risk_metrics
    assert "Volatility" in risk_metrics
    assert "Expected_Return" in risk_metrics
    
    # Test percentiles
    percentiles = analyzer.compute_percentiles()
    assert "p50" in percentiles
    assert percentiles["p5"] < percentiles["p95"]
    
    # Test probabilities
    probabilities = analyzer.compute_probability_metrics()
    assert "Prob_Positive_Return" in probabilities
    assert 0 <= probabilities["Prob_Positive_Return"] <= 1


def test_uncertainty_analyzer_summary():
    """Test uncertainty analyzer summary."""
    data = create_sample_data()
    
    forecast = MonteCarloForecast(n_simulations=1000, random_seed=42)
    results = forecast.run(data)
    
    analyzer = UncertaintyAnalyzer(results)
    summary = analyzer.get_summary()
    
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 1
    assert "Mean" in summary.columns
    assert "VaR_5pct" in summary.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

