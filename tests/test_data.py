"""Tests for data fetching module."""

import pytest
import pandas as pd
from src.data.fetch import fetch_stock_data, compute_returns, compute_volatility


def test_fetch_stock_data():
    """Test fetching stock data."""
    data = fetch_stock_data("NVDA", start="2020-01-01", end="2020-12-31")
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns
    assert "close" in data.columns
    assert "returns" in data.columns
    assert "log_returns" in data.columns
    assert "volatility" in data.columns


def test_compute_returns():
    """Test returns computation."""
    data = pd.DataFrame({
        "close": [100, 105, 110, 108]
    })
    
    result = compute_returns(data)
    
    assert "returns" in result.columns
    assert "log_returns" in result.columns
    assert result.iloc[0]["returns"] == 0  # First row should be NaN or 0
    assert result.iloc[1]["returns"] == pytest.approx(0.05, rel=1e-2)


def test_compute_volatility():
    """Test volatility computation."""
    data = pd.DataFrame({
        "log_returns": [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01]
    })
    
    result = compute_volatility(data, window=5)
    
    assert "volatility" in result.columns
    assert not result.iloc[0:4]["volatility"].isna().any()  # Should have values after window


def test_fetch_invalid_ticker():
    """Test handling of invalid ticker."""
    with pytest.raises(ValueError):
        fetch_stock_data("INVALID_TICKER_XYZ123", start="2020-01-01", end="2020-12-31")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

