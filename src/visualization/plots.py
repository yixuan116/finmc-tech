"""Plotting functions for visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_price_history(
    data: pd.DataFrame,
    ticker: str,
    show_indicators: bool = True,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """
    Plot historical price data with technical indicators.

    Parameters
    ----------
    data : pd.DataFrame
        Historical stock data
    ticker : str
        Stock ticker symbol
    show_indicators : bool, default True
        Whether to show moving averages
    figsize : tuple, default (14, 7)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Price with indicators
    ax1.plot(data["date"], data["close"], label="Close Price", linewidth=2)
    
    if show_indicators:
        if "sma_20" in data.columns:
            ax1.plot(data["date"], data["sma_20"], label="SMA 20", alpha=0.7)
        if "sma_50" in data.columns:
            ax1.plot(data["date"], data["sma_50"], label="SMA 50", alpha=0.7)
        if "sma_200" in data.columns:
            ax1.plot(data["date"], data["sma_200"], label="SMA 200", alpha=0.7)
    
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.set_title(f"{ticker} - Historical Price and Indicators", fontsize=14, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # Volume
    ax2.bar(data["date"], data["volume"], alpha=0.3, color="blue")
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.set_title("Trading Volume", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Returns
    ax3.plot(data["date"], data["returns"] * 100, alpha=0.6, color="green")
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax3.set_ylabel("Returns (%)", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_title("Daily Returns", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_monte_carlo_results(
    results: Dict[str, np.ndarray],
    data: pd.DataFrame,
    ticker: str,
    n_paths_show: int = 100,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot Monte Carlo simulation results (in percentage returns).

    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Monte Carlo simulation results
    data : pd.DataFrame
        Historical stock data
    ticker : str
        Stock ticker symbol
    n_paths_show : int, default 100
        Number of paths to show in plot
    figsize : tuple, default (14, 8)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    paths = results["paths"]
    S0 = results["S0"]
    final_prices = results["final_prices"]
    
    # Convert paths and final prices to percentage returns
    return_paths = (paths / S0 - 1) * 100
    final_returns = (final_prices / S0 - 1) * 100
    
    # Plot 1: Historical data (keep as prices) + Forecast paths (as returns)
    # Historical data - convert to returns relative to last price
    last_price = data["close"].iloc[-1]
    historical_returns = ((data["close"] / last_price - 1) * 100).values
    ax1.plot(data["date"], historical_returns, label="Historical Returns", linewidth=2, color="black")
    
    # Forecast paths (sample) - as percentage returns
    last_date = data["date"].iloc[-1]
    dates_forward = pd.date_range(
        start=last_date,
        periods=results["paths"].shape[1],
        freq="D"
    )
    
    n_show = min(n_paths_show, len(return_paths))
    for i in range(n_show):
        ax1.plot(dates_forward, return_paths[i], alpha=0.1, color="blue")
    
    # Mean path
    mean_return = np.mean(return_paths, axis=0)
    ax1.plot(dates_forward, mean_return, label="Mean Forecast", linewidth=2, color="red")
    
    # Confidence intervals
    ci_lower = np.percentile(return_paths, 2.5, axis=0)
    ci_upper = np.percentile(return_paths, 97.5, axis=0)
    ax1.fill_between(
        dates_forward,
        ci_lower,
        ci_upper,
        alpha=0.2,
        color="red",
        label="95% CI"
    )
    
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_title(f"{ticker} - Monte Carlo Forecast (Returns)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Return (%)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)
    
    # Plot 2: Distribution of final returns
    expected_return = (results["expected_price"] / S0 - 1) * 100
    ci_lower_return = (results["ci_lower"] / S0 - 1) * 100
    ci_upper_return = (results["ci_upper"] / S0 - 1) * 100
    
    ax2.hist(final_returns, bins=50, alpha=0.7, edgecolor="black")
    ax2.axvline(expected_return, color="red", linewidth=2, 
                label=f"Expected: {expected_return:.1f}%")
    ax2.axvline(0, color="black", linewidth=2, 
                label="Current (0%)")
    ax2.axvline(ci_lower_return, color="red", linestyle="--", 
                label=f"95% CI Lower: {ci_lower_return:.1f}%")
    ax2.axvline(ci_upper_return, color="red", linestyle="--", 
                label=f"95% CI Upper: {ci_upper_return:.1f}%")
    
    ax2.set_title("Distribution of Forecasted Returns", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Total Return (%)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_uncertainty_analysis(
    analyzer,
    ticker: str,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot uncertainty and risk metrics.

    Parameters
    ----------
    analyzer : UncertaintyAnalyzer
        Uncertainty analyzer instance
    ticker : str
        Stock ticker symbol
    figsize : tuple, default (14, 8)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get metrics
    risk_metrics = analyzer.compute_risk_metrics()
    percentiles = analyzer.compute_percentiles()
    probabilities = analyzer.compute_probability_metrics()
    
    # Plot 1: Risk metrics
    risk_labels = ["VaR_5pct", "CVaR_5pct", "Volatility", "Downside_Deviation"]
    risk_values = [risk_metrics[label] for label in risk_labels if label in risk_metrics]
    
    axes[0, 0].bar(range(len(risk_values)), risk_values)
    axes[0, 0].set_xticks(range(len(risk_labels)))
    axes[0, 0].set_xticklabels(risk_labels, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].set_title("Risk Metrics", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Price percentiles
    p_labels = list(percentiles.keys())
    p_values = list(percentiles.values())
    
    axes[0, 1].bar(p_labels, p_values)
    axes[0, 1].set_ylabel("Price ($)")
    axes[0, 1].set_title("Price Percentiles", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Probability metrics
    prob_labels = list(probabilities.keys())
    prob_values = [v * 100 for v in probabilities.values()]
    
    axes[1, 0].barh(prob_labels, prob_values)
    axes[1, 0].set_xlabel("Probability (%)")
    axes[1, 0].set_title("Scenario Probabilities", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Return distribution
    returns = (analyzer.final_prices - analyzer.S0) / analyzer.S0
    axes[1, 1].hist(returns * 100, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Return (%)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Return Distribution", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"{ticker} - Uncertainty Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_risk_metrics(
    results_list: List[Dict[str, Any]],
    tickers: List[str],
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Compare risk metrics across multiple assets.

    Parameters
    ----------
    results_list : List[Dict[str, Any]]
        List of Monte Carlo results dictionaries
    tickers : List[str]
        List of ticker symbols
    figsize : tuple, default (14, 6)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    from src.simulation.uncertainty import UncertaintyAnalyzer
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    var_values = []
    volatility_values = []
    
    for result in results_list:
        analyzer = UncertaintyAnalyzer(result)
        metrics = analyzer.compute_risk_metrics()
        var_values.append(metrics.get("VaR_5pct", 0))
        volatility_values.append(metrics.get("Volatility", 0))
    
    # VaR comparison
    ax1.bar(tickers, var_values)
    ax1.set_ylabel("VaR (5%)", fontsize=12)
    ax1.set_title("Value at Risk Comparison", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Volatility comparison
    ax2.bar(tickers, volatility_values)
    ax2.set_ylabel("Volatility", fontsize=12)
    ax2.set_title("Volatility Comparison", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def plot_ml_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "ML Model Predictions",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot ML model predictions vs actual values.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str, default "ML Model Predictions"
        Plot title
    figsize : tuple, default (14, 5)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Time series comparison
    ax1.plot(y_true, label="Actual", alpha=0.7, linewidth=2)
    ax1.plot(y_pred, label="Predicted", alpha=0.7, linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Return")
    ax1.set_title("Predictions vs Actual", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Predictions vs Actual (Scatter)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

