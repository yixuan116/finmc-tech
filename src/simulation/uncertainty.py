"""Uncertainty quantification and analysis for Monte Carlo simulations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class UncertaintyAnalyzer:
    """
    Analyze uncertainty and risk metrics from Monte Carlo simulation results.
    """
    
    def __init__(self, results: Dict[str, np.ndarray]):
        """
        Initialize uncertainty analyzer.

        Parameters
        ----------
        results : Dict[str, np.ndarray]
            Monte Carlo simulation results from MonteCarloForecast.run()
        """
        self.results = results
        self.final_prices = results["final_prices"]
        self.paths = results.get("paths")
        self.S0 = results.get("S0")
    
    def compute_risk_metrics(self) -> Dict[str, float]:
        """
        Compute various risk metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary of risk metrics:
            - VaR: Value at Risk (negative of 5th percentile)
            - CVaR: Conditional VaR (expected loss beyond VaR)
            - Expected Shortfall: Average loss in worst 5% scenarios
            - Volatility: Std dev of forecasted returns
            - Downside_Deviation: Std dev of negative returns only
        """
        if self.S0 is None:
            raise ValueError("S0 not found in results")
        
        # Compute returns
        returns = (self.final_prices - self.S0) / self.S0
        
        # Value at Risk (5th percentile)
        var_5 = np.percentile(returns, 5)
        var_5_value = -var_5  # Convert to positive (loss)
        
        # Conditional VaR / Expected Shortfall
        worst_5_percent = returns[returns <= np.percentile(returns, 5)]
        cvar = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else 0
        expected_shortfall = -cvar
        
        # Volatility
        volatility = np.std(returns)
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            np.std(downside_returns) if len(downside_returns) > 0 else 0
        )
        
        # Maximum Drawdown (if paths available)
        max_drawdown = None
        if self.paths is not None:
            max_drawdown = self._compute_max_drawdown(self.paths)
        
        metrics = {
            "VaR_5pct": var_5_value,
            "CVaR_5pct": expected_shortfall,
            "Expected_Shortfall": expected_shortfall,
            "Volatility": volatility,
            "Downside_Deviation": downside_deviation,
            "Expected_Return": np.mean(returns),
            "Median_Return": np.median(returns),
            "Max_Return": np.max(returns),
            "Min_Return": np.min(returns),
        }
        
        if max_drawdown is not None:
            metrics["Max_Drawdown"] = max_drawdown
        
        return metrics
    
    def _compute_max_drawdown(self, paths: np.ndarray) -> float:
        """
        Compute maximum drawdown across all paths.

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths

        Returns
        -------
        float
            Maximum drawdown as a fraction
        """
        drawdowns = []
        
        for path in paths:
            cumulative_max = np.maximum.accumulate(path)
            drawdown = (cumulative_max - path) / cumulative_max
            drawdowns.append(np.max(drawdown))
        
        return np.mean(drawdowns)
    
    def compute_percentiles(
        self, percentiles: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute price percentiles.

        Parameters
        ----------
        percentiles : List[float], optional
            Percentiles to compute. Default: [5, 25, 50, 75, 95]

        Returns
        -------
        Dict[str, float]
            Dictionary mapping percentile to price
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        
        result = {}
        for p in percentiles:
            result[f"p{p}"] = np.percentile(self.final_prices, p)
        
        return result
    
    def compute_probability_metrics(self) -> Dict[str, float]:
        """
        Compute probability-based metrics.

        Returns
        -------
        Dict[str, float]
            Probabilities for various scenarios
        """
        if self.S0 is None:
            raise ValueError("S0 not found in results")
        
        returns = (self.final_prices - self.S0) / self.S0
        
        prob_positive = np.mean(returns > 0)
        prob_negative = np.mean(returns < 0)
        prob_above_10pct = np.mean(returns > 0.10)
        prob_below_minus10pct = np.mean(returns < -0.10)
        prob_below_minus20pct = np.mean(returns < -0.20)
        
        return {
            "Prob_Positive_Return": prob_positive,
            "Prob_Negative_Return": prob_negative,
            "Prob_Return_Above_10pct": prob_above_10pct,
            "Prob_Return_Below_minus10pct": prob_below_minus10pct,
            "Prob_Return_Below_minus20pct": prob_below_minus20pct,
        }
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get comprehensive summary of results.

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        summary_dict = {}
        
        # Basic statistics
        summary_dict["Mean"] = np.mean(self.final_prices)
        summary_dict["Median"] = np.median(self.final_prices)
        summary_dict["Std"] = np.std(self.final_prices)
        summary_dict["Min"] = np.min(self.final_prices)
        summary_dict["Max"] = np.max(self.final_prices)
        
        # Risk metrics
        risk_metrics = self.compute_risk_metrics()
        summary_dict.update(risk_metrics)
        
        # Percentiles
        percentiles = self.compute_percentiles()
        summary_dict.update(percentiles)
        
        # Probabilities
        probabilities = self.compute_probability_metrics()
        summary_dict.update(probabilities)
        
        return pd.DataFrame([summary_dict])

