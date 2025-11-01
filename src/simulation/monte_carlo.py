"""Monte Carlo simulation for stock price forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    
    n_simulations: int = 10000
    days_ahead: int = 30
    confidence_level: float = 0.95
    dt: float = 1.0 / 252.0  # Daily time step (1 trading day)
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if self.days_ahead <= 0:
            raise ValueError("days_ahead must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


class MonteCarloForecast:
    """
    Monte Carlo simulation for stock price forecasting.
    
    Uses Geometric Brownian Motion (GBM) model:
    dS = μS dt + σS dW
    
    where:
    - S is the stock price
    - μ is the drift (expected return)
    - σ is the volatility
    - dW is a Wiener process
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        days_ahead: int = 30,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo forecast.

        Parameters
        ----------
        n_simulations : int, default 10000
            Number of simulation paths
        days_ahead : int, default 30
            Number of days to forecast ahead
        confidence_level : float, default 0.95
            Confidence level for interval estimates (e.g., 0.95 for 95% CI)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.config = MonteCarloConfig(
            n_simulations=n_simulations,
            days_ahead=days_ahead,
            confidence_level=confidence_level,
            random_seed=random_seed,
        )
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _compute_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute μ (drift) and σ (volatility) from historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Historical stock data with returns

        Returns
        -------
        Dict[str, float]
            Dictionary with 'mu' (annualized drift) and 'sigma' (annualized volatility)
        """
        # Use log returns for better statistical properties
        log_returns = data["log_returns"].dropna()
        
        # Drift (mean log return, annualized)
        mu = log_returns.mean() * 252
        
        # Volatility (std of log returns, annualized)
        sigma = log_returns.std() * np.sqrt(252)
        
        return {"mu": mu, "sigma": sigma}
    
    def _simulate_path(
        self,
        S0: float,
        mu: float,
        sigma: float,
        dt: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Simulate a single price path using GBM.

        Parameters
        ----------
        S0 : float
            Initial stock price
        mu : float
            Annualized drift (expected return)
        sigma : float
            Annualized volatility
        dt : float
            Time step
        n_steps : int
            Number of time steps

        Returns
        -------
        np.ndarray
            Simulated price path of length n_steps + 1
        """
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        
        # GBM formula
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        
        # Compute cumulative log returns
        cumsum_log_returns = np.cumsum(log_returns)
        
        # Convert back to prices
        path = S0 * np.exp(cumsum_log_returns)
        
        # Prepend initial price
        path = np.insert(path, 0, S0)
        
        return path
    
    def run(
        self,
        data: pd.DataFrame,
        S0: Optional[float] = None,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        data : pd.DataFrame
            Historical stock data
        S0 : float, optional
            Initial price. If None, uses latest close price from data
        mu : float, optional
            Drift parameter. If None, computed from data
        sigma : float, optional
            Volatility parameter. If None, computed from data

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'paths': All simulated paths (n_simulations × days_ahead+1)
            - 'final_prices': Final prices for each simulation
            - 'expected_return': Expected log return over forecast period
            - 'expected_price': Expected stock price at end of forecast
            - 'ci_lower': Lower bound of confidence interval
            - 'ci_upper': Upper bound of confidence interval
        """
        # Compute or use provided parameters
        params = self._compute_parameters(data)
        if mu is None:
            mu = params["mu"]
        if sigma is None:
            sigma = params["sigma"]
        if S0 is None:
            S0 = data["close"].iloc[-1]
        
        # Allocate arrays
        n_sims = self.config.n_simulations
        n_steps = self.config.days_ahead
        paths = np.zeros((n_sims, n_steps + 1))
        final_prices = np.zeros(n_sims)
        
        # Run simulations
        for i in range(n_sims):
            path = self._simulate_path(S0, mu, sigma, self.config.dt, n_steps)
            paths[i] = path
            final_prices[i] = path[-1]
        
        # Compute statistics
        expected_price = np.mean(final_prices)
        expected_return = np.log(expected_price / S0) / (self.config.days_ahead / 252)
        
        # Confidence intervals
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(final_prices, 100 * alpha / 2)
        ci_upper = np.percentile(final_prices, 100 * (1 - alpha / 2))
        
        results = {
            "paths": paths,
            "final_prices": final_prices,
            "expected_return": expected_return,
            "expected_price": expected_price,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "S0": S0,
            "mu": mu,
            "sigma": sigma,
        }
        
        return results
    
    def run_with_uncertainty(
        self,
        data: pd.DataFrame,
        n_bootstrap: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation with parameter uncertainty.

        Parameters
        ----------
        data : pd.DataFrame
            Historical stock data
        n_bootstrap : int, default 100
            Number of bootstrap samples for parameter estimation

        Returns
        -------
        Dict[str, np.ndarray]
            Results with uncertainty quantification
        """
        # Bootstrap parameter estimation
        log_returns = data["log_returns"].dropna()
        
        mu_samples = []
        sigma_samples = []
        
        for _ in range(n_bootstrap):
            boot_returns = np.random.choice(log_returns, size=len(log_returns), replace=True)
            mu_samples.append(boot_returns.mean() * 252)
            sigma_samples.append(boot_returns.std() * np.sqrt(252))
        
        # Use median parameters for main simulation
        mu = np.median(mu_samples)
        sigma = np.median(sigma_samples)
        
        # Run main simulation
        results = self.run(data, mu=mu, sigma=sigma)
        
        # Add uncertainty information
        results["mu_samples"] = np.array(mu_samples)
        results["sigma_samples"] = np.array(sigma_samples)
        results["mu_uncertainty"] = np.std(mu_samples)
        results["sigma_uncertainty"] = np.std(sigma_samples)
        
        return results

