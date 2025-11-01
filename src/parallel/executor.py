"""Parallel execution for Monte Carlo simulations."""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp


def run_parallel_simulations(
    data,
    configs: List[Dict[str, Any]],
    n_workers: Optional[int] = None,
    progress_bar: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple Monte Carlo simulations in parallel.

    Parameters
    ----------
    data : pd.DataFrame
        Historical stock data
    configs : List[Dict[str, Any]]
        List of configuration dictionaries for simulations.
        Each dict should contain keys like 'n_simulations', 'days_ahead', etc.
    n_workers : int, optional
        Number of parallel workers. If None, uses all available CPUs - 1.
    progress_bar : bool, default True
        Whether to show progress bar

    Returns
    -------
    List[Dict[str, Any]]
        List of simulation results for each configuration

    Examples
    --------
    >>> configs = [
    ...     {"n_simulations": 10000, "days_ahead": 30},
    ...     {"n_simulations": 10000, "days_ahead": 60},
    ...     {"n_simulations": 10000, "days_ahead": 90},
    ... ]
    >>> results = run_parallel_simulations(data, configs, n_workers=4)
    """
    from src.simulation.monte_carlo import MonteCarloForecast
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single simulation with given config."""
        forecast = MonteCarloForecast(**config)
        result = forecast.run(data)
        result["config"] = config
        return result
    
    # Run in parallel
    if progress_bar:
        results = Parallel(n_jobs=n_workers)(
            delayed(run_single_sim)(config)
            for config in tqdm(configs, desc="Running simulations")
        )
    else:
        results = Parallel(n_jobs=n_workers)(
            delayed(run_single_sim)(config) for config in configs
        )
    
    return results


def run_batch_simulations(
    data,
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    n_workers: Optional[int] = None,
    progress_bar: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run grid of Monte Carlo simulations (parameter sweep).

    Parameters
    ----------
    data : pd.DataFrame
        Historical stock data
    base_config : Dict[str, Any]
        Base configuration shared across all simulations
    param_grid : Dict[str, List[Any]]
        Dictionary mapping parameter names to lists of values to test.
        Example: {"days_ahead": [30, 60, 90], "n_simulations": [1000, 10000]}
    n_workers : int, optional
        Number of parallel workers. If None, uses all available CPUs - 1.
    progress_bar : bool, default True
        Whether to show progress bar

    Returns
    -------
    List[Dict[str, Any]]
        List of simulation results for each parameter combination

    Examples
    --------
    >>> base_config = {"confidence_level": 0.95}
    >>> param_grid = {
    ...     "days_ahead": [30, 60, 90],
    ...     "n_simulations": [5000, 10000],
    ... }
    >>> results = run_batch_simulations(data, base_config, param_grid)
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Create configs
    configs = []
    for combo in param_combinations:
        config = base_config.copy()
        for name, value in zip(param_names, combo):
            config[name] = value
        configs.append(config)
    
    # Run in parallel
    return run_parallel_simulations(
        data, configs, n_workers=n_workers, progress_bar=progress_bar
    )


def run_multiticker_simulations(
    tickers: List[str],
    data_func: Callable,
    config: Dict[str, Any],
    n_workers: Optional[int] = None,
    progress_bar: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run Monte Carlo simulations for multiple tickers in parallel.

    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols
    data_func : Callable
        Function to fetch data for a ticker: data_func(ticker) -> pd.DataFrame
    config : Dict[str, Any]
        Monte Carlo configuration
    n_workers : int, optional
        Number of parallel workers
    progress_bar : bool, default True
        Whether to show progress bar

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping ticker to simulation results

    Examples
    --------
    >>> tickers = ["NVDA", "AAPL", "MSFT"]
    >>> from src.data.fetch import fetch_stock_data
    >>> config = {"n_simulations": 10000, "days_ahead": 30}
    >>> results = run_multiticker_simulations(tickers, fetch_stock_data, config)
    """
    from src.simulation.monte_carlo import MonteCarloForecast
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    def run_ticker_sim(ticker: str) -> tuple:
        """Run simulation for a single ticker."""
        try:
            data = data_func(ticker)
            forecast = MonteCarloForecast(**config)
            result = forecast.run(data)
            return (ticker, result)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            return (ticker, None)
    
    # Run in parallel
    if progress_bar:
        results = Parallel(n_jobs=n_workers)(
            delayed(run_ticker_sim)(ticker)
            for ticker in tqdm(tickers, desc="Processing tickers")
        )
    else:
        results = Parallel(n_jobs=n_workers)(
            delayed(run_ticker_sim)(ticker) for ticker in tickers
        )
    
    # Convert to dictionary
    return {ticker: result for ticker, result in results if result is not None}

