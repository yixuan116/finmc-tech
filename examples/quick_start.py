"""
Quick Start Example for finmc-tech

This script demonstrates the core functionality of the ML + Monte Carlo
stock forecasting framework.
"""

import sys
sys.path.append("../")

from src.data.fetch import fetch_stock_data, compute_statistics
from src.simulation.monte_carlo import MonteCarloForecast
from src.simulation.uncertainty import UncertaintyAnalyzer
from src.parallel.executor import run_parallel_simulations


def main():
    print("=" * 60)
    print("finmc-tech: ML + Monte Carlo Stock Forecasting")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n[Step 1] Fetching NVIDIA stock data (2010-2025)...")
    try:
        data = fetch_stock_data("NVDA")
        print(f"✓ Successfully fetched {len(data)} data points")
        print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return
    
    # Step 2: Compute statistics
    print("\n[Step 2] Computing summary statistics...")
    stats = compute_statistics(data)
    print(f"  Annualized Return: {stats['annualized_return']:.2%}")
    print(f"  Volatility: {stats['volatility']:.2%}")
    if stats['sharpe_ratio']:
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    # Step 3: Monte Carlo simulation
    print("\n[Step 3] Running Monte Carlo simulation...")
    print("  Simulations: 10,000")
    print("  Forecast horizon: 30 days")
    
    forecast = MonteCarloForecast(
        n_simulations=10000,
        days_ahead=30,
        confidence_level=0.95,
        random_seed=42
    )
    
    results = forecast.run(data)
    
    print(f"\n  Results:")
    print(f"    Current Price: ${results['S0']:.2f}")
    print(f"    Expected Price: ${results['expected_price']:.2f}")
    print(f"    Expected Return: {results['expected_return']:.2%}")
    print(f"    95% CI: [${results['ci_lower']:.2f}, ${results['ci_upper']:.2f}]")
    
    # Step 4: Uncertainty analysis
    print("\n[Step 4] Computing risk metrics...")
    analyzer = UncertaintyAnalyzer(results)
    risk_metrics = analyzer.compute_risk_metrics()
    
    print(f"  VaR (5%): {risk_metrics['VaR_5pct']:.2%}")
    print(f"  CVaR (5%): {risk_metrics['CVaR_5pct']:.2%}")
    print(f"  Volatility: {risk_metrics['Volatility']:.2%}")
    print(f"  Max Drawdown: {risk_metrics.get('Max_Drawdown', 0):.2%}")
    
    # Step 5: Parallel simulations
    print("\n[Step 5] Running parallel simulations...")
    print("  Testing multiple forecast horizons: 30, 60, 90 days")
    
    configs = [
        {"n_simulations": 5000, "days_ahead": 30, "random_seed": 42},
        {"n_simulations": 5000, "days_ahead": 60, "random_seed": 43},
        {"n_simulations": 5000, "days_ahead": 90, "random_seed": 44},
    ]
    
    parallel_results = run_parallel_simulations(
        data, configs, n_workers=3, progress_bar=False
    )
    
    print(f"\n  Results by horizon:")
    for i, result in enumerate(parallel_results):
        days = configs[i]["days_ahead"]
        print(f"    {days}-day: {result['expected_return']:.2%} return")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ All simulations completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run the Jupyter notebook: notebooks/demo_nvda.ipynb")
    print("  - Try ML models: from src.ml.models import train_forecasting_model")
    print("  - Visualize results: from src.visualization.plots import plot_monte_carlo_results")


if __name__ == "__main__":
    main()

