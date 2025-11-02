# finmc-tech

**Machine Learning + Monte Carlo Simulation for Tech Stock Forecasting with HPC parallelization and uncertainty quantification**

`finmc-tech` is a minimal prototype for real-world stock forecasting that combines **machine learning signals** with **Monte Carlo uncertainty modeling** and **HPC-ready parallel execution**.

The initial demo focuses on **NVIDIA (NVDA)** using real daily data from the Yahoo Finance API to compute returns, volatility, and rolling μ–σ parameters as inputs for simulation.

This project serves as the foundation for scaling to multi-asset (Magnificent 7) analysis, integrating predictive modeling, uncertainty quantification, and performance benchmarking.

## Results

### Comprehensive Rolling Forecast Analysis (2018-2025)

**Methodology**: Year-by-year evaluation using **Linear Regression** with 30-day rolling windows across three training setups, compared against **Monte Carlo** GBM simulations.

![Year-by-Year Forecast Comparison](docs/images/comparison_yearly.png)

**Key Findings**:
- **Average Sign Accuracy**: ~50% (barely better than random)
- **Average R²**: ~-0.03 (worse than naive baseline)
- **MC Prediction Errors**: -100% to +48% (severely underestimates growth)
- **Best Year**: 2020 (59.6% sign accuracy, R² = 0.035)

**Monte Carlo vs Actual Prices** (Year-End Predictions):

| Year | Actual Close ($) | Predicted (P50) | ±90% Band | % Error | Sign Pred | Actual Dir | Coverage |
|------|------------------|-----------------|-----------|---------|-----------|------------|----------|
| 2018 | $3.31 | $4.92 | [4.7, 5.1] | +48.6% | ↓ | ↓ | ❌ |
| 2019 | $5.86 | $3.33 | [3.0, 3.6] | -43.2% | ↓ | ↑ | ❌ |
| 2020 | $13.02 | $6.00 | [5.8, 6.2] | -53.9% | ↑ | ↑ | ✅ |
| 2021 | $29.36 | $13.06 | [12.8, 13.3] | -55.5% | ↓ | ↑ | ❌ |
| 2022 | $14.60 | $30.00 | [28.2, 31.9] | +105.5% | ↓ | ↓ | ❌ |
| 2023 | $49.50 | $14.27 | [13.5, 15.1] | -71.2% | ↓ | ↑ | ❌ |
| 2024 | $134.26 | $48.08 | [46.7, 49.5] | -64.2% | ↓ | ↑ | ❌ |
| 2025 | $202.49 | $138.14 | [133.2, 143.5] | -31.8% | ↓ | ↑ | ❌ |

**Summary**: Average |% Error| = **59.2%**, Coverage Rate = **0%** (0/8 years), Direction Accuracy = **25%** (2/8).

**Critical Insight**: MC consistently **underestimates NVDA's exponential growth**; 90% confidence intervals fail to cover actual prices—**GBM fails to capture structural breaks** in tech stock evolution. Models predict bearish trends while actual returns are strongly positive.

**Three Training Window Setups**:
1. **Expanding-Long**: All historical data before test year (maximum context)
2. **Sliding-5y**: 5-year rolling window (balanced recent vs historical)
3. **Sliding-3y**: 3-year rolling window (most recent trends)

**Model Comparison**:
- **ML (Linear Regression)**: Predicts next-day log returns from 30-day window
  - Feature: Rolling 30-day log returns
  - Output: 1-day ahead return prediction
  - Metrics: R², MAE, Sign Accuracy, IC (information coefficient)

- **Monte Carlo (GBM)**: Simulates 1-year price distribution
  - Input: 30-day rolling μ/σ parameters from each year's first day
  - Output: P5/P50/P95 quantiles, VaR, CVaR, bandwidth
  - GBM equation: `S(t+dt) = S(t) × exp((μ-0.5σ²)dt + σ√dt×z)`

**Output Files**:
- `outputs/results_forecast.csv`: 24 year×setup combinations with ML metrics
- `outputs/results_mc.csv`: 8 years of MC simulations with risk metrics
- `outputs/results_all.csv`: Merged comprehensive results
- `outputs/comparison_yearly.png`: 6-panel visualization (see above)

**Conclusion**: Simple linear models fail to capture NVDA's growth dynamics; MC simulations consistently underestimate price appreciation, highlighting the need for advanced ML features and regime-aware models.

## Features

- **Data Pipeline**: Yahoo Finance API integration for historical stock data
- **Monte Carlo Simulation**: Probability-based forecasting with configurable parameters
- **Uncertainty Quantification**: Statistical analysis of returns, volatility, and confidence intervals
- **HPC-Ready**: Parallel execution using multiprocessing for scalable performance
- **ML Integration**: Feature engineering and predictive modeling pipeline
- **Visualization**: Results analysis and forecasting plots

# Appendix

## Technology Stack

**Language**: Python 3.9+  
**Core Libraries**: NumPy, Pandas, SciPy  
**Data Source**: yfinance (Yahoo Finance API)  
**Machine Learning**: scikit-learn (RF, GBM, Ridge, Lasso)  
**Simulation**: Monte Carlo (GBM), Uncertainty Quantification  
**Parallelization**: joblib, multiprocessing, numba  
**Visualization**: Matplotlib, Seaborn  
**Testing**: pytest, black, flake8, Jupyter

## Project Structure

```
finmc-tech/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── fetch.py           # Yahoo Finance data fetching
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── monte_carlo.py     # Monte Carlo core
│   │   └── uncertainty.py     # Uncertainty quantification
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py        # Feature engineering
│   │   └── models.py          # ML models
│   ├── parallel/
│   │   ├── __init__.py
│   │   └── executor.py        # HPC parallel execution
│   └── visualization/
│       ├── __init__.py
│       └── plots.py           # Results visualization
├── examples/
│   └── quick_start.py         # Quick start example
├── notebooks/
│   └── demo_nvda.ipynb        # Demo notebook
└── tests/
    ├── __init__.py
    ├── test_data.py
    └── test_simulation.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yixuan116/finmc-tech.git
cd finmc-tech

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from src.data.fetch import fetch_stock_data
from src.simulation.monte_carlo import MonteCarloForecast

# Fetch NVIDIA data (default: 2010-2025)
data = fetch_stock_data("NVDA")

# Run Monte Carlo simulation
forecast = MonteCarloForecast(n_simulations=10000, days_ahead=30)
results = forecast.run(data)

# Analyze results
print(f"Expected Return: {results['expected_return']:.2%}")
print(f"95% Confidence Interval: [${results['ci_lower']:.2f}, ${results['ci_upper']:.2f}]")
```

Or run the rolling forecast demo:

```bash
# Run comprehensive rolling forecast analysis (2018-2025)
python examples/demo_rolling_forecast.py

# Quick start example
python examples/quick_start.py
```

## Usage Examples

### Data Fetching

```python
from src.data.fetch import fetch_stock_data

# Fetch NVDA data (default: 2010-2025)
data = fetch_stock_data("NVDA")

# Custom date range
data = fetch_stock_data("AAPL", start="2020-01-01", end="2023-12-31")

# Use period parameter
data = fetch_stock_data("MSFT", period="max")

# Get additional tickers
magnificent_7 = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
```

**Note**: This project uses `yfinance` library, which fetches free data from Yahoo Finance. No API key required!

### Basic Monte Carlo Simulation

```python
from src.simulation.monte_carlo import MonteCarloForecast

forecast = MonteCarloForecast(
    n_simulations=10000,
    days_ahead=30,
    confidence_level=0.95
)
results = forecast.run(data)
```

### Parallel Execution

```python
from src.parallel.executor import run_parallel_simulations

# Run multiple simulations in parallel
configs = [
    {"n_simulations": 10000, "days_ahead": 30},
    {"n_simulations": 10000, "days_ahead": 60},
    {"n_simulations": 10000, "days_ahead": 90},
]

results = run_parallel_simulations(data, configs, n_workers=4)
```

### ML-Enhanced Forecasting

```python
from src.ml.models import train_forecasting_model
from src.ml.features import engineer_features

# Engineer features
features = engineer_features(data)

# Train model
model = train_forecasting_model(features)

# Make predictions
predictions = model.predict(features[-30:])
```

## Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `yfinance` - Yahoo Finance API
- `scipy` - Statistical functions
- `scikit-learn` - Machine learning
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `joblib` - Parallel processing

## License

MIT License - see LICENSE file for details

## Contributing

This is a research prototype. Contributions, suggestions, and feedback are welcome!

## Roadmap

- [ ] Multi-asset support (Magnificent 7)
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Real-time data streaming
- [ ] Risk metrics integration
- [ ] Performance benchmarking suite


## Running the Jupyter Notebook

The main demo is in `notebooks/demo_nvda.ipynb`.

### Quick Start

```bash
jupyter notebook notebooks/demo_nvda.ipynb
```

Then click **"Run All"** to execute all cells.

### What It Does

The notebook builds a complete **ML → Monte Carlo** pipeline in 5 steps:

1. **Data Loading**: Loads NVDA stock data, computes rolling statistics
2. **ML Baseline**: Predicts next-day returns using Linear Regression
3. **Monte Carlo**: Runs 5,000 simulated price paths (Serial + Numba backends)
4. **Visualizations**: Shows path samples and price distributions
5. **Sanity Checks**: Prints summary statistics

### Understanding Results

**ML Predictions**:
- R² Score: prediction accuracy (0-1, higher is better)
- Typical range: 0.01-0.05 for noisy returns

**Monte Carlo Results**:
- Terminal prices show where NVDA might be in 1 year
- P5/P50/P95: downside/expected/upside scenarios
- Speedup: Numba vs serial performance comparison

**Output Files** (saved in `outputs/`):
- `nvda_ml_pred.csv` - ML predictions (date, actual, predicted)
- `nvda_mc_terminals.csv` - MC terminal prices
- `nvda_mc_meta.json` - Simulation metadata


