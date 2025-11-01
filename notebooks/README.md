# How to Run the Demo

## Quick Start

```bash
# Make sure you're in the project root
cd /path/to/finmc-tech

# Start Jupyter
jupyter notebook notebooks/demo_nvda.ipynb
```

This will open the notebook in your browser. Then click "Run All" to execute all cells.

## What the Notebook Does

The notebook builds a complete **ML → Monte Carlo** pipeline in 5 steps:

### Step 1: Data Loading
- Loads NVDA stock data from CSV (2010-2025)
- Computes 30-day rolling mean (μ) and std (σ) of log returns
- Prepares clean dataset with essential columns

**Output**: Clean dataframe with date, price, returns, and rolling statistics

### Step 2: ML Baseline
- Predicts next-day log returns using 30-day window of past returns
- Uses Linear Regression (simple but effective)
- Trains on 80% of data, tests on 20%
- Shows actual vs predicted plot
- Saves predictions to `outputs/nvda_ml_pred.csv`

**Output**: R² score and predictions CSV file

### Step 3: Monte Carlo Simulation
Runs TWO implementations for comparison:

**Serial Backend** (`simulate_paths_serial`):
- Plain Python/NumPy loop
- Baseline for comparison

**Numba HPC Backend** (`simulate_paths_numba`):
- GPU-ready parallel execution using `@njit(parallel=True)`
- Uses `prange` for multi-threading
- Same signature, just faster

**What it simulates**:
- 252 trading days (1 year)
- 5,000 random price paths
- Uses latest rolling μ and σ from the data
- Geometric Brownian Motion: `S_new = S_old * exp((μ-½σ²)dt + σ√dt*Z)`

**Output**: Speed comparison (typically 3-10x faster with Numba)

### Step 4: Visualizations
Two plots:
1. **Sample Paths**: Random 50 paths showing price evolution
2. **Terminal Distribution**: Histogram of final prices with P5/P50/P95

Also saves:
- `outputs/nvda_mc_terminals.csv` - Final prices for all 5,000 paths
- `outputs/nvda_mc_meta.json` - Simulation parameters and runtime

### Step 5: Sanity Checks
Prints summary statistics to verify everything works.

## Understanding the Results

### ML Predictions
- **R² Score**: How well the model predicts (0-1, higher is better)
- Typical range: 0.01-0.05 (returns are noisy!)
- The plot shows actual vs predicted returns

### Monte Carlo Results
- **Terminal Prices**: Where NVDA might be in 1 year
- **P5/P50/P95**: 
  - P5 = 5th percentile (conservative downside)
  - P50 = Median (expected scenario)
  - P95 = 95th percentile (optimistic upside)
- **Speedup**: How much faster Numba is vs serial

### Example Interpretation
If you see:
```
P50 = $250.00
P5  = $180.00
P95 = $350.00
```

This means:
- There's a 50% chance NVDA is above $250 in 1 year
- There's a 95% chance it's above $180
- There's only a 5% chance it reaches $350+

## Next Steps

After the notebook runs successfully, you can:
1. Scale to multiple stocks (Magnificent 7)
2. Add more ML models (LSTM, Transformer)
3. Implement true HPC backends (CUDA, OpenMP)
4. Add real-time streaming
5. Build interactive dashboards

## Troubleshooting

**Import errors**: Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

**Numba warnings**: Normal on first run (JIT compilation)

**CSV not found**: Run `python examples/export_data.py` first

