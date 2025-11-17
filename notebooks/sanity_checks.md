# Sanity Checks for Macro-Driven NVDA Simulation

## Data Quality Checks

### 1. Macro Data
- [ ] VIX data is non-negative
- [ ] TNX yield is between 0% and 20% (reasonable range)
- [ ] SP500 returns are reasonable (not >100% in single period)
- [ ] No missing values in critical periods
- [ ] Date alignment is correct (all features on same date)

### 2. Firm Data
- [ ] Revenue data is positive
- [ ] Price data is aligned to quarterly frequency
- [ ] Forward returns are calculated correctly (12 months ahead)
- [ ] No data leakage (future data not used in past predictions)

### 3. Feature Engineering
- [ ] All features are on same time scale (quarterly)
- [ ] No infinite or NaN values in features
- [ ] Feature distributions are reasonable
- [ ] Correlation between features is not too high (>0.95)

## Model Validation

### 1. Random Forest
- [ ] Training R² > 0 (model learns something)
- [ ] Test R² is not too negative (< -5 suggests major issues)
- [ ] Feature importance makes sense (macro features should be important)
- [ ] Predictions are in reasonable range

### 2. LSTM (if used)
- [ ] Sequence length is appropriate (4 quarters = 1 year)
- [ ] Training loss decreases over epochs
- [ ] Validation loss doesn't diverge from training loss (no overfitting)
- [ ] Predictions are in reasonable range

## Simulation Checks

### 1. Macro Scenarios
- [ ] Scenario probabilities sum to 1.0
- [ ] Scenario parameters are reasonable (VIX, TNX, SP500)
- [ ] Baseline scenario matches historical averages

### 2. Monte Carlo
- [ ] Number of simulations is sufficient (>=10,000)
- [ ] Results are stable across runs (with same seed)
- [ ] Percentiles are ordered correctly (5th < 25th < 50th < 75th < 95th)
- [ ] Mean prediction is reasonable

## Visualization Checks

### 1. Plots
- [ ] All plots are generated without errors
- [ ] Axes labels are clear and correct
- [ ] Legends are present and accurate
- [ ] Plots are saved to correct location

## Performance Checks

### 1. Runtime
- [ ] Data fetching completes in reasonable time (<5 min)
- [ ] Model training completes in reasonable time (<10 min)
- [ ] Simulation completes in reasonable time (<5 min)

### 2. Memory
- [ ] No memory errors during execution
- [ ] Data fits in memory (check DataFrame sizes)

## Reproducibility

### 1. Random Seeds
- [ ] Random seed is set for all random operations
- [ ] Results are reproducible across runs

### 2. Caching
- [ ] Data is cached correctly
- [ ] Cache is used when available
- [ ] Cache invalidation works correctly

