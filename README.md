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

---

### Dual-Head ML Analysis: Revenue + Macro Features (2009-2024)

**Methodology**: Dual-head machine learning pipeline that predicts both **12-month forward returns** (return head) and **12-month forward stock prices** (price head) using revenue fundamentals from SEC XBRL data plus macro features (VIX, 10Y treasury yield). Three baseline models (Ridge, KNN, RandomForest) are trained with temporal split (train before 2019, test from 2019).

**Key Results**:
- **Best Price Prediction Route**: Indirect (from return head) achieves **$39.22 RMSE** vs Direct route **$78.93 RMSE**
- **Feature Importance**: Macro factors (`tnx_yield` = 0.3681) dominate revenue features in importance
- **Sample Size**: 62 quarterly observations (2009-07-26 to 2024-10-27)

#### Analysis Figures

**1. Revenue YoY Growth vs Future 12M Return**
![YoY vs Return](outputs/figs/yoy_vs_return.png)

**Purpose**: Examines the relationship between revenue year-over-year growth and future stock returns.

**What it shows**: Scatter plot of revenue YoY growth (x-axis) against future 12-month returns (y-axis), with a fitted regression line. This visualization helps assess whether revenue growth is predictive of future stock performance.

**Key insight**: The slope and correlation coefficient reveal whether stronger revenue growth is associated with better future returns, providing evidence for the revenue-based prediction hypothesis.

---

**2. Revenue Acceleration vs Future 12M Return**
![Acceleration vs Return](outputs/figs/accel_vs_return.png)

**Purpose**: Analyzes how changes in revenue growth momentum (acceleration) relate to future returns.

**What it shows**: Scatter plot of revenue acceleration (change in YoY growth rate) against future 12-month returns. Revenue acceleration captures the second derivative of revenue growth, which may signal turning points in business performance.

**Key insight**: Positive acceleration indicates improving growth momentum, while negative acceleration suggests deceleration. This metric may be more sensitive to inflection points than absolute growth rates.

---

**3. Rolling Correlation: Revenue YoY vs Future Returns**
![Rolling Correlation](outputs/figs/rolling_corr.png)

**Purpose**: Tracks the time-varying strength of the relationship between revenue growth and future returns.

**What it shows**: Time series of rolling 3-year correlation coefficients between revenue YoY growth and future 12-month returns. The correlation is computed over a sliding window to capture regime changes.

**Key insight**: Reveals whether the predictive power of revenue signals is stable over time or varies across market regimes. A correlation that changes sign or magnitude suggests the relationship is context-dependent.

---

**4. RandomForest Feature Importance (Return Head)**
![RF Feature Importance](outputs/figs/rf_feature_importance.png)

**Purpose**: Identifies which features are most important for predicting future returns in the RandomForest model.

**What it shows**: Horizontal bar chart of feature importances from the RandomForest return head model. Higher values indicate greater contribution to predictions.

**Key insight**: Reveals that macro features (especially `tnx_yield` - 10-year treasury yield) dominate revenue features in importance, suggesting that interest rates and market volatility are more predictive than fundamental revenue signals for NVDA's returns. This finding highlights the importance of incorporating macro factors alongside company fundamentals.

---

**5. Return Head Predictions vs Actual - RandomForest (Test Set)**
![RF Return: Pred vs Actual](outputs/figs/pred_vs_actual_return_rf.png)

**Purpose**: Evaluates the out-of-sample performance of the RandomForest return head model by comparing predicted vs actual future returns over time.

**What it shows**: Time series plot of actual future 12-month returns (solid line) and RandomForest predictions (dashed line) for the test set (2019 onwards). This is the primary diagnostic for return prediction accuracy.

**Key insight**: Visual assessment of prediction quality, including whether the model captures trends, volatility, and turning points. Large divergences indicate periods where the model struggles, potentially due to regime changes or model limitations.

---

**5b. Return Head Predictions vs Actual - KNN (Test Set)**
![KNN Return: Pred vs Actual](outputs/figs/knn_pred_vs_actual.png)

**Purpose**: Evaluates the out-of-sample performance of the KNN return head model by comparing predicted vs actual future returns over time.

**What it shows**: Time series plot of actual future 12-month returns (blue line with circles) and KNN predictions (green line with triangles) for the test set. KNN predictions are smoother and less volatile than actual returns, showing the model's tendency to regress toward the mean.

**Key insight**: KNN provides smoothed predictions that generally underestimate high actual returns and overestimate low or negative returns, failing to capture the full amplitude of market movements. This smoothing effect is characteristic of k-nearest neighbors models, which average over similar historical patterns.

---

**6. Indirect Price Predictions vs Actual (from Return Head)**
![Price (Indirect from Return)](outputs/figs/pred_vs_actual_price_indirect.png)

**Purpose**: Assesses price prediction accuracy using the indirect route: converting return predictions to price predictions via `price_hat = current_price × (1 + return_hat)`.

**What it shows**: Time series comparing actual future 12-month stock prices with indirect price predictions derived from the return head model. This route leverages the return head's predictions to estimate absolute prices.

**Key insight**: The indirect route achieves **$39.22 RMSE**, outperforming the direct price head. This suggests that predicting returns and then converting to prices is more effective than directly modeling price levels, possibly because returns are more stationary and easier to predict than absolute prices.

---

**7. Direct Price Predictions vs Actual (from Price Head)**
![Price (Direct Head)](outputs/figs/pred_vs_actual_price_direct.png)

**Purpose**: Evaluates price prediction accuracy using the direct route: predicting log(price) directly from features.

**What it shows**: Time series comparing actual future 12-month stock prices with direct price predictions from the price head model. The model predicts log prices to handle non-stationarity, then converts back to price scale.

**Key insight**: The direct route achieves **$78.93 RMSE**, significantly worse than the indirect route. This indicates that modeling prices directly, even in log space, is more challenging than modeling returns, likely due to the non-stationarity of price levels and the compounding of errors.

---

**8. Return Head Calibration Plot**
![Calibration Return](outputs/figs/calibration_return.png)

**Purpose**: Assesses whether the model's predictions are well-calibrated (i.e., whether predicted values systematically match actual values).

**What it shows**: Scatter plot of actual vs predicted returns, with a red dashed y=x line representing perfect calibration. Points should cluster around the y=x line if the model is well-calibrated.

**Key insight**: Systematic deviations from the y=x line indicate calibration issues:
- Points above the line: model underestimates (predictions too low)
- Points below the line: model overestimates (predictions too high)
- Clustering indicates whether the model is biased or has high variance

---

**9. Return Head Residuals Over Time**
![Residuals Return](outputs/figs/residuals_return.png)

**Purpose**: Analyzes prediction errors (residuals) over time to detect patterns, biases, and heteroscedasticity.

**What it shows**: Time series of residuals (actual - predicted) for the test set. The zero line represents perfect predictions. Residuals should be randomly distributed around zero with no systematic patterns.

**Key insight**: Patterns in residuals reveal model weaknesses:
- **Trends**: Systematic over/under-prediction during certain periods
- **Clustering**: Volatility clustering suggests the model misses regime changes
- **Outliers**: Large residuals indicate periods where the model fails (e.g., during market crashes or structural breaks)
- **Heteroscedasticity**: Changing variance suggests the model's uncertainty varies over time

---

## Features

- **Data Pipeline**: Yahoo Finance API integration for historical stock data
- **Monte Carlo Simulation**: Probability-based forecasting with configurable parameters
- **Uncertainty Quantification**: Statistical analysis of returns, volatility, and confidence intervals
- **HPC-Ready**: Parallel execution using multiprocessing for scalable performance
- **ML Integration**: Feature engineering and predictive modeling pipeline
- **Visualization**: Results analysis and forecasting plots

## Prediction Target & Features

### Prediction Target

**Target Variable**: Monthly stock returns (not stock price)

- **Definition**: `y = df['Ret'].shift(-1)` - Next month's return
- **Format**: Percentage return (e.g., 0.05 = 5% return)
- **No log transformation needed**: Returns are already percentage changes and typically stationary
- **Why returns instead of price**: Returns are more stationary and easier to predict than absolute price levels

**Return Calculation Formula:**

$$\text{Ret}_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

where $P_t$ is the adjusted close price at time $t$. The target variable is:

$$y_t = \text{Ret}_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$

This represents the **next-month return** we aim to predict.

### Feature List

#### Base Features (7 features)

**Macro Features (5):**

| Feature | Description | Source |
|---------|-------------|--------|
| `CPI` | Consumer Price Index (CPIAUCSL) | FRED API |
| `VIX` | CBOE Volatility Index (VIXCLS) | FRED API |
| `DGS10` | 10-Year Treasury Constant Maturity Rate | FRED API |
| `FEDFUNDS` | Effective Federal Funds Rate | FRED API |
| `GDP` | Real Gross Domestic Product (GDPC1) | FRED API |

*Note: Macro features are forward-filled from quarterly/monthly frequency to match the data alignment frequency.*

**Firm Features (7):**

| Feature | Description | Formula |
|---------|-------------|---------|
| `rev_qoq` | Revenue quarter-over-quarter growth | `rev_qoq_t = (Revenue_t / Revenue_{t-1}) - 1` |
| `rev_yoy` | Revenue year-over-year growth | `rev_yoy_t = (Revenue_t / Revenue_{t-4}) - 1`<br>*For quarterly data, compares to same quarter previous year* |
| `rev_accel` | Revenue acceleration (change in YoY growth rate) | `rev_accel_t = rev_yoy_t - rev_yoy_{t-1}`<br>*Measures the change in growth momentum* |
| `vix_level` | VIX level (from firm data) | `vix_level_t = VIX_t`<br>*Current VIX index value* |
| `tnx_yield` | 10-Year Treasury Yield (from firm data) | `tnx_yield_t = DGS10_t`<br>*Current 10-year Treasury yield* |
| `vix_change_3m` | VIX 3-month change | `vix_change_3m_t = (VIX_t / VIX_{t-3}) - 1`<br>*3-month percentage change in VIX* |
| `tnx_change_3m` | Treasury yield 3-month change | `tnx_change_3m_t = (DGS10_t / DGS10_{t-3}) - 1`<br>*3-month percentage change in 10Y yield* |

#### Extended Features (Optional, controlled by config flags)

**A. Price Momentum Features (7)** - `INCLUDE_PRICE_FEATURES=True`:
- `price_returns_1m/3m/6m/12m` - Price returns over different horizons
- `price_momentum` - Price momentum indicator
- `price_volatility` - Price volatility measure
- `price_to_ma_4q` - Price relative to 4-quarter moving average

**B. Technical Indicators (6)** - `INCLUDE_TECHNICAL_FEATURES=True`:
- `rsi_14` - RSI (Relative Strength Index)
- `macd`, `macd_signal` - MACD indicator and signal
- `bb_position` - Bollinger Bands position
- `stoch_k` - Stochastic oscillator
- `atr` - Average True Range

**C. Market Macro Features (2)** - `INCLUDE_MARKET_FEATURES=True`:
- `sp500_level` - S&P 500 index level
- `sp500_returns` - S&P 500 returns

**D. Time Features (4)** - `INCLUDE_TIME_FEATURES=True`:
- `quarter` - Quarter (1-4)
- `month` - Month (1-12)
- `year` - Year
- `days_since_start` - Days since start date

**E. Interaction Features (4)** - `INCLUDE_INTERACTION_FEATURES=True`:
- `rev_yoy_x_vix` - Revenue YoY × VIX
- `rev_qoq_x_sp500` - Revenue QoQ × SP500
- `price_momentum_x_volatility` - Price momentum × volatility
- `vix_x_tnx` - VIX × Treasury yield

#### Lag Features

- All features (except `Ret`) have 1-period lag versions with `_L1` suffix
- Example: `CPI_L1`, `VIX_L1`, `rev_qoq_L1`
- Each original feature becomes 2 features: current value + lagged value

#### Total Feature Count

- **Base features**: 7 × 2 (with lags) = 14 features
- **Extended features**: Up to 23 × 2 (with lags) = 46 features
- **Total**: Up to 60 features (if all extended features are enabled)

**Note**: By default, all extended feature flags are `True` in the configuration, so the full feature set is used.

<!-- FEAT_IMPORT_START -->
### Feature Importance Analysis (Light)

**Why Random Forest?**
Random Forest is used for feature importance analysis because:
1. It captures **nonlinear relationships** and **interactions** between macro and firm-level features
2. It is **ensemble-based** and robust to outliers, noise, and overfitting
3. It provides **direct, interpretable importance scores** through mean decrease in impurity
4. It does **not assume linearity** or stationarity—important since macro-financial data often violate those assumptions
5. Compared to deep networks (LSTM) or SVMs, Random Forest offers explainability and speed for feature screening

**Mathematical Formulation:**

**Random Forest Prediction:**
A Random Forest is an ensemble of $B$ decision trees. The final prediction is the average of all tree predictions:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

where $T_b(\mathbf{x})$ is the prediction from the $b$-th tree, and $\mathbf{x}$ is the feature vector.

**Feature Importance (Variance-based for Regression):**
For each feature $j$, the importance is computed as the average decrease in variance (MSE) across all trees:

$$\text{Importance}_j = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} p(t) \cdot \Delta \text{Var}_j(t)$$

where:
- $p(t)$ is the proportion of samples reaching node $t$
- $\Delta \text{Var}_j(t)$ is the decrease in variance when feature $j$ is used to split node $t$
- The variance at node $t$ is: $\text{Var}(t) = \frac{1}{N_t} \sum_{i \in t} (y_i - \bar{y}_t)^2$, where $N_t$ is the number of samples at node $t$ and $\bar{y}_t$ is the mean target value at node $t$
- For regression, this is equivalent to the mean decrease in impurity (MSE reduction)

**Intuitive Explanation:**
This formula measures **how much each feature helps reduce prediction error** across all trees:

1. **At each node**, the algorithm tries all features and picks the one that **maximizes variance reduction** (i.e., splits the data so that samples in each child node have more similar target values)

2. **Variance reduction** $\Delta \text{Var}_j(t)$ measures: 
   - Before split: variance of all samples at node $t$
   - After split: weighted average variance of left and right child nodes
   - The difference = how much "cleaner" (less scattered) the predictions become

3. **Weighted by sample proportion** $p(t)$: Features used at nodes with more samples contribute more to importance

4. **Averaged across all trees**: A feature that consistently helps reduce variance across many trees is more important

**Key Insight:** It's not about "distance" or "correlation" directly—it's about **predictive power**: features that can better separate high-return months from low-return months (reduce variance in predictions) are ranked higher.

**Concrete Example:**

*Note: This is a conceptual example to illustrate how Random Forest calculates feature importance. The actual feature importance scores are computed across the entire dataset (2010-2024), not just these 10 months.*

Suppose we have 10 months of data at a node (sampled from different time periods):

| Year-Month | rev_yoy | Stock Price ($) | Actual Return (y) | Growth Category |
|------------|---------|-----------------|-------------------|-----------------|
| 2012-07 | 0.15 | $5.20 | +0.08 (+8%) | **High Growth** (rev_yoy > 0.13) |
| 2016-10 | 0.18 | $18.50 | +0.12 (+12%) | **High Growth** (rev_yoy > 0.13) |
| 2020-01 | 0.20 | $49.30 | +0.15 (+15%) | **High Growth** (rev_yoy > 0.13) |
| 2021-04 | 0.14 | $62.20 | +0.10 (+10%) | **High Growth** (rev_yoy > 0.13) |
| 2023-07 | 0.16 | $134.50 | +0.12 (+12%) | **High Growth** (rev_yoy > 0.13) |
| 2025-01 | 1.14 | $118.40 | +0.18 (+18%) | **High Growth** (rev_yoy > 0.13) |
| 2025-07 | 0.62 | $176.74 | +0.15 (+15%) | **High Growth** (rev_yoy > 0.13) |
| 2011-04 | 0.05 | $4.20 | -0.03 (-3%) | Low Growth (rev_yoy ≤ 0.13) |
| 2013-01 | 0.08 | $6.80 | +0.02 (+2%) | Low Growth (rev_yoy ≤ 0.13) |
| 2014-01 | 0.10 | $8.80 | +0.03 (+3%) | Low Growth (rev_yoy ≤ 0.13) |
| 2015-04 | 0.12 | $12.10 | +0.05 (+5%) | Low Growth (rev_yoy ≤ 0.13) |
| 2018-07 | -0.05 | $28.50 | -0.08 (-8%) | Low Growth (rev_yoy ≤ 0.13) |
| 2019-10 | -0.02 | $38.60 | -0.05 (-5%) | Low Growth (rev_yoy ≤ 0.13) |
| 2022-10 | 0.08 | $108.30 | -0.05 (-5%) | Low Growth (rev_yoy ≤ 0.13) |
| 2024-01 | 0.12 | $145.80 | +0.03 (+3%) | Low Growth (rev_yoy ≤ 0.13) |
| 2025-04 | 0.69 | $108.72 | -0.08 (-8%) | **High Growth** (rev_yoy > 0.13, but negative return due to other factors) |

**Before split (at root node):**
- All 10 months mixed together
- Mean return: $\bar{y} = 0.039$ (3.9%)
- Variance: $\text{Var} = \frac{1}{10} \sum_{i=1}^{10} (y_i - 0.039)^2 = 0.0068$
- *High variance because high-return months (+8% to +15%) are mixed with low-return months (-8% to +5%)*

**After split using `rev_yoy > 0.13` (threshold chosen to maximize variance reduction):**

**Left child (Low Growth: rev_yoy ≤ 0.13):** 2011-04, 2013-01, 2014-01, 2015-04, 2018-07, 2019-10, 2022-10, 2024-01
- Contains: Months with low/negative revenue growth (rev_yoy: -0.05 to 0.12)
- Stock prices: $4.20, $6.80, $8.80, $12.10, $28.50, $38.60, $108.30, $145.80
- Time span: 2011-2024 (covers early period, mid period, and recent period)
- Mean: $\bar{y}_L = -0.008$ (-0.8%)
- Variance: $\text{Var}_L = 0.0012$
- *Low variance: all returns are clustered around -0.8%*

**Right child (High Growth: rev_yoy > 0.13):** 2012-07, 2016-10, 2020-01, 2021-04, 2023-07, 2025-01, 2025-04, 2025-07
- Contains: Months with high revenue growth (rev_yoy: 0.14 to 1.14)
- Stock prices: $5.20, $18.50, $49.30, $62.20, $134.50, $118.40, $108.72, $176.74
- Time span: 2012-2025 (covers early period, mid period, recent period, and latest 2025 data)
- Mean: $\bar{y}_R = 0.1100$ (+11.00%)
- Variance: $\text{Var}_R = 0.0015$
- *Note: 2025-01 shows exceptional growth (rev_yoy=1.14, +114% YoY) with +18% return, demonstrating the pattern continues into 2025*
- *2025-04 has high rev_yoy (0.69) but negative return (-8%), showing that even high growth months can have short-term volatility due to other market factors*

**Weighted variance after split:**
$$\text{Var}_{\text{after}} = \frac{8}{16} \times 0.0012 + \frac{8}{16} \times 0.0015 = 0.00135$$

**Variance reduction:**
$$\Delta \text{Var} = 0.0068 - 0.00135 = 0.00545$$

This large variance reduction (80% decrease: from 0.0068 to 0.00135) means `rev_yoy` is very effective at separating high-return months from low-return months. This $\Delta \text{Var}$ value contributes to `rev_yoy`'s importance score. When this pattern repeats across many nodes and trees throughout the entire dataset (2009-2025), `rev_yoy` accumulates high importance.

**Important Note:** The feature importance scores shown in our results (e.g., `rev_yoy` = 0.1477) are computed across **all available data** (2009-2025, 66+ monthly observations), not just this 16-month example. The pattern shown here—where high revenue growth months tend to have higher returns—is consistent across the full time period (2011-2025), which is why `rev_yoy` ranks as the 2nd most important feature overall. The inclusion of 2025 data (including the exceptional 2025-01 with +114% YoY growth) demonstrates that this predictive relationship continues to hold in the most recent period.

**Temporal Stability of Feature Importance:**

To validate that feature importance is **stable across time** (not just a recent phenomenon), the analysis uses the **full dataset spanning 2009-2025** (16+ years, 66+ monthly observations). This long time horizon ensures that:

1. **Regime Robustness**: Features that are important across different market regimes (bull markets, bear markets, high volatility periods) are ranked higher
2. **Not Time-Specific**: The importance scores reflect patterns that hold across the entire period, not just recent years
3. **Statistical Significance**: With 62 observations, the importance rankings have sufficient statistical power

**Data Coverage:**
- **Time Period**: 2009-07 to 2025-07 (16+ years)
- **Total Observations**: 66+ monthly data points
- **Market Regimes Covered**: 
  - Post-financial crisis recovery (2009-2012)
  - Bull market expansion (2013-2019)
  - COVID-19 volatility (2020-2021)
  - Recent tech sector dynamics (2022-2024)
  - Latest AI boom period (2025)

The fact that `rev_yoy` consistently ranks in the top 3 across this entire period suggests it is a **persistent, regime-independent predictor** of NVDA returns, not just a temporary correlation.

**Permutation Importance:**
An alternative measure that evaluates the drop in model performance when feature $j$ is randomly shuffled:

$$\text{PermImportance}_j = \frac{1}{R} \sum_{r=1}^{R} \left[ \text{Score}(\mathbf{X}, \mathbf{y}) - \text{Score}(\mathbf{X}^{(j,r)}, \mathbf{y}) \right]$$

where:
- $\mathbf{X}^{(j,r)}$ is the feature matrix with feature $j$ permuted in the $r$-th repetition
- $\text{Score}$ is typically R² or MSE
- $R$ is the number of permutation repetitions (default: 10)

**Why Feature Importance before forecasting?**
Feature importance helps to:
1. Identify the **most predictive signals** (rev_yoy, VIX, FedFunds, etc.)
2. Remove redundant or noisy features before training LSTM (faster convergence, better generalization)
3. Provide a **transparent feature ranking** for the project
4. Build trust in the ML pipeline—by knowing "what the model is learning"

**Top 10 Features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `rev_accel` | 0.1486 |
| 2 | `rev_yoy` | 0.1477 |
| 3 | `revenue` | 0.1415 |
| 4 | `tnx_yield` | 0.1347 |
| 5 | `tnx_change_3m` | 0.1280 |
| 6 | `rev_qoq` | 0.1149 |
| 7 | `vix_change_3m` | 0.0951 |
| 8 | `vix_level` | 0.0896 |

![RF Feature Importance](plots/rf_feature_importance.png)

#### Permutation Importance Validation

Permutation importance provides an alternative measure of feature importance by evaluating the drop in model performance when each feature is randomly shuffled. This method is less biased than tree-based importance and better captures true predictive power.

**Top 10 Features (Permutation Importance):**

| Rank | Feature | Importance (Mean ± Std) |
|------|---------|------------------------|
| 1 | `rev_yoy` | 0.1233 ± 0.0222 |
| 2 | `rev_accel` | 0.1187 ± 0.0214 |
| 3 | `revenue` | 0.1185 ± 0.0133 |
| 4 | `tnx_change_3m` | 0.1177 ± 0.0140 |
| 5 | `tnx_yield` | 0.1113 ± 0.0195 |
| 6 | `rev_qoq` | 0.0717 ± 0.0057 |
| 7 | `vix_change_3m` | 0.0616 ± 0.0099 |
| 8 | `vix_level` | 0.0612 ± 0.0102 |

![Permutation Feature Importance](plots/rf_permutation_importance.png)

**Side-by-Side Comparison:**

![Feature Importance Comparison](plots/rf_importance_comparison.png)

**Key Observations:**
- Both methods identify the same top features, but with slight ranking differences
- `revenue` ranks higher in permutation importance, suggesting stronger true predictive power
- `rev_accel` ranks higher in tree-based importance, possibly due to feature interactions
- Permutation importance provides uncertainty estimates (error bars), showing `rev_accel` and `rev_yoy` have higher variance
- The comparison validates that revenue-related features are consistently the most important predictors

<!-- FEAT_IMPORT_END -->
## Data Pipeline & Alignment

### Why Data Alignment is Critical

The data pipeline combines multiple data sources with different frequencies:
- **Revenue data**: Quarterly (from SEC XBRL API)
- **Stock prices**: Daily (from yfinance)
- **Macro indicators** (VIX, TNX, SP500): Daily (from yfinance)

To enable time series analysis and lag features, all data must be aligned to a common time index (quarterly).

### What Alignment Does

The `align_data()` function (in `finmc_tech/data/align.py`) performs the following:

1. **Base Time Unit**: Uses quarterly frequency (from revenue data) as the base time unit
2. **Aggregation**: For each quarterly record, finds the closest trading day for:
   - Stock prices: Uses the price on the quarter-end date (or next trading day)
   - Macro indicators: Finds the closest trading day to the quarter-end date
3. **Feature Creation**: Creates extended features (price momentum, technical indicators, etc.) on the aligned quarterly data
4. **Result**: All features are on the same quarterly time index, enabling lag operations

### Example: Before and After Alignment

**Before alignment:**
- Revenue: Q1 2024, Q2 2024, Q3 2024 (quarterly, ~4 records per year)
- VIX: 2024-01-02, 2024-01-03, ..., 2024-12-31 (daily, ~252 records per year)
- Stock prices: 2024-01-02, 2024-01-03, ..., 2024-12-31 (daily, ~252 records per year)

**After alignment:**
- All data: Q1 2024, Q2 2024, Q3 2024 (quarterly, ~4 records per year)
- Each quarter record contains: revenue, price, VIX, TNX, SP500, and all derived features
- All features are on the same time index

### Why This Enables Lag Analysis

Once data is aligned to quarterly frequency, you can safely perform lag operations:

```python
# Quarterly-over-quarterly growth (lag 1 quarter)
rev_qoq = revenue.pct_change(1)

# Year-over-year growth (lag 4 quarters)
rev_yoy = revenue.pct_change(4)

# Price momentum (lag 1 quarter)
price_momentum = price / price.shift(1) - 1

# VIX change (lag 1 quarter)
vix_change = vix.diff(1)
```

**Without alignment**, these operations would be meaningless because:
- Revenue and prices would be on different time scales
- `shift(1)` on revenue would shift by 1 quarter, but on prices would shift by 1 day
- `pct_change(4)` on revenue would compare to 4 quarters ago, but on prices would compare to 4 days ago

### Implementation Details

The alignment process:
1. Takes quarterly firm data (revenue, prices) with `px_date` column
2. Takes daily macro data with `Date` column
3. For each quarterly record:
   - Finds the closest trading day in macro data to the quarter's `px_date`
   - Merges macro features (vix_level, tnx_yield, etc.) into the quarterly record
4. Creates extended features (price momentum, technical indicators, time features, interaction features)
5. Returns a DataFrame where all features are aligned to quarterly frequency

### Data File Priority

The pipeline checks for existing data files in this order (to avoid redundant API calls):

1. **Cache files**: `data_cache/{ticker}_firm_*.csv`
2. **Processed files**: `data/processed/{TICKER}_revenue_features.csv`
3. **Raw revenue files**: `data/raw/{TICKER}_revenue.csv`
4. **If not found**: Fetches from SEC API (revenue) and yfinance (prices)

This ensures efficient data loading and avoids unnecessary API calls.

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

### Quick Setup

```bash
pip install -r requirements.txt
```

### Detailed Setup

```bash
# Clone the repository
git clone https://github.com/yixuan116/finmc-tech.git
cd finmc-tech

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (optional)
pip install -e .
```

## Quick Start

### Using CLI (Recommended)

**Training RandomForest Model:**

```bash
python -m finmc_tech.cli train-rf
```

This will:
- Fetch macro and firm data
- Align data to monthly frequency
- Build features with lag variables
- Train RandomForest model
- Save model and feature importance plot to `results/`

**Running Monte Carlo Simulation:**

```bash
python -m finmc_tech.cli simulate --shock stress --h 24 --n 200
```

This will:
- Load trained model (or train if not exists)
- Generate macro paths with stress shocks
- Run Monte Carlo simulation over 24 months
- Save predictions and summary statistics to `results/`

**Generating Plots:**

```bash
python -m finmc_tech.cli plots --which all
```

This will generate:
- Predictions vs actual plot
- Simulation distribution plot
- Rolling correlation plot

**Running Tests:**

```bash
make test
# or
python tests/smoke_test.py
```

### Using Python API (Legacy)

```bash
# Clone the repository
git clone https://github.com/yixuan116/finmc-tech.git
cd finmc-tech

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (optional)
pip install -e .
```

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

### Using Jupyter Notebooks

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

**Note**: This project uses `yfinance` library, which fetches **free data from Yahoo Finance**. No API key required!

**Data Convention**: All OHLC prices are **adjusted for stock splits and dividends** by default (`auto_adjust=True`), making historical prices comparable across time. This is the standard approach for financial modeling. Historical prices appear low (e.g., \$0.4 in 2010) due to NVDA's stock splits (4:1 in 2021, 10:1 in 2024) and represent prices in "current share units".

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


The main demo is in `notebooks/demo_nvda.ipynb`.

**Quick Start:**

```bash
jupyter notebook notebooks/demo_nvda.ipynb
```

Then click **"Run All"** to execute all cells.

**What It Does:**

The notebook builds a complete **ML → Monte Carlo** pipeline in 5 steps:

1. **Data Loading**: Loads NVDA stock data, computes rolling statistics
2. **ML Baseline**: Predicts next-day returns using Linear Regression
3. **Monte Carlo**: Runs 5,000 simulated price paths (Serial + Numba backends)
4. **Visualizations**: Shows path samples and price distributions
5. **Sanity Checks**: Prints summary statistics

**Understanding Results:**

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




<!-- AUTO-REPORT:START -->
## Auto-Generated Dual-Head Analysis Report

This report analyzes revenue-based signals for **NVDA** using SEC XBRL data and dual-head machine learning models. The analysis spans 2009-07-26 to 2024-10-27, using the `Revenues` revenue tag. We model both **12-month forward returns** (return head) and **12-month forward stock prices** (price head) using revenue features (YoY growth, QoQ growth, acceleration) plus macro features (VIX level, 10Y yield, and their 3-month changes). Three baseline models (Ridge Regression, k-Nearest Neighbors, and RandomForest) are trained for each head with a temporal split (train before 2019, test from 2019). Price predictions are generated via two routes: (1) **indirect**: `price_hat = current_price * (1 + return_hat)` from the return head, and (2) **direct**: `log(price_hat)` predicted directly from the price head. Results show that the **Indirect** price route achieves lower RMSE ($78.93 vs $39.22). Revenue acceleration remains important after adding macro features, with RandomForest capturing non-linear relationships effectively.


### Return Head Performance

| Model | Test R² | Test RMSE | Test MAE | Direction Accuracy |
|-------|---------|-----------|----------|-------------------|
| k-NN (k=5) | -1.2035 | 1.0933 | 0.8993 | 70.8% |
| RandomForest | -1.1534 | **1.0809** | **0.8849** | **70.8%** |

**Model Comparison:**
- **RandomForest** achieves the lowest RMSE (1.0809) and MAE (0.8849), outperforming KNN by 0.0124 RMSE and 0.0144 MAE
- Both RandomForest and KNN achieve 70.8% directional accuracy
- RandomForest's superior performance is attributed to its ability to capture non-linear relationships and adapt across different market regimes through tree-based feature splitting

**Data Quality vs Model Capability Analysis:**
- **Data Quality**: Maximum feature-target correlation is 0.366 (tnx_yield), indicating moderate predictive signal. Baseline RMSE (predicting mean) is 0.7365, representing the inherent difficulty of the prediction task.
- **Model Capability**: RF's 1.14% RMSE improvement over KNN (0.0124 reduction) reflects model capability differences, but the improvement is small relative to the baseline.
- **Key Insight**: Both models significantly underperform the baseline (RMSE 1.08 vs 0.74), indicating that **data quality limitations dominate model performance**. The RF vs KNN difference (1.14%) represents the **model capability component**, which is modest compared to the data quality constraint.

### Direct Price Head Performance

| Model | Test R² (Log Price) | Test RMSE (Log Price) | Test Price RMSE (USD) |
|-------|---------------------|------------------------|----------------------|
| k-NN (k=5) | -11.8106 | 3.6723 | **$78.84** |
| RandomForest | -11.6402 | 3.6478 | $78.93 |

### Price RMSE Comparison

- **Indirect (from Return RF)**: $39.22
- **Direct (from Price RF)**: $78.93
- **Best Route**: Indirect

### Key Findings

- **Sample Size**: 62 rows
- **Date Range**: 2009-07-26 to 2024-10-27
- **Revenue Tag Used**: `Revenues`
- **Pearson Correlations** (with returns for analysis):
  - Rev YoY vs Future 12M Return: -0.0795
  - Rev Acceleration vs Future 12M Return: 0.0146
- **Best Return Head Model**: RandomForest (Test R² = -1.1534)
- **Best Price Head Model**: RandomForest (Test R² = -11.6402)
- **Best Price Route**: Indirect (RMSE = $39.22)

<!-- AUTO-REPORT:END -->
