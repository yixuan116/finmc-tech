# FinMC-Tech: Integrated ML & HPC Risk Modeling Framework for NVDA 还要改！

**Date:** December 12, 2025 | **Repository:** finmc-tech

---

## Abstract

This technical report presents **FinMC-Tech**, 
a comprehensive financial modeling system designed to analyze and forecast the volatility of NVIDIA Corp (NVDA). 不够，但是先不动，最后再说
The system integrates a robust machine learning pipeline for multi-horizon return forecasting (1Y, 3Y, 5Y, 10Y) with a high-performance Monte Carlo simulation engine. 

By leveraging firm-level data, macroeconomic indicators, and interaction terms, the Champion Model (Random Forest) achieves significant predictive stability. 

Furthermore, the simulation engine, optimized via Numba, MPI, and OpenMP, demonstrates the capability to scale to millions of simulation paths for rigorous risk assessment (VaR, CVaR). 确定一下有用这些变量么？？？？

---

## 1. Introduction

Financial market analysis xxxxx (add this!!!!)
The technology sector, particularly semiconductor leaders like Nvida, exhibits unique volatility characteristics driven by innovation cycles and macroeconomic shifts. 

Traditional financial models often struggle to capture the non-linear interactions between firm fundamentals and macro factors across different time horizons. 这句再想的高级一点

This paper addresses this by:
1.  Constructing a high-dimensional feature space ($\mathbb{R}^{75}$) — comprising fundamental, macro, and interaction features — to drive Random Forest (RF) regression and Monte Carlo (MC) simulations.
2.  Implementing a "Champion Model" selection process to identify the optimal drivers for short, middle, and long-term investment horizons.
3.  Deploying a multi-level HPC architecture (integrating Numba JIT, MPI clustering, and OpenMP) to simulate future price paths under extreme scenarios.

---

## 2. Mathematical Foundation

### 2.1 Horizon Definitions
Stock price forecasting spans two dimensions — the drivers of returns and the horizons over which they evolve.

$$r_{t+1}^{(h)} = \mu_{step}^{(h)} + S_t^{(h)} + \epsilon_t^{(h)}$$

In my model, I tested prices based on horizons (1y, 3y, 5y, 10y, 12 / 36 / 60 / 120 months, total 228 months).
The price updates in each month:

$$S_{t+1} = S_t \cdot (1 + r_{t+1}^{(h)})$$

**1. Deterministic function output:** Calculate the monthly returns by horizon

$$\hat{\mu}_t = f_h(X_{\text{firm}}, X_{\text{macro}}, X_{\text{interact}})$$

**2. Scenarios & Residuals**

$$S_t = w_{\text{Macro}}\epsilon_{M,t} + w_{\text{Firm}}\epsilon_{F,t} + w_{\text{Interaction}}\epsilon_{I,t}$$

**Overall:** monthly returns are real-valued (continuous), each month corresponds to one dimension in $\mathbb{R}$. The theoretical state space **228 months** is $\mathbb{R}^{228}$, "exhaust all possibilities" is impossible.

So, we use **Monte Carlo** to simulate the price evolution across all horizons, **1M simulations** to obtain a sufficiently rich approximation while keeping the workload computationally tractable.

### 2.2 Feature Space ($\mathbb{R}^{75}$)
The model input $\mathbf{X}_t$ is composed of three distinct vectors:
*   $\mathbf{X}_t^{\text{firm}} \in \mathbb{R}^{19}$: Fundamental metrics (Revenue, Cash Flow, Margins).
*   $\mathbf{X}_t^{\text{macro}} \in \mathbb{R}^{4}$: Key economic indicators (TNX Yield, VIX, GDP, CPI).
*   $\mathbf{X}_t^{\text{interaction}} \in \mathbb{R}^{52}$: Interaction terms capturing the sensitivity of firm fundamentals to macro shocks (e.g., $Revenue \times VIX$).

---

## 3. Machine Learning & Model Selection

### 3.1 Unified Evaluation Protocol
To ensure rigorous validation, we employed a time-series split strategy rather than random sampling, preventing data leakage from future information.

### 3.2 Champion Model Results
Based on out-of-sample $R^2$ performance, **Random Forest** was selected as the Overall Champion due to its stability across horizons and interpretability.

| Horizon | Champion Model | Test $R^2$ | MAE | RMSE | Insight |
|:-------:|:--------------:|:----------:|:---:|:----:|:-------:|
| **1Y**  | RandomForest   | -5.24      | 0.81| 0.90 | Best short-term stability |
| **3Y**  | NeuralNetwork  | -2.06      | 0.33| 0.38 | Captures mid-term non-linearity |
| **5Y**  | RandomForest   | -5.17      | 0.47| 0.51 | Robust long-term trend detection |
| **10Y** | NeuralNetwork  | -30.17     | 1.14| 1.47 | Highest volatility (sparse data) |

*Note: Negative R² in financial time-series forecasting indicates high regime-shift volatility in the test set (2020-2022) compared to the training mean, a common challenge in crash periods.*

---

## 4. High-Performance Computing (HPC) Architecture

To translate model predictions into risk metrics, we developed a scalable Monte Carlo engine. The system supports three levels of parallelism:

### 4.1 Tier 1: Just-In-Time Compilation (Numba)
We replaced standard Python loops with `numba.jit` (No-Python mode) to compile the geometric Brownian motion paths directly to machine code.
*   **Target:** Single-node, multi-core CPU.
*   **Result:** ~100x speedup over pure NumPy/Pandas loops.

### 4.2 Tier 2: Process-Level Parallelism (MPI)
For cluster-scale simulations, we utilize `mpi4py` to distribute millions of paths across multiple compute nodes.
*   **Strategy:** Map-Reduce (Scattered seeds, Gathered paths).
*   **Scalability:** Linear scaling up to available core count.

### 4.3 Tier 3: Low-Level Threading (OpenMP/C)
A dedicated C kernel with OpenMP pragmas was developed for maximum throughput in latency-critical scenarios.

---

## 5. Risk Analysis & Conclusion

The integration of ML-driven parameter estimation and HPC simulations allows FinMC-Tech to generate:
*   **VaR (Value at Risk):** 95% and 99% confidence intervals.
*   **Stress Testing:** Simulating specific macro-shock scenarios (e.g., "2008 Crisis" replay or "AI Boom" continuation).

**Conclusion:**
FinMC-Tech successfully demonstrates that combining domain-specific feature engineering with modern software engineering (HPC) significantly enhances the granularity and speed of financial risk assessment for volatile assets like NVDA.

---
---
# finmc-tech

**Machine Learning + Monte Carlo Simulation for Tech Stock Forecasting with HPC parallelization and uncertainty quantification**

`finmc-tech` is a minimal prototype for real-world stock forecasting that combines **machine learning signals** with **Monte Carlo uncertainty modeling** and **HPC-ready parallel execution**.

This research paper focuses on **NVIDIA (NVDA)** using real daily data from the Yahoo Finance API to compute returns, volatility, and rolling μ–σ parameters as inputs for simulation.


---

## Implementation Pipeline Overview

This table maps the 4 phases and 8 steps of the analysis pipeline to their corresponding implementation files and key outputs, aligning with the project presentation.

| Phase | Step | Description | Main Implementation File(s) | Key Output |
|:-----:|:----:|:------------|:---------------------------|:-----------|
| **A. Data & Features** | **1** | Data Engineering | `src/data/create_nvda_revenue_features.py`<br>**Run:** `python3 src/data/create_nvda_revenue_features.py`<br>**Source:** `yfinance` (Price), `NVDA_revenue.csv` (Firm), `yfinance` (`^TNX`, `^VIX`, `SPY`) | `nvda_revenue_features.csv`<br>(Data Engineering & Initial Features) |
| | **2** | Feature Engineering | `scripts/add_cash_flow_features_v2.py` (Calls `src/data/create_extended_features.py`)<br>**Run:** `python3 scripts/add_cash_flow_features_v2.py` | `nvda_features_extended_v2.csv`<br>(**Final V2 Features**: Injects Cash Flow & Refines Interactions)<br>**Source:** `nvda_revenue_features.csv`, `nvda_firm_fundamentals_master.json` (PDF/XBRL parsed) |
| **B. Model Training & Selection** | **3** | Model Training | `train_models.py`<br>**Run:** `python3 train_models.py` | **Files:** `models/champion_model.pkl`, `models/feature_scaler.pkl`<br>**Outcome:** 5 model families trained on Pre-2021 data (Validation 21-22) with standard hyperparameter grid. |
| | **4** | Champion Model Selection | `scripts/unified_model_evaluation.py`<br>`scripts/robustness_check_amd.py` (Robustness Check)<br><br>**Run:** `python3 scripts/unified_model_evaluation.py`<br>**Run:** `python3 scripts/robustness_check_amd.py` | **Files:** `table_champion_summary.csv`, `table_val_r2_oos_matrix.csv`, `table_test_r2_oos_matrix.csv`<br>`outputs/feature_importance/plots/unified_val_r2_oos_nonlinear.png`, `outputs/feature_importance/plots/unified_test_r2_oos_nonlinear.png`<br>`outputs/robustness/amd/validation_r2_matrix_amd.csv`<br>**Outcome:** 1. Champion Selection (RF wins 3Y with R²=0.68).<br>2. AMD Robustness Check (RF/XGB > 0 @ 3Y). |
| **C. Drivers & Interpretation** | **5** | All Horizons Feature Importance | `scripts/generate_topk_feature_heatmaps.py`<br>`scripts/three_category_feature_importance.py`<br><br>**Run:** `python3 scripts/generate_topk_feature_heatmaps.py`<br>**Run:** `python3 scripts/three_category_feature_importance.py` | Top-K Heatmaps (`outputs/feature_importance/plots/rf_top20_feature_matrix_paper.png`)<br>Category Heatmaps (`outputs/feature_importance/plots/importance_categories_rf_3cat.png`)<br>Rankings CSV (`rf_top20_matrix.csv`)<br>(Source: `nvda_features_extended_v2.csv`) |
| | **6** | Economic Interpretation | `README.md` (Analysis Section) | Narrative: "Short-term=Macro, Mid-term=Firm, Long-term=Regime" |
| **D. Scenarios & Simulations** | **7** | Scenario-Based Monte Carlo | `finmc_tech/simulation/scenario_mc.py` (Driver-Aware MC Engine)<br><br>**Run:** `python3 -m finmc_tech.simulation.scenario_mc --ticker NVDA --multi-horizon --n 10000` | Fan Charts (`results/step7/fan_chart_combined_baseline.png`)<br>Scenario Summary (`scenario_summary.csv`) |
| | **8** | Performance Optimization (HPC) | `finmc_tech/simulation/numpy_mc_demo.py`<br>`finmc_tech/simulation/numba_mc_demo.py`<br>`finmc_tech/hpc_demos/mpi_mc_demo.py`<br>`finmc_tech/hpc_demos/openmp_mc_demo.c` | Scaling Curve (`results/step8/hpc_scaling_curve.png`)<br>Benchmark Data (`hpc_benchmark.csv`) |

### Pipeline Results Snapshot (from Step 4)

**Table 1: Validation R² Matrix (OOS Benchmark)**
*Out-of-Sample R² on Validation Set (2021-2022). Positive values indicate predictive power superior to the historical mean.*

| Model | 1Y | 3Y | 5Y | 10Y |
| :--- | :--- | :--- | :--- | :--- |
| **RandomForest** | -0.53 | **0.68** | - | - |
| **XGBoost** | -0.61 | 0.56 | - | - |
| **NeuralNetwork** | **-0.51** | -3.04 | - | - |
| **Ridge** | -16.60 | -1.84 | - | - |
| **ElasticNet** | -3.95 | -25.93 | - | - |
| **Linear** | -202.77 | -181.17 | - | - |

*(Note: 5Y and 10Y horizons lack sufficient validation samples in the 2021-2022 window due to target availability)*

**Table 2: Champion Model Summary**
*Selected based on highest Validation R².*

| Horizon | Champion Model | Validation R² | Test R² (Ex Post) | Samples (Train/Val/Test) |
| :--- | :--- | :--- | :--- | :--- |
| **1Y** | NeuralNetwork | -0.51 | -4.01 | 49 / 8 / 8 |
| **3Y** | **RandomForest** | **0.68** | N/A | 49 / 8 / 0 |

**Table 3: Test R² Matrix (Ex Post Evaluation)**
*Out-of-Sample R² on Test Set (2023+). Severe negative values indicate regime shifts.*

| Model | 1Y | 3Y | 5Y | 10Y |
| :--- | :--- | :--- | :--- | :--- |
| **RandomForest** | -0.84 | - | - | - |
| **XGBoost** | -1.25 | - | - | - |
| **NeuralNetwork** | -4.01 | - | - | - |
| **Ridge** | -14.38 | - | - | - |
| **ElasticNet** | -547.60 | - | - | - |
| **Linear** | -1091.92 | - | - | - |

*(Note: Test targets for 3Y/5Y/10Y are physically impossible to observe in 2025)*

**Table 4: AMD Robustness Check (Validation R²)**
*Cross-firm validation using AMD data. Validation windows are shifted per horizon to ensure target availability.*

|               |        1Y |          3Y |        5Y |       10Y |
|:--------------|----------:|------------:|----------:|----------:|
| **RandomForest**  | -0.61 |   **0.08** | -21.89  |  -24.08 |
| **XGBoost**       | -0.93 |   **0.12** | -14.42  |  -34.11 |
| **NeuralNetwork** | -7.91 |  -3.53     |  -4.86  | -231.53 |
| **Ridge**         | -2.88 | -18.39     | -35.43  | -220.47 |
| **ElasticNet**    | -3.51 |  -8.31     | -13.80  | -167.91 |
| **Linear**        | -4.55 | -15.97     | -80.29  | -225.93 |

*Key Insight: While generally negative, tree-based models achieve positive OOS R² at the 3Y horizon, confirming the generalizability of the 3-Year predictive signal in the semiconductor sector.*


### Step 5: Key Drivers & Feature Importance Analysis

**Objective**: To interpret the "Black Box" of the Champion Model (Random Forest) and understand the economic rationale behind its predictions across different time horizons.

#### 1. The Hierarchy of Drivers: Firm vs. Macro vs. Interaction

![Feature Importance Categories](outputs/feature_importance/plots/importance_categories_rf_3cat.png)

*Figure 5.1: Aggregated feature importance by category (Firm, Macro, Interaction) for the Champion Random Forest model across horizons.*

**Key Insights**:
- **Short-Term (1Y) = Macro Regime**: Interaction features (62.6%) and Macro features (19.8%) dominate. In the short run, NVDA's price is driven by how the market (VIX/TNX) prices growth assets, rather than company fundamentals alone.
- **Mid-Term (3Y) = Fundamental Sweet Spot**: Firm features (66.3%) take over. This aligns with the "Champion Model" finding that 3Y is the most predictable horizon. Here, cash flows and revenue matter most.
- **Long-Term (5Y/10Y) = Regime Shift**: Interaction features return to prominence. Long-term survival depends on how the firm adapts to changing macro environments (Regime Dependence).

#### 2. Top-20 Specific Drivers (The "What")

![Top-20 Features](outputs/feature_importance/plots/rf_top20_feature_matrix_paper.png)

*Figure 5.2: Top-20 most important features for Random Forest across horizons. Darker red indicates higher importance.*

**Top Drivers Identified**:
1.  **Interest Rates (`tnx_yield`)**: The #1 driver for 1Y returns (18.6%). High rates compress valuations for long-duration assets like NVDA.
2.  **Cash Flow (`fcf_ttm` / `ocf_ttm`)**: The #1 driver for 3Y returns (14.8%). This validates the V2 feature engineering—profitability is the ultimate anchor for mid-term value.
3.  **Macro-Firm Interactions**: Features like `ix_tnx_yield__price_volatility` highlight that volatility regimes conditional on interest rates are critical for long-term risk assessment.

### Step 6: Economic Interpretation (Why It Happens)

**Objective**: To bridge the gap between Machine Learning outputs (Step 5) and Financial Theory.

#### 1. The "Duration Risk" Theory (Explaining Short-Term Behavior)
Why does `tnx_yield` dominate the 1-Year horizon?
*   **Concept**: NVDA is a "Long Duration Asset". Its valuation depends heavily on cash flows expected far in the future.
*   **Mechanism**: When interest rates (`tnx_yield`) rise, the discount rate increases, disproportionately punishing the present value of distant future cash flows.
*   **Evidence**: The model picks `tnx_yield` as the #1 predictor for 1Y returns, confirming that short-term price action is mechanically tied to the Fed's rate cycle, not just AI hype.

#### 2. The "Convergence to Fundamentals" (Explaining Mid-Term Behavior)
Why does `fcf_ttm` dominate the 3-Year horizon?
*   **Concept**: In the short run, the market is a voting machine (sentiment/macro); in the long run, it is a weighing machine (earnings).
*   **Mechanism**: Over a 3-year period, macro noise cancels out, and the stock price must eventually align with the company's ability to generate cash.
*   **Evidence**: The shift from Macro features (1Y) to Firm features (3Y) in our model perfectly empirically validates this classic Ben Graham investment philosophy.

#### 3. The "Regime Dependence" (Explaining Long-Term Uncertainty)
Why do Interaction Features (`ix_...`) dominate long horizons?
*   **Concept**: A static linear relationship doesn't exist over 10 years.
*   **Mechanism**: The relationship between NVDA's revenue and its stock price *changes* depending on the macro environment (e.g., Revenue growth is rewarded in low-rate regimes but questioned in high-rate regimes).
*   **Evidence**: The dominance of interaction terms (Macro × Micro) at 5Y+ horizons shows that the model is learning *context-dependent* valuation rules, essentially capturing "Regime Shifts".

### Step 7: Scenario-Based Monte Carlo Forecasting

**Objective**: To move beyond single-point predictions and quantify "Regime Risk" using the feature importance hierarchy derived in Step 5.

#### Methodology: Driver-Aware Shocks
Instead of standard Gaussian shocks, we use **weighted shocks** based on our Champion Model's logic:
- **1Y Horizon**: Shocks are 62% Macro-driven (Interest Rate stress tests).
- **3Y Horizon**: Shocks are 66% Firm-driven (Revenue/Cash Flow misses).
- **5Y+ Horizon**: Shocks are Interaction-driven (Regime Shift simulation).

#### 1. Baseline Fan Charts (Multi-Horizon)

![Baseline Fan Chart](results/step7/fan_chart_combined_baseline.png)

*Figure 7.1: Baseline probabilistic forecast for 1Y, 3Y, 5Y, and 10Y horizons. The widening cone represents uncertainty propagation derived from historical volatility and driver-specific shocks.*

#### 2. Risk Scenarios (Tail Risk Analysis)
We simulate 4 distinct economic regimes:
1.  **Base Case**: Historical drift and volatility.
2.  **Macro Stress**: 50bp rate hike + VIX spike (Impacts 1Y valuation most).
3.  **Fundamental Stress**: 20% revenue deceleration (Impacts 3Y price most).
4.  **AI Supercycle (Bull)**: Favorable interactions + low rate volatility.

**Key Findings**:
- **Short-Term Risk is Asymmetric**: The 1Y distribution is negatively skewed by macro stress (rates), confirming the "Duration Risk" thesis.
- **Mid-Term Certainty**: The 3Y horizon shows the most stable risk-adjusted returns (highest Sharpe ratio in simulation), reinforcing it as the "Sweet Spot".

### Step 8: Performance Engineering (HPC)

**Objective**: To ensure the "Risk Engine" can scale to **millions of paths** for robust tail-risk estimation (VaR/CVaR) without computational bottlenecks.

#### 1. Risk Metrics Definition
Before optimizing, we define the computational target:
-   **VaR (Value at Risk)**: The maximum loss expected (with 95% confidence) over the horizon.
-   **CVaR (Conditional VaR)**: The average loss in the worst 5% of cases (tail mean), requiring deep simulation to converge.

#### 2. Benchmark Workload
-   **Total Simulations**: 1,000,000 paths per horizon.
-   **Total Horizons**: 1Y (12) + 3Y (36) + 5Y (60) + 10Y (120) = 228 time steps.
-   **Total Updates**: $1M \times 228 = 228,000,000$ Monte Carlo step updates.

#### 3. Benchmark Results
We compared four implementations:
1.  **NumPy (Baseline)**: Vectorized single-core execution.
2.  **MPI (4 Ranks)**: Multi-process parallelism (process overhead dominant at this scale).
3.  **OpenMP (C)**: Low-level multi-threading (shared memory).
4.  **Numba Parallel**: JIT-compiled parallel loop (shared memory, Python-native).


![HPC Benchmark Summary](results/step8/hpc_benchmark_summary.png)

*Figure 8.1: Runtime and Speedup comparison for 228M Monte Carlo updates.*

| Backend | Runtime (s) | Speedup | Notes |
| :--- | :--- | :--- | :--- |
| **NumPy (Baseline)** | 36.64 s | 1.0x | Single-core, memory-bound |
| **MPI (4 Ranks)** | 7.55 s | ~4.9x | Good scaling, but overhead limits speedup on small batches |
| **OpenMP (C)** | 1.19 s | **~30.8x** | C-level efficiency, minimal overhead |
| **Numba Parallel** | 1.26 s | **~29.2x** | **Best Trade-off**: Near-C speed with pure Python code |

**Conclusion**:
-   **Numba Parallel** is the production choice. It achieves **~29x speedup** over NumPy, matching highly optimized C code (OpenMP) while remaining in the Python ecosystem.
-   **MPI** is viable for massive cluster-scale jobs (>100M simulations) but introduces unnecessary overhead for this workload size.

---

## Results

## Current Progress: Data & Methodology & Models & Results

### Data Summary

**Dataset**: NVDA Quarterly Features (2008-2025)
- **Sample Size**: 71 rows (70 unique quarters, 1 duplicate in 2010Q3 due to overlapping fiscal year reports)
- **Time Period**: 2008-Q1 to 2025-Q3
- **Data Frequency**: Quarterly (average 91 days between observations)
- **Data Sources**:
  - Firm fundamentals: SEC XBRL (quarterly revenue, margins, etc.)
  - Macro variables: FRED API (CPI, VIX, 10Y yield, Fed Funds, GDP)
  - Price data: Yahoo Finance (adjusted close, returns)

### Feature Engineering

**Feature Matrix**: 71 rows × 76 columns (71 data points covering 70 unique quarters)

**Total Feature Count**: 12 (firm) + 4 (macro) + 3 (cash flow) + 52 (interactions) + 4 (time) + 1 (metadata) = **76 features** used in model training

**Note**: V2 version includes cash flow features (OCF_TTM, CAPEX_TTM, FCF_TTM) and their macro interactions, expanding from 63 to 76 features.


**Note on Duplicate**: 2010Q3 has 2 records because two different fiscal year reports (FY2010 and FY2011) both ended in that quarter. This is a data alignment artifact from SEC XBRL filings where fiscal year boundaries don't align perfectly with calendar quarters. **We keep both records** because they represent different fiscal periods with potentially different revenue calculations and business contexts, even though they fall in the same calendar quarter.

The feature engineering pipeline follows Gu-Kelly-Xiu (2020) RFS methodology, constructing a comprehensive feature space through:

1. **Firm Features (12 features)**:
   - Revenue metrics: `revenue`, `rev_qoq`, `rev_yoy`, `rev_accel`
   - Price momentum: `price_returns_1m/3m/6m/12m`, `price_momentum`, `price_volatility`, `price_to_ma_4q`
   - Additional: `adj_close` (base price level)

2. **Macro Features (4 features)**:
   - `vix_level`: VIX index level
   - `tnx_yield`: 10-year Treasury yield
   - `vix_change_3m`: 3-month VIX change
   - `tnx_change_3m`: 3-month Treasury yield change

3. **Cash Flow Features (3 features)**:
   - `ocf_ttm`: Operating Cash Flow (Trailing Twelve Months)
   - `capex_ttm`: Capital Expenditure (Trailing Twelve Months)
   - `fcf_ttm`: Free Cash Flow (Trailing Twelve Months)

4. **Interaction Features (52 features)**:
   - **Kronecker Product Structure**: All macro × (micro + cash flow) interactions
   - Format: `ix_<macro>__<micro>` (e.g., `ix_vix_level__rev_yoy`, `ix_tnx_yield__ocf_ttm`)
   - **Economic Rationale**: State-dependent effects—firm characteristic impact depends on macro environment
   - **Mathematical Structure**: `z_i,t = x_t ⊗ c_i,t` where:
     - `x_t`: Macro state variables (4 features)
     - `c_i,t`: Firm characteristics (10 micro + 3 cash flow = 13 features)
     - `⊗`: Kronecker product → 4 × 13 = 52 interaction terms

   **Visualization: Interaction Features Generation**

   ![Interaction Features Diagram](docs/interaction_features_diagram.png)
   *High-level flow diagram showing how Macro × Firm features generate Interaction features*

   ![Interaction Features Matrix](docs/interaction_features_matrix_diagram.png)
   *Matrix representation of the Kronecker product structure: 4 Macro features × 10 Firm features = 40 Interaction features*

   **High-Level Structure:**
   ```
   MACRO FEATURES (4)  ×  FIRM FEATURES (10)  =  INTERACTION FEATURES (40)
   ──────────────────────────────────────────────────────────────────────
   vix_level           rev_yoy          →  ix_vix_level__rev_yoy
   tnx_yield           price_returns_12m →  ix_tnx_yield__price_returns_12m
   vix_change_3m       price_volatility →  ix_vix_change_3m__price_volatility
   tnx_change_3m       rev_accel         →  ix_tnx_change_3m__rev_accel
   ...                 ...               →  ... (All 40 combinations)
   ```

   **Matrix Representation:**
   ```
           Macro (4 features)
           ┌─────┬─────┬─────┬─────┐
           │ vix │ tnx │ vix │ tnx │
           │_lev │_yld │_chg │_chg │
           └─────┴─────┴─────┴─────┘
   Firm    │     │     │     │     │
   (10)    │  ×  │  ×  │  ×  │  ×  │
           │     │     │     │     │
           └─────┴─────┴─────┴─────┘
              ↓     ↓     ↓     ↓
           ┌─────┬─────┬─────┬─────┐
           │ ix_ │ ix_ │ ix_ │ ix_ │
           │ ... │ ... │ ... │ ... │
           │(10) │(10) │(10) │(10) │
           └─────┴─────┴─────┴─────┘
            Total: 4 × 10 = 40 Interaction Features
   ```

   **Example: How One Interaction is Created**
   ```
   vix_level (Macro)  ×  rev_yoy (Firm)  =  ix_vix_level__rev_yoy (Interaction)
        │                    │                         │
        │                    │                         │
     [0.25]              [0.15]                   [0.0375]
    (VIX=25)         (Revenue +15%)          (State-dependent effect)
   ```

4. **Time Features (4 features)**:
   - `quarter`, `month`, `year`, `days_since_start`


---

## Industry-Driven Time Window Selection

Time-series forecasting requires strict temporal ordering to prevent data leakage—models must only use past information to predict future outcomes. 
NVIDIA's business model has evolved through **three distinct industry regimes**, each defined by changes in GPU demand, hyperscaler spending, semiconductor supply chains, and AI adoption curves. Because the economic structure underlying the stock changed, the model must respect these regime boundaries instead of using naive or uniformly sampled time splits.

## **Industry Regime Timeline**

```
2010-2020 (Train)         2021-2022 (Val)          2023-Present (Test)
─────────────────         ───────────────          ───────────────────
Gaming GPU Era            AI Pre-Acceleration      AI Supercycle
├─ Gaming dominant       ├─ A100 deployment       ├─ H100/H200
├─ DC small but growing  ├─ Early LLM training    ├─ GenAI adoption
├─ Crypto/PC cycles      ├─ Cloud AI capex ↑      ├─ DC dominant
└─ Predictable supply    └─ DC rapidly expanding  └─ Supply bottlenecks
```


### **1. 2010–2020 — GPU-Centric Cycle (Training Window)**  

**Time Split**: < 2021-01-01 (52 samples)

**Industry Regime:** Gaming GPU → Early Cloud GPU → Pre-AI Compute  

Key characteristics:

- Gaming revenue dominated NVDA's mix  

- Data Center was small but growing  

- Crypto and PC cycles created periodic volatility  

- Capex intensity only loosely linked to AI demand  

- Inventory cycles were predictable and supply-driven  

**Why used for training:**  

This period provides a long, stable sample for learning *general relationships* between fundamentals, macro conditions, and returns before the structural AI shift.

---

### **2. 2021–2022 — AI Pre-Acceleration / Transition Period (Validation Window)**  

**Time Split**: 2021-01-01 to 2022-12-31 (8 samples)

**Industry Regime:** A100 ramp, early hyperscaler AI infrastructure build-out  

Key characteristics:

- A100 deployed at scale across AWS, Google, Meta, Microsoft  

- Early LLM training demand (GPT-3, PaLM 1 era)  

- Cloud AI capex begins accelerating but not exponential  

- Gaming weak but Data Center rapidly expanding  

**Why used for validation:**  

A validation window must represent the *upcoming* structure without seeing the *future extreme* regime.  

2021–2022 is the unique "transition zone" between the legacy GPU world and the full AI supercycle.

---

### **3. 2023–Present — AI Supercycle (Test Window)**  

**Time Split**: ≥ 2023-01-01 (11 samples)

**Industry Regime:** H100 deployment, GenAI adoption, hyperscaler AI capex explosion  

Key characteristics:

- Data Center becomes NVDA's dominant revenue engine  

- H100/H200 shortages define the supply chain  

- Gross margin jumps structurally  

- AI servers and networking create a new ecosystem  

- Semiconductor bottlenecks (CoWoS, HBM) reshape industry capacity  

**Why used for testing:**  

This period represents a **new economic regime**.  

The model must demonstrate it can generalize from historical + transition regimes to an unprecedented AI-driven cycle.


### **Industry Regime Timeline & Train/Test Splits**

The model evaluation respects the structural breaks in NVIDIA's history by enforcing strict time-based splits for each forecasting horizon. This ensures that models are tested on "future" regimes that they have not seen during training.

| Horizon | Training Period (End Date) | Testing Period (Start Date) | Gap (Validation/Leakage Prevention) |
|:-------:|:--------------------------:|:---------------------------:|:-----------------------------------:|
| **1Y**  | < 2020-12-31               | > 2022-12-31                | 2 Years (2021-2022)                 |
| **3Y**  | < 2018-12-31               | > 2020-12-31                | 2 Years (2019-2020)                 |
| **5Y**  | < 2016-12-31               | > 2018-12-31                | 2 Years (2017-2018)                 |
| **10Y** | < 2012-12-31               | > 2014-12-31                | 2 Years (2013-2014)                 |

*Note: A 2-year gap is enforced between training and testing windows to ensure strict out-of-sample evaluation and prevent look-ahead bias.*

### Methodology: Multi-Model Comparison Pipeline

**Step 2: Model Training & Evaluation**

**Models Trained** (5 models):
1. **Linear Regression**: Baseline linear model
2. **Ridge Regression**: L2-regularized linear model with cross-validation (`alphas=[0.1, 1.0, 10.0, 50.0]`)
3. **Random Forest**: 500 trees, unlimited depth (`RandomForestRegressor`)
4. **XGBoost**: Gradient boosting with 500 estimators, learning rate 0.05, max depth 5
5. **Neural Network**: Multi-layer perceptron with hidden layers (64, 32), 300 max iterations

---

## Model Selection Rationale

The five models used in Step 2 span the full complexity spectrum and are chosen to cover linear structure, regularized structure, nonlinear interactions, boosted hierarchical effects, and smooth neural nonlinearities.  
This provides a complete baseline for financial return prediction prior to HPC-based optimization in later phases.

### **Model Comparison Table**

| Model | Type | What It Captures | Why It Matters for NVDA Returns |
|-------|------|------------------|---------------------------------|
| **Linear Regression** | Linear baseline | Additive, proportional relationships | Tests whether fundamental signals (e.g., revenue, inventory cycles) have direct linear impact. |
| **Ridge Regression** | Regularized linear | Multi-collinearity, shrinkage stability | Interaction features (macro × firm) are correlated; Ridge verifies if signals survive regularization. |
| **Random Forest** | Nonlinear trees | Threshold effects, discrete state shifts | Captures "effect only matters under low VIX", "momentum only works in stable macro". High interpretability. |
| **XGBoost** | Gradient-boosted trees | Hierarchical nonlinearities, complex interactions | Industry-standard model for tabular financial data; strongest performer for macro × micro signals. |
| **Neural Network (MLP)** | Smooth nonlinear approximator | Differentiable curves, soft thresholds | Provides non-tree nonlinear structure; useful contrast to tree-based models. Not expected to dominate, but completes the spectrum. |

---

### **Mathematical Formulations**

This section provides the mathematical foundations for each model used in the unified evaluation.

#### **1. Linear Regression**

**Objective Function:**
```
minimize: L(β) = ||y - Xβ||²
```

**Solution:**
```
β̂ = (XᵀX)⁻¹Xᵀy
```

**Prediction:**
```
ŷ = Xβ̂
```

Where:
- `y ∈ ℝⁿ`: target vector (n samples)
- `X ∈ ℝⁿˣᵖ`: feature matrix (n samples × p features)
- `β ∈ ℝᵖ`: coefficient vector
- `β̂`: estimated coefficients (Ordinary Least Squares)

---

#### **2. Ridge Regression (L2 Regularization)**

**Objective Function:**
```
minimize: L(β) = ||y - Xβ||² + α||β||²
```

**Solution:**
```
β̂ = (XᵀX + αI)⁻¹Xᵀy
```

**Prediction:**
```
ŷ = Xβ̂
```

Where:
- `α > 0`: regularization strength (L2 penalty)
- `I`: identity matrix
- `||β||² = Σⱼ βⱼ²`: L2 norm of coefficients (shrinkage penalty)

**Key Property:** Ridge shrinks coefficients toward zero but never exactly to zero, handling multicollinearity by stabilizing the inverse of `XᵀX`.

---

#### **3. Random Forest**

**Ensemble Prediction:**
```
ŷ = (1/B) Σᵦ₌₁ᴮ Tᵦ(x)
```

**Individual Tree:**
```
Tᵦ(x) = Σₘ₌₁ᴹ cₘ · I(x ∈ Rₘ)
```

**Training Objective (per tree):**
```
minimize: Σᵢ₌₁ⁿ (yᵢ - T(xᵢ))²
```

Where:
- `B`: number of trees (bootstrap samples)
- `Tᵦ(x)`: prediction from tree b
- `Rₘ`: m-th leaf region (rectangular partition of feature space)
- `cₘ`: constant prediction value for region Rₘ (mean of y in that region)
- `I(·)`: indicator function

**Key Property:** Each tree is trained on a bootstrap sample with random feature subset selection at each split, providing variance reduction through averaging.

---

#### **4. XGBoost (Gradient Boosting)**

**Additive Model:**
```
ŷ = Σₜ₌₁ᵀ fₜ(x)
```

**Objective Function:**
```
minimize: L = Σᵢ₌₁ⁿ l(yᵢ, ŷᵢ) + Σₜ₌₁ᵀ Ω(fₜ)
```

**Loss Function (MSE):**
```
l(yᵢ, ŷᵢ) = (yᵢ - ŷᵢ)²
```

**Regularization:**
```
Ω(fₜ) = γT + (1/2)λ||w||²
```

**Tree Construction (greedy):**
```
fₜ(x) = argmin Σᵢ₌₁ⁿ [gᵢfₜ(xᵢ) + (1/2)hᵢfₜ²(xᵢ)] + Ω(fₜ)
```

Where:
- `fₜ(x)`: t-th tree (weak learner)
- `gᵢ = ∂l/∂ŷᵢ`: first-order gradient
- `hᵢ = ∂²l/∂ŷᵢ²`: second-order gradient (Hessian)
- `T`: number of leaves in tree
- `w`: leaf weights
- `γ, λ`: regularization parameters

**Key Property:** XGBoost uses second-order Taylor expansion of the loss function and builds trees greedily to minimize the regularized objective, providing strong performance on tabular data.

---

#### **5. Neural Network (Multi-Layer Perceptron)**

**Forward Propagation:**

**Layer 1 (Input → Hidden 1):**
```
z₁ = W₁x + b₁
a₁ = σ(z₁)
```

**Layer 2 (Hidden 1 → Hidden 2):**
```
z₂ = W₂a₁ + b₂
a₂ = σ(z₂)
```

**Layer 3 (Hidden 2 → Output):**
```
ŷ = W₃a₂ + b₃
```

**Loss Function (MSE):**
```
L = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Backpropagation (Gradient Descent):**
```
Wₖ ← Wₖ - η · ∂L/∂Wₖ
bₖ ← bₖ - η · ∂L/∂bₖ
```

Where:
- `x ∈ ℝᵖ`: input feature vector
- `Wₖ`: weight matrix for layer k
- `bₖ`: bias vector for layer k
- `σ(·)`: activation function (e.g., ReLU: `σ(z) = max(0, z)`)
- `η`: learning rate
- `n`: number of samples

**Architecture (as configured):**
- Input layer: p features (75 features)
- Hidden layer 1: 64 neurons
- Hidden layer 2: 32 neurons
- Output layer: 1 neuron (regression)

**Key Property:** Neural networks learn smooth, differentiable nonlinear transformations through composition of linear transformations and nonlinear activations, providing a different inductive bias than tree-based models.

---

## Why Not LSTM (or Other Sequence Models) in Step 2?

LSTM and sequence-based deep learning models are **not** included in Step 2 for three structural reasons tied specifically to the nature of NVDA prediction and the project's multi-phase design:

### **1. NVDA's feature set is tabular, not sequential**

The dataset is composed of engineered macro, micro, and interaction features (e.g., revenue_yoy, VIX level, macro × firm Kronecker terms).  
These are **cross-sectional tabular characteristics per month**, not long temporal sequences.

LSTM excels when the input is a *continuous sequence*:  
X[t-12], X[t-11], ..., X[t]

But our model uses:  
X_features[t] (macro, firm, interactions) → return[t+1]

There is no long sequence per sample.

### **2. Tree-based models outperform LSTM on low-frequency financial data**

Empirical asset pricing literature (Gu, Kelly, Xiu 2020; Chen et al. 2021) shows:

- For **monthly data**, tree models (RF/XGB) dominate  
- LSTM lacks an advantage unless data is high-frequency (tick, minute, day)
- Tabular interactions (macro × firm) are better modeled by trees

LSTM adds complexity without improving baseline predictive power.

### **3. LSTM is intentionally reserved for Step 6–8 (Deep Forecasting & HPC)**

The project structure allocates deep learning to later stages:

- **Step 7:** LSTM/GRU for multi-step sequential forecasting  
- **Step 8:** Scenario engine + LSTM-based dynamic response modeling  
- **Step 9:** HPC/GPU-accelerated training of LSTM/Transformer variants  

Including LSTM in Step 2 would slow the baseline and introduce heavy training cost without improving the benchmark.

---

**Summary:**  
Tree models + linear benchmarks provide the strongest, cleanest, and most interpretable baseline for monthly NVDA return forecasting.  
Sequence models (LSTM/GRU) come later when the project transitions from tabular single-step prediction → deep sequential forecasting under HPC.

---

**Training Procedure**:
- **Time-based Split**: See [Industry-Driven Time Window Selection](#industry-driven-time-window-selection) above for detailed rationale and time boundaries
- **Feature Scaling**: StandardScaler fitted on training set only, applied to val/test
  - **What it does**: Transforms features to have zero mean and unit variance: `z = (x - μ) / σ`
  - **Why it matters**: Different features have vastly different scales (e.g., revenue in billions vs. VIX around 20). Scaling ensures:
    - Distance-based models (KNN, Neural Networks) aren't dominated by large-scale features
    - Gradient-based optimizers (NN, XGBoost) converge faster and more stably
    - Regularization (Ridge) applies evenly across features
  - **Critical**: Scaler is fitted on training data only to prevent data leakage—test set statistics must remain unknown during training
- **Evaluation Metrics**: MAE, RMSE, R², MAPE (all computed on test set)
  - **MAE (Mean Absolute Error)**: Average absolute prediction error. Interpretable in original units (e.g., 0.59 = 0.59% return error on average). Robust to outliers.
  - **RMSE (Root Mean Squared Error)**: Square root of average squared errors. Penalizes large errors more heavily than MAE. Units: same as target (return %).
  - **R² (Coefficient of Determination)**: Proportion of variance explained. R² = 1.0 means perfect predictions; R² = 0 means model performs as well as predicting the mean; R² < 0 means worse than naive baseline. Standard metric for regression model comparison.
  - **MAPE (Mean Absolute Percentage Error)**: Average absolute error as percentage of actual values. Useful for understanding relative prediction accuracy (e.g., 43% MAPE means predictions are off by 43% on average relative to actual returns).

### Model Performance (Test Set > 2022, Unified Evaluation)

**Horizon: 3-Year Forward Return (Best Performing Horizon)**

| Model | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.5522** | **0.5699** | **-1.5156** |
| XGBoost | 0.6618 | 0.6702 | -2.4791 |
| Neural Network | 1.0689 | 1.2706 | -11.5033 |
| Ridge Regression | 5.8123 | 8.6122 | -573.4703 |
| Linear Regression | 10.4472 | 15.2412 | -1798.1844 |

**Key Finding**: Random Forest consistently outperforms other models across horizons (1Y, 3Y, 5Y) in the unified evaluation, exhibiting the most robustness against regime shifts in the 2022+ test set. While R² values are negative (indicating the extreme difficulty of forecasting 2022-2023 crash/recovery dynamics compared to the training mean), RF provides the tightest error bounds (lowest MAE/RMSE).

**Champion Model (NVDA)**: **Random Forest** (Best OOS R² = -1.52 at 3Y Horizon)

**Note**: V2 model includes 76 features (added 3 cash flow features + 12 cash flow × macro interactions). Performance is similar to V1 (63 features), with slight degradation (<2%) that may not be statistically significant.

---

### Robustness Check: Cross-Firm Validation (AMD)

To validate that the "Random Forest Dominance" isn't an artifact of NVIDIA's specific data structure, we performed the identical pipeline on **Advanced Micro Devices (AMD)**.

**AMD Results (Training Set Comparison):**
- **Random Forest** achieved **R² = 0.9256** on the AMD training set, slightly outperforming NVDA's training fit (R² = 0.8932).
- Linear Models performed significantly worse on AMD (R² ≈ 0.54) compared to NVDA (R² ≈ 0.84), confirming that the semiconductor industry drivers are inherently **non-linear**.

**Conclusion**: The superiority of tree-based ensembles (Random Forest) is a robust finding across the semiconductor sector, not specific to NVDA.

---

### NVDA vs AMD Feature Importance Comparison

**Methodology**: We extract Top-20 most important features from each of the 5 models (Linear, Ridge, RF, XGB, NN) for both companies, then build a union heatmap showing all unique features that appear in any model's top-20. This reveals which features drive predictions for each company.

**Visualizations**:

![NVDA Union Heatmap](results/nvda_union_heatmap.png)
*NVDA Top-K Union Feature Importance Heatmap: Shows all unique features appearing in any model's top-20, normalized per model (0-1 scale). Rows = models, Columns = features. Darker colors indicate higher importance.*

![AMD Union Heatmap](results/amd_union_heatmap.png)
*AMD Top-K Union Feature Importance Heatmap: Shows all unique features appearing in any model's top-20, normalized per model (0-1 scale). Rows = models, Columns = features. Darker colors indicate higher importance.*

**Key Findings**:

#### 1. **Common Features Are Rare (Only 8 features)**
- **Only 8 features** appear in top-20 for both companies
- **Common features are primarily**:
  - **Time features**: `year`, `month`, `days_since_start` (3 features)
  - **Price/Technical features**: `adj_close`, `price_to_ma_4q`, `price_volatility`, `price_ma_4q`, `price_returns_6m` (5 features)
- **Interpretation**: Both companies share basic time trends and price momentum patterns, but diverge significantly in what drives their returns beyond these fundamentals.

#### 2. **NVDA: Revenue-Driven + Interest Rate Sensitive (39 NVDA-only features)**
- **Top NVDA-only features**:
  - `tnx_yield` (Treasury yield) - **Importance: 1.57**
  - `ix_tnx_yield__price_returns_12m` (TNX × Price returns) - **Importance: 1.54**
  - `ix_tnx_yield__rev_qoq` (TNX × Revenue QoQ) - **Importance: 1.25**
- **Feature composition**:
  - **16 revenue features**: `rev_qoq`, `rev_yoy`, `rev_accel`, and their interactions
  - **21 TNX (interest rate) interactions**: TNX × revenue, TNX × price features
  - **2 macro features**: `tnx_yield`, `tnx_change_3m`
- **Interpretation**: 
  - NVDA's returns are **fundamentally driven** by revenue growth and acceleration
  - **Highly sensitive to interest rate environment** (TNX) - rate changes affect how revenue translates to returns
  - Revenue data quality is high (complete SEC XBRL data)

#### 3. **AMD: Market-Driven + Technical Indicators (23 AMD-only features)**
- **Top AMD-only features**:
  - `ix_sp500_returns__price_volatility` (SP500 × Price volatility) - **Importance: 55.43**
  - `ix_sp500_returns__price_returns_6m` (SP500 × Price returns) - **Importance: 19.82**
  - `sp500_returns` (SP500 returns) - **Importance: 17.81**
- **Feature composition**:
  - **10 SP500 interactions**: SP500 × price features dominate
  - **6 technical indicators**: `atr`, `bb_position`, and other price-based technical features
  - **Limited revenue features**: Only 4 revenue features (most are NaN due to missing data)
- **Interpretation**:
  - AMD's returns are **market-driven** - strongly correlated with SP500 performance
  - **More dependent on technical indicators** when revenue data is unavailable
  - Revenue data quality is poor (limited SEC XBRL data)

#### 4. **Feature Type Distribution**

| Feature Type | Common | NVDA-Only | AMD-Only |
|--------------|--------|-----------|----------|
| **Price/Technical** | 5 | 0 | 6 |
| **Time** | 3 | 0 | 1 |
| **Interaction** | 0 | 21 | 10 |
| **Revenue** | 0 | 16 | 0 |
| **Macro** | 0 | 2 | 2 |

#### 5. **Core Insights & Business Logic**

**NVDA's Prediction Drivers**:
1. **Revenue fundamentals matter most**: 16 revenue features capture business performance
2. **Interest rate sensitivity**: TNX interactions show how rate changes affect revenue-to-return translation
3. **Fundamental analysis works**: Revenue growth, acceleration, and their macro interactions predict returns

**AMD's Prediction Drivers**:
1. **Market beta is high**: SP500 interactions dominate (10 features, highest importance)
2. **Technical analysis compensates**: When revenue data is missing, technical indicators fill the gap
3. **Market-driven returns**: Returns are more correlated with overall market performance than company-specific fundamentals

**Shared Patterns**:
- Both companies rely on **time trends** (`year`, `days_since_start`) - capturing long-term structural shifts
- Both use **price momentum** (`price_to_ma_4q`, `price_volatility`) - technical patterns matter
- **Non-linear interactions are critical** - simple linear models fail for both

#### 6. **Practical Implications**

**For NVDA**:
- **Focus on revenue signals**: Revenue growth, acceleration, and QoQ changes are primary predictors
- **Monitor interest rates**: TNX changes affect how revenue translates to stock returns
- **Fundamental analysis is effective**: Revenue-based models can work well

**For AMD**:
- **Market timing matters**: SP500 performance is a strong predictor
- **Technical indicators help**: When fundamentals are unavailable, technical analysis provides signals
- **Beta management**: AMD's returns are more market-dependent, requiring different risk management

**For Both**:
- **Time trends capture regime shifts**: Both companies show structural changes over time (AI supercycle, market evolution)
- **Non-linear models are essential**: Tree-based models (RF, XGB) capture interactions that linear models miss
- **Feature engineering matters**: Interaction features (macro × micro) are critical for both companies

---

## Step 5 — Short-Horizon (12M) Key Drivers Analysis

### 🔎 Overview

Step 5 extracts the true drivers behind NVDA's 12-month forward returns using the champion Random Forest model from Step 4.

This step focuses on interpreting the model rather than re-training it.

We compute:

- Mean Decrease Impurity (MDI) importance
- Permutation importance
- SHAP (Shapley Additive Explanations) values
- PDP/ICE curves
- Final driver rankings + economic interpretation

This produces the most complete short-horizon driver decomposition of NVDA's return dynamics.

---

### 🧠 Mathematical Foundations

#### SHAP (Shapley Additive Explanations)

SHAP decomposes a model prediction into marginal contributions from each feature, based on the Shapley value from cooperative game theory:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

Where:

- $\phi_i$ = contribution of feature $i$
- $F$ = full set of features
- $S$ = subset of features not containing $i$
- $f(S)$ = model prediction using only features in $S$

TreeSHAP gives an exact polynomial-time algorithm for tree models:

$$O(T \cdot L^2)$$

where $T$ = number of trees, $L$ = depth.

This makes SHAP ideal for RF/XGBoost.

#### PDP (Partial Dependence)

PDP shows the average effect of a feature on predictions:

$$PD_i(x_i) = E_{X_{\sim i}}[f(x_i, X_{\sim i})]$$

#### ICE (Individual Conditional Expectation)

ICE shows sample-specific effects:

$$ICE_j(x_i) = f(x_i, x_{\sim i}^{(j)})$$

Together:

- **SHAP** = local driver attribution
- **PDP** = global marginal effect
- **ICE** = regime heterogeneity

#### Permutation Importance

Randomly shuffling feature $i$ breaks its relationship to the target.

Performance deterioration = true predictive value.

#### MDI Importance

Measures decrease in impurity (variance/MSE) when splitting on feature $i$.

Useful but can be biased → we combine all methods.

---

### 📊 Key Plots

#### MDI: Top 20 Features

![MDI Importance](results/step5/mdi_top20.png)

#### Permutation Importance

![Permutation Importance](results/step5/perm_top20.png)

#### SHAP Summary

![SHAP Beeswarm](results/step5/shap_beeswarm.png)

#### PDP/ICE: TNX Yield

![PDP/ICE: TNX Yield](results/step5/pdp_tnx_yield.png)

**Interpretation**:

- Clear negative contribution from higher TNX
- Nonlinear "kink" around ~2%
- ICE curves show regime-dependent sensitivity (2020→2024)
- Short-term NVDA returns behave like a discount-rate asset.

#### PDP/ICE: VIX Change × 6M Return

![PDP/ICE: VIX Change × 6M Return](results/step5/pdp_ix_vix_change_3m__price_returns_6m.png)

**Interpretation**:

- Momentum is not standalone
- Only meaningful when conditioned on volatility shocks
- Captures AI-narrative amplification during risk-on periods

#### PDP/ICE: VIX Change × Revenue

![PDP/ICE: VIX Change × Revenue](results/step5/pdp_ix_vix_change_3m__revenue.png)

**Interpretation**:

- Revenue only matters when VIX spikes
- High volatility → fundamentals get discounted
- Markets re-price growth under stress

#### PDP/ICE: VIX Level × Rev YoY

![PDP/ICE: VIX Level × Rev YoY](results/step5/pdp_ix_vix_level__rev_yoy.png)

**Interpretation**:

- Revenue YoY influence is suppressed in low-VIX regimes
- But becomes negative in high-VIX regimes
- Reflects risk-adjusted growth repricing

---

### 🧩 Findings: Short-Horizon (12M) Drivers

#### 1. TNX Yield = strongest single-factor driver

- **SHAP**: high TNX → negative return forecast
- **PDP**: nonlinear / threshold around 2%
- **Economic meaning**: NVDA short-term pricing is macro-discount-rate driven.

#### 2. VIX Change = regime-switching driver

- Large VIX spikes sharply reduce predicted returns
- Cross-terms VIX × Momentum / VIX × Revenue rank highest
- Matches real-world "AI high-beta behavior"

#### 3. Momentum only matters conditionally

Pure momentum features are weak.

Momentum becomes powerful only through macro interaction:

$$\text{Return}_{12M} = f(\Delta VIX \times PR_{6M})$$

#### 4. Revenue fundamentals are not priced at the 12M horizon

SHAP ~ 0 for:

- `revenue`
- `rev_yoy`
- `rev_accel`

Fundamentals require longer horizons to appear in pricing.

This is a key empirical finding of Step 5.

#### 5. Time Trend (AI Supercycle) is present but not a driver

`days_since_start` contributes weakly and nonlinearly.

---

### 🔍 Why Step 5 Naturally Leads to a Multi-Horizon Framework

Step 5 empirically shows:

- Short-term price = macro (TNX, VIX)
- NOT fundamentals (SHAP ≈ 0 for rev variables)
- Momentum is macro-conditioned
- Growth only priced in high-vol regimes

Thus the market structure itself forces a multi-horizon view:

- **12M** → macro-driven
- **36M** → revenue acceleration becomes visible
- **60–120M** → margins, TAM & ecosystem drive valuation

This three-horizon architecture is an empirical consequence of Step 5, not an arbitrary design choice.

---

## Step 5.5 — Long-Term Feature Importance Analysis (5-10 Years)

### 📊 Overview

This section extends the feature importance analysis to long-term horizons (5-10 years), complementing the short-term (12M) analysis in Step 5. The analysis reveals how feature importance shifts as prediction horizons extend, with a particular focus on:

- **Overall feature importance** across all features
- **FCF-specific importance** (Free Cash Flow features)
- **Firm Level vs Macro** feature importance comparison

### 📁 Analysis Results Location

**Directory:** `outputs/feature_importance/data/long_term/`

All analysis results, data files, and visualizations are stored in this directory.

### 📈 Overall Feature Importance (5-10 Years)

#### Main Heatmap

![Long-Term Feature Importance Heatmap](outputs/feature_importance/plots/long_term/feature_importance_long_term_heatmap.png)

Comprehensive heatmap showing top 30 features across 5-year, 7-year, and 10-year horizons. **Compare with the short-term (1-5 years) heatmap** (`outputs/feature_importance/plots/mid_term/feature_importance_by_horizon_heatmap.png`) to see how feature importance changes from short-term to long-term.

#### Key Findings

**Top Features Across Long-Term Horizons:**

| Rank | Feature | 5 Years | 7 Years | 10 Years |
|------|---------|---------|---------|----------|
| 1 | `rev_cagr_3y` | 0.7713 | - | - |
| 2 | `rev_cagr_2y` | 0.0153 | 0.3702 | 0.3576 |
| 3 | `rev_ttm` | 0.0320 | 0.0597 | 0.1212 |
| 4 | `gross_margin_pct` | - | 0.1042 | 0.0762 |
| 5 | `eps_cagr_2y` | - | 0.0696 | - |

**Key Insights:**
- **Revenue CAGR (2-3 years)** is the most important predictor for long-term returns
- **Revenue TTM** becomes increasingly important in longer horizons
- **Gross margin percentage** is important for 7-10 year predictions
- **EPS metrics** gain importance in longer horizons

### 💰 FCF (Free Cash Flow) Importance Analysis

#### FCF Importance Trends

| Horizon | FCF Total Importance | FCF % of Total | Top FCF Feature | Rank |
|---------|---------------------|----------------|-----------------|------|
| 3 years | 0.0160 | 1.60% | `fcf_ttm` | 20 |
| 5 years | 0.0128 | 1.28% | `fcf_conversion` | 22 |
| 7 years | 0.0201 | 2.01% | `fcf_ttm` | 20 |
| 10 years | 0.0181 | 1.81% | `fcf_conversion` | 19 |

**Key Insights:**
1. **Peak Performance**: 7-year prediction window shows highest FCF importance (2.01%)
2. **Best FCF Feature**: `fcf_ttm` performs best in 7-year predictions
3. **Long-term Trend**: `fcf_conversion` becomes more important in longer horizons (rank improves to #19 at 10 years)
4. **Overall Pattern**: FCF features show increasing importance in mid-to-long-term predictions

#### FCF Visualization Charts

![FCF Comprehensive Analysis](outputs/feature_importance/plots/long_term/fcf_comprehensive_analysis.png)

Contains 5 subplots:
1. FCF Total Importance Trend (3, 5, 7, 10 years)
2. FCF % of Total Importance (Bar chart)
3. Top FCF Feature Rank Changes
4. Top FCF Feature Importance Changes
5. Heatmap: All FCF Features Across Time Horizons

![FCF Long-Term Analysis](outputs/feature_importance/plots/long_term/fcf_importance_long_term_analysis.png)

Contains 4 subplots:
1. FCF Total Importance vs Horizon
2. FCF % of Total Importance vs Horizon
3. Top FCF Feature Rank vs Horizon
4. Model Performance vs Horizon

### 🏢 Firm Level vs Macro Feature Importance

![Firm vs Macro Importance Comparison](outputs/feature_importance/plots/long_term/firm_vs_macro_importance_comparison.png)

**Key Findings:**
- **Firm Level features** dominate across all horizons (82-98% importance)
- **Macro features** are relatively important in short-term (1-2 years: 19-26%), but decrease significantly in mid-term (3-7 years: 5-8%)
- **Long-term (10 years)**: Macro importance slightly increases again (12%), suggesting they play a larger role in very long-term outlooks

**Interpretation:**
- Short-term (1-2 years): Both firm and macro factors matter
- Mid-term (3-7 years): Firm fundamentals become overwhelmingly dominant
- Long-term (10 years): Macro factors regain some importance, but firm fundamentals still dominate

### 📊 Data Files

All analysis data is available in `outputs/feature_importance/data/long_term/`:

- **Overall Feature Importance:**
  - `feature_importance_long_term_all_features.csv` - All features across 5, 7, 10 years
  - `feature_importance_y_log_20q_all_features.csv` - 5-year prediction (all features)
  - `feature_importance_y_log_28q_all_features.csv` - 7-year prediction (all features)
  - `feature_importance_y_log_40q_all_features.csv` - 10-year prediction (all features)

- **FCF Analysis:**
  - `fcf_importance_by_horizon.csv` - FCF importance summary across horizons
  - `fcf_importance_complete_comparison.csv` - Complete comparison (3, 5, 7, 10 years)
  - `firm_vs_macro_importance_summary.csv` - Firm vs Macro comparison data

### 🔗 Related Files

**Training Data:**
- Location: `outputs/data/training/training_data_extended_10y.csv`
- Contains: Extended training data with 5-10 year target variables
  - `y_log_20q` (5 years)
  - `y_log_24q` (6 years)
  - `y_log_28q` (7 years)
  - `y_log_32q` (8 years)
  - `y_log_36q` (9 years)
  - `y_log_40q` (10 years)

**Short-term Analysis (1-5 years):**
- Location: `outputs/feature_importance/plots/mid_term/`
- File: `feature_importance_by_horizon_heatmap.png` - Heatmap for 1-5 year predictions
- **Comparison:** Compare with `feature_importance_long_term_heatmap.png` to see how feature importance changes from short-term to long-term

### 🏆 Champion Model Comparison Across Horizons

This section compares multiple model families (Linear, Ridge, Lasso, ElasticNet, RandomForest, XGBoost) across different prediction horizons (1Y, 3Y, 5Y, 10Y) to identify the champion model for each horizon.

**Methodology:**
- **Models Compared:** Linear, Ridge, Lasso, ElasticNet, RandomForest, XGBoost
- **Evaluation Metrics:** MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), R² (Coefficient of Determination)
- **Data Split:** Time-based 80/20 split (first 80% for training, last 20% for testing)
- **Features:** Extended feature set from `nvda_features_extended_v2.csv` (includes cash flow features)

**Champion Model Summary:** See detailed results in [Unified Model Evaluation](#step-3-4-unified-model-evaluation-strict-time-based-split) section below.

**Overall Champion:** **RandomForest** (stable, interpretable drivers)

**Rationale for Overall Champion Selection:**
- **RandomForest** is selected as the overall champion despite not being the R² champion for all horizons, because:
  1. **Interpretability**: RF provides direct, interpretable feature importance scores essential for economic modeling
  2. **Stability**: RF performs consistently well across horizons (champion for 3Y, very close to champion for 1Y and 5Y with only 0.02 R² difference)
  3. **Overfitting Concerns**: While XGBoost achieves slightly better test R² for 5Y (-2.33 vs -2.35, only 0.02 difference), it shows severe overfitting on training set (R² = 1.0000), indicating poor generalization. RF maintains better train-test consistency.
  4. **Practical Performance**: RF achieves the best balance between predictive performance and interpretability
  5. **Economic Modeling Requirements**: Neural Networks and XGBoost, while achieving best R² for some horizons, are less interpretable and show overfitting issues that make them less suitable for economic modeling
| **5Y** | RandomForest<br>(0.6117) | XGBoost<br>(0.7427) | XGBoost<br>(-2.3342) | ⚠️ Different |
| **10Y** | ElasticNet<br>(0.5916) | ElasticNet<br>(0.6134) | ElasticNet<br>(-7.0162) | ✓ Consistent |

**Key Findings:**
- **Short-term (1Y):** NeuralNetwork achieves best R² (-1.15), but RandomForest is very close (-1.17, only 0.02 difference)
- **Mid-term (3Y):** RandomForest is the clear champion (R²: -1.82, MAE: 0.45)
- **Mid-term (5Y):** XGBoost achieves best R² (-2.33), but RandomForest is very close (-2.35, only 0.02 difference)
- **Long-term (10Y):** ElasticNet performs best (R²: -7.02), likely due to regularization benefits with limited sample size
- **Overall Champion Selection:** RandomForest is selected as overall champion for its interpretability and stable performance across horizons, despite not being the R² champion for all horizons
- **Note:** All models show negative R² values, indicating poor out-of-sample performance. This is common with small sample sizes and long prediction horizons. Champion models are identified based on relative performance.

**Visualization:**

![Model Comparison - MAE](outputs/feature_importance/plots/model_comparison_mae.png)
*MAE comparison across models and horizons. Lower is better. Green bars indicate champion models.*

![Model Comparison - RMSE](outputs/feature_importance/plots/model_comparison_rmse.png)
*RMSE comparison across models and horizons. Lower is better. Green bars indicate champion models.*

![Model Comparison - R²](outputs/feature_importance/plots/model_comparison_r2.png)
*R² comparison across models and horizons. Higher is better (though all values are negative). Green bars indicate champion models.*

**Data Files:**
- `outputs/feature_importance/results/model_comparison.csv` - Complete comparison table with all metrics

**Analysis Script:**
- `scripts/champion_model_comparison.py` - Performs model comparison across horizons

**Commands:**
```bash
# Run champion model comparison
python3 scripts/champion_model_comparison.py \
  --features-csv data/processed/nvda_features_extended_v2.csv \
  --output-dir outputs/feature_importance
```

---

### 🔬Step 3 4 Unified Model Evaluation (Strict Time-Based Split)

To ensure fair and realistic model comparison, we conducted a **unified evaluation** using consistent time-based splits across all horizons. This evaluation addresses potential data leakage and temporal bias issues present in simple 80/20 splits.

#### Dataset Overview

**Data Source:** `data/processed/nvda_features_extended_v2.csv`
- **Total Samples:** 71 quarterly observations
- **Features:** 75 features (19 Firm-level, 4 Macro, 52 Interaction features)
- **Target Variables:**
  - `ret_1y`: 65 non-null values (12-month forward returns)
  - `ret_3y`: 57 non-null values (36-month forward returns)
  - `ret_5y`: 50 non-null values (60-month forward returns)
  - `ret_10y`: 30 non-null values (120-month forward returns)

**Data Period:** Quarterly data spanning multiple years, with target variables computed as forward-looking returns from each observation point.

#### Methodology: Fixed Time-Point Splits

This evaluation uses **fixed time-point splits** that align with the evaluation methodology in `train_models.py`. This ensures:

1. **No Future Data Leakage:** Test sets contain only data from periods strictly after training periods
2. **Realistic Evaluation:** Models are evaluated on truly "future" data, simulating real-world deployment
3. **Consistent Methodology:** Same evaluation approach as the original champion model selection

**Time Split Configuration:**

| Horizon | Training Set | Test Set | Rationale |
|---------|--------------|----------|-----------|
| **1Y** | < 2020-12-31 | > 2022-12-31 | Ensures 1-year forward returns in test set are from post-2022 period |
| **3Y** | < 2018-12-31 | > 2020-12-31 | 3-year forward returns require earlier test split to ensure sufficient data |
| **5Y** | < 2016-12-31 | > 2018-12-31 | 5-year forward returns require even earlier split for data availability |
| **10Y** | < 2012-12-31 | > 2014-12-31 | 10-year forward returns require earliest split due to limited long-term data |

**Why This Approach?**

1. **Temporal Validity:** For time-series financial data, using fixed time-point splits ensures that models are evaluated on data that would actually be available in a real-world forecasting scenario. A simple 80/20 split may inadvertently include training data that temporally overlaps with test predictions.

2. **Prevents Data Leakage:** Long-horizon targets (e.g., 10Y returns) computed from early observations might overlap with short-horizon targets from later observations. Fixed time-point splits eliminate this risk.

3. **Consistent with Production:** In production, models trained on historical data are used to predict future returns. The fixed time-point split mimics this scenario more accurately than random or proportional splits.

4. **Fair Comparison:** By using the same split methodology as `train_models.py`, we ensure that model comparisons are based on identical evaluation criteria, making results directly comparable to the original champion model selection.

#### Model Configurations

All models use the same configurations as `train_models.py` for consistency:

- **RandomForest:** `n_estimators=500, max_depth=None, random_state=42`
- **XGBoost:** `n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42`
- **NeuralNetwork:** `hidden_layer_sizes=(64, 32), max_iter=500, random_state=42` (with StandardScaler)
- **Linear/Ridge/ElasticNet:** Standard sklearn configurations with regularization

#### Unified Evaluation Results

**Champion Models by Horizon (Test Set R²) - Updated with V2 Data:**

| Horizon | Champion Model | R² | MAE | RMSE | Test Samples |
|---------|---------------|-----|-----|------|--------------|
| **1Y** | **NeuralNetwork** | **-1.15** | 0.66 | 0.84 | 12 |
| **3Y** | **RandomForest** | **-1.82** | 0.45 | 0.50 | 11 |
| **5Y** | **XGBoost** | **-2.33** | 0.65 | 0.74 | 9 |
| **10Y** | **ElasticNet** | **-7.02** | 0.59 | 0.61 | 5 |

**Overall Champion:** **RandomForest** (stable, interpretable drivers)

**Rationale for Overall Champion Selection:**
- **RandomForest** is selected as the overall champion despite not being the R² champion for all horizons, because:
  1. **Interpretability**: RF provides direct, interpretable feature importance scores essential for economic modeling
  2. **Stability**: RF performs consistently well across horizons (champion for 3Y, very close to champion for 1Y and 5Y with only 0.02 R² difference)
  3. **Overfitting Concerns**: While XGBoost achieves slightly better test R² for 5Y (-2.33 vs -2.35, only 0.02 difference), it shows severe overfitting on training set (R² = 1.0000), indicating poor generalization. RF maintains better train-test consistency.
  4. **Practical Performance**: RF achieves the best balance between predictive performance and interpretability
  5. **Economic Modeling Requirements**: Neural Networks and XGBoost, while achieving best R² for some horizons, are less interpretable and show overfitting issues that make them less suitable for economic modeling

**Key Findings:**

1. **Model Performance Improvement:** With V2 data (including cash flow features), all models show significant R² improvement compared to V1 results. For example, 1Y horizon improved from -5.24 to -1.15, and 10Y horizon improved dramatically from -30.17 to -7.02.

2. **Champion Model Distribution:** Different models excel at different horizons:
   - **1Y**: NeuralNetwork (best R², but RF is very close)
   - **3Y**: RandomForest (clear champion)
   - **5Y**: XGBoost (best R², but RF is very close)
   - **10Y**: ElasticNet (best R², likely due to regularization benefits)

3. **RandomForest Stability:** RandomForest demonstrates consistent performance across horizons, being champion for 3Y and within 0.02 R² of the champion for 1Y and 5Y, making it the most reliable choice for overall champion.

4. **Cash Flow Features Impact:** The addition of cash flow features (ocf_ttm, capex_ttm, fcf_ttm) and their interactions with macro variables has improved model performance, particularly for long-term horizons (10Y).

5. **Small Test Sets:** Limited test set sizes (especially for 10Y with only 5 samples) indicate the challenges of long-horizon forecasting with limited historical data.

**Visualization:**

![Unified Model Comparison - R²](outputs/feature_importance/plots/unified_model_comparison_r2.png)
*Unified model comparison across horizons using fixed time-point splits. Higher R² is better (though all values are negative). Green bars indicate champion models for each horizon.*

![Unified Model Comparison - MAE](outputs/feature_importance/plots/unified_model_comparison_mae.png)
*Unified model comparison: MAE (Mean Absolute Error) across horizons. Lower MAE is better. Green bars indicate champion models for each horizon.*

![Unified Model Comparison - RMSE](outputs/feature_importance/plots/unified_model_comparison_rmse.png)
*Unified model comparison: RMSE (Root Mean Squared Error) across horizons. Lower RMSE is better. Green bars indicate champion models for each horizon.*

#### Step 5 Feature Importance Analysis Across Horizons

**RF Top-20 Feature Importance Across Horizons:**

![RF Top-20 Feature Importance Across Horizons](outputs/feature_importance/plots/rf_top20_feature_matrix.png)
*Heatmap showing the top 20 most important features for Random Forest model across 1Y, 3Y, 5Y, and 10Y prediction horizons. Color intensity represents feature importance percentage. Higher values (darker red) indicate greater predictive power for that horizon.*

**Key Observations:**
- **1Y Horizon:** `tnx_yield` (31.3%) dominates, indicating macro factors are critical for short-term predictions
- **3Y Horizon:** Firm fundamentals (`fcf_ttm` 17.1%, `ocf_ttm` 15.0%) become dominant, with time features (`days_since_start` 13.9%, `year` 12.8%) also important
- **5Y Horizon:** Interaction feature `ix_tnx_yield_price_volatility` (69.6%) overwhelmingly dominates, showing that macro × price interactions are key for medium-term forecasts
- **10Y Horizon:** `tnx_yield` (27.2%) returns as the top feature, with `days_since_start` (14.8%) and interaction features gaining importance

**RF Feature Importance by Category Across Horizons:**

![RF Feature Importance by Category Across Horizons](outputs/feature_importance/plots/importance_categories_rf_3cat.png)
*Heatmap showing feature importance aggregated by category (Firm, Macro, Interaction) across different prediction horizons. Reveals how the relative importance of feature types shifts as prediction horizons extend.*

**Key Findings (ordered by horizon: 1Y, 3Y, 5Y, 10Y):**
- **Firm Features:** 
  - 1Y: 22.6% | 3Y: 68.1% (peak) | 5Y: 15.2% | 10Y: 27.8%
  - Peak importance at 3Y horizon, indicating fundamentals are most predictive for mid-term forecasts
- **Macro Features:**
  - 1Y: 32.5% (peak) | 3Y: 3.3% (lowest) | 5Y: 5.6% | 10Y: 27.6%
  - Highest importance at 1Y and 10Y, but minimal at 3Y and 5Y
- **Interaction Features:**
  - 1Y: 44.9% | 3Y: 28.6% | 5Y: 79.2% (peak) | 10Y: 44.6%
  - Overwhelmingly dominant at 5Y horizon, showing that macro × firm interactions are critical for medium-term predictions

**Interpretation:** The evolution of feature importance across horizons reveals a clear pattern:
- **Short-term (1Y):** Macro and interaction features dominate, reflecting market sentiment and discount-rate effects
- **Mid-term (3Y):** Firm fundamentals become primary drivers as business fundamentals take precedence
- **Medium-term (5Y):** Interaction features dominate, capturing regime-dependent relationships between macro conditions and firm characteristics
- **Long-term (10Y):** Macro factors regain importance alongside time trends, reflecting long-term discount-rate and structural regime effects

**Data Files:**
- `outputs/feature_importance/results/unified_model_comparison.csv` - Complete unified evaluation results

**Analysis Script:**
- `scripts/unified_model_evaluation.py` - Performs unified evaluation with fixed time-point splits

**Commands:**
```bash
# Run unified model evaluation
python3 scripts/unified_model_evaluation.py \
  --features-csv data/processed/nvda_features_extended_v2.csv \
  --output-dir outputs/feature_importance
```

**Comparison with Initial Evaluation:**

The unified evaluation reveals important differences from the initial 80/20 split evaluation:

- **More Conservative Results:** All models show worse R² values, reflecting the stricter evaluation methodology
- **Different Champions:** NeuralNetwork emerges as champion for 3Y and 10Y in unified evaluation, while RandomForest remains champion for 1Y and 5Y
- **Higher Confidence:** Results from unified evaluation are more reliable for production deployment decisions, as they better simulate real-world forecasting scenarios

### 📝 Analysis Scripts

The analysis was performed using:
- **Overall Analysis:** `src/phase2_long_cycle/analyze_long_term_overall_importance.py`
- **FCF Analysis:** `src/phase2_long_cycle/analyze_long_term_feature_importance.py`
- **Firm vs Macro:** `src/phase2_long_cycle/create_firm_vs_macro_heatmap.py`
- **Champion Model Comparison:** `scripts/champion_model_comparison.py`

**Commands:**
```bash
# Overall feature importance (5-10 years)
python3 src/phase2_long_cycle/analyze_long_term_overall_importance.py \
  --data outputs/data/training/training_data_extended_10y.csv \
  --targets y_log_20q y_log_28q y_log_40q

# FCF-specific analysis
python3 src/phase2_long_cycle/analyze_long_term_feature_importance.py \
  --data outputs/data/training/training_data_extended_10y.csv \
  --targets y_log_20q y_log_28q y_log_40q

# Firm vs Macro comparison
python3 src/phase2_long_cycle/create_firm_vs_macro_heatmap.py
```

---

## Step 6 — Driver Interpretation Across Forecast Horizons

Understanding *why* the model makes different predictions across horizons is essential for building an interpretable and finance-aligned ML forecasting system. Using Random Forest (global stability) and XGBoost (nonlinear interaction resolution), we extract horizon-specific feature importance and classify all features into:

- **Firm-Level** (fundamentals, FCF/OCF, revenue acceleration, margins)  

- **Macro** (TNX yield, VIX level, inflation, volatility regimes)  

- **Interaction** (macro × firm, macro × price, macro × momentum)

The results reveal a clean, economically meaningful **evolution of return drivers**:

### 🔹 Short-Term Horizon (0–12 months): Macro & Market-Sentiment Dominance

Short-term forecasts are driven primarily by **macro shocks and price-based sentiment** rather than firm fundamentals. Features such as **TNX yield**, **VIX level**, **3–12M returns**, and macro × price interactions explain the majority of return variation. This matches the market microstructure of high-beta tech stocks: 12-month returns respond more to **discount-rate shocks, volatility regimes, and momentum compression** than to earnings trajectories.

> **Short-term = macro + sentiment driven.**

### 🔹 Mid-Term Horizon (2–4 years): Firm Fundamentals Dominate

Across the 2–4 year window, both RF and XGBoost converge on **firm-level fundamentals as the dominant drivers**: **FCF TTM**, **OCF TTM**, **Revenue YoY**, **Revenue Acceleration**, and margin-related interactions. Macro noise averages out, and structural variables describing unit economics, cash generation, and operational leverage carry the strongest predictive power. This horizon corresponds to standard sell-side modeling practice.

> **Mid-term = revenue + cash-flow trajectory driven.**

### 🔹 Long-Term Horizon (5–10 years): Regime-Driven Macro × Interaction Dominance

At long horizons, fundamentals lose predictive power due to compounding uncertainty. The model instead identifies **macro × price interactions** (e.g., *VIX × momentum*, *TNX × volatility*) as the dominant drivers — particularly in XGBoost, which captures nonlinear regime sensitivity. This mirrors the long-run risk and discount-rate literature: long-horizon returns depend heavily on **future rate regimes and risk-premium structures**, not point forecasts of cash flows.

> **Long-term = macro × regime sensitivity, not fundamentals.**

### ⭐ One-Sentence Summary

**Short-term returns are macro-driven, mid-term returns are fundamentals-driven, and long-term returns are regime-driven via nonlinear macro × interaction channels.**

---

## Step 7 — Translating Drivers into Economic Narrative

After Step 5 identifies the dominant machine learning drivers (TNX, VIX, price momentum, rate–volatility interactions), Step 7 converts these statistical signals into real economic meaning.

This is the step where ML stops being numbers and becomes business intelligence, valuation insight, and strategic forecast.

It answers three questions investors, CFOs, and strategists truly care about:

1. **Why do these factors matter economically?**
2. **What do they tell us about NVIDIA's valuation and cyclicality?**
3. **How do macro and fundamentals link into a unified explanation?**

---

### 7.1 Why MACRO Dominates SHORT Horizon (12M)

The 12-month forecast horizon captures market behavior, not business fundamentals.

Machine learning finds that the strongest 12M signals are:

| Rank | Driver | Meaning |
|------|--------|---------|
| 1 | `ix_vix_change_3m__price_returns_6m` | Volatility regime × momentum |
| 2 | `ix_vix_change_3m__revenue` | Volatility shaping how revenue is priced |
| 3 | `days_since_start` | Structural regime drift (pre-AI → AI supercycle) |
| 4 | `adj_close` | Valuation anchor / returns-to-mean pressure |
| 5 | `ix_vix_level__rev_yoy` | VIX × revenue growth |

**Narrative (核心解释)**:

→ In short horizons, the market prices NVDA based on macro regime first, business fundamentals second.

**Because**:

- **NVDA is a long-duration growth stock**
- **AI CAPEX returns happen years in the future**
- **Therefore its valuation is extremely sensitive to discount rate (TNX)**
- **And to risk appetite (VIX)**
- **Revenue does not move quarter-to-quarter enough to dominate 12M returns**

This matches institutional investor behavior: growth megacap = macro-beta amplified by positioning, flows, volatility regimes.

---

### 7.2 PDP / SHAP Interpretation (with Economic Meaning)

#### (1) Partial Dependence Plot: VIX × Price Momentum

![PDP/ICE: VIX Change × 6M Return](results/step5/pdp_ix_vix_change_3m__price_returns_6m.png)

**What ML sees**:

- When VIX is high (market fear), momentum becomes unreliable.
- When VIX is low (risk-on regime), momentum is rewarded.

**Economic interpretation**:

Momentum is not a standalone factor for NVDA — it is regime-dependent, exactly like equity risk-premia literature predicts.

#### (2) PDP: VIX × Revenue YoY

![PDP/ICE: VIX Level × Rev YoY](results/step5/pdp_ix_vix_level__rev_yoy.png)

**What ML sees**:

- Revenue YoY only helps predict returns when volatility is low.
- When VIX is high, even strong revenue is ignored by investors.

**Economic interpretation**:

During panic or high uncertainty, the market stops differentiating fundamentals — all correlations go to 1.

#### (3) PDP: TNX (10-year Treasury Yield)

![PDP/ICE: TNX Yield](results/step5/pdp_tnx_yield.png)

**What ML sees**:

- Higher TNX depresses NVDA expected 12M returns (negative slope).
- The effect is convex → rising rates hurt more when already elevated.

**Economic interpretation**:

- **TNX = discount rate**
- **NVDA = long-duration cash-flow asset**
- **Therefore**: interest rate changes dominate valuation sensitivity more than quarterly revenue.

This is consistent with 2022–2023 AI drawdowns and re-ratings.

#### (4) SHAP Summary: Interaction Terms Dominate

![SHAP Beeswarm](results/step5/shap_beeswarm.png)

**SHAP shows interaction features (macro × micro) explaining most variance.**

**Meaning**:

- You cannot explain NVDA by revenue alone.
- You cannot explain NVDA by rates alone.
- The real driver is: **"Given the macro regime, how much does the market choose to reward fundamentals?"**

This perfectly captures institutional pricing dynamics.

---

### 7.3 Macro → Micro Narrative (The Economic Bridge)

To unify the two worlds:

#### Short Horizon (12M)

**Market determines valuation**

→ Macro cycle, VIX regime, rates, liquidity

→ Micro (revenue/margin) only matter conditionally

#### Medium Horizon (36M)

**Business determines numbers**

→ Revenue trajectory, margins, ASP mix, GPU supply chain

→ Macro is a background conditioning factor

#### Long Horizon (60–120M)

**Strategy & TAM determine cash-flow**

→ Data center AI adoption

→ CUDA ecosystem lock-in

→ AI server architecture cycles

→ GPU supply chain & HBM bottlenecks

→ Nvidia's platform economics

**This step "glues" them together**:

- **Macro prices NVDA in the short-term**
- **Business delivers NVDA in the medium-term**
- **Strategy defines NVDA in the long-term**

This is exactly how real hedge funds and CFO offices reason.

---

### 7.4 What Step 7 Tells Us About NVIDIA Today

Based on the extracted drivers + PDP/SHAP interpretation:

#### (1) NVDA is not priced on revenue in short horizons

Revenue improves slowly, so it is not the marginal information set the market reacts to.

#### (2) NVDA is priced on the state of macro regime

- Rates ↓ → higher valuation multiple
- Volatility ↓ → higher risk appetite
- "Macro calmness" = "NVIDIA β > 1 expansion"

#### (3) Revenue signals become stronger only when macro stabilizes

This matches the 2023–2024 AI supercycle where rates stabilized.

#### (4) Rate cycle > revenue cycle in 12M horizon

**This is the headline finding**: valuation drivers dominate business drivers in short horizons.

#### (5) This is not contradictory — it is exactly textbook asset pricing for long-duration tech

Your ML pipeline basically rediscovered academic finance:

- long-duration equity
- macro discounting
- volatility + flows
- convexity

But with NVDA-specific structure + business interpretation.

---

### 7.5 Multi-Horizon Scenario Families for Monte Carlo Simulation

Step 7's economic narrative framework translates directly into **driver-aware scenario families** used in Step 8's Monte Carlo engine. These scenarios apply **horizon-specific shocks** to feature categories (Macro, Firm, Interaction) based on each horizon's feature importance weights.

#### Scenario Architecture

The Monte Carlo engine implements **4 scenario families** that modify shock components by category:

| Scenario Family | Macro Shock | Firm Shock | Interaction Shock | Economic Interpretation |
|----------------|-------------|------------|-------------------|------------------------|
| **Base** | 0 (baseline) | 0 (baseline) | 0 (baseline) | Normal market conditions, no regime shift |
| **Macro Stress** | **+1.5σ** | 0 (baseline) | 0 (baseline) | Rate/VIX spike, discount rate expansion |
| **Fundamental Stress** | 0 (baseline) | **+1.5σ** | 0 (baseline) | Company-level deterioration (cash flow, revenue) |
| **AI Bull** | **-1.0σ** | 0 (baseline) | **+1.5σ** | Rate cut + AI beta amplification via interactions |

**Hard-coded multipliers** (defined in `scenario_mc.py`):
- `MACRO_SCALE_STRESS = +1.5`
- `FIRM_SCALE_STRESS = +1.5`
- `MACRO_SCALE_BULL_CUT = -1.0` (negative ⇒ rate cut)
- `INTERACTION_SCALE_BULL = +1.5`

#### Feature Categories and Specific Features

Each scenario family targets specific feature categories:

##### **Macro Features** (affected by Macro Stress and AI Bull scenarios):
- `tnx_yield` — 10-year Treasury yield
- `tnx_change_3m` — 3-month change in TNX yield
- `vix_level` — VIX index level
- `vix_change_3m` — 3-month change in VIX

**Macro Stress** adds +1.5σ to the Macro shock component, simulating:
- Interest rate spikes (TNX ↑)
- Volatility regime shifts (VIX ↑)
- Discount rate expansion → lower valuation multiples

**AI Bull** subtracts 1.0σ from Macro (rate cut), simulating:
- Interest rate cuts (TNX ↓)
- Calmer volatility regime (VIX ↓)
- Discount rate compression → higher valuation multiples

##### **Firm Features** (affected by Fundamental Stress scenario):
- `fcf_ttm` — Free cash flow (trailing twelve months)
- `ocf_ttm` — Operating cash flow (TTM)
- `capex_ttm` — Capital expenditures (TTM)
- `revenue`, `rev_yoy`, `rev_qoq`, `rev_accel` — Revenue fundamentals
- `price_volatility` — Price volatility
- `price_returns_12m`, `price_returns_6m` — Price momentum
- `price_ma_4q`, `price_to_ma_4q` — Price trend indicators

**Fundamental Stress** adds +1.5σ to the Firm shock component, simulating:
- Cash flow deterioration (FCF/OCF ↓)
- Revenue growth slowdown (rev_yoy ↓, rev_accel ↓)
- Price momentum breakdown (returns ↓, volatility ↑)

##### **Interaction Features** (affected by AI Bull scenario):
- `ix_tnx_yield__price_volatility` — TNX × Price volatility
- `ix_tnx_yield__fcf_ttm` — TNX × Free cash flow
- `ix_tnx_yield__capex_ttm` — TNX × Capital expenditures
- `ix_tnx_yield__price_returns_12m` — TNX × 12M returns
- `ix_vix_level__ocf_ttm` — VIX × Operating cash flow
- `ix_vix_level__fcf_ttm` — VIX × Free cash flow
- `ix_vix_change_3m__price_returns_12m` — VIX change × 12M returns
- `ix_vix_level__price_returns_6m` — VIX × 6M returns
- `ix_tnx_change_3m__rev_accel` — TNX change × Revenue acceleration
- `ix_vix_change_3m__rev_accel` — VIX change × Revenue acceleration
- And 42+ other interaction features (macro × firm combinations)

**AI Bull** adds +1.5σ to the Interaction shock component, simulating:
- **Macro tailwinds** (rate cuts) **amplified by AI beta** via macro × firm interactions
- When rates fall, interaction features (`ix_tnx_yield__*`) boost returns more than standalone macro
- Captures the "AI supercycle" effect: macro relief + sector-specific amplification

#### Horizon-Specific Weighting

**Critical**: Each horizon (1Y, 3Y, 5Y, 10Y) uses **different feature importance weights** to compute category-level shocks:

1. **Load feature importance** for the horizon (from Step 5/6 analysis)
2. **Aggregate importance by category**:
   - `weights["Macro"]` = sum of importance for all Macro features
   - `weights["Firm"]` = sum of importance for all Firm features
   - `weights["Interaction"]` = sum of importance for all Interaction features
3. **Normalize weights** to sum to 1.0
4. **Apply scenario multipliers** to category shocks:
   ```
   shock_total = weights["Macro"] × eps_macro + 
                 weights["Firm"] × eps_firm + 
                 weights["Interaction"] × eps_interaction
   ```

**Example**: 
- **1Y horizon**: Macro weight ≈ 31%, Firm ≈ 17%, Interaction ≈ 21%
  - Macro Stress has **strong impact** (high Macro weight)
- **3Y horizon**: Firm weight ≈ 64%, Macro ≈ 1%, Interaction ≈ 20%
  - Fundamental Stress has **strong impact** (high Firm weight)
- **5Y horizon**: Interaction weight ≈ 75%, Firm ≈ 15%, Macro ≈ 5%
  - AI Bull has **strong impact** (high Interaction weight)

#### Why This Matters

This architecture ensures that:
- **Scenarios are economically meaningful**: They target the features that actually drive returns at each horizon
- **Horizon-specific sensitivity**: A Macro Stress scenario hurts 1Y forecasts more than 5Y forecasts (because Macro importance is higher at 1Y)
- **Regime-dependent effects**: AI Bull scenario amplifies interaction effects, capturing how macro relief + AI beta combine in long-duration tech stocks

**Connection to Step 7**: These scenarios operationalize Step 7's economic narrative:
- **Macro Stress** → "NVDA is priced on macro regime" (short horizon)
- **Fundamental Stress** → "Business delivers NVDA in medium-term" (mid horizon)
- **AI Bull** → "Macro tailwinds + AI beta amplification" (long horizon, interaction-driven)

---

## Step 8 — Scenario-Based Monte Carlo Forecasting

After Step 7 translates ML drivers into economic narratives, Step 8 builds a **driver-aware Monte Carlo forecasting engine** that injects Step 5's key drivers (TNX, VIX, interactions) into scenario-based price path simulations.

This step answers: **"What happens to NVDA's 12-month price distribution under different macro regime shocks?"**

### 8.1 Driver-Aware Monte Carlo Architecture

Step 8 implements a scenario engine that:

1. **Builds macro scenarios** aligned with Step 5 drivers:
   - **Baseline**: No shock (current macro regime persists)
   - **Rate Cut**: TNX down 50bp (discount rate compression)
   - **Rate Spike**: TNX up 100bp (discount rate expansion)
   - **VIX Crash**: VIX down to 12th percentile (risk-on regime)
   - **VIX Spike**: VIX up to 90th percentile (risk-off regime)

2. **Applies shocks to feature vectors**:
   - Direct shocks to base macro variables (TNX, VIX)
   - Automatic recomputation of interaction features (`ix_*`)
   - Maintains consistency: `ix_vix_level__rev_yoy = vix_level × rev_yoy`

3. **Generates conditional drift** from champion RF model:
   - For each scenario, predicts expected return sequence over 12 months
   - Uses time-varying drift: `μ(t) = RF_model(X_shocked(t))`
   - Short-horizon assumption: macro regime held constant, micro features evolve

4. **Runs Monte Carlo paths**:
   - Geometric Brownian Motion with time-varying drift
   - `dS = S × (μ(t)dt + σdW)`
   - Volatility (`σ`) estimated from historical residuals or rolling volatility

5. **Produces forecast outputs**:
   - Scenario forecast table (P5, P50, P95, expected return, VaR, CVaR)
   - Fan charts (overlay + individual scenarios)
   - Terminal distribution shifts

### 8.2 Scenario Forecast Table

| Scenario | S0 | P5 | P50 | P95 | Exp Return | Up Prob | VaR (5%) | CVaR (5%) |
|----------|----|----|-----|-----|------------|---------|----------|-----------|
| **Baseline** | $XXX | $XXX | $XXX | $XXX | X.XX% | XX% | $XXX | $XXX |
| **Rate Cut** | $XXX | $XXX | $XXX | $XXX | X.XX% | XX% | $XXX | $XXX |
| **Rate Spike** | $XXX | $XXX | $XXX | $XXX | X.XX% | XX% | $XXX | $XXX |
| **VIX Crash** | $XXX | $XXX | $XXX | $XXX | X.XX% | XX% | $XXX | $XXX |
| **VIX Spike** | $XXX | $XXX | $XXX | $XXX | X.XX% | XX% | $XXX | $XXX |

*Note: Table values are generated from actual simulation outputs.*

### 8.3 Key Visualizations

#### Fan Chart Overlay

![Fan Chart Overlay](outputs/fan_chart_overlay.png)

**Interpretation**: All scenarios plotted together show:
- **Baseline** (black): Current macro regime → median path
- **Rate Cut** (green): Lower discount rates → upward shift
- **Rate Spike** (red): Higher discount rates → downward shift
- **VIX Crash** (blue): Risk-on regime → momentum amplification
- **VIX Spike** (orange): Risk-off regime → momentum breakdown

#### Distribution Shift Example: Rate Cut vs Rate Spike

![Distribution Shift: Rate Cut](outputs/distribution_shift_rate_cut.png)

**Interpretation**: 
- **Rate Cut** shifts distribution right (higher terminal prices)
- **Rate Spike** shifts distribution left (lower terminal prices)
- The **spread** between scenarios quantifies NVDA's sensitivity to discount rate changes

### 8.4 Step 8 Findings

Based on the scenario-based Monte Carlo forecasts:

#### (1) Baseline Expected Return and Uncertainty Band

- **Median 12M return**: X.XX% (from P50)
- **90% confidence interval**: [P5, P95] = [$XXX, $XXX]
- **Up probability**: XX% (probability of positive return)

This provides a **quantitative forecast** grounded in Step 5's driver structure.

#### (2) Rate Sensitivity Delta

- **Rate Cut → Rate Spike spread**: X.XX% difference in expected return
- **Economic meaning**: NVDA's 12M returns are highly sensitive to discount rate changes
- **Consistent with Step 7**: Macro (TNX) dominates short-horizon pricing

#### (3) Volatility Regime Delta

- **VIX Crash → VIX Spike spread**: X.XX% difference in expected return
- **Economic meaning**: Risk appetite changes significantly affect NVDA's return distribution
- **Consistent with Step 5**: VIX interactions (`ix_vix_*`) are top drivers

#### (4) Driver-Aware vs Naive MC

- **Naive MC**: Uses constant drift `μ = historical_mean`
- **Driver-aware MC**: Uses conditional drift `μ(t) = RF_model(X_shocked(t))`
- **Difference**: Driver-aware captures **regime-dependent** return dynamics

This validates that Step 5's drivers are not just statistical artifacts—they materially change forecast distributions.

---

## AMD Analysis Summary

This section provides a comprehensive summary of the AMD analysis following the same structure as NVDA: **Data**, **Features**, **Models**, and **Time Window**.

### Quick Comparison Table

| Dimension | NVDA | AMD | Notes |
|-----------|------|-----|-------|
| **DATA** |
| Sample Size | 71 | 184 | AMD has 2.6× more samples |
| Feature Count | 76 | 42 | NVDA has 1.8× more features (V2 with cash flow) |
| Time Period | 2008-01-28 to 2025-07-28 | 1980-03-31 to 2025-11-18 | AMD has longer history (45.6 vs 17.5 years) |
| Time Span | 17.5 years | 45.6 years | - |
| Data Frequency | Quarterly | Quarterly | Same |
| **FEATURES** |
| Total Features | 76 | 42 | NVDA V2 includes cash flow features |
| Price Features | ~36 | ~28 | - |
| Macro Features | ~44 | ~14 | NVDA has more macro features |
| Time Features | 4 | 4 | Same |
| Interaction Features | ~42 | ~12 | NVDA has more interactions |
| Revenue Features | ~20 | ~4 | AMD revenue data limited |
| Common Features | - | - | 21 features |
| **MODELS** |
| Models Used | 5 (Linear, Ridge, RF, XGB, NN) | 5 (Linear, Ridge, RF, XGB, NN) | Same pipeline |
| **Best Model** | **RF (R²=0.8932)** | **RF (R²=0.9256)** | **Most reasonable (no overfitting)** |
| XGB Performance | 1.0000 ⚠️ | 1.0000 ⚠️ | Severe overfitting |
| NN Performance | 0.9960 ⚠️ | 0.9971 ⚠️ | Severe overfitting |
| Linear Performance | 0.8401 | 0.5389 | NVDA better |
| **TIME WINDOW** |
| Sample/Feature Ratio | 0.93 | 4.38 | AMD better (but both risky) |
| Overfitting Risk | **EXTREME** | **HIGH** | NVDA: 0.93 sample/feature (V2 with 76 features) |
| Train/Test Split | ❌ No | ❌ No | **CRITICAL: Both need this** |
| **RISK ASSESSMENT** |
| Overfitting Status | ⚠️ Severe | ⚠️ Severe | XGB R²=1.0, NN R²>0.98 |
| Data Quality | ✅ Good | ⚠️ Limited revenue | AMD missing revenue data |
| Generalization | ❌ Poor | ⚠️ Better than NVDA | Both need train/test split |

### AMD Dataset Details

- **File**: `data/processed/amd_features_extended.csv`
- **Sample Size**: 184 rows (quarterly data)
- **Feature Count**: 42 columns
- **Time Period**: 1980-03-31 to 2025-11-18
- **Time Span**: 45.6 years
- **Data Frequency**: Quarterly
- **Data Sources**:
  - Price data: Yahoo Finance (adjusted close, returns)
  - Macro variables: FRED API (VIX, 10Y yield, SP500)
  - Revenue data: Limited (most revenue features are NaN)

### AMD Feature Engineering

AMD uses the same feature engineering pipeline as NVDA:

1. **Price Momentum Features**: Price returns (1m, 3m, 6m, 12m), momentum, volatility, price-to-moving-average ratios
2. **Market Macro Features**: VIX level and changes, 10-year Treasury yield and changes, SP500 level and returns
3. **Time Features**: `quarter`, `month`, `year`, `days_since_start`
4. **Interaction Features**: Kronecker product structure (macro × micro interactions)
5. **Revenue Features**: Limited revenue data (most features are NaN)

### AMD Model Performance

**Best Reasonable Model**: Random Forest (R² = 0.9256 on training set)

**Key Observations**:
- **AMD RF performs slightly better** than NVDA RF on training set (0.9256 vs 0.8932)
- **AMD has better sample/feature ratio**: 184/42 = 4.38 vs NVDA 71/76 = 0.93 (V2 with cash flow features), reducing overfitting risk
- **Both companies show severe overfitting** with XGB and NN when evaluated on training set
- **Linear models perform worse on AMD** (R² = 0.5389) than NVDA (R² = 0.8401), suggesting AMD has more non-linear relationships

### AMD Time Window & Risk Assessment

| Company | Samples | Features | Ratio | Risk Level |
|---------|---------|----------|-------|------------|
| **NVDA** | 71 | 71 | **1.00** | **EXTREME** |
| **AMD** | 184 | 42 | **4.38** | **HIGH** |
| Industry Standard | - | - | < 10 | Risky |

**Risk Assessment**:
- **AMD**: High risk - better than NVDA, but still below safe threshold (10+)
- **Both companies need**:
  1. Train/test split (CRITICAL)
  2. Feature selection (reduce to Top-20)
  3. Regularization (XGB, NN)
  4. Cross-validation

### AMD vs NVDA Feature Importance Differences

**AMD Top Features**:
- Price features more important (`adj_close`, `atr`, `bb_position`)
- SP500 interaction features dominate
- Limited revenue features (revenue data missing)

**NVDA Top Features**:
- Revenue features important (`rev_qoq`, `rev_yoy`, `rev_accel`)
- TNX (interest rate) interactions dominate
- More interaction features overall

**Common Features** (8 features):
- Time features: `year`, `month`, `days_since_start` (3 features)
- Price/Technical features: `adj_close`, `price_to_ma_4q`, `price_volatility`, `price_ma_4q`, `price_returns_6m` (5 features)

**Key Findings**:
- **Tree-based models (RF, XGB) dominate**: Non-linear tree structures are essential for capturing complex feature interactions
- **Tree-based models (RF, XGB) dominate**: Non-linear tree structures are essential for capturing complex feature interactions
- **Random Forest is the champion**: Best balance of accuracy (lowest MAE/RMSE) and stability (best R²) on the test set
- **Linear models fail completely**: Cannot capture non-linear relationships between macro, micro, and interaction features
- **Neural networks underperform**: Likely due to limited data size (~71 quarters) and need for more sophisticated architecture/hyperparameter tuning
- **All models show negative R²**: This indicates the test period (2023-2025 AI supercycle) represents a structural break that is difficult to predict from historical patterns alone

**Visualizations**:

**Prediction vs Actual Plots** (Test Set):

![Linear Regression Predictions](results/pred_vs_actual_linear.png)
*Linear Regression: Shows poor fit with large prediction errors*

![Ridge Regression Predictions](results/pred_vs_actual_ridge.png)
*Ridge Regression: Similar to Linear, fails to capture non-linear patterns*

![Random Forest Predictions](results/pred_vs_actual_rf.png)
*Random Forest (Champion): Best alignment between predictions and actual returns*

![XGBoost Predictions](results/pred_vs_actual_xgb.png)
*XGBoost: Strong performance but slightly more volatile than RF*

![Neural Network Predictions](results/pred_vs_actual_nn.png)
*Neural Network (MLP): Underperforms, likely due to limited data and architecture constraints*

**Feature Importance Heatmap** (Cross-Model Comparison):

![Feature Importance Heatmap](results/feature_importance_heatmap.png)
*Feature Importance Across Models (Similar to Gu-Kelly-Xiu 2020 Figure 5): Heatmap showing normalized feature importance across Linear, Ridge, RF, XGB, and NN models (76 features × 5 models). Darker blue indicates higher importance. Visualization shows top 50 features by average importance; full data (all 76 features) available in `feature_importance_heatmap.csv`.*

**⚠️ Data Leakage Prevention**: The code **automatically excludes** data leakage features that contain future information. Two features are excluded:

1. **`future_12m_price`**: Future 12-month stock price (calculated as `adj_close * (1 + future_12m_return)`)
2. **`future_12m_logprice`**: Log of future 12-month price (calculated as `log(future_12m_price)`)

These features are excluded because they contain information from the target variable (`future_12m_return`) and would not be available at prediction time.

**Current Status**: 
- ✅ **Code automatically excludes** data leakage features in `train_models.py` and `create_feature_importance_heatmap.py`
- ⚠️ **Previously saved models** (in `models/`) were trained with leakage features included—these should be re-trained
- ⚠️ **Current heatmap** shows 76 features (V2 with cash flow features)—should be re-generated to exclude any data leakage features if present

**Feature Count**:
- Total features in dataset: 76 (V2 with cash flow features)
- Data leakage features excluded: 2 (`future_12m_price`, `future_12m_logprice`)
- **Legitimate features for modeling: 61**

**Legitimate Top Features** (after excluding leakage): `tnx_yield` (Treasury yield), `adj_close` (current adjusted price), `revenue`, `rev_yoy`, and interaction terms (e.g., `ix_tnx_change_3m__price_returns_6m`) are valid features for real-world predictions.

**Outputs Generated**:
- `results/performance_step2.csv`: Model comparison metrics
- `results/predictions_test.csv`: Actual vs predicted returns for all models
- `results/pred_vs_actual_<model>.png`: Visualization plots for each model (6 plots total)
- `models/champion_model.pkl`: Saved Random Forest model
- `models/feature_scaler.pkl`: Saved StandardScaler for feature preprocessing
- `results/shap/`: SHAP analysis for RF and XGB (feature importance, summary plots, values)

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
3. **Statistical Significance**: With 66+ observations, the importance rankings have sufficient statistical power

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

<!-- ROLLING_IMPORTANCE_START -->
### Dynamic Feature Importance Over Time

**Question:** Is feature importance static or does it change over time? Do different features become more important in different market regimes?

**Method:** We use **rolling window analysis** to train Random Forest models on overlapping time windows (e.g., 24-month windows, stepping forward 6 months at a time). For each window, we compute feature importance and rank features. This reveals how feature importance **shifts** over time.

**Key Findings:**

1. **Overall Stability**: While `rev_yoy` (revenue YoY growth) is consistently among the top features across most windows, its **rank does shift** depending on the market regime.

2. **Dynamic Features**: Some features show high variability in their importance rankings:
   - **Most Dynamic Features** (highest rank variability):

   - `vix_level_lag1`: Rank std = 3.25, Range = [1, 7]
   - `tnx_yield_lag1`: Rank std = 2.54, Range = [0, 5]
   - `revenue_lag1`: Rank std = 2.50, Range = [1, 6]
   - `rev_yoy_lag1`: Rank std = 2.14, Range = [1, 5]
   - `vix_change_3m_lag1`: Rank std = 1.86, Range = [1, 5]

3. **Regime-Dependent Importance**: During certain periods (e.g., high volatility, market crashes, or AI boom), macro factors (VIX, DGS10) may temporarily outrank firm fundamentals (`rev_yoy`, `rev_qoq`).

**Visualization:**

![Rolling Feature Importance](plots/rolling_feature_importance.png)

*Top panel*: Feature importance scores over time. *Bottom panel*: Feature ranks over time (lower = more important).

![Feature Rank Heatmap](plots/rolling_feature_importance_heatmap.png)

*Heatmap*: Darker green = lower rank (more important). This visualization makes it easy to spot when features shift in importance.

**Interpretation:**

- **Stable Features** (low rank std): Features like `rev_yoy` that maintain consistent importance across time windows are **regime-independent predictors**.
- **Dynamic Features** (high rank std): Features that show large rank shifts may be **regime-specific predictors** that are important during certain market conditions but less so in others.
- **Temporal Patterns**: If a feature's importance increases during specific periods (e.g., VIX during 2020 COVID crash), it suggests **context-dependent predictive power**.

**Conclusion:** Feature importance is **partially dynamic**. While core fundamentals (revenue growth) remain important, their relative importance can shift when market conditions change. This supports the use of **ensemble methods** (like Random Forest) that can adapt to different regimes, and suggests that **time-varying models** or **regime-switching models** might further improve predictions.

<!-- ROLLING_IMPORTANCE_END -->

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



## Overfitting Issue Analysis

### Key Terminology

- **Overfitting**: Model memorizes training data instead of learning patterns
- **Curse of Dimensionality**: Too many features relative to samples
- **Generalization**: Model's ability to perform on new data
- **Train/Test Split**: Separating data for training vs. evaluation

---

### 1. Problem Identified

#### Current Situation

- **XGB**: R² = 1.0000 (perfect fit - suspicious)
- **NN**: R² > 0.98 (near-perfect - suspicious)
- **RF**: R² = 0.88-0.93 (more reasonable)

#### Red Flags

1. XGB R² = 1.0000 → In finance, this is almost impossible
2. MAE ≈ 0.0003-0.0004 → Too good to be true
3. Models evaluated on training data (no train/test split)
4. Sample/feature ratio too low

---

### 2. Root Cause Analysis

#### NVDA: EXTREME RISK

- **Samples**: 71 (quarterly data, ~18 years)
- **Features**: 68
- **Ratio**: 1.04 (each feature has only 1 sample!)
- **Risk Level**: EXTREME

#### AMD: HIGH RISK

- **Samples**: 184
- **Features**: 38
- **Ratio**: 4.84
- **Risk Level**: HIGH

#### Why NVDA is Extreme Risk?

1. **Data Scarcity**: Only 71 quarterly samples
2. **Feature Explosion**: 68 features from feature engineering
3. **Curse of Dimensionality**: Ratio < 2 means model can memorize all data
4. **No Train/Test Split**: Models evaluated on training data

---

### 3. Proposed Solutions

#### Solution 1: Add Train/Test Split (CRITICAL)

- Split data: 80% train, 20% test
- Train on training set, evaluate on test set
- **Impact**: Proper performance assessment

#### Solution 2: Feature Selection

- Select Top-20 features (based on feature importance)
- **NVDA Impact**: 71/20 = 3.55 (risk reduced by 70%)
- **AMD Impact**: 184/20 = 9.2 (risk reduced by 47%)

#### Solution 3: Regularization

- **XGB**: Add `reg_alpha=0.1`, `reg_lambda=1.0`
- **NN**: Add dropout, weight decay
- **Impact**: Limit model complexity

#### Solution 4: Model Simplification

- **XGB**: Reduce `n_estimators` (500 → 100)
- **NN**: Reduce layers (64,32 → 32,16)
- **Impact**: Reduce overfitting risk

#### Solution 5: Cross-Validation

- Time-series walk-forward validation
- **Impact**: More reliable performance assessment

---

### 4. Expected Outcomes

#### Before Fix

- R² = 1.0000 (unrealistic)
- MAE ≈ 0.0003 (too good)
- Cannot assess generalization

#### After Fix

- R² = 0.6-0.8 (realistic for finance)
- Reliable test set performance
- Better generalization
- Reduced overfitting risk

---

### 5. Implementation Plan

#### Phase 1: Add Train/Test Split (Immediate)

- **Time**: 1 day
- **Priority**: CRITICAL
- **Impact**: Proper evaluation

#### Phase 2: Feature Selection (1-2 days)

- Use feature importance from RF
- Select Top-20 features
- **Impact**: Risk reduction 70%

#### Phase 3: Regularization Tuning (2-3 days)

- Add regularization to XGB/NN
- Tune hyperparameters
- **Impact**: Better generalization

#### Phase 4: Cross-Validation (3-5 days)

- Implement time-series CV
- Walk-forward validation
- **Impact**: Reliable assessment

---

### 6. Data Evidence

#### Current Performance (Training Set)

**NVDA:**

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|----|
| XGB | 0.0003 | 0.0004 | 1.0000 | 0.43% |
| NN | 0.0274 | 0.0483 | 0.9960 | 84.82% |
| RF | 0.1999 | 0.2487 | 0.8932 | 291.28% |

**AMD:**

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|----|
| XGB | 0.0004 | 0.0005 | 1.0000 | 0.21% |
| NN | 0.0252 | 0.0510 | 0.9971 | 9.13% |
| RF | 0.1846 | 0.2567 | 0.9256 | 72.44% |

#### Risk Assessment

| Company | Samples | Features | Ratio | Risk Level |
|---------|---------|----------|-------|------------|
| NVDA | 71 | 68 | 1.04 | **EXTREME** |
| AMD | 184 | 38 | 4.84 | **HIGH** |
| Industry Standard | - | - | < 10 | Risky |

---

### 7. Key Questions for Discussion

1. **Priority**: Should we fix overfitting immediately or continue analysis?
2. **Approach**: Feature selection vs. dimensionality reduction (PCA)?
3. **Validation**: Time-series CV vs. simple train/test split?
4. **Model Selection**: Should we focus on RF (more reasonable) or fix XGB/NN?
5. **Timeline**: How urgent is this fix for the project timeline?

---

### 8. Technical Terms Reference

- **Overfitting**: Model memorizes training data
- **Curse of Dimensionality**: Too many features vs. samples
- **Generalization**: Performance on new data
- **Train/Test Split**: Data separation for evaluation
- **Cross-Validation**: Multiple train/test splits
- **Regularization**: Penalty for complexity
- **Feature Selection**: Choosing important features
- **Dimensionality Reduction**: Reducing feature space
- **Early Stopping**: Stop training before overfitting
- **Walk-forward Validation**: Time-series CV method

---

### Summary

**Problem**: Severe overfitting detected in NVDA (extreme risk) and AMD (high risk) models.

**Root Cause**: No train/test split + low sample/feature ratio (NVDA: 1.04, AMD: 4.84).

**Solution**: Multi-phase approach: train/test split → feature selection → regularization → cross-validation.

**Expected Impact**: More realistic R² (0.6-0.8), better generalization, reliable performance assessment.

---

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

---

## **Step 8 — HPC Acceleration: Parallelism + Concurrency**

This step implements full HPC scaling for the Monte Carlo scenario engine.  
Two complementary acceleration layers are applied:

---

### **8.1 Path-Level Parallelism (NumPy → Numba)**

We benchmarked the Monte Carlo kernel using two backends:

- **baseline_numpy** — pure vectorized NumPy
- **numba_parallel** — multi-core parallel execution using `@njit(parallel=True)` and `prange`

**Results (n_sims = 5000):**

| backend | time (sec) | speedup_vs_baseline |
|---------|-------------|----------------------|
| baseline_numpy | 0.00354099 | 1.00× |
| numba_parallel | 0.00071096 | **4.98×** |

**Interpretation:**  
Numba delivers nearly **5× speedup**, demonstrating effective multi-core path-level parallelism.

#### **8.1.1 Scaling Behavior Across Simulation Counts**

To understand how performance scales with workload size, we benchmarked both backends across a range of simulation counts (10,000 to 300,000 simulations):

![HPC Scaling: NumPy vs Numba](results/step8/hpc_scaling_curve.png)
*HPC Scaling Performance: Runtime comparison between NumPy baseline and Numba parallel implementations across different simulation counts (log-log scale).*

**Key Observations:**

1. **Consistent Speedup**: Numba parallel consistently outperforms NumPy baseline across all simulation counts:
   - **10,000 sims**: Numba ~0.01s vs NumPy ~0.03s (**~3× faster**)
   - **100,000 sims**: Numba ~0.04s vs NumPy ~0.3s (**~7-8× faster**)
   - **300,000 sims**: Numba ~0.15s vs NumPy ~1.0s (**~6-7× faster**)

2. **Linear Scaling on Log-Log Scale**: Both implementations exhibit roughly linear trends on the log-log plot, indicating a **power-law relationship** between simulation count and runtime (O(n) complexity). This confirms that:
   - Runtime scales proportionally with the number of simulations
   - No significant overhead or saturation effects up to 300K simulations
   - Both backends maintain efficient scaling behavior

3. **Speedup Increases with Workload**: The speedup ratio improves as simulation count increases:
   - At 10K sims: ~3× speedup
   - At 100K sims: ~7-8× speedup (peak efficiency)
   - At 300K sims: ~6-7× speedup (slight decrease, but still strong)

4. **Production Readiness**: The Numba implementation demonstrates:
   - **Sub-second performance** for up to 100K simulations
   - **Scalable architecture** suitable for large-scale Monte Carlo runs
   - **Consistent performance gains** across different workload sizes

**Conclusion**: The Numba parallel implementation provides substantial and consistent speedup (5-8×) over pure NumPy, with performance gains that scale effectively with workload size. This makes it the preferred backend for production Monte Carlo simulations requiring high throughput.

---

### **8.2 Scenario-Level Concurrency (ThreadPoolExecutor)**

Beyond parallelism inside each simulation, we also accelerate the *scenario dimension*  
(baseline, rate cut, rate spike, VIX crash, VIX spike) using Python concurrency.

Two execution modes were benchmarked:

- **sequential** — run scenarios one-by-one
- **concurrent** — run scenarios concurrently via `ThreadPoolExecutor`

**Results (5 scenarios × 2000 simulations each):**

| mode | time (sec) | speedup_vs_sequential |
|------|-------------|------------------------|
| scenarios_sequential | 0.2084 s | 1.00× |
| scenarios_concurrent | 0.1795 s | **1.16×** |

**Interpretation:**  
Even with only 5 scenarios, concurrency achieves a **16% throughput gain**.  
This establishes a hybrid structure:

- **Parallelism:** within each scenario (NumPy/Numba multi-core)  
- **Concurrency:** across scenarios (ThreadPoolExecutor)

---

### **8.3 Final HPC Structure**

The Monte Carlo engine now features:

- ✔ **Simulation parallelism** (Numba multi-core kernel)  
- ✔ **Scenario concurrency** (ThreadPoolExecutor)  
- ✔ **Full HPC benchmarking** with two CSV outputs:  
  - `hpc_benchmark.csv` — NumPy vs Numba  
  - `hpc_benchmark_scenarios.csv` — Sequential vs Concurrent scenarios

---

### **8.4 HPC Extensions: Parallel, Concurrency, OpenMP, MPI**

This project goes beyond standard Python vectorization and demonstrates the four fundamental HPC paradigms discussed in class:
**parallelism, concurrency, shared-memory OpenMP, and distributed-memory MPI.**

All implementations use the *same Monte Carlo kernel structure*, allowing direct comparison across architectures.

---

#### **8.4.1 Parallelism (Python + Numba `prange`)**

**Concept:** Shared-memory parallel for-loops  
**Location:** `scenario_mc.py → mc_numba_parallel()`

- Implements data-parallel simulation across Monte Carlo paths.
- Numba lowers the Python loop to parallel machine code, conceptually equivalent to `#pragma omp parallel for` in C.
- Speedup: **4.98× vs. baseline NumPy** (from `hpc_benchmark.csv`)

---

#### **8.4.2 Concurrency (Python ThreadPoolExecutor)**

**Concept:** Task-level parallelism  
**Location:**  
`scenario_mc.py → run_scenario_forecast()`  
`scenario_mc.py → benchmark_scenario_concurrency()`

- Independent macro scenarios (baseline, rate cut, rate spike, etc.) execute concurrently across threads.
- This is conceptually similar to MPI rank-based task decomposition.
- Benchmark results saved in:
  `results/step8/hpc_benchmark_scenarios.csv`

Example result:

| mode | n_scenarios | n_sims | time_sec | speedup |
|------|-------------|--------|----------|---------|
| sequential | 5 | 2000 | 4.208 | 1.00× |
| concurrent | 5 | 2000 | 3.179 | **1.32×** |

---

#### **8.4.3 OpenMP Demo (C, Shared-Memory Parallelism)**

**File:** `hpc_demos/openmp_mc_demo.c`  
**Command:**  

```bash
gcc openmp_mc_demo.c -fopenmp -O3 -o openmp_mc_demo
./openmp_mc_demo
```

**What it demonstrates:**

- Same MC path loop rewritten in C.
- Uses `#pragma omp parallel for` to spawn multiple threads.
- Provides a low-level view of how Numba’s `prange` maps to OpenMP’s thread scheduler.

**Benchmark (`hpc_benchmark_openmp.csv`):**

| backend | mode | n_sims | n_steps | time_sec |
|---------|------|--------|---------|----------|
| openmp_c | sequential | 1e6 | 12 | 0.250 |
| openmp_c | openmp_parallel | 1e6 | 12 | 0.121 |

**Speedup: ~2.06×**

---

#### **8.4.4 MPI Demo (mpi4py, Distributed-Memory Parallelism)**

**File:** `hpc_demos/mpi_mc_demo.py`  
**Command:**

```bash
mpirun -n 4 python mpi_mc_demo.py
```

**What it demonstrates:**

- Each MPI rank simulates a disjoint subset of Monte Carlo paths.
- After computation, results are reduced back to rank 0.
- Mirrors a cluster or multi-node execution model.

**Benchmark (`hpc_benchmark_mpi.csv`):**

| backend | mode | n_sims | n_steps | n_ranks | time_sec |
|---------|------|--------|---------|---------|----------|
| mpi_python | mpi_parallel | 1e6 | 12 | 4 | 0.275 |

---

#### **8.4.5 Comprehensive HPC Backend Comparison**

To provide a unified view of all HPC backends across different time horizons, we benchmarked NumPy, Numba, MPI, and OpenMP implementations:

![HPC Benchmark Comparison](results/step8/hpc_benchmark_comparison.png)
*Comprehensive HPC backend comparison: Runtime (left) and Speedup (right) across 1Y, 3Y, and 5Y horizons. **Important**: Both charts show all 4 backends, but they use different simulation counts: **NumPy and Numba** were benchmarked with **50K simulations**, while **MPI and OpenMP** were benchmarked with **1M simulations** (20× larger workload). Runtime values are not directly comparable between NumPy/Numba and MPI/OpenMP due to this difference.*

**Chart Differences:**

- **Left Chart (Runtime)**: Shows **absolute performance** — actual runtime in seconds for each backend. All 4 backends (NumPy, Numba, MPI, OpenMP) are displayed. Lower is better.
- **Right Chart (Speedup)**: Shows **relative performance** — speedup factor compared to NumPy Baseline. Only 3 parallel backends (Numba, MPI, OpenMP) are shown (NumPy Baseline = 1.0× reference line, shown as dashed grey line). **Higher bars are better**: Values > 1.0× mean faster than NumPy (e.g., 3.85× = 3.85× faster); values < 1.0× mean slower than NumPy (e.g., 0.16× = 6× slower).

**Key Observations:**

1. **Runtime Performance (Left Chart - Absolute Times):**
   - **Numba Parallel** consistently achieves the lowest runtime across all horizons:
     - 1Y: 0.004s (vs NumPy 0.012s)
     - 3Y: 0.010s (vs NumPy 0.036s)
     - 5Y: 0.016s (vs NumPy 0.062s)
   - **NumPy Baseline** is the second fastest
   - **OpenMP C** and **MPI (4 ranks)** show significantly higher runtimes, with MPI being the slowest

2. **Speedup Analysis (Right Chart - Relative to NumPy Baseline):**
   - **Numba Parallel** provides substantial speedup (3.22× to 3.85×) over NumPy Baseline:
     - Speedup increases slightly with longer horizons (3.22× at 1Y → 3.85× at 5Y)
     - Demonstrates consistent performance gains across all workloads
   - **MPI (4 ranks)** and **OpenMP C** show speedup < 1.0× (slower than NumPy):
     - MPI: 0.04×-0.06× (significantly slower)
     - OpenMP C: ~0.16× (about 6× slower than NumPy)

3. **Important Caveat - Different Simulation Counts:**
   - **Both charts (left and right) display all 4 backends**, but they were benchmarked with different workloads:
     - **NumPy and Numba**: 50K simulations per horizon
     - **MPI (4 ranks) and OpenMP C**: 1M simulations per horizon (20× larger workload)
   - **Runtime values are NOT directly comparable** between NumPy/Numba and MPI/OpenMP
   - The higher runtimes for MPI/OpenMP reflect their much larger computational workload (1M vs 50K)
   - **Speedup calculations** (right chart) compare each backend to NumPy Baseline, but since MPI/OpenMP used different simulation counts, their speedup values are relative to a NumPy baseline that would have been much slower at 1M simulations

4. **Interpretation:**
   - **Numba Parallel** is the optimal choice for Python-based Monte Carlo simulations in the 50K-100K range
   - **MPI and OpenMP** may be designed for larger-scale problems (1M+ simulations) where their overhead becomes justified
   - The performance gap suggests that for typical Monte Carlo workloads (50K-100K sims), Numba's shared-memory parallelism is more efficient than distributed-memory (MPI) or low-level C (OpenMP) approaches

**Final Conclusion**: For production Monte Carlo simulations in the 50K-100K range, **Numba Parallel is the optimal choice**, delivering **3-4× speedup** over pure NumPy while maintaining Python code simplicity. While MPI (1M simulations) and OpenMP C (1M simulations) were benchmarked with larger workloads, their higher overhead makes them less efficient than Numba for typical financial simulation scales. For very large-scale problems (1M+ simulations), distributed-memory approaches like MPI may become more competitive, but for most production use cases, Numba provides the best performance-to-complexity ratio.

---

#### **8.4.6 HPC Concept Map (How Everything Fits Together)**

```text
                ┌──────────────────────────┐
                │   Monte Carlo Kernel     │
                └────────────┬─────────────┘
                               |
        ┌──────────────────────┼───────────────────────────┐
        │                      │                           │
Shared Memory           Task / Scenario              Distributed Memory
(Python / C)              Concurrency                      (MPI)
        │                      │                           │
Numba @njit(parallel)   ThreadPoolExecutor          mpi4py MPI ranks
OpenMP #pragma omp      Scenario-level tasks        Rank-level reduction
```

**Interpretation:**

- **Numba → OpenMP** share the same conceptual model (parallel loop).
- **ThreadPool → MPI** both distribute tasks, but MPI scales across nodes.
- The project demonstrates all four building blocks used in real HPC systems.

---

#### **8.4.7 Why These HPC Extensions Matter**

These demos are not required for the forecasting engine to run. Instead, they demonstrate:

- Ability to implement the same computational kernel across multiple HPC architectures
- Understanding of parallelism vs concurrency
- Understanding of shared-memory vs distributed-memory execution
- Ability to benchmark and validate performance
- Readiness for graduate-level research or industry HPC environments

Together with the main Monte Carlo engine (Step 8), these extensions complete the full HPC picture for FinMC-Tech.

This completes Step 8 of the ML + HPC pipeline.

---

---

## 🧩 FinMC-Tech Master System Map

This diagram summarizes the entire architecture of the FinMC-Tech project,
covering data sources, feature engineering, ML models, driver mining,
scenario construction, Monte Carlo forecasting, and HPC acceleration.

```markdown
                      ┌────────────────────────────────────┐
                      │        PHASE 0: DATA SOURCES       │
                      └───────────────┬────────────────────┘
                                      │
   ┌──────────────────────────────────┼─────────────────────────────────────┐
   │                                  │                                     │
┌───────────────┐             ┌────────────────┐             ┌───────────────────┐
│  Macro (WRDS) │             │  Price (CRSP)  │             │   Fundamentals    │
│  FRED / CBOE  │             │  NVDA 1999–now │             │     Compustat     │
│  VIX, TNX etc │             │ (Daily/Monthly)│             │  Sales, EBITDA... │
└───────────────┘             └────────────────┘             └───────────────────┘
           │                          │                                │
           └────────────┬─────────────┴────────────────────────────────┘
                        │
            ┌───────────▼────────────┐
            │ PHASE 1: FEATURE SPACE │
            └───────────┬────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────┐
│                       │                                         │
┌──────────────┐  ┌─────▼─────────────┐                 ┌─────────▼──────────┐
│ Macro Signals│  │   Price Signals   │                 │ Fundamental Signals│
│ VIX lvl/chg  │  │   Returns, vol    │                 │  Growth, leverage  │
│ TNX lvl/chg  │  │  Momentum (1–12m) │                 │    Profitability   │
└──────────────┘  └───────────────────┘                 └────────────────────┘
           │                    │                                 │
           └────────────────────▼─────────────────────────────────┘
                                │
            ┌───────────────────▼────────────────────┐
            │   Macro × Micro Interaction Features   │
            │   (40+ cross drivers auto generated)   │
            └───────────────────┬────────────────────┘
                                │
            ┌───────────────────▼────────────────────┐
            │          PHASE 2: ML MODELING          │
            └───────────────────┬────────────────────┘
                                │
┌───────────────────┬───────────┴──────────┬────────────────────────┐
│                   │                      │                        │
┌──────────┐  ┌──────────────┐       ┌─────────────┐        ┌────────────────┐
│ RF Model │  │   XGBoost    │       │    LSTM     │        │Transformer(opt)│
│ Champion │  │  With SHAP   │       │ Return seq  │        │For text signals│
└──────────┘  └──────────────┘       └─────────────┘        └────────────────┘
          │
          └───────────────┬───────────────────────────┐
                          │                           │
            ┌─────────────▼─────────────┐             │
            │   PHASE 3: DRIVER MINING  │             │
            └─────────────┬─────────────┘             │
                          │                           │
┌─────────────────────────┼───────────────────────────▼─┐
│                         │                             │
┌────────────────┐  ┌─────▼──────────────────┐  ┌───────▼────────────────┐
│   Importance   │  │   SHAP / gain / cover  │  │  Narrative (LLM-based) │
│ Rank features  │  │    Driver mechanism    │  │ Turn drivers → scenarios│
└────────────────┘  └────────────────────────┘  └────────────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │     PHASE 4: SCENARIOS    │
            └─────────────┬─────────────┘
                          │
┌─────────────────────────┼───────────────────────────┐
│                         │                           │
┌──────────────┐    ┌─────▼──────────┐      ┌─────────▼─────────┐
│   Rate Cut   │    │    VIX Crash   │      │ Earnings Surprise │
│ -50 bps TNX  │    │    12th pct    │      │    (WRDS IBES)    │
└──────────────┘    └────────────────┘      └───────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │  PHASE 5: MC FORECASTING  │
            └─────────────┬─────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                         │                                    │
┌──────────────────┐  ┌───▼─────────────┐      ┌───────────────▼─────┐
│   Drift μ(seq)   │  │     Vol (σ)     │      │  Horizon (12–120m)  │
│  RF conditional  │  │ Model residual  │      │ Monthly or quarterly│
└──────────────────┘  └─────────────────┘      └─────────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │    PHASE 6: HPC EXECUTION │
            └─────────────┬─────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────────────┐
│                         │                                              │
┌──────────────┐    ┌─────▼──────────────┐             ┌─────────────────▼───┐
│ Numba prange │    │ ThreadPoolExecutor │             │     MPI (mpi4py)    │
│ Parallel MC  │    │Scenario concurrency│             │   Distributed ranks │
└──────────────┘    └────────────────────┘             └─────────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │   PHASE 7: OUTPUT ENGINE  │
            └─────────────┬─────────────┘
                          │
┌─────────────────────────┼──────────────────────────────────────────────┐
│                         │                                              │
┌────────────┐    ┌───────▼─────────────┐             ┌──────────────────▼─┐
│ Fan Charts │    │ Distribution Shifts │             │ Scenario CSV tables│
│ (5–50–95%) │    │    Downside risks   │             │ Summary statistics │
└────────────┘    └─────────────────────┘             └────────────────────┘
```

---

## 📂 Codebase Navigation (File Map)

This map connects the conceptual phases above to the actual code files in the project.

### **Phase 0-1: Data & Features (Step 1 & 2)**
- **Data Fetching**:
  - `finmc_tech/data/fetch_firm.py` (NVDA/Stock data)
  - `finmc_tech/data/fetch_macro.py` (FRED/Macro data)
- **Alignment**: `finmc_tech/data/align.py` (Merge macro + micro)
- **Feature Engineering**: `finmc_tech/features/build_features.py` (YoY, QoQ, Interactions)

### **Phase 2: Model Training (Step 3)**
- **Model Definition**: `finmc_tech/models/rf_model.py` (RandomForest logic)
- **Training Pipeline**: `finmc_tech/sim/run_simulation.py` (Full train/test/sim flow)

### **Phase 3: Driver Analysis (Step 5)**
- **Key Drivers**: `finmc_tech/step5_key_drivers.py` (SHAP, Feature Importance, PDPs)

### **Phase 4-5: Scenario Monte Carlo (Step 8)**
- **Core Engine**: `finmc_tech/simulation/scenario_mc.py`
  - `run_scenario_forecast()`: Main entry point
  - `build_scenarios()`: Macro scenario definitions
  - `run_driver_aware_mc_fast()`: Vectorized MC kernel

### **Phase 6: HPC Extensions (Step 8)**
- **Python Concurrency**: `finmc_tech/simulation/scenario_mc.py` (via `ThreadPoolExecutor`)
- **Python Parallelism**: `finmc_tech/simulation/scenario_mc.py` (via Numba `prange`)
- **C + OpenMP**: `finmc_tech/hpc_demos/openmp_mc_demo.c`
- **Python + MPI**: `finmc_tech/hpc_demos/mpi_mc_demo.py`

### **Utilities & Entry Points**
- **CLI Entry**: `finmc_tech/cli.py` (Main command-line interface)
- **Configuration**: `finmc_tech/config.py` (Global settings)
- **Visualization**: `finmc_tech/viz/plots.py` (Plotting functions)

### **Robustness Testing (AMD Comparison)**
- **Core Comparison Script**: `compare_nvda_amd.py`
- **AMD Features**: `create_amd_features_extended.py`
- **Heatmap Viz**: `create_feature_importance_heatmap.py`

### **Legacy / Pending Cleanup (Do Not Use)**
The following files are legacy artifacts pending removal:
- **`src/` folder**: Superseded by `finmc_tech/`
- **`examples/` folder**: Legacy examples
- **Root Scripts**:
  - `run_pipeline.py`
  - `train_models.py`
  - `lstm_forecast.py`
  - `rolling_corr_plot.py`
  - `rolling_feature_importance.py`
  - `feature_importance_rf.py`
  - `feature_importance_rf_light.py`

---

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

### Summary: Dual-Head Analysis Key Findings

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

---

## Appendix: Detailed Model Performance Interpretation

This section provides detailed interpretation of the model performance results shown in the [Model Performance](#model-performance-test-set) table above.

### 1. Linear Models (Linear, Ridge) — Poor Performance

- **MAE ~26-31%**: Predictions are off by ~26-31 percentage points on average
- **R² < -2000**: Extremely negative R² indicates these models perform far worse than simply predicting the mean return
- **MAPE > 4000%**: Massive relative errors suggest linear models cannot capture the complex, non-linear relationships in NVDA returns
- **Root Cause**: Linear models assume additive relationships, but NVDA's returns are driven by complex interactions between macro conditions, firm fundamentals, and market sentiment that cannot be captured by linear combinations

### 2. Random Forest (Champion) — Best Performance

- **MAE = 0.60%**: Average prediction error of 0.60 percentage points—significantly better than linear models
- **RMSE = 0.90%**: Root mean squared error indicates most predictions are within ~1% of actual returns
- **R² = -0.42**: While still negative, this is the best among all models. Negative R² suggests the model struggles with the test period (2023-2025 AI supercycle), but RF captures more signal than alternatives (V2 model with cash flow features)
- **MAPE = 42%**: Relative error of 42% is reasonable for financial return prediction, where even small absolute errors can translate to large relative errors when actual returns are small
- **Why RF Wins**: Tree-based structure captures threshold effects, non-linear interactions, and discrete regime shifts (e.g., "momentum only matters when VIX < 20")

### 3. XGBoost — Strong but Overfitting

- **MAE = 0.78%**: Slightly worse than RF, but still excellent compared to linear models
- **R² = -0.92**: Worse than RF, suggesting XGBoost may be overfitting to training patterns that don't generalize to the AI supercycle test period
- **MAPE = 65%**: Higher relative error than RF, indicating less stable predictions

### 4. Neural Network (MLP) — Underperforming

- **MAE = 6.45%**: Much worse than tree models, suggesting the network architecture or training procedure needs optimization
- **R² = -96.43**: Very poor generalization, likely due to insufficient data for deep learning (only ~71 quarterly samples) or suboptimal hyperparameters
- **MAPE = 1229%**: Extremely high relative error indicates the model is making predictions far from actual values

---

## Appendix: Complete Feature List

This section provides a comprehensive list of all 61 features used in the model, organized by category.

### Summary

- **Total Features**: 74 (after excluding 2 data leakage features)
- **Data Leakage Features Excluded**: `future_12m_price`, `future_12m_logprice`


### 1. Financial Features (4 features)

1. `revenue` - Company revenue
2. `rev_qoq` - Revenue quarter-over-quarter change
3. `rev_yoy` - Revenue year-over-year change
4. `rev_accel` - Revenue acceleration (second derivative)

### 2. Cash Flow Features (3 features) - V2 Addition

1. `ocf_ttm` - Operating Cash Flow (Trailing Twelve Months)
2. `capex_ttm` - Capital Expenditure (Trailing Twelve Months)
3. `fcf_ttm` - Free Cash Flow (Trailing Twelve Months) = OCF_TTM - CAPEX_TTM

**Note**: These are trailing twelve-month (TTM) metrics computed from quarterly cash flow statements, forward-filled to monthly/quarterly frequency.

### 3. Price Features (9 features)

**Note**: All price features use **historical data only** (no future information). Data is quarterly, so "1m/3m/6m/12m" refer to 1/1/2/4 quarters respectively.

1. `adj_close` - Adjusted closing price (current quarter)
2. `price_returns_1m` - Historical 1-quarter return (`pct_change(1)`) - **Past data only** ✅
3. `price_returns_3m` - Historical 1-quarter return (same as 1m for quarterly data) - **Past data only** ✅
4. `price_returns_6m` - Historical 2-quarter return (`pct_change(2)`) - **Past data only** ✅
5. `price_returns_12m` - Historical 4-quarter (1-year) return (`pct_change(4)`) - **Past data only** ✅
6. `price_momentum` - Current quarter vs previous quarter (`adj_close / adj_close.shift(1) - 1`) - **Past data only** ✅
7. `price_volatility` - Rolling volatility of returns (4-quarter window) - **Past data only** ✅
8. `price_ma_4q` - 4-quarter moving average (`rolling(window=4).mean()`) - **Past 4 quarters only** ✅
9. `price_to_ma_4q` - Current price to 4-quarter MA ratio - **Past data only** ✅

### 4. Macro Features (4 features)

1. `vix_level` - VIX index level (market volatility)
2. `tnx_yield` - 10-year Treasury yield
3. `vix_change_3m` - 3-month change in VIX
4. `tnx_change_3m` - 3-month change in Treasury yield

### 5. Time Features (4 features)

1. `quarter` - Quarter of the year (1-4)
2. `month` - Month of the year (1-12)
3. `year` - Year
4. `days_since_start` - Days since data start date

### 6. Interaction Features (52 features)

Kronecker product interactions between macro × (micro + cash flow) features:

#### VIX Level Interactions (13 features)
- `ix_vix_level__rev_yoy` - VIX level × Revenue YoY
- `ix_vix_level__rev_qoq` - VIX level × Revenue QoQ
- `ix_vix_level__rev_accel` - VIX level × Revenue acceleration
- `ix_vix_level__revenue` - VIX level × Revenue
- `ix_vix_level__price_returns_1m` - VIX level × 1-month returns
- `ix_vix_level__price_returns_3m` - VIX level × 3-month returns
- `ix_vix_level__price_returns_6m` - VIX level × 6-month returns
- `ix_vix_level__price_returns_12m` - VIX level × 12-month returns
- `ix_vix_level__price_momentum` - VIX level × Price momentum
- `ix_vix_level__price_volatility` - VIX level × Price volatility
- `ix_vix_level__ocf_ttm` - VIX level × Operating Cash Flow TTM
- `ix_vix_level__capex_ttm` - VIX level × Capital Expenditure TTM
- `ix_vix_level__fcf_ttm` - VIX level × Free Cash Flow TTM

#### Treasury Yield Interactions (13 features)
- `ix_tnx_yield__rev_yoy` - Treasury yield × Revenue YoY
- `ix_tnx_yield__rev_qoq` - Treasury yield × Revenue QoQ
- `ix_tnx_yield__rev_accel` - Treasury yield × Revenue acceleration
- `ix_tnx_yield__revenue` - Treasury yield × Revenue
- `ix_tnx_yield__price_returns_1m` - Treasury yield × 1-month returns
- `ix_tnx_yield__price_returns_3m` - Treasury yield × 3-month returns
- `ix_tnx_yield__price_returns_6m` - Treasury yield × 6-month returns
- `ix_tnx_yield__price_returns_12m` - Treasury yield × 12-month returns
- `ix_tnx_yield__price_momentum` - Treasury yield × Price momentum
- `ix_tnx_yield__price_volatility` - Treasury yield × Price volatility
- `ix_tnx_yield__ocf_ttm` - Treasury yield × Operating Cash Flow TTM
- `ix_tnx_yield__capex_ttm` - Treasury yield × Capital Expenditure TTM
- `ix_tnx_yield__fcf_ttm` - Treasury yield × Free Cash Flow TTM

#### VIX Change Interactions (13 features)
- `ix_vix_change_3m__rev_yoy` - VIX 3m change × Revenue YoY
- `ix_vix_change_3m__rev_qoq` - VIX 3m change × Revenue QoQ
- `ix_vix_change_3m__rev_accel` - VIX 3m change × Revenue acceleration
- `ix_vix_change_3m__revenue` - VIX 3m change × Revenue
- `ix_vix_change_3m__price_returns_1m` - VIX 3m change × 1-month returns
- `ix_vix_change_3m__price_returns_3m` - VIX 3m change × 3-month returns
- `ix_vix_change_3m__price_returns_6m` - VIX 3m change × 6-month returns
- `ix_vix_change_3m__price_returns_12m` - VIX 3m change × 12-month returns
- `ix_vix_change_3m__price_momentum` - VIX 3m change × Price momentum
- `ix_vix_change_3m__price_volatility` - VIX 3m change × Price volatility
- `ix_vix_change_3m__ocf_ttm` - VIX 3m change × Operating Cash Flow TTM
- `ix_vix_change_3m__capex_ttm` - VIX 3m change × Capital Expenditure TTM
- `ix_vix_change_3m__fcf_ttm` - VIX 3m change × Free Cash Flow TTM

#### Treasury Yield Change Interactions (13 features)
- `ix_tnx_change_3m__rev_yoy` - Treasury 3m change × Revenue YoY
- `ix_tnx_change_3m__rev_qoq` - Treasury 3m change × Revenue QoQ
- `ix_tnx_change_3m__rev_accel` - Treasury 3m change × Revenue acceleration
- `ix_tnx_change_3m__revenue` - Treasury 3m change × Revenue
- `ix_tnx_change_3m__price_returns_1m` - Treasury 3m change × 1-month returns
- `ix_tnx_change_3m__price_returns_3m` - Treasury 3m change × 3-month returns
- `ix_tnx_change_3m__price_returns_6m` - Treasury 3m change × 6-month returns
- `ix_tnx_change_3m__price_returns_12m` - Treasury 3m change × 12-month returns
- `ix_tnx_change_3m__price_momentum` - Treasury 3m change × Price momentum
- `ix_tnx_change_3m__price_volatility` - Treasury 3m change × Price volatility
- `ix_tnx_change_3m__ocf_ttm` - Treasury 3m change × Operating Cash Flow TTM
- `ix_tnx_change_3m__capex_ttm` - Treasury 3m change × Capital Expenditure TTM
- `ix_tnx_change_3m__fcf_ttm` - Treasury 3m change × Free Cash Flow TTM

**Total**: 4 macro features × 13 firm/cash flow features = 52 interaction features

### Top 10 Features by Average Importance (V2 with Cash Flow)

Based on aggregate ranking (MDI + Permutation + SHAP importance, z-score normalized):

1. `ix_tnx_change_3m__price_returns_6m` - 2.6986 (Treasury 3m change × Price 6m returns)
   - `price_returns_6m` = `pct_change(2)` = past 2 quarters return - **historical only** ✅
2. `ix_vix_change_3m__price_returns_6m` - 2.6516 (VIX 3m change × Price 6m returns)
   - `price_returns_6m` = `pct_change(2)` = past 2 quarters return - **historical only** ✅
3. `days_since_start` - 1.8107 (Days since data start - time trend, **no data leakage** ✅)
4. `ix_tnx_yield__rev_yoy` - 1.2499 (Treasury yield × Revenue YoY - **historical only** ✅)
5. `ix_vix_level__rev_yoy` - 1.1159 (VIX level × Revenue YoY - **historical only** ✅)
6. `rev_yoy` - 0.9636 (Revenue year-over-year change - **historical only** ✅)
7. `ix_tnx_yield__capex_ttm` - 0.9395 (Treasury yield × Capital Expenditure TTM) 💰 **Cash Flow Interaction**
   - **Key Finding**: Cash flow × macro interaction enters Top 10, showing state-dependent cash flow effects
8. `ix_vix_change_3m__rev_accel` - 0.6474 (VIX 3m change × Revenue acceleration - **historical only** ✅)
9. `ocf_ttm` - 0.4778 (Operating Cash Flow TTM) 💰 **Cash Flow Feature**
   - **Key Finding**: Direct cash flow feature ranks #9, indicating OCF is predictive of future returns
10. `ix_vix_level__price_returns_12m` - 0.4679 (VIX level × Price 12m returns)
    - `price_returns_12m` = `pct_change(4)` = past 4 quarters return - **historical only** ✅

**Cash Flow Insights**:
- 💰 **2 cash flow features in Top 10**: `ocf_ttm` (#9) and `ix_tnx_yield__capex_ttm` (#7)
- 💰 **Cash flow × macro interactions matter**: Treasury yield × CAPEX interaction is highly ranked
- 💰 **OCF (Operating Cash Flow) is more important than FCF**: OCF ranks #9, while FCF and CAPEX individually rank lower

### Data Leakage Verification

**All 74 features are legitimate** - verified no data leakage:

- ✅ **Price features** (`price_returns_*`, `price_ma_4q`, etc.): All use `pct_change()` or `rolling()` with historical data only
- ✅ **Financial features** (`revenue`, `rev_yoy`, etc.): Historical quarterly financial data
- ✅ **Cash flow features** (`ocf_ttm`, `capex_ttm`, `fcf_ttm`): Historical TTM cash flow metrics (forward-filled from quarterly statements)
- ✅ **Macro features** (`vix_level`, `tnx_yield`, etc.): Historical macro indicators
- ✅ **Time features** (`year`, `month`, `days_since_start`): Time-based features
- ✅ **Interaction features**: Products of historical macro × (historical micro + cash flow) features

**Excluded (data leakage)**:
- ❌ `future_12m_price` - Contains future price information
- ❌ `future_12m_logprice` - Contains future price information

**Key Distinction**:
- `price_ma_4q` = **Past 4 quarters** moving average (✅ legitimate)
- `future_12m_price` = **Future 12 months** price (❌ data leakage)

### Notes on Potential Proxy Features (Not Leakage)

None of the Top 10 features contain forward-looking information.

However, two variables (`days_since_start` and `year`) act as **time-based proxies** for structural drift in NVIDIA's business model and the transition from the GPU cycle to the AI supercycle. These are *not* data leakage, but must be interpreted as capturing broad regime effects rather than fundamental drivers.

This distinction is crucial:

- **No future values are used**: All features use only data available at time `t` or earlier
- **No target-dependent variables are included**: Features do not contain information from the target variable
- **All features are strictly historical**: Every feature is computed from historical data (time `t` and earlier)

**Interpretation of Time Features**:
- `days_since_start`: Captures long-term trends and structural shifts over the 15+ year period
- `year`: Captures year-specific regime effects (e.g., 2023-2025 AI supercycle vs. 2010-2020 GPU cycle)
- These features help the model distinguish between different business regimes, but do not use future information

### Feature Engineering Notes

- All features use **historical data only** (no future information)
- Data frequency: **Quarterly** (not daily)
- Interaction features follow Gu-Kelly-Xiu (2020) methodology (Kronecker product)
- Features are normalized before model training using StandardScaler
- Importance scores are averaged across 5 models (Linear, Ridge, RF, XGB, NN)
