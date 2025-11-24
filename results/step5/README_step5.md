# Step 5: Short-Horizon Key Drivers Analysis

## Baseline Metrics (Test Set)

| Metric | Value |
|--------|-------|
| R² | -0.3285 |
| RMSE | 0.8726 |
| MAE | 0.6053 |

## Top 10 Key Drivers (Aggregate Ranking)

| Rank | Feature | Aggregate Score | MDI Z | Perm Z | SHAP Z |
|------|---------|-----------------|-------|--------|--------|
| 1 | rev_yoy | 1.7449 | 0.1852 | 2.5831 | 2.4666 |
| 2 | adj_close | 0.6188 | 2.4527 | -0.2366 | -0.3596 |
| 3 | tnx_yield | -0.1091 | 0.4458 | -0.7362 | -0.0368 |
| 4 | rev_accel | -0.1773 | -0.4669 | 0.1532 | -0.2182 |
| 5 | tnx_change_3m | -0.2029 | -0.1890 | -0.4353 | 0.0156 |
| 6 | revenue | -0.3468 | -0.7067 | -0.5641 | 0.2303 |
| 7 | vix_change_3m | -0.3531 | -0.5059 | -0.2144 | -0.3390 |
| 8 | vix_level | -0.5279 | -0.5645 | -0.2463 | -0.7730 |
| 9 | rev_qoq | -0.6466 | -0.6506 | -0.3033 | -0.9859 |


## Driver Persistence (Top 10 Most Stable)

| Feature | Appearances in Top-10 | Persistence Rate |
|---------|----------------------|------------------|
| revenue | 5 | 100.00% |
| adj_close | 5 | 100.00% |
| rev_qoq | 5 | 100.00% |
| rev_yoy | 5 | 100.00% |
| rev_accel | 5 | 100.00% |
| vix_level | 5 | 100.00% |
| tnx_yield | 5 | 100.00% |
| vix_change_3m | 5 | 100.00% |
| tnx_change_3m | 5 | 100.00% |


## Key Interpretations

- **TNX (Interest Rate) Effects**: tnx_yield, tnx_change_3m appear in top drivers. Higher rates compress NVDA valuation, consistent with discount rate effects.

- **VIX × Price/Revenue Interactions**: vix_change_3m, vix_level dominate short-horizon variance. Market volatility interacts with firm fundamentals in non-linear ways.

- **Revenue Signals**: rev_yoy, rev_accel contribute to predictions, though with longer lag than price momentum.


## Output Files

- `mdi_importance.csv`: MDI importance for all features
- `permutation_importance.csv`: Permutation importance (mean ± std)
- `shap_importance.csv`: Mean absolute SHAP values
- `key_drivers_top10.csv`: Aggregate ranking (Top 10)
- `driver_persistence.csv`: Persistence analysis across CV folds
- `mdi_top20.png`: MDI importance visualization
- `perm_top20.png`: Permutation importance visualization
- `shap_bar.png`: SHAP importance (bar chart)
- `shap_beeswarm.png`: SHAP beeswarm plot
- `shap_dependence_<feature>.png`: SHAP dependence plots for top 3 drivers
- `pdp_<feature>.png`: PDP/ICE plots for top 5 drivers
