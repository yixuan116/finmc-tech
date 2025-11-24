# Step 5: Key Drivers Analysis Results

This directory contains key driver analysis results from two different feature sets:

## Analysis 1: Extended Features (61 features)

**Script**: `finmc_tech/step5_key_drivers.py`  
**Data**: `data/processed/nvda_features_extended.csv`  
**Features**: 61 features including macro × micro interactions, price momentum, time features

**Output Files**:
- `driver_ranking_extended_features.csv` - Comprehensive ranking with MDI, Permutation, SHAP
- `mdi_importance_extended.png` - MDI importance visualization
- `permutation_importance_extended.png` - Permutation importance visualization
- `shap_summary_extended.png` - SHAP summary plot
- `pdp_*.png` - PDP/ICE plots for top drivers (extended feature set)
- `shap_values_test.npy` - SHAP values array

**Top Drivers** (from extended features):
1. `ix_vix_change_3m__price_returns_6m` - VIX change × Price returns interaction
2. `ix_vix_change_3m__revenue` - VIX change × Revenue interaction
3. `days_since_start` - Time trend feature

---

## Analysis 2: Short-Horizon (9 features)

**Script**: `src/step5_key_drivers_short.py`  
**Data**: `data/processed/NVDA_revenue_features.csv`  
**Features**: 9 basic features (revenue, macro, price)

**Output Files**:
- `key_drivers_top10.csv` - Top 10 drivers (aggregate ranking)
- `mdi_importance.csv` - MDI importance for all features
- `permutation_importance.csv` - Permutation importance (mean ± std)
- `shap_importance.csv` - Mean absolute SHAP values
- `driver_persistence.csv` - Persistence analysis across CV folds
- `mdi_top20.png` - MDI importance visualization
- `perm_top20.png` - Permutation importance visualization
- `shap_bar.png` - SHAP importance (bar chart)
- `shap_beeswarm.png` - SHAP beeswarm plot
- `shap_dependence_*.png` - SHAP dependence plots for top 3 drivers
- `pdp_*.png` - PDP/ICE plots for top 5 drivers
- `README_step5.md` - Detailed summary

**Top Drivers** (from short-horizon analysis):
1. `rev_yoy` - Revenue year-over-year growth
2. `adj_close` - Current adjusted closing price
3. `tnx_yield` - 10-year Treasury yield

---

## Key Differences

| Aspect | Extended Features | Short-Horizon |
|--------|------------------|---------------|
| **Feature Count** | 61 features | 9 features |
| **Feature Types** | Macro × Micro interactions, price momentum, time | Basic revenue, macro, price |
| **Analysis Depth** | Comprehensive ranking | Aggregate ranking + persistence |
| **Use Case** | Full feature importance analysis | Quick short-horizon driver identification |

---

## Recommendations

- **For comprehensive analysis**: Use extended features results (`driver_ranking_extended_features.csv`)
- **For quick insights**: Use short-horizon results (`key_drivers_top10.csv`)
- **For stability check**: See `driver_persistence.csv` (short-horizon only)

