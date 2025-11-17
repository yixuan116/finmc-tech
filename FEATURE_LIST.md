# Feature Importance Heatmap - Complete Feature List

## Summary
- **Total Features**: 61 (after excluding 2 data leakage features)
- **Data Leakage Features Excluded**: `future_12m_price`, `future_12m_logprice`

## Feature Categories

### 1. Financial Features (4 features)
1. `revenue` - Company revenue
2. `rev_qoq` - Revenue quarter-over-quarter change
3. `rev_yoy` - Revenue year-over-year change
4. `rev_accel` - Revenue acceleration (second derivative)

### 2. Price Features (9 features)
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

### 3. Macro Features (4 features)
1. `vix_level` - VIX index level (market volatility)
2. `tnx_yield` - 10-year Treasury yield
3. `vix_change_3m` - 3-month change in VIX
4. `tnx_change_3m` - 3-month change in Treasury yield

### 4. Time Features (4 features)
1. `quarter` - Quarter of the year (1-4)
2. `month` - Month of the year (1-12)
3. `year` - Year
4. `days_since_start` - Days since data start date

### 5. Interaction Features (40 features)
Kronecker product interactions between macro and micro features:

#### VIX Level Interactions (10 features)
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

#### Treasury Yield Interactions (10 features)
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

#### VIX Change Interactions (10 features)
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

#### Treasury Yield Change Interactions (10 features)
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

## Top 10 Features by Average Importance

1. `tnx_yield` - 0.7926 (10-year Treasury yield - macro feature, **no data leakage** ✅)
2. `days_since_start` - 0.6519 (Days since data start - time trend, **no data leakage** ✅)
3. `price_ma_4q` - 0.5774 (4-quarter moving average of past prices - **historical only** ✅)
   - **Definition**: `adj_close.rolling(window=4).mean()` 
   - **Calculation**: Average of current quarter + previous 3 quarters
   - **Not data leakage**: Only uses past 4 quarters of historical prices
4. `year` - 0.5037 (Year - time feature, **no data leakage** ✅)
5. `ix_tnx_change_3m__price_returns_6m` - 0.3631 (Treasury 3m change × Price 6m returns)
   - `price_returns_6m` = `pct_change(2)` = past 2 quarters return - **historical only** ✅
6. `adj_close` - 0.3104 (Current adjusted closing price - **no data leakage** ✅)
7. `price_to_ma_4q` - 0.2936 (Current price / 4-quarter MA - **historical only** ✅)
8. `month` - 0.2843 (Month - time feature, **no data leakage** ✅)
9. `ix_vix_change_3m__rev_qoq` - 0.2834 (VIX 3m change × Revenue QoQ - **historical only** ✅)
10. `ix_vix_level__price_returns_12m` - 0.2803 (VIX level × Price 12m returns)
    - `price_returns_12m` = `pct_change(4)` = past 4 quarters return - **historical only** ✅

## Data Leakage Verification

**All 61 features are legitimate** - verified no data leakage:

- ✅ **Price features** (`price_returns_*`, `price_ma_4q`, etc.): All use `pct_change()` or `rolling()` with historical data only
- ✅ **Financial features** (`revenue`, `rev_yoy`, etc.): Historical quarterly financial data
- ✅ **Macro features** (`vix_level`, `tnx_yield`, etc.): Historical macro indicators
- ✅ **Time features** (`year`, `month`, `days_since_start`): Time-based features
- ✅ **Interaction features**: Products of historical macro × historical micro features

**Excluded (data leakage)**:
- ❌ `future_12m_price` - Contains future price information
- ❌ `future_12m_logprice` - Contains future price information

**Key Distinction**:
- `price_ma_4q` = **Past 4 quarters** moving average (✅ legitimate)
- `future_12m_price` = **Future 12 months** price (❌ data leakage)

## Notes

- All features use **historical data only** (no future information)
- Data frequency: **Quarterly** (not daily)
- Interaction features follow Gu-Kelly-Xiu (2020) methodology (Kronecker product)
- Features are normalized before model training using StandardScaler
- Importance scores are averaged across 5 models (Linear, Ridge, RF, XGB, NN)

