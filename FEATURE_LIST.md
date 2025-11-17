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
1. `adj_close` - Adjusted closing price
2. `price_returns_1m` - 1-month price returns
3. `price_returns_3m` - 3-month price returns
4. `price_returns_6m` - 6-month price returns
5. `price_returns_12m` - 12-month price returns
6. `price_momentum` - Price momentum indicator
7. `price_volatility` - Price volatility measure
8. `price_ma_4q` - 4-quarter moving average of price
9. `price_to_ma_4q` - Price to 4-quarter moving average ratio

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

1. `tnx_yield` - 0.7926 (Treasury yield - most important)
2. `days_since_start` - 0.6519 (Time trend)
3. `price_ma_4q` - 0.5774 (4-quarter moving average)
4. `year` - 0.5037 (Year effect)
5. `ix_tnx_change_3m__price_returns_6m` - 0.3631 (Treasury change × 6m returns)
6. `adj_close` - 0.3104 (Adjusted closing price)
7. `price_to_ma_4q` - 0.2936 (Price to MA ratio)
8. `month` - 0.2843 (Month effect)
9. `ix_vix_change_3m__rev_qoq` - 0.2834 (VIX change × Revenue QoQ)
10. `ix_vix_level__price_returns_12m` - 0.2803 (VIX level × 12m returns)

## Notes

- All features are legitimate (no data leakage)
- Interaction features follow Gu-Kelly-Xiu (2020) methodology
- Features are normalized before model training
- Importance scores are averaged across 5 models (Linear, Ridge, RF, XGB, NN)

