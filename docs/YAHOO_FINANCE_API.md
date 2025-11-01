# Yahoo Finance API Usage Guide

This project uses `yfinance` library to fetch stock data. yfinance is an unofficial Python API for Yahoo Finance.

## Installation

```bash
pip install yfinance
```

Already included in `requirements.txt`: `yfinance>=0.2.0`

## Basic Usage

### 1. Basic Data Fetching

```python
import yfinance as yf

# Create ticker object
stock = yf.Ticker("NVDA")  # Use stock ticker symbol

# Get historical data
data = stock.history(period="1y")  # Fetch past 1 year data
print(data.head())
```

### 2. Using Date Range

```python
# Specify start and end dates
data = stock.history(start="2010-01-01", end="2025-01-01")

# Date format: "YYYY-MM-DD"
```

### 3. Time Period Parameters

**period** options:
- `"1d"` - Past 1 day
- `"5d"` - Past 5 days
- `"1mo"` - Past 1 month
- `"3mo"` - Past 3 months
- `"6mo"` - Past 6 months
- `"1y"` - Past 1 year
- `"2y"` - Past 2 years
- `"5y"` - Past 5 years
- `"10y"` - Past 10 years
- `"ytd"` - Year to date
- `"max"` - Full history

**interval** options:
- `"1m"`, `"2m"`, `"5m"`, `"15m"`, `"30m"`, `"60m"` - Minute data
- `"90m"` - 90 minutes
- `"1h"` - Hourly data
- `"1d"`, `"5d"`, `"1wk"` - Daily/Weekly data
- `"1mo"`, `"3mo"` - Monthly data

### 4. Data Columns

Returned DataFrame contains the following columns:
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price
- `Close` - Closing price
- `Volume` - Trading volume
- `Dividends` - Dividends
- `Stock Splits` - Stock splits

## Project Wrapper

The project wraps data fetching functionality in `src/data/fetch.py`:

```python
from src.data.fetch import fetch_stock_data

# Fetch NVDA data (default: 2010-2025)
data = fetch_stock_data("NVDA")

# Custom date range
data = fetch_stock_data("AAPL", start="2020-01-01", end="2023-12-31")

# Use period parameter
data = fetch_stock_data("MSFT", period="max")
```

### Automatic Additional Columns

The wrapper function automatically computes:
- `returns` - Simple returns
- `log_returns` - Log returns (suitable for Monte Carlo)
- `volatility` - Rolling volatility (annualized)

## Common Stock Ticker Examples

- **US Tech Stocks**: `NVDA`, `AAPL`, `MSFT`, `GOOGL`, `META`, `AMZN`, `TSLA`
- **Magnificent 7**: `NVDA`, `AAPL`, `MSFT`, `GOOGL`, `META`, `AMZN`, `TSLA`
- **Market Indices**: `^GSPC` (S&P 500), `^DJI` (Dow Jones), `^IXIC` (NASDAQ)

## Error Handling

```python
try:
    data = stock.history(start="2010-01-01", end="2025-01-01")
    if data.empty:
        print("No data retrieved")
except Exception as e:
    print(f"Error: {e}")
```

## Other Common Methods

```python
stock = yf.Ticker("NVDA")

# Get basic info
info = stock.info
print(f"Company: {info.get('longName')}")
print(f"Market Cap: {info.get('marketCap')}")
print(f"P/E Ratio: {info.get('trailingPE')}")

# Get financial statements
financials = stock.financials
balance_sheet = stock.balance_sheet
cashflow = stock.cashflow

# Get news
news = stock.news
```

## Important Notes

1. **Free Limitations**: yfinance uses Yahoo Finance's free API, which may have rate limits
2. **Network Connection**: Stable internet connection required
3. **Data Delay**: Free data typically has 15-20 minute delay
4. **Data Quality**: Recommend verifying accuracy of critical data points
5. **Split Adjustment**: yfinance automatically handles stock splits and dividend adjustments

## Additional Resources

- yfinance official docs: https://github.com/ranaroussi/yfinance
- Yahoo Finance: https://finance.yahoo.com/
- Stock ticker lookup: Search company name on Yahoo Finance

## Example: Fetch Multiple Stocks

```python
import yfinance as yf

# Fetch multiple stocks
tickers = ["NVDA", "AAPL", "MSFT"]
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")

# data is a multi-level indexed DataFrame
# First level is ticker symbol, second level is data columns
print(data["NVDA"]["Close"].head())
```
