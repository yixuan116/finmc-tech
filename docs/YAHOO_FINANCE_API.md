# Yahoo Finance API 使用指南

本项目使用 `yfinance` 库获取股票数据。yfinance 是 Yahoo Finance 的非官方 Python API。

## 安装

```bash
pip install yfinance
```

已在 `requirements.txt` 中包含: `yfinance>=0.2.0`

## 基本用法

### 1. 基本数据获取

```python
import yfinance as yf

# 创建股票对象
stock = yf.Ticker("NVDA")  # 使用股票代码

# 获取历史数据
data = stock.history(period="1y")  # 获取过去1年数据
print(data.head())
```

### 2. 使用日期范围

```python
# 指定开始和结束日期
data = stock.history(start="2010-01-01", end="2025-01-01")

# 日期格式: "YYYY-MM-DD"
```

### 3. 时间周期参数

**period** 选项:
- `"1d"` - 过去1天
- `"5d"` - 过去5天
- `"1mo"` - 过去1个月
- `"3mo"` - 过去3个月
- `"6mo"` - 过去6个月
- `"1y"` - 过去1年
- `"2y"` - 过去2年
- `"5y"` - 过去5年
- `"10y"` - 过去10年
- `"ytd"` - 今年至今
- `"max"` - 全部历史数据

**interval** 选项:
- `"1m"`, `"2m"`, `"5m"`, `"15m"`, `"30m"`, `"60m"` - 分钟数据
- `"90m"` - 90分钟
- `"1h"` - 小时数据
- `"1d"`, `"5d"`, `"1wk"` - 日/周数据
- `"1mo"`, `"3mo"` - 月数据

### 4. 数据列说明

返回的 DataFrame 包含以下列:
- `Open` - 开盘价
- `High` - 最高价
- `Low` - 最低价
- `Close` - 收盘价
- `Volume` - 成交量
- `Dividends` - 分红
- `Stock Splits` - 股票分割

## 本项目的封装

项目在 `src/data/fetch.py` 中封装了数据获取功能:

```python
from src.data.fetch import fetch_stock_data

# 获取 NVDA 数据 (默认 2010-2025)
data = fetch_stock_data("NVDA")

# 自定义日期范围
data = fetch_stock_data("AAPL", start="2020-01-01", end="2023-12-31")

# 使用 period 参数
data = fetch_stock_data("MSFT", period="max")
```

### 自动计算额外列

封装函数会自动计算:
- `returns` - 简单收益率
- `log_returns` - 对数收益率 (适用于蒙特卡洛)
- `volatility` - 滚动波动率 (年化)

## 常见股票代码示例

- **美国科技股**: `NVDA`, `AAPL`, `MSFT`, `GOOGL`, `META`, `AMZN`, `TSLA`
- **科技七巨头**: `NVDA`, `AAPL`, `MSFT`, `GOOGL`, `META`, `AMZN`, `TSLA`
- **大盘指数**: `^GSPC` (S&P 500), `^DJI` (Dow Jones), `^IXIC` (NASDAQ)

## 错误处理

```python
try:
    data = stock.history(start="2010-01-01", end="2025-01-01")
    if data.empty:
        print("未获取到数据")
except Exception as e:
    print(f"错误: {e}")
```

## 其他常用方法

```python
stock = yf.Ticker("NVDA")

# 获取基本信息
info = stock.info
print(f"公司名: {info.get('longName')}")
print(f"市值: {info.get('marketCap')}")
print(f"市盈率: {info.get('trailingPE')}")

# 获取财务报表
financials = stock.financials
balance_sheet = stock.balance_sheet
cashflow = stock.cashflow

# 获取新闻
news = stock.news
```

## 注意事项

1. **免费限制**: yfinance 使用 Yahoo Finance 的免费 API，可能有访问频率限制
2. **网络连接**: 需要稳定的网络连接
3. **数据延迟**: 免费数据通常有 15-20 分钟的延迟
4. **数据质量**: 建议验证关键数据点的准确性
5. **分拆调整**: yfinance 会自动处理股票分拆和分红调整

## 更多资源

- yfinance 官方文档: https://github.com/ranaroussi/yfinance
- Yahoo Finance: https://finance.yahoo.com/
- 股票代码查询: 在 Yahoo Finance 搜索公司名称

## 示例: 获取多个股票

```python
import yfinance as yf

# 获取多个股票
tickers = ["NVDA", "AAPL", "MSFT"]
data = yf.download(tickers, start="2020-01-01", end="2024-12-31")

# data 是多级索引的 DataFrame
# 第一级是股票代码，第二级是数据列
print(data["NVDA"]["Close"].head())
```

