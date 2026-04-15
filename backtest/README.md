# Backtesting

## Overview

Backtesting is a crucial component of the RBI (Research, Backtest, Implement) system. This process involves analyzing historical open, high, low, close, and volume data to evaluate if a trading strategy would have been successful in the past. While past performance doesn't guarantee future results, strategies that worked historically have a higher probability of success going forward.

## Resources

### Template

- **template.py**: A starter template for backtesting that can be used with any trading idea. This is designed to work with strategies developed in the research phase. Simply examine the template and integrate your researched strategy to see how it would have performed historically.

### Backtesting Libraries

This repository includes support for three popular backtesting libraries:

1. **Backtesting.py**: Our recommended library and the one used in the template.py file. It offers a good balance of simplicity and power.
2. **Backtrader**: A comprehensive Python framework for backtesting and trading.
3. **Zipline**: An event-driven backtesting system developed by Quantopian.

### Data Acquisition

- **data.py**: A utility for obtaining market data from Finnhub when `FINNHUB_API_KEY` is set, with a Hyperliquid fallback for the existing crypto examples. It saves CSV files into `backtest/data/`.

## Getting Started

1. Review your trading idea from the research phase
2. Use data.py to gather historical market data and save it locally
3. Set `BACKTEST_DATA_PATH` if you want `template.py` or `bb_squeeze_adx.py` to load a different CSV
4. Modify template.py to implement your strategy
5. Run the backtest and analyze the results
6. Refine your strategy based on performance metrics

### Environment Variables

- `FINNHUB_API_KEY`: required when you want `data.py` to pull stock or ETF candles from Finnhub.
- `DATA_SYMBOL`: optional symbol override for `data.py`.
- `DATA_TIMEFRAME`: optional timeframe override for `data.py`, such as `1d`, `1h`, or `15m`.
- `BACKTEST_DATA_PATH`: optional CSV path override for the strategy scripts.
