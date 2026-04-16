# Production Alpha Pipeline

This module upgrades the project to a more production-grade AI/ML architecture focused on alpha generation, robustness, and risk-adjusted returns.

## What it includes

- Multi-source fusion: local multi-asset OHLCV + macro/sentiment context (`fear_greed`, `fred`, `gdelt`, market-news score)
- Advanced features:
  - volatility clustering (`vol_cluster`, `vol_ewma`)
  - momentum factors (`momentum_5`, `momentum_20`, `trend_factor`)
  - RSI / MACD
  - rolling benchmark correlation
  - drawdown + ulcer index
  - regime indicators (high/low vol, trend up/down)
  - order-book proxies (spread/liquidity/imbalance proxies when L2 not available)
- Modeling:
  - Ensemble tabular models (GradientBoosting + RandomForest + MLP)
  - GRU sequence model
  - Walk-forward model scoring and ensemble weighting
  - RL allocator (Q-learning) for dynamic capital allocation across 3 assets + cash reserve
- Validation and safety:
  - Time-series walk-forward validation
  - Leakage-safe chronological split
  - Metrics: Sharpe, Sortino, max drawdown, win/loss ratio, CAGR, plus accuracy/F1
- Strategy and execution simulation:
  - Buy/Sell/Hold classification
  - target-based exits, dynamic stop-loss/take-profit
  - risk sizing (Kelly-inspired + volatility scaling)
  - high-volatility de-risking
  - realistic backtest knobs (fees, slippage, latency)
- Monitoring and continuous learning hooks:
  - feature drift flags
  - performance decay report
  - periodic retrain entrypoint
  - prediction/trade logs

## Run

```python
cd c:\Users\lastf\Desktop\ai-trader
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe research/prod_alpha/run_prod_alpha.py
```

## Hyperparameter Optimization (fast/prod modes)

```python
cd c:\Users\lastf\Desktop\ai-trader\research\prod_alpha
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe run_prod_alpha_hyperopt.py --trials 24 --seed 42 --parallel-workers 2
```

Recommended production workflow (separate phases):

1. Run hyperopt.
2. Save best config/artifacts (`best_config_<timestamp>.json`, `best_config_latest.json`).
3. Run production separately with the saved best config.

```python
cd c:\Users\lastf\Desktop\ai-trader\research\prod_alpha
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe run_prod_alpha.py --config-json ../artifacts/prod_alpha/best_config_latest.json
```

Notes:

- `run_prod_alpha_hyperopt.py` does not chain production by default.
- If you pass `--run-production`, the script prints an explicit transition banner and per-phase elapsed time.
- `run_prod_alpha.py` prints production phase elapsed time.

Profile-driven tuning (default / fast-tabular / very-light-debug):

```python
cd c:\Users\lastf\Desktop\ai-trader\research\prod_alpha
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe run_prod_alpha_hyperopt.py --profile fast-tabular --trials 12 --seed 42 --parallel-workers 1 --debug
```

Quick debug run:

```python
cd c:\Users\lastf\Desktop\ai-trader\research\prod_alpha
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe run_prod_alpha_hyperopt.py --trials 8 --seed 42 --parallel-workers 2 --debug
```

Notes:

- GRU uses CUDA automatically when available, otherwise falls back to CPU.
- Raw data, engineered features, and walk-forward splits are cached within process to reduce repeat trial overhead.
- Runtime logs include per-stage timing, cache hits, and trial ETA estimates.
- `--profile` controls tabular model profile without changing default production behavior unless explicitly selected.
- `--very-light-debug` remains supported and maps to the `very-light-debug` profile.

## Tabular Profile A/B Benchmark

```python
cd c:\Users\lastf\Desktop\ai-trader\research\prod_alpha
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe benchmark_tabular_profiles.py --trials 4 --seed 42 --cpu-threads 12 --parallel-workers 1
```

This writes `research/artifacts/prod_alpha/tabular_profile_benchmark_latest.json` with:

- runtime/trial and split-stage timing comparison
- objective + Sharpe/turnover/drawdown comparison
- decision recommendation for default tuning profile adoption

## Output artifacts

Saved under `research/artifacts/prod_alpha/`:

- `comparison_<timestamp>.json` (baseline vs upgraded metrics)
- `predictions_<timestamp>.csv`
- `trades_<timestamp>.json`
