# Research: The Foundation of Algorithmic Trading

## Why Research Matters

Research is the critical first step in the RBI (Research, Backtest, Implement) system. Without proper research, you're essentially gambling rather than trading.

> "As an algorithmic trader, you should be doing one of three things every day: researching, backtesting, or implementing code into trading bots." - Moon Dev 🌙

## Sources for Trading Strategies and Alpha Generation

### Academic Resources

- **Google Scholar** - Search for peer-reviewed papers written by PhDs on trading strategies, market efficiency, and alpha generation
- **University Repositories** - Many universities publish finance and trading research
- **Financial Journals** - Journal of Finance, Review of Financial Studies, Journal of Financial Economics

### AI and Machine Learning

- **ChatGPT** - Great for generating broad ideas that you can then research more deeply
- **Research Papers with Code** - Find implementations of trading algorithms

### Trader Insights

- **Chat with Traders** - Podcast featuring 300+ professional traders sharing their methods
- **Market Wizards Series** - Interviews with top traders
- **Trading Conferences** - Recordings often available online

### Online Platforms

- **YouTube** - Search for technical analysis, factor investing, statistical arbitrage, etc.
- **Trading Forums** - Quantopian archives, QuantConnect community, Elite Trader
- **Finance Blogs** - Seeking Alpha, Quantocracy, Alpha Architect

## Essential Trading Books

### Trading Psychology & Philosophy

- **Trading in the Zone** by Mark Douglas
- **The Black Swan** by Nassim Nicholas Taleb
- **Fooled by Randomness** by Nassim Nicholas Taleb
- **What I Learned Losing a Million Dollars** by Jim Paul
- **Best Loser Wins** by Tom Hougaard
- **A Random Walk Down Wall Street** by Burton Malkiel

### Market Structure & History

- **Flash Boys** by Michael Lewis
- **Dark Pools** by Scott Patterson
- **The Quants** by Scott Patterson
- **The Man Who Solved the Market** (about Jim Simons) by Gregory Zuckerman
- **One Up on Wall Street** by Peter Lynch

### Trading Systems & Methods

- **Systematic Trading** by Robert Carver
- **Trading Systems and Methods** by Perry Kaufman
- **How to Trade in Stocks** by Jesse Livermore
- **The Universal Tactics of Successful Trend Trading** by Brent Penfold
- **Stocks on the Move** by Andreas Clenow
- **The Complete Turtle Trader** by Michael Covel

### Technical & Quantitative Trading

- **Advances in Financial Machine Learning** by Marcos Lopez de Prado
- **Option Volatility and Pricing** by Sheldon Natenberg
- **The Mathematics of Money Management** by Ralph Vince
- **The Leveraged Trading Model** by Howard B. Bandy
- **Rocket Science for Traders** by John Ehlers
- **Cybernetic Analysis for Stocks and Futures** by John Ehlers
- **Cycle Analytics for Traders** by John Ehlers
- **Statistical Sound Indicators for Financial Market Prediction** by Timothy Masters
- **Cybernetic Trading Strategies** by Murray Ruggiero
- **Testing and Tuning Market Trading Systems** by Timothy Masters
- **Permutation and Randomization Tests of Trading System Development** by Timothy Masters
- **Detecting Regime Change in Computational Finance** by Philip Roni

## Research Process Best Practices

1. **Start Broad, Then Narrow** - Begin with general strategy types, then focus on specific implementations
2. **Document Everything** - Keep detailed notes on every strategy idea and its source
3. **Evaluate Against Your Resources** - Consider data requirements, computational needs, and capital constraints
4. **Cross-Reference Multiple Sources** - Validate strategies across different books and papers
5. **Identify Key Assumptions** - Understand what market conditions each strategy requires
6. **Consider Risk First** - Evaluate potential drawdowns before potential returns
7. **Look for Counter-Arguments** - Seek out criticisms of strategies you're considering

## Finding Your Trading Style

Remember that the goal of research is to find strategies that match:

- Your personality and risk tolerance
- Your available capital
- Your technical capabilities
- Your time commitment

No strategy works forever, and what worked in the past doesn't always work in the future. The goal is to develop a deep understanding of market mechanics that allows you to adapt as conditions change.

## Next Steps

Once you have thoroughly researched multiple trading strategies, proceed to the Backtest (B) phase to validate your ideas with historical data.

> "🚀 Moon Dev's Research Tip: Read one trading book and listen to one trading podcast each week to continuously expand your knowledge base!"

## Model Training

If you want to train a simple machine-learning model on Finnhub candles, use [train_model.py](train_model.py). It downloads OHLCV data, builds technical features, labels each bar by the next-bar direction, trains a classifier, and saves the fitted model plus metrics into `research/artifacts/`.

If Finnhub access is not available, the script falls back to the local CSV configured by `TRAIN_DATA_PATH` and can still train a model from the repo's existing historical data.

If you want richer training inputs, use [build_datasets.py](build_datasets.py). It creates multiple CSVs from the repo's local price history plus Finnhub news and earnings endpoints, including company news, market news, earnings calendar, earnings surprises, and daily news features. The combined training dataset currently joins Finnhub news features with the repo's existing local price history.

For a model that actually uses the richer feature set, run [train_combined_model.py](train_combined_model.py). It trains on the merged price + news dataset produced by `build_datasets.py`.

If you want a simple action recommendation from the latest trained model, run [generate_trade_signal.py](generate_trade_signal.py). It prints BUY, HOLD, or SELL from the latest combined dataset row.

If you want the interactive app, run the separated stack from the repo root with `docker compose up --build` and open the React frontend on `http://localhost:3000`.

If you want to train an LLM on your app data, run [prepare_llm_training_data.py](prepare_llm_training_data.py). It converts your combined dataset into chat fine-tuning JSONL examples for OpenAI-compatible providers.

If you want a deployable API endpoint for your live stack, run `python -m backend.app` from the repo root. It serves `/health`, `/signal`, `/dashboard`, and the trading control endpoints used by the React frontend.

### Training Environment Variables

- `FINNHUB_API_KEY`: required for data download.
- `TRAIN_SYMBOL`: stock symbol to train on, default `AAPL`.
- `TRAIN_TIMEFRAME`: Finnhub resolution such as `1d` or `1h`.
- `TRAIN_LOOKBACK_DAYS`: how much history to fetch.
- `TRAIN_TARGET_HORIZON`: how many bars ahead to predict.
- `TRAIN_DATA_PATH`: local CSV used when Finnhub access is unavailable.
- `TRAIN_OUTPUT_DIR`: where to save the model and metrics.

### Dataset Builder Environment Variables

- `DATA_OUTPUT_DIR`: where the CSV datasets are written, default `research/data_sets/`.
- `DATA_SYMBOL`: symbol used for company news and earnings data, default `AAPL`.
- `PRICE_DATA_PATH`: local OHLCV file used as the price history base.
- `NEWS_LOOKBACK_DAYS`: number of days of company news to collect.
- `MARKET_EXCHANGE`: exchange code used for the market status snapshot.

### Combined Trainer Environment Variables

- `COMBINED_DATASET_PATH`: path to the merged training dataset, default `research/data_sets/aapl_training_dataset.csv`.
- `TARGET_COLUMN`: label column name, default `target`.
- `TEST_SIZE`: fraction of rows reserved for testing, default `0.2`.

### Signal Generator Environment Variables

- `MODEL_PATH`: optional path to a specific `.joblib` model file.
- `DATASET_PATH`: optional path to a specific combined dataset CSV.
- `BUY_THRESHOLD`: probability threshold for BUY, default `0.55`.
- `SELL_THRESHOLD`: probability threshold for SELL, default `0.45`.
- `LLM_PROVIDER`: `deepseek`, `openai_compatible`, or `huggingface_inference`.
- `LLM_MODEL`: model name for the selected provider.
- `LLM_BASE_URL`: provider base URL.
- `LLM_API_KEY`: provider token/key (for Hugging Face use your HF token).
- `LLM_MERGE_ENABLED`: set `true` to merge LLM view with ML probability.
- `LLM_MERGE_WEIGHT`: merge weight (0.0-0.5).
- `LLM_CONFIDENCE_FLOOR`: minimum confidence required for merge.
- `LLM_CONFIDENCE_SOFT_GATE`: if `true`, low confidence scales merge weight down instead of fully disabling merge.
- `LLM_TIMEOUT_SECONDS`: HTTP timeout for LLM requests, default `30`.
- `LLM_BYPASS_ENV_PROXY`: if `true`, retries direct HTTPS without environment proxy when proxy tunnel fails.

### Free LLM Setup (Hugging Face)

Use environment variables:

- `LLM_ENABLED=true`
- `LLM_PROVIDER=huggingface_inference`
- `LLM_MODEL=Qwen/Qwen2.5-3B-Instruct`
- `LLM_BASE_URL=https://api-inference.huggingface.co`
- `LLM_API_KEY=<your_hugging_face_token>`
- `LLM_MERGE_ENABLED=true`
- `LLM_MERGE_WEIGHT=0.20`
- `LLM_CONFIDENCE_FLOOR=0.40`
- `LLM_CONFIDENCE_SOFT_GATE=true`

This keeps the Python backend local or self-hosted while the LLM inference runs on Hugging Face free infrastructure.

### LLM Dataset Builder Environment Variables

- `LLM_TRAIN_OUTPUT_PATH`: destination JSONL used for fine-tuning uploads.
- `LLM_MAX_FEATURES`: max numeric fields per training sample.
- `LLM_MAX_SAMPLES`: optional row cap (uses latest rows).

### Backend API Environment Variables

- `SIGNAL_API_HOST`: bind host for API server, default `0.0.0.0`.
- `SIGNAL_API_PORT`: bind port for API server, default `8765`.

### App Runtime

Run the backend with `python -m backend.app`, or run the full app with `docker compose up --build` from the repo root.

Optional Binance connectivity settings for restrictive networks:

- `BINANCE_TIMEOUT_SECONDS`: timeout for Binance API requests, default `30`.
- `BINANCE_BYPASS_ENV_PROXY`: if `true`, bypasses environment proxy settings for Binance API calls.

The backend uses the latest model artifact and combined dataset, while the React frontend handles controls and visualization.
