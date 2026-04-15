"""Build multiple CSV datasets for model training.

Outputs:
- price_history.csv: local historical OHLCV data from the repo
- company_news.csv: Finnhub company news for a symbol
- market_news.csv: Finnhub market news snapshot
- earnings_calendar.csv: Finnhub earnings calendar snapshot
- earnings_surprises.csv: Finnhub earnings surprises snapshot
- quote.csv: Finnhub latest quote snapshot
- company_profile2.csv: Finnhub company profile snapshot
- recommendation_trends.csv: Finnhub analyst recommendation snapshot
- peers.csv: Finnhub peer list snapshot
- market_status.csv: Finnhub market status snapshot
- daily_news_features.csv: aggregated daily counts + simple sentiment proxy
- fred_macro_features.csv: free macro series from FRED with daily forward-filled values
- gdelt_event_features.csv: free crypto/macro news intensity from GDELT
- training_dataset.csv: merged price + news + macro feature dataset for supervised learning

This script is designed around free-access / high-usage Finnhub endpoints plus the
repo's existing local price CSVs.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = Path(os.getenv("DATA_OUTPUT_DIR", BASE_DIR / "data_sets"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
DATA_SYMBOL = os.getenv("DATA_SYMBOL", "AAPL")
DATA_SYMBOLS = [symbol.strip().upper() for symbol in os.getenv("DATA_SYMBOLS", DATA_SYMBOL).split(",") if symbol.strip()]
PRICE_DATA_PATH = Path(
    os.getenv(
        "PRICE_DATA_PATH",
        str(PROJECT_ROOT / "backtest" / "data" / "BTC-6h-1000wks-data.csv"),
    )
)
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "365"))
MARKET_EXCHANGE = os.getenv("MARKET_EXCHANGE", "US")
DATA_TIMEFRAME = os.getenv("DATA_TIMEFRAME", "1d")
DATA_LOOKBACK_DAYS = int(os.getenv("DATA_LOOKBACK_DAYS", "730"))
TARGET_HORIZON = int(os.getenv("TARGET_HORIZON", "1"))
TARGET_RETURN_THRESHOLD = float(os.getenv("TARGET_RETURN_THRESHOLD", "0.003"))
TARGET_DOWNSIDE_THRESHOLD = float(os.getenv("TARGET_DOWNSIDE_THRESHOLD", "-0.003"))
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred").rstrip("/")
FRED_TIMEOUT_SECONDS = int(os.getenv("FRED_TIMEOUT_SECONDS", "20"))
FRED_BYPASS_ENV_PROXY = os.getenv("FRED_BYPASS_ENV_PROXY", "true").lower() == "true"
FRED_SERIES_IDS = [
    series.strip().upper()
    for series in os.getenv("FRED_SERIES_IDS", "DFF,CPIAUCSL,UNRATE,DGS10,VIXCLS").split(",")
    if series.strip()
]
EVENT_LOOKBACK_DAYS = int(os.getenv("EVENT_LOOKBACK_DAYS", "1095"))
GDELT_CRYPTO_QUERY = os.getenv(
    "GDELT_CRYPTO_QUERY",
    'bitcoin OR ethereum OR crypto OR stablecoin OR ETF OR SEC OR hack OR bankruptcy OR exchange',
)
GDELT_MACRO_QUERY = os.getenv(
    "GDELT_MACRO_QUERY",
    'Fed OR inflation OR CPI OR rates OR recession OR rate hike OR rate cut',
)
GDELT_TIMEOUT_SECONDS = int(os.getenv("GDELT_TIMEOUT_SECONDS", "30"))
GDELT_BYPASS_ENV_PROXY = os.getenv("GDELT_BYPASS_ENV_PROXY", "true").lower() == "true"

TIMEFRAME_TO_RESOLUTION = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "1d": "D",
    "1w": "W",
    "1mo": "M",
    "1mth": "M",
    "1M": "M",
}

POSITIVE_WORDS = {
    "beat",
    "bull",
    "bullish",
    "buy",
    "growth",
    "gain",
    "good",
    "great",
    "improve",
    "increased",
    "outperform",
    "profit",
    "record",
    "strong",
    "surge",
    "up",
    "upgrade",
}

NEGATIVE_WORDS = {
    "bear",
    "bearish",
    "below",
    "cut",
    "decline",
    "down",
    "downgrade",
    "drop",
    "fall",
    "loss",
    "miss",
    "negative",
    "sell",
    "weak",
    "warning",
}


def require_finnhub_key() -> None:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is not set")


def fetch_json(url: str, params: dict) -> object:
    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with {response.status_code}: {response.text}")
    return response.json()


def timeframe_to_resolution(timeframe: str) -> str:
    if timeframe not in TIMEFRAME_TO_RESOLUTION:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_TO_RESOLUTION[timeframe]


def slugify_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def fetch_price_history_for_symbol(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    require_finnhub_key()
    resolution = timeframe_to_resolution(timeframe)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    payload = fetch_json(
        "https://finnhub.io/api/v1/stock/candle",
        {
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start_time.timestamp()),
            "to": int(end_time.timestamp()),
            "token": FINNHUB_API_KEY,
        },
    )
    if not isinstance(payload, dict) or payload.get("s") != "ok":
        raise RuntimeError(f"Unexpected candle payload for {symbol}: {payload}")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert(None),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"],
        }
    )
    frame["symbol"] = symbol
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_price_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price history not found: {path}")

    data = pd.read_csv(path)
    timestamp_column = "datetime" if "datetime" in data.columns else "timestamp" if "timestamp" in data.columns else None
    if timestamp_column is None:
        raise ValueError(f"Expected a datetime or timestamp column in {path}")

    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data = data.rename(columns={column: column.lower() for column in data.columns})
    required_columns = ["open", "high", "low", "close", "volume"]
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f"Price history must include columns: {required_columns}")

    data = data.rename(columns={timestamp_column: "timestamp"})
    data = data[["timestamp"] + required_columns]
    data["symbol"] = DATA_SYMBOL
    return data.sort_values("timestamp").reset_index(drop=True)


def fetch_company_news(symbol: str, lookback_days: int) -> pd.DataFrame:
    require_finnhub_key()
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    payload = fetch_json(
        "https://finnhub.io/api/v1/company-news",
        {
            "symbol": symbol,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "token": FINNHUB_API_KEY,
        },
    )

    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected company-news payload: {payload}")

    rows = []
    for item in payload:
        rows.append(
            {
                "symbol": symbol,
                "datetime": pd.to_datetime(item.get("datetime"), unit="s", utc=True).tz_convert(None)
                if item.get("datetime")
                else pd.NaT,
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "category": item.get("category", "company news"),
                "url": item.get("url", ""),
                "related": item.get("related", ""),
            }
        )

    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def fetch_market_news() -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/news",
        {
            "category": "general",
            "token": FINNHUB_API_KEY,
        },
    )

    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected market news payload: {payload}")

    rows = []
    for item in payload:
        rows.append(
            {
                "datetime": pd.to_datetime(item.get("datetime"), unit="s", utc=True).tz_convert(None)
                if item.get("datetime")
                else pd.NaT,
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "category": item.get("category", "general"),
                "url": item.get("url", ""),
                "related": item.get("related", ""),
                "id": item.get("id"),
            }
        )

    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def fetch_earnings_calendar() -> pd.DataFrame:
    require_finnhub_key()
    end_date = date.today()
    start_date = end_date - timedelta(days=90)
    payload = fetch_json(
        "https://finnhub.io/api/v1/calendar/earnings",
        {
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "token": FINNHUB_API_KEY,
        },
    )

    rows = []
    for item in payload.get("earningsCalendar", []):
        rows.append(item)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_earnings_surprises(symbol: str) -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/stock/earnings",
        {
            "symbol": symbol,
            "token": FINNHUB_API_KEY,
        },
    )

    return pd.DataFrame(payload).sort_values("period").reset_index(drop=True)


def fetch_quote(symbol: str) -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/quote",
        {
            "symbol": symbol,
            "token": FINNHUB_API_KEY,
        },
    )
    row = {"symbol": symbol, **payload, "fetched_at": datetime.now().astimezone()}
    return pd.DataFrame([row])


def fetch_company_profile2(symbol: str) -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/stock/profile2",
        {
            "symbol": symbol,
            "token": FINNHUB_API_KEY,
        },
    )
    row = {"symbol": symbol, **payload}
    return pd.DataFrame([row])


def fetch_recommendation_trends(symbol: str) -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/stock/recommendation",
        {
            "symbol": symbol,
            "token": FINNHUB_API_KEY,
        },
    )
    return pd.DataFrame(payload).sort_values("period").reset_index(drop=True)


def fetch_company_peers(symbol: str) -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/stock/peers",
        {
            "symbol": symbol,
            "token": FINNHUB_API_KEY,
        },
    )
    return pd.DataFrame({"symbol": payload})


def fetch_market_status(exchange: str) -> pd.DataFrame:
    require_finnhub_key()
    payload = fetch_json(
        "https://finnhub.io/api/v1/stock/market-status",
        {
            "exchange": exchange,
            "token": FINNHUB_API_KEY,
        },
    )
    row = {"exchange": exchange, **payload, "fetched_at": datetime.now().astimezone()}
    return pd.DataFrame([row])


def fetch_fred_series(series_id: str, lookback_days: int) -> pd.DataFrame:
    if not FRED_API_KEY:
        return pd.DataFrame(columns=["date", slugify_name(series_id)])

    series = (series_id or "").strip().upper()
    if not series:
        return pd.DataFrame(columns=["date", "value"])

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": series,
        "observation_start": start_date.isoformat(),
        "observation_end": end_date.isoformat(),
        "sort_order": "asc",
        "limit": 5000,
    }
    with requests.Session() as session:
        if FRED_BYPASS_ENV_PROXY:
            session.trust_env = False
        response = session.get(f"{FRED_BASE_URL}/series/observations", params=params, timeout=FRED_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()

    rows = []
    column_name = f"fred_{slugify_name(series)}"
    for item in payload.get("observations", []):
        value_text = str(item.get("value", "."))
        if value_text in {".", "", "nan", "NaN"}:
            value = None
        else:
            try:
                value = float(value_text)
            except ValueError:
                value = None
        rows.append({"date": pd.to_datetime(item.get("date")).date(), column_name: value})

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_fred_features(series_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series_id in series_ids:
        frame = fetch_fred_series(series_id, max(DATA_LOOKBACK_DAYS, 365))
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date"])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    fred_columns = [column for column in merged.columns if column != "date"]
    merged[fred_columns] = merged[fred_columns].ffill().bfill()

    for column in fred_columns:
        merged[f"{column}_diff_7"] = merged[column].diff(7)
        merged[f"{column}_diff_30"] = merged[column].diff(30)

    return merged


def fetch_gdelt_timeline(query: str, lookback_days: int, column_name: str) -> pd.DataFrame:
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "timelinesmooth": 0,
    }

    with requests.Session() as session:
        if GDELT_BYPASS_ENV_PROXY:
            session.trust_env = False
        response = session.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=GDELT_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()

    rows = []
    for item in payload.get("timeline", []):
        date_text = str(item.get("date", ""))
        parsed_date = None
        if len(date_text) >= 8 and date_text[:8].isdigit():
            parsed_date = datetime.strptime(date_text[:8], "%Y%m%d").date()
        else:
            try:
                parsed_date = pd.to_datetime(date_text).date()
            except Exception:
                parsed_date = None
        if parsed_date is None:
            continue
        raw_value = item.get("value", item.get("count", 0))
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 0.0
        rows.append({"date": parsed_date, column_name: value})

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_gdelt_features() -> pd.DataFrame:
    crypto = fetch_gdelt_timeline(GDELT_CRYPTO_QUERY, EVENT_LOOKBACK_DAYS, "gdelt_crypto_news_count")
    macro = fetch_gdelt_timeline(GDELT_MACRO_QUERY, EVENT_LOOKBACK_DAYS, "gdelt_macro_news_count")

    frames = [frame for frame in [crypto, macro] if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["date"])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    for column in [col for col in merged.columns if col != "date"]:
        merged[column] = merged[column].fillna(0)

    if "gdelt_crypto_news_count" in merged.columns and "gdelt_macro_news_count" in merged.columns:
        merged["gdelt_total_news_count"] = merged["gdelt_crypto_news_count"] + merged["gdelt_macro_news_count"]
        merged["gdelt_crypto_share"] = merged["gdelt_crypto_news_count"] / merged["gdelt_total_news_count"].replace(0, pd.NA)
    elif "gdelt_crypto_news_count" in merged.columns:
        merged["gdelt_total_news_count"] = merged["gdelt_crypto_news_count"]
        merged["gdelt_crypto_share"] = 1.0
    else:
        merged["gdelt_total_news_count"] = merged["gdelt_macro_news_count"]
        merged["gdelt_crypto_share"] = 0.0

    merged["gdelt_total_news_7d"] = merged["gdelt_total_news_count"].rolling(7, min_periods=1).sum()
    merged["gdelt_total_news_30d"] = merged["gdelt_total_news_count"].rolling(30, min_periods=1).sum()
    return merged


def text_sentiment_score(text: str) -> int:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    positive_score = sum(word in POSITIVE_WORDS for word in words)
    negative_score = sum(word in NEGATIVE_WORDS for word in words)
    return positive_score - negative_score


def build_daily_news_features(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["date", "news_count", "headline_score", "summary_score", "total_score"])

    frame = news.copy()
    frame = frame.dropna(subset=["datetime"])
    frame["date"] = frame["datetime"].dt.date
    frame["headline_score"] = frame["headline"].fillna("").map(text_sentiment_score)
    frame["summary_score"] = frame["summary"].fillna("").map(text_sentiment_score)
    frame["total_score"] = frame["headline_score"] + frame["summary_score"]

    daily = (
        frame.groupby("date")
        .agg(
            news_count=("headline", "count"),
            headline_score=("headline_score", "sum"),
            summary_score=("summary_score", "sum"),
            total_score=("total_score", "sum"),
            unique_sources=("source", pd.Series.nunique),
        )
        .reset_index()
    )
    return daily


def build_daily_context(
    price_history: pd.DataFrame,
    daily_news: pd.DataFrame,
    fred_features: pd.DataFrame,
    gdelt_features: pd.DataFrame,
) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame(columns=["date"])

    price_dates = pd.to_datetime(price_history["timestamp"]).dt.date
    date_index = pd.DataFrame({"date": pd.date_range(price_dates.min(), price_dates.max(), freq="D").date})

    context = date_index.copy()
    for frame in [daily_news, fred_features, gdelt_features]:
        if frame is not None and not frame.empty:
            context = context.merge(frame, on="date", how="left")

    news_columns = ["news_count", "headline_score", "summary_score", "total_score", "unique_sources"]
    for column in news_columns:
        if column in context.columns:
            context[column] = context[column].fillna(0)
            context[f"{column}_7d"] = context[column].rolling(7, min_periods=1).sum()

    fred_columns = [column for column in context.columns if column.startswith("fred_")]
    if fred_columns:
        context[fred_columns] = context[fred_columns].ffill().bfill()

    gdelt_columns = [column for column in context.columns if column.startswith("gdelt_")]
    for column in gdelt_columns:
        context[column] = context[column].fillna(0)

    return context


def build_training_dataset(
    price_history: pd.DataFrame,
    daily_news: pd.DataFrame,
    fred_features: pd.DataFrame,
    gdelt_features: pd.DataFrame,
    horizon: int = 1,
    target_return_threshold: float = 0.003,
    target_downside_threshold: float = -0.003,
) -> pd.DataFrame:
    frame = price_history.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.date
    frame["symbol_code"] = frame["symbol"].astype("category").cat.codes

    feature_frames: list[pd.DataFrame] = []
    for symbol_value, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").copy()
        group["return_1"] = group["close"].pct_change(1)
        group["return_2"] = group["close"].pct_change(2)
        group["return_4"] = group["close"].pct_change(4)
        group["sma_3"] = group["close"].rolling(3).mean()
        group["sma_6"] = group["close"].rolling(6).mean()
        group["sma_10"] = group["close"].rolling(10).mean()
        group["sma_20"] = group["close"].rolling(20).mean()
        group["close_sma_3"] = group["close"] / group["sma_3"] - 1
        group["close_sma_6"] = group["close"] / group["sma_6"] - 1
        group["close_sma_10"] = group["close"] / group["sma_10"] - 1
        group["close_sma_20"] = group["close"] / group["sma_20"] - 1
        group["volatility_6"] = group["return_1"].rolling(6).std()
        group["volatility_10"] = group["return_1"].rolling(10).std()
        group["volatility_20"] = group["return_1"].rolling(20).std()
        group["range_pct"] = (group["high"] - group["low"]) / group["close"]
        group["body_pct"] = (group["close"] - group["open"]) / group["open"]
        group["volume_change_1"] = group["volume"].pct_change(1)
        group["volume_sma_10"] = group["volume"] / group["volume"].rolling(10).mean() - 1
        group["volume_sma_20"] = group["volume"] / group["volume"].rolling(20).mean() - 1
        group["trend_strength"] = group["close_sma_3"] - group["close_sma_10"]
        group["momentum_10"] = group["close"].pct_change(10)
        group["momentum_20"] = group["close"].pct_change(20)
        group["drawdown_20"] = group["close"] / group["close"].rolling(20).max() - 1
        group["future_return"] = group["close"].shift(-horizon) / group["close"] - 1
        feature_frames.append(group)

    frame = pd.concat(feature_frames, ignore_index=True)
    daily_context = build_daily_context(frame, daily_news, fred_features, gdelt_features)
    merged = frame.merge(daily_context, on="date", how="left")

    for column in [col for col in merged.columns if col.startswith("fred_") or col.startswith("gdelt_")]:
        merged[column] = merged[column].fillna(0)

    merged["target"] = pd.NA
    merged.loc[merged["future_return"] >= target_return_threshold, "target"] = 1
    merged.loc[merged["future_return"] <= target_downside_threshold, "target"] = 0
    merged["target"] = merged["target"].astype("float")
    merged["target_label"] = merged["target"].map({1.0: "trade_long", 0.0: "avoid_or_short"})
    merged["target_return_threshold"] = float(target_return_threshold)
    merged["target_downside_threshold"] = float(target_downside_threshold)

    feature_columns = [
        "symbol_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1",
        "return_2",
        "return_4",
        "sma_3",
        "sma_6",
        "sma_10",
        "sma_20",
        "close_sma_3",
        "close_sma_6",
        "close_sma_10",
        "close_sma_20",
        "volatility_6",
        "volatility_10",
        "volatility_20",
        "range_pct",
        "body_pct",
        "volume_change_1",
        "volume_sma_10",
        "volume_sma_20",
        "trend_strength",
        "momentum_10",
        "momentum_20",
        "drawdown_20",
        "news_count",
        "headline_score",
        "summary_score",
        "total_score",
        "unique_sources",
        "news_count_7d",
        "headline_score_7d",
        "summary_score_7d",
        "total_score_7d",
        "unique_sources_7d",
        "gdelt_crypto_news_count",
        "gdelt_macro_news_count",
        "gdelt_total_news_count",
        "gdelt_total_news_7d",
        "gdelt_total_news_30d",
        "gdelt_crypto_share",
    ]

    feature_columns.extend([column for column in merged.columns if column.startswith("fred_")])

    dataset = merged[["timestamp", "date", "symbol"] + feature_columns + ["future_return", "target", "target_label"]].dropna().reset_index(drop=True)
    dataset["target"] = dataset["target"].astype(int)
    return dataset


def save_dataset(data: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    data.to_csv(path, index=False)
    return path


def main() -> None:
    print(f"Saving datasets to: {OUTPUT_DIR}")
    symbol_frames: list[pd.DataFrame] = []
    all_company_news: list[pd.DataFrame] = []

    market_news = fetch_market_news()
    market_news_path = save_dataset(market_news, "market_news.csv")
    print(f"Saved {market_news_path.name} ({len(market_news)} rows)")

    fred_features = build_fred_features(FRED_SERIES_IDS)
    fred_path = save_dataset(fred_features, "fred_macro_features.csv")
    print(f"Saved {fred_path.name} ({len(fred_features)} rows)")

    gdelt_features = build_gdelt_features()
    gdelt_path = save_dataset(gdelt_features, "gdelt_event_features.csv")
    print(f"Saved {gdelt_path.name} ({len(gdelt_features)} rows)")

    earnings_calendar = fetch_earnings_calendar()
    earnings_calendar_path = save_dataset(earnings_calendar, "earnings_calendar.csv")
    print(f"Saved {earnings_calendar_path.name} ({len(earnings_calendar)} rows)")

    market_status = fetch_market_status(MARKET_EXCHANGE)
    market_status_path = save_dataset(market_status, f"market_status_{MARKET_EXCHANGE.lower()}.csv")
    print(f"Saved {market_status_path.name} ({len(market_status)} rows)")

    for index, symbol in enumerate(DATA_SYMBOLS):
        if FINNHUB_API_KEY:
            price_history = fetch_price_history_for_symbol(symbol, DATA_TIMEFRAME, DATA_LOOKBACK_DAYS)
        elif index == 0:
            price_history = load_price_history(PRICE_DATA_PATH)
            price_history["symbol"] = symbol
        else:
            raise RuntimeError("Multi-symbol dataset build requires FINNHUB_API_KEY so each symbol can fetch its own candles.")

        symbol_frames.append(price_history)
        company_news = fetch_company_news(symbol, NEWS_LOOKBACK_DAYS)
        all_company_news.append(company_news)
        company_news_path = save_dataset(company_news, f"{symbol.lower()}_company_news.csv")
        print(f"Saved {company_news_path.name} ({len(company_news)} rows)")

        earnings_surprises = fetch_earnings_surprises(symbol)
        earnings_surprises_path = save_dataset(earnings_surprises, f"{symbol.lower()}_earnings_surprises.csv")
        print(f"Saved {earnings_surprises_path.name} ({len(earnings_surprises)} rows)")

        quote = fetch_quote(symbol)
        quote_path = save_dataset(quote, f"{symbol.lower()}_quote.csv")
        print(f"Saved {quote_path.name} ({len(quote)} rows)")

        company_profile2 = fetch_company_profile2(symbol)
        profile_path = save_dataset(company_profile2, f"{symbol.lower()}_company_profile2.csv")
        print(f"Saved {profile_path.name} ({len(company_profile2)} rows)")

        recommendation_trends = fetch_recommendation_trends(symbol)
        recommendation_path = save_dataset(recommendation_trends, f"{symbol.lower()}_recommendation_trends.csv")
        print(f"Saved {recommendation_path.name} ({len(recommendation_trends)} rows)")

        peers = fetch_company_peers(symbol)
        peers_path = save_dataset(peers, f"{symbol.lower()}_peers.csv")
        print(f"Saved {peers_path.name} ({len(peers)} rows)")

    price_history = pd.concat(symbol_frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    price_path = save_dataset(price_history, "price_history.csv")
    print(f"Saved {price_path.name} ({len(price_history)} rows)")

    company_news_all = pd.concat(all_company_news, ignore_index=True) if all_company_news else pd.DataFrame()
    daily_company_news = build_daily_news_features(company_news_all)
    daily_market_news = build_daily_news_features(market_news)
    daily_news = (
        daily_company_news.merge(daily_market_news, on="date", how="outer", suffixes=("_company", "_market"))
        .fillna(0)
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily_news["news_count"] = daily_news.get("news_count_company", 0) + daily_news.get("news_count_market", 0)
    daily_news["headline_score"] = daily_news.get("headline_score_company", 0) + daily_news.get("headline_score_market", 0)
    daily_news["summary_score"] = daily_news.get("summary_score_company", 0) + daily_news.get("summary_score_market", 0)
    daily_news["total_score"] = daily_news.get("total_score_company", 0) + daily_news.get("total_score_market", 0)
    daily_news["unique_sources"] = daily_news.get("unique_sources_company", 0) + daily_news.get("unique_sources_market", 0)
    daily_news = daily_news[["date", "news_count", "headline_score", "summary_score", "total_score", "unique_sources"]]
    daily_news_path = save_dataset(daily_news, f"{DATA_SYMBOL.lower()}_daily_news_features.csv")
    print(f"Saved {daily_news_path.name} ({len(daily_news)} rows)")

    training_dataset = build_training_dataset(
        price_history,
        daily_news,
        fred_features,
        gdelt_features,
        horizon=TARGET_HORIZON,
        target_return_threshold=TARGET_RETURN_THRESHOLD,
        target_downside_threshold=TARGET_DOWNSIDE_THRESHOLD,
    )
    dataset_label = DATA_SYMBOL.lower() if len(DATA_SYMBOLS) == 1 else "multi_symbol"
    training_dataset_path = save_dataset(training_dataset, f"{dataset_label}_training_dataset.csv")
    print(f"Saved {training_dataset_path.name} ({len(training_dataset)} rows)")

    print("Done")


if __name__ == "__main__":
    main()
