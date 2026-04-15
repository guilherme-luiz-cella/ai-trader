"""Download historical OHLCV data for backtesting.

This script supports two data sources:
- Finnhub, if FINNHUB_API_KEY is set in the environment.
- Hyperliquid, as a fallback for the repo's existing crypto examples.

The output CSV is always written into backtest/data/ inside this workspace.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOL = os.getenv("DATA_SYMBOL", "AAPL")
DEFAULT_TIMEFRAME = os.getenv("DATA_TIMEFRAME", "1d")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "finnhub").lower()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

BATCH_SIZE = 5000


def timeframe_to_finnhub_resolution(timeframe: str) -> str:
    mapping = {
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
    if timeframe not in mapping:
        raise ValueError(f"Unsupported Finnhub timeframe: {timeframe}")
    return mapping[timeframe]


def fetch_finnhub_historical_data(symbol: str, timeframe: str, lookback_days: int = 365) -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is not set")

    resolution = timeframe_to_finnhub_resolution(timeframe)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    print("\nFetching Finnhub data")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe} -> {resolution}")
    print(f"From: {start_time.isoformat()}")
    print(f"To:   {end_time.isoformat()}")

    response = requests.get(
        "https://finnhub.io/api/v1/stock/candle",
        params={
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start_time.timestamp()),
            "to": int(end_time.timestamp()),
            "token": FINNHUB_API_KEY,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("s") != "ok":
        raise RuntimeError(f"Finnhub returned no data: {payload}")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert(None),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"],
        }
    )

    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_hyperliquid_historical_data(symbol: str, timeframe: str) -> pd.DataFrame:
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=60)

    response = requests.post(
        "https://api.hyperliquid.xyz/info",
        headers={"Content-Type": "application/json"},
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": timeframe,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000),
                "limit": BATCH_SIZE,
            },
        },
        timeout=20,
    )
    response.raise_for_status()
    snapshot_data = response.json()

    if not snapshot_data:
        return pd.DataFrame()

    rows = []
    for snapshot in snapshot_data:
        rows.append(
            [
                datetime.utcfromtimestamp(snapshot["t"] / 1000),
                snapshot["o"],
                snapshot["h"],
                snapshot["l"],
                snapshot["c"],
                snapshot["v"],
            ]
        )

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_historical_data(symbol: str, timeframe: str, provider: str = DATA_PROVIDER) -> pd.DataFrame:
    print("\nHistorical data fetcher")
    print(f"Provider: {provider}")

    if provider == "finnhub":
        if FINNHUB_API_KEY:
            return fetch_finnhub_historical_data(symbol, timeframe)

        print("FINNHUB_API_KEY is not set; falling back to Hyperliquid.")
        return fetch_hyperliquid_historical_data(symbol, timeframe)

    if provider == "hyperliquid":
        return fetch_hyperliquid_historical_data(symbol, timeframe)

    if FINNHUB_API_KEY:
        return fetch_finnhub_historical_data(symbol, timeframe)

    raise ValueError(f"Unknown data provider: {provider}")


def main() -> None:
    symbol = DEFAULT_SYMBOL
    timeframe = DEFAULT_TIMEFRAME

    all_data = fetch_historical_data(symbol, timeframe)

    if all_data.empty:
        print("No data to save.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = DATA_DIR / f"{symbol}_{timeframe}_{timestamp}_historical.csv"
    all_data.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")


if __name__ == "__main__":
    main()
