"""Train a simple supervised model from Finnhub OHLCV data.

This script is intentionally lightweight:
- downloads stock candles from Finnhub
- builds technical features from price/volume history
- labels each row by the next-bar direction
- trains a classifier
- saves the fitted model and evaluation metrics

Environment variables:
- FINNHUB_API_KEY: required
- TRAIN_SYMBOL: stock symbol to train on, default AAPL
- TRAIN_TIMEFRAME: Finnhub resolution, default 1d
- TRAIN_LOOKBACK_DAYS: lookback window, default 730
- TRAIN_TARGET_HORIZON: bars ahead to predict, default 1
- TRAIN_DATA_PATH: optional local CSV path for offline training fallback
- TRAIN_OUTPUT_DIR: output directory, default research/artifacts
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
OUTPUT_DIR = Path(os.getenv("TRAIN_OUTPUT_DIR", BASE_DIR / "artifacts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TRAIN_SYMBOL = os.getenv("TRAIN_SYMBOL", "AAPL")
TRAIN_TIMEFRAME = os.getenv("TRAIN_TIMEFRAME", "1d")
TRAIN_LOOKBACK_DAYS = int(os.getenv("TRAIN_LOOKBACK_DAYS", "730"))
TRAIN_TARGET_HORIZON = int(os.getenv("TRAIN_TARGET_HORIZON", "1"))
TRAIN_TARGET_RETURN_THRESHOLD = float(os.getenv("TRAIN_TARGET_RETURN_THRESHOLD", "0.003"))
TRAIN_TARGET_DOWNSIDE_THRESHOLD = float(os.getenv("TRAIN_TARGET_DOWNSIDE_THRESHOLD", "-0.003"))
TRAIN_DATA_PATH = os.getenv(
    "TRAIN_DATA_PATH",
    str(BASE_DIR.parent / "backtest" / "data" / "BTC-6h-1000wks-data.csv"),
)
TRAIN_DATA_PATHS = [item.strip() for item in os.getenv("TRAIN_DATA_PATHS", TRAIN_DATA_PATH).split(",") if item.strip()]

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


@dataclass(frozen=True)
class TrainingResult:
    symbol: str
    timeframe: str
    samples: int
    train_accuracy: float
    test_accuracy: float
    train_path: str
    test_path: str
    model_path: str
    metrics_path: str


def timeframe_to_resolution(timeframe: str) -> str:
    if timeframe not in TIMEFRAME_TO_RESOLUTION:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_TO_RESOLUTION[timeframe]


def fetch_finnhub_candles(symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is not set")

    resolution = timeframe_to_resolution(timeframe)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    response = requests.get(
        "https://finnhub.io/api/v1/stock/candle",
        params={
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start_time.timestamp()),
            "to": int(end_time.timestamp()),
            "token": FINNHUB_API_KEY,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Finnhub request failed with {response.status_code}: {response.text}")
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


def load_local_candles(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Local training data not found: {path}")

    data = pd.read_csv(path)
    timestamp_column = "datetime" if "datetime" in data.columns else "timestamp" if "timestamp" in data.columns else None
    if timestamp_column is None:
        raise ValueError(f"Expected a datetime or timestamp column in {path}")

    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data = data.rename(columns={column: column.lower() for column in data.columns})

    required_columns = ["open", "high", "low", "close", "volume"]
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f"Local training data must include columns: {required_columns}")

    data = data.rename(columns={timestamp_column: "timestamp"})
    data = data[["timestamp"] + required_columns]
    data["symbol"] = path.stem.upper()
    return data.sort_values("timestamp").reset_index(drop=True)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()

    frame["return_1"] = frame["close"].pct_change(1)
    frame["return_3"] = frame["close"].pct_change(3)
    frame["return_5"] = frame["close"].pct_change(5)

    frame["sma_5"] = frame["close"].rolling(5).mean()
    frame["sma_10"] = frame["close"].rolling(10).mean()
    frame["sma_20"] = frame["close"].rolling(20).mean()

    frame["close_sma_5"] = frame["close"] / frame["sma_5"] - 1
    frame["close_sma_10"] = frame["close"] / frame["sma_10"] - 1
    frame["close_sma_20"] = frame["close"] / frame["sma_20"] - 1

    frame["volatility_10"] = frame["return_1"].rolling(10).std()
    frame["volatility_20"] = frame["return_1"].rolling(20).std()

    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["body_pct"] = (frame["close"] - frame["open"]) / frame["open"]
    frame["upper_wick_pct"] = (frame["high"] - frame[["close", "open"]].max(axis=1)) / frame["close"]
    frame["lower_wick_pct"] = (frame[["close", "open"]].min(axis=1) - frame["low"]) / frame["close"]

    frame["volume_change_1"] = frame["volume"].pct_change(1)
    frame["volume_sma_10"] = frame["volume"] / frame["volume"].rolling(10).mean() - 1

    frame["rolling_high_10"] = frame["high"].rolling(10).max()
    frame["rolling_low_10"] = frame["low"].rolling(10).min()
    frame["distance_from_high_10"] = frame["close"] / frame["rolling_high_10"] - 1
    frame["distance_from_low_10"] = frame["close"] / frame["rolling_low_10"] - 1

    return frame


def build_dataset(
    data: pd.DataFrame,
    target_horizon: int,
    target_return_threshold: float,
    target_downside_threshold: float,
) -> Tuple[pd.DataFrame, pd.Series]:
    frame = build_features(data)
    frame["symbol_code"] = frame.get("symbol", "DEFAULT").astype("category").cat.codes
    frame["future_return"] = frame["close"].shift(-target_horizon) / frame["close"] - 1
    frame["target"] = pd.NA
    frame.loc[frame["future_return"] >= target_return_threshold, "target"] = 1
    frame.loc[frame["future_return"] <= target_downside_threshold, "target"] = 0

    feature_columns = [
        "symbol_code",
        "return_1",
        "return_3",
        "return_5",
        "close_sma_5",
        "close_sma_10",
        "close_sma_20",
        "volatility_10",
        "volatility_20",
        "range_pct",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "volume_change_1",
        "volume_sma_10",
        "distance_from_high_10",
        "distance_from_low_10",
    ]

    dataset = frame[["timestamp", "symbol"] + feature_columns + ["future_return", "target"]].dropna().reset_index(drop=True)
    dataset["target"] = dataset["target"].astype(int)
    features = dataset[feature_columns]
    labels = dataset["target"].astype(int)
    return dataset, labels


def chronological_split(features: pd.DataFrame, labels: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(features) * (1 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Not enough data after feature engineering to split train/test")

    x_train = features.iloc[:split_index].copy()
    x_test = features.iloc[split_index:].copy()
    y_train = labels.iloc[:split_index].copy()
    y_test = labels.iloc[split_index:].copy()
    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series):
    gradient_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[
            ("hist_gb", gradient_model),
            ("rf", forest_model),
        ],
        voting="soft",
    )
    calibrated_model = CalibratedClassifierCV(estimator=ensemble, method="sigmoid", cv=3)
    calibrated_model.fit(x_train, y_train)
    return calibrated_model


def evaluate_model(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    test_probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "classification_report": classification_report(y_test, test_predictions, output_dict=True, zero_division=0),
        "mean_test_probability": float(pd.Series(test_probabilities).mean()),
    }
    return metrics


def save_outputs(model, metrics: Dict[str, object], features: pd.DataFrame, symbol: str, timeframe: str) -> TrainingResult:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = OUTPUT_DIR / f"{symbol}_{timeframe}_{stamp}_model.joblib"
    metrics_path = OUTPUT_DIR / f"{symbol}_{timeframe}_{stamp}_metrics.json"
    train_path = OUTPUT_DIR / f"{symbol}_{timeframe}_{stamp}_dataset.csv"
    test_path = OUTPUT_DIR / f"{symbol}_{timeframe}_{stamp}_dataset_preview.csv"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    features.to_csv(train_path, index=False)
    features.tail(50).to_csv(test_path, index=False)

    return TrainingResult(
        symbol=symbol,
        timeframe=timeframe,
        samples=len(features),
        train_accuracy=metrics["train_accuracy"],
        test_accuracy=metrics["test_accuracy"],
        train_path=str(train_path),
        test_path=str(test_path),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )


def main() -> None:
    source_symbol = TRAIN_SYMBOL
    source_timeframe = TRAIN_TIMEFRAME

    try:
        if not FINNHUB_API_KEY:
            raise RuntimeError("FINNHUB_API_KEY is not set")

        candles = fetch_finnhub_candles(TRAIN_SYMBOL, TRAIN_TIMEFRAME, TRAIN_LOOKBACK_DAYS)
        candles["symbol"] = TRAIN_SYMBOL
    except Exception as error:
        print(f"Finnhub training data unavailable: {error}")
        print(f"Falling back to local CSV path(s): {', '.join(TRAIN_DATA_PATHS)}")
        candles = pd.concat([load_local_candles(path) for path in TRAIN_DATA_PATHS], ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        source_symbol = "multi_symbol_local" if len(TRAIN_DATA_PATHS) > 1 else Path(TRAIN_DATA_PATHS[0]).stem
        source_timeframe = "local"

    dataset, labels = build_dataset(
        candles,
        TRAIN_TARGET_HORIZON,
        TRAIN_TARGET_RETURN_THRESHOLD,
        TRAIN_TARGET_DOWNSIDE_THRESHOLD,
    )
    feature_columns = [column for column in dataset.columns if column not in {"timestamp", "symbol", "future_return", "target"}]
    features = dataset[feature_columns]
    x_train, x_test, y_train, y_test = chronological_split(features, labels)

    model = train_model(x_train, y_train)
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
    result = save_outputs(model, metrics, dataset, source_symbol, source_timeframe)

    print("Training complete")
    print(f"Symbol: {result.symbol}")
    print(f"Timeframe: {result.timeframe}")
    print(f"Samples: {result.samples}")
    print(f"Train accuracy: {result.train_accuracy:.4f}")
    print(f"Test accuracy: {result.test_accuracy:.4f}")
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.metrics_path}")


if __name__ == "__main__":
    main()
