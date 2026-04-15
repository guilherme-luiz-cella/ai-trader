"""Generate a simple trade signal from the latest combined model.

The signal is based on the most recent row in the combined training dataset.
It prints BUY / HOLD / SELL with the model probability for the positive class.

Environment variables:
- MODEL_PATH: optional path to a trained .joblib model
- DATASET_PATH: optional path to a combined dataset CSV
- BUY_THRESHOLD: default 0.55
- SELL_THRESHOLD: default 0.45
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data_sets"

MODEL_PATH = Path(os.getenv("MODEL_PATH", "")) if os.getenv("MODEL_PATH") else None
DATASET_PATH = Path(os.getenv("DATASET_PATH", DATA_DIR / "aapl_training_dataset.csv"))
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.55"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.45"))


def latest_file_with_suffix(folder: Path, suffix: str) -> Optional[Path]:
    files = sorted(folder.glob(f"*{suffix}"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def resolve_model_path() -> Path:
    if MODEL_PATH is not None and MODEL_PATH.exists():
        return MODEL_PATH

    latest_model = latest_file_with_suffix(ARTIFACTS_DIR, "_model.joblib")
    if latest_model is None:
        raise FileNotFoundError("No model artifact found in research/artifacts")
    return latest_model


def load_model(model_path: Path):
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'joblib'. Run the dashboard with the project venv: "
            "./.venv/bin/python -m streamlit run research/streamlit_dashboard.py"
        ) from exc

    return joblib.load(model_path)


def load_latest_row(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    excluded_columns = {"timestamp", "date", "target"}
    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(data[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found in dataset")

    latest_row = data.iloc[[-1]][feature_columns]
    return latest_row, data.iloc[-1]


def main() -> None:
    model_path = resolve_model_path()
    model = load_model(model_path)
    features, full_row = load_latest_row(DATASET_PATH)

    probability = float(model.predict_proba(features)[0][1])
    if probability >= BUY_THRESHOLD:
        signal = "BUY"
    elif probability <= SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    result = {
        "signal": signal,
        "probability_up": probability,
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "model_path": str(model_path),
        "dataset_path": str(DATASET_PATH),
        "latest_timestamp": str(full_row.get("timestamp", "")),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
