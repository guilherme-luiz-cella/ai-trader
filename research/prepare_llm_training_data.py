"""Build chat fine-tuning examples from the local training dataset.

This script converts your existing numeric market dataset into instruction-style
JSONL examples so you can fine-tune an LLM with any OpenAI-compatible provider.

Environment variables:
- COMBINED_DATASET_PATH: input CSV path
- LLM_TRAIN_OUTPUT_PATH: output JSONL path
- LLM_MAX_FEATURES: max numeric fields included per sample
- LLM_MAX_SAMPLES: optional cap on generated rows (latest rows kept)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = Path(
    os.getenv(
        "COMBINED_DATASET_PATH",
        str(BASE_DIR / "data_sets" / f"{os.getenv('DATA_SYMBOL', 'AAPL').lower()}_training_dataset.csv"),
    )
)
OUTPUT_PATH = Path(
    os.getenv(
        "LLM_TRAIN_OUTPUT_PATH",
        str(BASE_DIR / "artifacts" / f"{os.getenv('DATA_SYMBOL', 'AAPL').lower()}_llm_finetune.jsonl"),
    )
)
MAX_FEATURES = int(os.getenv("LLM_MAX_FEATURES", "24"))
MAX_SAMPLES = int(os.getenv("LLM_MAX_SAMPLES", "0"))


def label_to_action(label: int) -> str:
    if int(label) == 1:
        return "BUY"
    if int(label) == 0:
        return "SELL"
    return "HOLD"


def numeric_preview(row: pd.Series, max_items: int) -> dict[str, float]:
    preview: dict[str, float] = {}
    for key, value in row.items():
        if len(preview) >= max_items:
            break
        if pd.api.types.is_number(value):
            preview[str(key)] = float(value)
    return preview


def build_example(row: pd.Series, feature_columns: list[str], target_column: str) -> dict:
    feature_payload = numeric_preview(row[feature_columns], max_items=MAX_FEATURES)
    action = label_to_action(int(row[target_column]))

    system_prompt = (
        "You are a conservative trading model. "
        "Analyze the numeric market feature snapshot and return only one action: BUY, SELL, or HOLD."
    )
    user_prompt = (
        "Predict the next action from this market snapshot.\n"
        f"features={json.dumps(feature_payload, separators=(',', ':'))}"
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": action},
        ]
    }


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    if "target" not in data.columns:
        raise ValueError("Expected target column named 'target' in dataset.")

    excluded_columns = {"target", "timestamp", "date"}
    feature_columns = [
        col
        for col in data.columns
        if col not in excluded_columns and pd.api.types.is_numeric_dtype(data[col])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    if MAX_SAMPLES > 0:
        data = data.tail(MAX_SAMPLES).copy()

    rows = []
    for _, row in data.iterrows():
        rows.append(build_example(row=row, feature_columns=feature_columns, target_column="target"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    print(f"Created {len(rows)} chat fine-tune samples.")
    print(f"Input dataset: {DATASET_PATH}")
    print(f"Output JSONL: {OUTPUT_PATH}")
    print("Next step: upload this JSONL to your LLM provider fine-tuning API.")


if __name__ == "__main__":
    main()
