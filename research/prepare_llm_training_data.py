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
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.generate_trade_signal import build_llm_market_snapshot, clamp  # noqa: E402

DATASET_PATH = Path(
    os.getenv(
        "COMBINED_DATASET_PATH",
        str(BASE_DIR / "data_sets" / f"{os.getenv('DATA_SYMBOL', 'AAPL').lower()}_training_dataset.csv"),
    )
)
OUTPUT_PATH = Path(
    os.getenv(
        "LLM_TRAIN_OUTPUT_PATH",
        str(BASE_DIR / "artifacts" / f"{DATASET_PATH.stem.replace('_training_dataset', '')}_llm_finetune.jsonl"),
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


def row_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        value = row.get(column, default)
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def probability_from_row(row: pd.Series) -> float:
    future_return = row_float(row, "future_return", 0.0)
    if abs(future_return) > 0:
        return clamp(0.5 + (future_return * 20.0), 0.05, 0.95)
    return 0.68 if int(row.get("target", 0) or 0) == 1 else 0.32


def confidence_from_row(row: pd.Series, action: str) -> float:
    future_return = abs(row_float(row, "future_return", 0.0))
    data_confidence = row_float(row, "data_context_coverage", 0.5)
    edge_confidence = clamp(future_return / 0.025, 0.15, 0.90)
    confidence = (0.70 * edge_confidence) + (0.30 * clamp(data_confidence, 0.0, 1.0))
    if action == "HOLD":
        confidence *= 0.85
    return round(clamp(confidence, 0.05, 0.95), 2)


def risk_flags_from_row(row: pd.Series, probability_up: float, buy_threshold: float, sell_threshold: float) -> list[str]:
    flags: list[str] = []
    if min(abs(probability_up - buy_threshold), abs(probability_up - sell_threshold)) < 0.02:
        flags.append("threshold_proximity")
    if row_float(row, "data_context_coverage", 0.5) < 0.70:
        flags.append("missing_context")
    if row_float(row, "volume_sma_20", 0.0) < -0.35:
        flags.append("thin_volume")
    if row_float(row, "volatility_20", 0.0) > 0.025:
        flags.append("elevated_volatility")
    return flags[:5]


def action_rationale(action: str, probability_up: float, risk_flags: list[str]) -> str:
    if action == "BUY":
        base = "Positive forward outcome and model edge support a cautious BUY."
    elif action == "SELL":
        base = "Negative forward outcome and weak upside probability support SELL."
    else:
        base = "Mixed or borderline evidence supports HOLD."
    if risk_flags:
        return f"{base} Key risks: {', '.join(risk_flags[:3])}."
    return base


def numeric_preview(row: pd.Series, max_items: int) -> dict[str, float]:
    preview: dict[str, float] = {}
    for key, value in row.items():
        if len(preview) >= max_items:
            break
        if pd.api.types.is_number(value):
            preview[str(key)] = float(value)
    return preview


def build_example(row: pd.Series, feature_columns: list[str], target_column: str) -> dict:
    action = label_to_action(int(row[target_column]))
    buy_threshold = float(os.getenv("BUY_THRESHOLD", "0.55"))
    sell_threshold = float(os.getenv("SELL_THRESHOLD", "0.45"))
    probability_up = probability_from_row(row)
    data_confidence = row_float(row, "data_context_coverage", 0.5)
    market_snapshot = build_llm_market_snapshot(
        probability_up=probability_up,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        latest_row=row,
        market_regime={},
        data_confidence=data_confidence,
    )
    risk_flags = risk_flags_from_row(row, probability_up, buy_threshold, sell_threshold)
    assistant_payload = {
        "llm_signal": action,
        "confidence": confidence_from_row(row, action),
        "rationale": action_rationale(action, probability_up, risk_flags),
        "risk_flags": risk_flags,
    }

    system_prompt = (
        "You are a conservative crypto trading overlay validator for an ML signal. "
        "Validate the ML probability using only the JSON snapshot. "
        "Return exactly one JSON object with keys llm_signal, confidence, rationale, risk_flags."
    )
    user_prompt = json.dumps(
        {
            "market_snapshot": market_snapshot,
            "legacy_feature_preview": numeric_preview(row[feature_columns], max_items=MAX_FEATURES),
            "instructions": {
                "llm_signal": "BUY|SELL|HOLD",
                "confidence": "numeric float between 0 and 1",
                "rationale": "short plain-English reason",
                "risk_flags": ["array of short risk strings"],
            },
        },
        separators=(",", ":"),
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant_payload, separators=(",", ":"))},
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
