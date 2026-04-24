"""Generate a simple trade signal from the latest combined model.

The signal is based on the most recent row in the combined training dataset.
It prints BUY / HOLD / SELL with the model probability for the positive class.

Environment variables:
- MODEL_PATH: optional path to a trained .joblib model
- DATASET_PATH: optional path to a combined dataset CSV
- BUY_THRESHOLD: default 0.55
- SELL_THRESHOLD: default 0.45
- LLM_ENABLED: set true to enable optional LLM overlay
- PRIMARY_MODEL / PRIMARY_MODEL_PATH: the trained model to use for local AI
- LLM_PROVIDER: runtime provider such as local_transformers, openai_compatible, groq, ollama, or huggingface_inference
- LLM_BASE_URL: runtime server base URL when using a server-based provider
- LLM_API_KEY: API key for selected provider
- ALLOW_MODEL_FALLBACK: set true to allow an explicit fallback model
- LLM_MERGE_ENABLED: set true to merge LLM with ML probability
- LLM_MERGE_WEIGHT: 0.0-0.5 weight assigned to LLM probability view
- LLM_CONFIDENCE_FLOOR: minimum LLM confidence required for merge
- ADAPTIVE_THRESHOLD_ENABLED: set true to auto-adjust thresholds
- VOLATILITY_REGIME_ENABLED: enable volatility regime switcher
- VOLATILITY_LOOKBACK_ROWS: recent rows used for volatility estimate
- VOLATILITY_LOW_PCT: low-volatility threshold
- VOLATILITY_HIGH_PCT: high-volatility threshold
- VOLATILITY_EXTREME_PCT: extreme-volatility threshold
- LLM_MERGE_WEIGHT_HIGH: elevated LLM merge weight in high-volatility regime
- HIGH_REGIME_REQUIRE_LLM_CONFIRMATION: require LLM confirmation in high-volatility regime
- HIGH_REGIME_MIN_LLM_CONFIDENCE: minimum LLM confidence for high-volatility confirmation
- EXTREME_REGIME_FORCE_HOLD: force HOLD in extreme volatility
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data_sets"

from backend.llm_client import ensure_llm_startup_logged, get_llm_status, llm_chat

MODEL_PATH = Path(os.getenv("MODEL_PATH", "")) if os.getenv("MODEL_PATH") else None
_dataset_path_env = os.getenv("DATASET_PATH", "").strip()
DATASET_PATH = Path(_dataset_path_env) if _dataset_path_env else (DATA_DIR / "aapl_training_dataset.csv")
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.55"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.45"))
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_MERGE_ENABLED = os.getenv("LLM_MERGE_ENABLED", "false").lower() == "true"
LLM_MERGE_WEIGHT = float(os.getenv("LLM_MERGE_WEIGHT", "0.20"))
LLM_CONFIDENCE_FLOOR = float(os.getenv("LLM_CONFIDENCE_FLOOR", "0.55"))
LLM_CONFIDENCE_SOFT_GATE = os.getenv("LLM_CONFIDENCE_SOFT_GATE", "false").lower() == "true"
ADAPTIVE_THRESHOLD_ENABLED = os.getenv("ADAPTIVE_THRESHOLD_ENABLED", "true").lower() == "true"
VOLATILITY_REGIME_ENABLED = os.getenv("VOLATILITY_REGIME_ENABLED", "true").lower() == "true"
VOLATILITY_LOOKBACK_ROWS = int(os.getenv("VOLATILITY_LOOKBACK_ROWS", "120"))
VOLATILITY_LOW_PCT = float(os.getenv("VOLATILITY_LOW_PCT", "1.20"))
VOLATILITY_HIGH_PCT = float(os.getenv("VOLATILITY_HIGH_PCT", "2.50"))
VOLATILITY_EXTREME_PCT = float(os.getenv("VOLATILITY_EXTREME_PCT", "4.00"))
LLM_MERGE_WEIGHT_HIGH = float(os.getenv("LLM_MERGE_WEIGHT_HIGH", "0.35"))
HIGH_REGIME_REQUIRE_LLM_CONFIRMATION = os.getenv("HIGH_REGIME_REQUIRE_LLM_CONFIRMATION", "true").lower() == "true"
HIGH_REGIME_MIN_LLM_CONFIDENCE = float(os.getenv("HIGH_REGIME_MIN_LLM_CONFIDENCE", "0.60"))
EXTREME_REGIME_FORCE_HOLD = os.getenv("EXTREME_REGIME_FORCE_HOLD", "true").lower() == "true"
MARKET_REGIME_TREND_LOOKBACK_ROWS = int(os.getenv("MARKET_REGIME_TREND_LOOKBACK_ROWS", "20"))
MARKET_REGIME_VOLUME_LOOKBACK_ROWS = int(os.getenv("MARKET_REGIME_VOLUME_LOOKBACK_ROWS", "20"))
MARKET_REGIME_TREND_THRESHOLD_PCT = float(os.getenv("MARKET_REGIME_TREND_THRESHOLD_PCT", "1.0"))
MARKET_REGIME_VOLUME_SPIKE_RATIO = float(os.getenv("MARKET_REGIME_VOLUME_SPIKE_RATIO", "1.25"))
CONFIDENCE_MERGE_ENABLED = os.getenv("CONFIDENCE_MERGE_ENABLED", "true").lower() == "true"
CONFIDENCE_MODEL_WEIGHT = float(os.getenv("CONFIDENCE_MODEL_WEIGHT", "0.50"))
CONFIDENCE_LLM_WEIGHT = float(os.getenv("CONFIDENCE_LLM_WEIGHT", "0.30"))
CONFIDENCE_DATA_WEIGHT = float(os.getenv("CONFIDENCE_DATA_WEIGHT", "0.20"))
CONFIDENCE_AGREEMENT_BONUS = float(os.getenv("CONFIDENCE_AGREEMENT_BONUS", "0.10"))

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
            "Missing dependency 'joblib'. Run this project with the repo virtual environment installed."
        ) from exc

    return joblib.load(model_path)


def load_latest_row(path: Path, model_feature_names: list[str] | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("Dataset is empty; cannot generate signal.")

    if model_feature_names:
        feature_columns = [str(name) for name in model_feature_names]
    else:
        excluded_columns = {"timestamp", "date", "target"}
        feature_columns = [
            column
            for column in data.columns
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(data[column])
        ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found in dataset")

    aligned = data.iloc[[-1]].copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0

    latest_row = aligned[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return latest_row, data.iloc[-1]


def _row_preview(row: pd.Series, max_items: int = 20) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    count = 0
    for key, value in row.items():
        if count >= max_items:
            break
        if isinstance(value, (int, float)):
            preview[str(key)] = float(value)
            count += 1
    return preview


def _extract_json_object(text: str) -> dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        candidate = content[start : end + 1]
        return json.loads(candidate)

    raise ValueError("LLM response did not contain valid JSON object.")


def generate_llm_overlay(
    probability_up: float,
    buy_threshold: float,
    sell_threshold: float,
    latest_row: pd.Series,
) -> dict[str, Any]:
    ensure_llm_startup_logged()
    if not LLM_ENABLED:
        return {
            "enabled": False,
            "status": "disabled",
            "message": "LLM overlay disabled.",
        }

    system_prompt = (
        "You are a conservative trading overlay validator. "
        "Your job is to validate an ML trading probability, not to invent extra conviction. "
        "If the evidence is weak, uncertain, malformed, or borderline, choose HOLD. "
        "Return exactly one JSON object with these keys only: llm_signal, confidence, rationale, risk_flags. "
        "llm_signal must be one of BUY, SELL, HOLD. "
        "confidence must be a JSON number between 0 and 1, not text, not a range, and not a percentage string. "
        "rationale must be a short plain-English sentence. "
        "risk_flags must be an array of short strings."
    )
    user_payload = {
        "probability_up": probability_up,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "latest_row_preview": _row_preview(latest_row),
        "instructions": {
            "llm_signal": "BUY|SELL|HOLD",
            "confidence": "numeric float only (e.g. 0.63)",
            "rationale": "short plain-English reason",
            "risk_flags": ["array of short risk strings"],
        },
    }

    try:
        llm_response = llm_chat(
            system_prompt=system_prompt,
            user_content=json.dumps(user_payload),
            max_new_tokens=220,
            response_format={"type": "json_object"},
        )
        if llm_response.get("status") != "ok":
            return {
                "enabled": True,
                "status": "error",
                "provider": llm_response.get("provider"),
                "model": llm_response.get("active_model") or llm_response.get("active_model_path"),
                "message": str(llm_response.get("error") or "LLM overlay failed."),
                "fallback_active": bool(llm_response.get("fallback_active", False)),
                "rate_limit": dict(llm_response.get("rate_limit") or {}),
            }

        content = str(llm_response.get("content", ""))
        parsed = _extract_json_object(content)
        validated = validate_llm_overlay_payload(parsed)
        if validated.get("status") != "ok":
            return {
                "enabled": True,
                "status": "invalid_response",
                "provider": llm_response.get("provider"),
                "model": llm_response.get("active_model") or llm_response.get("active_model_path"),
                "endpoint": llm_response.get("endpoint", ""),
                "is_trained_model": bool(llm_response.get("is_trained_model", False)),
                "fallback_active": bool(llm_response.get("fallback_active", False)),
                "llm_signal": "HOLD",
                "confidence": 0.0,
                "rationale": str(validated.get("message") or "Invalid LLM overlay response."),
                "risk_flags": ["invalid_response"],
                "raw_content": content,
                "rate_limit": dict(llm_response.get("rate_limit") or {}),
            }
        return {
            "enabled": True,
            "status": "ok",
            "provider": llm_response.get("provider"),
            "model": llm_response.get("active_model") or llm_response.get("active_model_path"),
            "endpoint": llm_response.get("endpoint", ""),
            "is_trained_model": bool(llm_response.get("is_trained_model", False)),
            "fallback_active": bool(llm_response.get("fallback_active", False)),
            "llm_signal": str(validated.get("llm_signal") or "HOLD"),
            "confidence": float(validated.get("confidence") or 0.0),
            "rationale": str(validated.get("rationale") or ""),
            "risk_flags": list(validated.get("risk_flags") or []),
            "rate_limit": dict(llm_response.get("rate_limit") or {}),
        }
    except Exception as exc:
        status = get_llm_status()
        return {
            "enabled": True,
            "status": "error",
            "provider": status.get("provider"),
            "model": status.get("active_model") or status.get("active_model_path"),
            "message": str(exc),
            "fallback_active": bool(status.get("fallback_active", False)),
        }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_llm_signal(value: Any) -> str | None:
    signal = str(value or "").strip().upper()
    if signal in {"BUY", "SELL", "HOLD"}:
        return signal
    return None


def parse_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return clamp(numeric, 0.0, 1.0)
        return None

    text = str(value or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text):
        return None
    try:
        return clamp(float(text), 0.0, 1.0)
    except ValueError:
        return None


def sanitize_risk_flags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text or text in cleaned:
            continue
        cleaned.append(text[:80])
    return cleaned[:5]


def validate_llm_overlay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    signal = normalize_llm_signal(payload.get("llm_signal"))
    if signal is None:
        return {"status": "error", "message": "llm_signal must be BUY, SELL, or HOLD."}

    confidence = parse_confidence(payload.get("confidence"))
    if confidence is None:
        return {"status": "error", "message": "confidence must be a numeric float between 0 and 1."}

    rationale = str(payload.get("rationale") or "").strip()
    if not rationale:
        return {"status": "error", "message": "rationale is required."}

    return {
        "status": "ok",
        "llm_signal": signal,
        "confidence": float(confidence),
        "rationale": rationale[:240],
        "risk_flags": sanitize_risk_flags(payload.get("risk_flags", [])),
    }


def compute_recent_volatility_pct(dataset_path: Path, lookback_rows: int) -> float:
    try:
        raw = pd.read_csv(dataset_path)
        if "close" not in raw.columns or len(raw) < 20:
            return 0.0
        close = pd.to_numeric(raw["close"], errors="coerce").dropna().tail(max(20, lookback_rows))
        if len(close) < 20:
            return 0.0
        returns = close.pct_change().dropna()
        if returns.empty:
            return 0.0
        return float(returns.std() * 100)
    except Exception:
        return 0.0


def compute_market_regime(dataset_path: Path) -> dict[str, Any]:
    volatility_pct = compute_recent_volatility_pct(dataset_path, VOLATILITY_LOOKBACK_ROWS)
    trend_bias = "neutral"
    volume_state = "normal"
    market_state = "range_bound"
    momentum_pct = 0.0
    trend_strength_pct = 0.0
    volume_ratio = 1.0
    range_position_pct = 0.5
    if not VOLATILITY_REGIME_ENABLED:
        return {
            "enabled": False,
            "regime": "disabled",
            "volatility_pct": volatility_pct,
            "notes": "Volatility regime controller disabled.",
        }

    try:
        raw = pd.read_csv(dataset_path)
        if {"close", "high", "low"}.issubset(raw.columns):
            close = pd.to_numeric(raw["close"], errors="coerce").dropna()
            high = pd.to_numeric(raw["high"], errors="coerce").dropna()
            low = pd.to_numeric(raw["low"], errors="coerce").dropna()
            lookback = max(5, MARKET_REGIME_TREND_LOOKBACK_ROWS)
            if len(close) >= lookback + 5:
                recent_close = close.iloc[-1]
                past_close = close.iloc[-lookback]
                sma_fast = close.tail(10).mean()
                sma_slow = close.tail(max(20, lookback)).mean()
                momentum_pct = float(((recent_close / past_close) - 1.0) * 100.0) if past_close else 0.0
                trend_strength_pct = float(((sma_fast / sma_slow) - 1.0) * 100.0) if sma_slow else 0.0
                rolling_high = high.tail(lookback).max()
                rolling_low = low.tail(lookback).min()
                if rolling_high > rolling_low:
                    range_position_pct = float((recent_close - rolling_low) / (rolling_high - rolling_low))
                if trend_strength_pct >= MARKET_REGIME_TREND_THRESHOLD_PCT and momentum_pct >= 0:
                    trend_bias = "bullish"
                elif trend_strength_pct <= -MARKET_REGIME_TREND_THRESHOLD_PCT and momentum_pct <= 0:
                    trend_bias = "bearish"
                else:
                    trend_bias = "neutral"
        if "volume" in raw.columns:
            volume = pd.to_numeric(raw["volume"], errors="coerce").dropna()
            volume_lookback = max(5, MARKET_REGIME_VOLUME_LOOKBACK_ROWS)
            if len(volume) >= volume_lookback:
                trailing = volume.tail(volume_lookback)
                trailing_mean = float(trailing.mean() or 0.0)
                latest_volume = float(trailing.iloc[-1] or 0.0)
                volume_ratio = (latest_volume / trailing_mean) if trailing_mean > 0 else 1.0
                if volume_ratio >= MARKET_REGIME_VOLUME_SPIKE_RATIO:
                    volume_state = "expanding"
                elif volume_ratio <= (2.0 - MARKET_REGIME_VOLUME_SPIKE_RATIO):
                    volume_state = "thin"
    except Exception:
        pass

    if volatility_pct >= VOLATILITY_EXTREME_PCT:
        regime = "extreme"
    elif volatility_pct >= VOLATILITY_HIGH_PCT:
        regime = "high"
    elif volatility_pct <= VOLATILITY_LOW_PCT:
        regime = "low"
    else:
        regime = "medium"

    if trend_bias == "bullish" and range_position_pct >= 0.60:
        market_state = "trend_up"
    elif trend_bias == "bearish" and range_position_pct <= 0.40:
        market_state = "trend_down"
    elif regime in {"high", "extreme"}:
        market_state = "volatile"

    notes = [
        f"volatility regime={regime}",
        f"trend_bias={trend_bias}",
        f"market_state={market_state}",
        f"volume_state={volume_state}",
    ]

    return {
        "enabled": True,
        "regime": regime,
        "volatility_pct": volatility_pct,
        "trend_bias": trend_bias,
        "market_state": market_state,
        "volume_state": volume_state,
        "momentum_pct": momentum_pct,
        "trend_strength_pct": trend_strength_pct,
        "volume_ratio": volume_ratio,
        "range_position_pct": range_position_pct,
        "notes": notes,
        "thresholds": {
            "low_pct": VOLATILITY_LOW_PCT,
            "high_pct": VOLATILITY_HIGH_PCT,
            "extreme_pct": VOLATILITY_EXTREME_PCT,
        },
    }


def llm_signal_to_probability(llm_signal: str, llm_confidence: float) -> float:
    signal = (llm_signal or "HOLD").upper()
    conf = clamp(float(llm_confidence or 0.0), 0.0, 1.0)
    if signal == "BUY":
        return 0.5 + (0.5 * conf)
    if signal == "SELL":
        return 0.5 - (0.5 * conf)
    return 0.5


def merge_ml_llm_decision(
    ml_probability_up: float,
    buy_threshold: float,
    sell_threshold: float,
    llm_overlay: dict[str, Any],
    market_regime: dict[str, Any],
    data_confidence: float,
) -> dict[str, Any]:
    base = {
        "enabled": False,
        "status": "disabled",
        "merged_probability_up": float(ml_probability_up),
        "merge_weight": 0.0,
        "llm_probability_view": 0.5,
        "reason": "LLM merge disabled.",
    }

    if not LLM_MERGE_ENABLED:
        return base

    if llm_overlay.get("status") != "ok":
        return {
            "enabled": True,
            "status": "llm_unavailable",
            "merged_probability_up": float(ml_probability_up),
            "merge_weight": 0.0,
            "llm_probability_view": 0.5,
            "reason": "LLM overlay unavailable, using ML only.",
        }

    llm_conf = clamp(float(llm_overlay.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    llm_prob_view = llm_signal_to_probability(str(llm_overlay.get("llm_signal", "HOLD")), llm_conf)
    regime = str(market_regime.get("regime", "disabled"))
    base_merge_weight = clamp(LLM_MERGE_WEIGHT, 0.0, 0.5)
    data_boost = 0.70 + (0.60 * clamp(data_confidence, 0.0, 1.0))
    base_merge_weight = clamp(base_merge_weight * data_boost, 0.0, 0.5)
    if market_regime.get("enabled"):
        if regime == "low":
            base_merge_weight = 0.0
        elif regime == "high":
            base_merge_weight = max(base_merge_weight, clamp(LLM_MERGE_WEIGHT_HIGH, 0.0, 0.5))
        elif regime == "extreme":
            base_merge_weight = 0.0

    if base_merge_weight <= 0:
        return {
            "enabled": True,
            "status": "regime_disabled_merge",
            "merged_probability_up": float(ml_probability_up),
            "merge_weight": 0.0,
            "llm_probability_view": llm_prob_view,
            "reason": f"LLM merge disabled for {regime} volatility regime.",
        }

    floor = clamp(LLM_CONFIDENCE_FLOOR, 0.0, 1.0)

    status = "ok"
    reason = "Merged ML and LLM using confidence-gated weighted blend."
    if llm_conf < floor:
        if LLM_CONFIDENCE_SOFT_GATE and floor > 0:
            # Keep a reduced LLM influence instead of dropping it entirely.
            confidence_ratio = clamp(llm_conf / floor, 0.0, 1.0)
            merge_weight = base_merge_weight * confidence_ratio
            status = "confidence_scaled"
            reason = "LLM confidence below floor, merge weight scaled down proportionally."
        else:
            return {
                "enabled": True,
                "status": "confidence_too_low",
                "merged_probability_up": float(ml_probability_up),
                "merge_weight": 0.0,
                "llm_probability_view": llm_prob_view,
                "reason": "LLM confidence below floor, using ML only.",
            }
    else:
        merge_weight = base_merge_weight

    merged_prob = ((1.0 - merge_weight) * float(ml_probability_up)) + (merge_weight * llm_prob_view)

    merged_signal = "HOLD"
    if merged_prob >= buy_threshold:
        merged_signal = "BUY"
    elif merged_prob <= sell_threshold:
        merged_signal = "SELL"

    return {
        "enabled": True,
        "status": status,
        "merged_probability_up": float(merged_prob),
        "merge_weight": float(merge_weight),
        "llm_probability_view": float(llm_prob_view),
        "llm_signal": str(llm_overlay.get("llm_signal", "HOLD")).upper(),
        "llm_confidence": float(llm_conf),
        "merged_signal": merged_signal,
        "data_confidence": float(clamp(data_confidence, 0.0, 1.0)),
        "data_boost": float(data_boost),
        "reason": reason,
    }


def compute_data_context_confidence(latest_row: pd.Series) -> float:
    if "data_context_coverage" in latest_row.index:
        try:
            return clamp(float(latest_row.get("data_context_coverage", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            pass

    contextual_columns = [
        col
        for col in latest_row.index
        if str(col).startswith(("fred_", "gdelt_", "fng_", "cg_"))
        or str(col) in {"news_count", "headline_score", "summary_score", "total_score", "unique_sources"}
    ]
    if not contextual_columns:
        return 0.5

    values: list[float] = []
    for column in contextual_columns:
        raw_value = latest_row.get(column)
        if pd.isna(raw_value):
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            continue

    if not values:
        return 0.0

    nonnull_ratio = len(values) / max(1, len(contextual_columns))
    nonzero_ratio = sum(abs(item) > 1e-12 for item in values) / len(values)
    return clamp((0.5 * nonnull_ratio) + (0.5 * nonzero_ratio), 0.0, 1.0)


def compute_model_confidence(probability_up: float, buy_threshold: float, sell_threshold: float) -> float:
    probability = clamp(float(probability_up), 0.0, 1.0)
    edge_from_mid = abs(probability - 0.5) * 2.0
    nearest_boundary = min(abs(probability - buy_threshold), abs(probability - sell_threshold))
    threshold_band = max(0.01, (buy_threshold - sell_threshold) / 2.0)
    boundary_strength = clamp(nearest_boundary / threshold_band, 0.0, 1.0)
    return clamp((0.6 * edge_from_mid) + (0.4 * boundary_strength), 0.0, 1.0)


def normalize_confidence_weights() -> tuple[float, float, float]:
    model_weight = max(0.0, float(CONFIDENCE_MODEL_WEIGHT))
    llm_weight = max(0.0, float(CONFIDENCE_LLM_WEIGHT))
    data_weight = max(0.0, float(CONFIDENCE_DATA_WEIGHT))
    total = model_weight + llm_weight + data_weight
    if total <= 0:
        return 0.5, 0.3, 0.2
    return model_weight / total, llm_weight / total, data_weight / total


def compose_decision_confidence(
    final_signal: str,
    ml_signal: str,
    final_probability_up: float,
    buy_threshold: float,
    sell_threshold: float,
    llm_overlay: dict[str, Any],
    data_confidence: float,
) -> tuple[float, dict[str, Any]]:
    model_confidence = compute_model_confidence(final_probability_up, buy_threshold, sell_threshold)
    llm_confidence = 0.0
    llm_signal = str(llm_overlay.get("llm_signal", "HOLD")).upper()
    if llm_overlay.get("status") == "ok":
        llm_confidence = clamp(float(llm_overlay.get("confidence", 0.0) or 0.0), 0.0, 1.0)

    model_weight, llm_weight, data_weight = normalize_confidence_weights()
    merged_confidence = (
        (model_confidence * model_weight)
        + (llm_confidence * llm_weight)
        + (clamp(data_confidence, 0.0, 1.0) * data_weight)
    )

    if str(ml_signal).upper() == str(final_signal).upper() and llm_signal == str(final_signal).upper() and str(final_signal).upper() in {"BUY", "SELL"}:
        merged_confidence += clamp(CONFIDENCE_AGREEMENT_BONUS, 0.0, 0.20)

    if str(final_signal).upper() == "HOLD":
        merged_confidence *= 0.90

    merged_confidence = clamp(merged_confidence, 0.0, 1.0)
    return merged_confidence, {
        "enabled": CONFIDENCE_MERGE_ENABLED,
        "model_confidence": model_confidence,
        "llm_confidence": llm_confidence,
        "data_confidence": clamp(data_confidence, 0.0, 1.0),
        "weights": {
            "model": model_weight,
            "llm": llm_weight,
            "data": data_weight,
        },
        "agreement_bonus": clamp(CONFIDENCE_AGREEMENT_BONUS, 0.0, 0.20),
        "ml_signal": str(ml_signal).upper(),
        "llm_signal": llm_signal,
        "final_signal": str(final_signal).upper(),
        "merged_confidence": merged_confidence,
    }


def compute_adaptive_thresholds(
    dataset_path: Path,
    base_buy_threshold: float,
    base_sell_threshold: float,
) -> tuple[float, float, dict[str, float | str]]:
    volatility_pct = compute_recent_volatility_pct(dataset_path, VOLATILITY_LOOKBACK_ROWS)

    vol_factor = clamp(volatility_pct / 2.5, 0.0, 1.0)
    dynamic_margin = 0.015 + (0.065 * vol_factor)

    adaptive_buy = clamp(base_buy_threshold + dynamic_margin, 0.50, 0.95)
    adaptive_sell = clamp(base_sell_threshold - dynamic_margin, 0.05, 0.50)

    min_gap = 0.08
    if adaptive_buy - adaptive_sell < min_gap:
        midpoint = (adaptive_buy + adaptive_sell) / 2
        adaptive_buy = clamp(midpoint + (min_gap / 2), 0.50, 0.95)
        adaptive_sell = clamp(midpoint - (min_gap / 2), 0.05, 0.50)

    return adaptive_buy, adaptive_sell, {
        "mode": "adaptive",
        "volatility_pct": volatility_pct,
        "dynamic_margin": dynamic_margin,
        "buy_threshold": adaptive_buy,
        "sell_threshold": adaptive_sell,
        "base_buy_threshold": base_buy_threshold,
        "base_sell_threshold": base_sell_threshold,
    }


def generate_trade_decision(
    model_path: Path,
    dataset_path: Path,
    base_buy_threshold: float,
    base_sell_threshold: float,
    adaptive_threshold_enabled: bool,
) -> dict[str, Any]:
    model = load_model(model_path)
    model_feature_names = list(getattr(model, "feature_names_in_", []))
    features, full_row = load_latest_row(dataset_path, model_feature_names=model_feature_names or None)

    effective_buy_threshold = float(base_buy_threshold)
    effective_sell_threshold = float(base_sell_threshold)
    threshold_context: dict[str, Any] = {
        "mode": "manual",
        "dynamic_margin": 0.0,
        "buy_threshold": effective_buy_threshold,
        "sell_threshold": effective_sell_threshold,
        "base_buy_threshold": base_buy_threshold,
        "base_sell_threshold": base_sell_threshold,
    }

    if adaptive_threshold_enabled:
        adaptive_buy, adaptive_sell, adaptive_ctx = compute_adaptive_thresholds(
            dataset_path=dataset_path,
            base_buy_threshold=base_buy_threshold,
            base_sell_threshold=base_sell_threshold,
        )
        effective_buy_threshold = adaptive_buy
        effective_sell_threshold = adaptive_sell
        threshold_context = adaptive_ctx

    market_regime = compute_market_regime(dataset_path)

    probability = float(model.predict_proba(features)[0][1])
    if probability >= effective_buy_threshold:
        signal = "BUY"
    elif probability <= effective_sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    llm_overlay = generate_llm_overlay(
        probability_up=probability,
        buy_threshold=effective_buy_threshold,
        sell_threshold=effective_sell_threshold,
        latest_row=full_row,
    )
    data_confidence = compute_data_context_confidence(full_row)
    merge_result = merge_ml_llm_decision(
        ml_probability_up=probability,
        buy_threshold=effective_buy_threshold,
        sell_threshold=effective_sell_threshold,
        llm_overlay=llm_overlay,
        market_regime=market_regime,
        data_confidence=data_confidence,
    )

    final_probability_up = float(merge_result.get("merged_probability_up", probability))
    if final_probability_up >= effective_buy_threshold:
        final_signal = "BUY"
    elif final_probability_up <= effective_sell_threshold:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    safety_guard: dict[str, Any] = {
        "triggered": False,
        "reason": "",
    }
    regime = str(market_regime.get("regime", "disabled"))
    ml_signal = signal
    llm_signal = str(llm_overlay.get("llm_signal", "HOLD")).upper()
    llm_confidence = clamp(float(llm_overlay.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    llm_ok = llm_overlay.get("status") == "ok"

    if market_regime.get("enabled") and regime == "extreme" and EXTREME_REGIME_FORCE_HOLD:
        final_signal = "HOLD"
        safety_guard = {
            "triggered": True,
            "reason": "Extreme volatility regime: forced HOLD.",
        }
    elif market_regime.get("enabled") and regime == "high" and HIGH_REGIME_REQUIRE_LLM_CONFIRMATION:
        if not llm_ok or llm_confidence < clamp(HIGH_REGIME_MIN_LLM_CONFIDENCE, 0.0, 1.0):
            final_signal = "HOLD"
            safety_guard = {
                "triggered": True,
                "reason": "High volatility regime: LLM confirmation unavailable or confidence too low.",
            }
        elif ml_signal in {"BUY", "SELL"} and llm_signal not in {ml_signal, "HOLD"}:
            final_signal = "HOLD"
            safety_guard = {
                "triggered": True,
                "reason": "High volatility regime: ML and LLM disagree, forced HOLD.",
            }

    decision_confidence = 0.0
    confidence_breakdown: dict[str, Any] = {
        "enabled": False,
        "reason": "Confidence composer disabled.",
    }
    if CONFIDENCE_MERGE_ENABLED:
        decision_confidence, confidence_breakdown = compose_decision_confidence(
            final_signal=final_signal,
            ml_signal=signal,
            final_probability_up=final_probability_up,
            buy_threshold=effective_buy_threshold,
            sell_threshold=effective_sell_threshold,
            llm_overlay=llm_overlay,
            data_confidence=data_confidence,
        )

    return {
        "signal": final_signal,
        "probability_up": final_probability_up,
        "buy_threshold": effective_buy_threshold,
        "sell_threshold": effective_sell_threshold,
        "threshold_context": threshold_context,
        "decision_engine": "ml_regime_hybrid" if VOLATILITY_REGIME_ENABLED else ("ml_llm_merged" if LLM_MERGE_ENABLED else "ml_primary"),
        "market_regime": market_regime,
        "ml_signal": signal,
        "ml_probability_up": probability,
        "llm_overlay": llm_overlay,
        "llm_merge": merge_result,
        "decision_confidence": decision_confidence,
        "confidence_breakdown": confidence_breakdown,
        "safety_guard": safety_guard,
        "latest_timestamp": str(full_row.get("timestamp", "")),
    }

def main() -> None:
    model_path = resolve_model_path()
    decision = generate_trade_decision(
        model_path=model_path,
        dataset_path=DATASET_PATH,
        base_buy_threshold=BUY_THRESHOLD,
        base_sell_threshold=SELL_THRESHOLD,
        adaptive_threshold_enabled=ADAPTIVE_THRESHOLD_ENABLED,
    )

    result = {
        **decision,
        "adaptive_threshold_enabled": ADAPTIVE_THRESHOLD_ENABLED,
        "model_path": str(model_path),
        "dataset_path": str(DATASET_PATH),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
