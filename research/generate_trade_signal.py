"""Generate a simple trade signal from the latest combined model.

The signal is based on the most recent row in the combined training dataset.
It prints BUY / HOLD / SELL with the model probability for the positive class.

Environment variables:
- MODEL_PATH: optional path to a trained .joblib model
- DATASET_PATH: optional path to a combined dataset CSV
- BUY_THRESHOLD: default 0.55
- SELL_THRESHOLD: default 0.45
- LLM_ENABLED: set true to enable optional LLM overlay
- LLM_PROVIDER: ollama (default), openai_compatible, or huggingface_inference
- LLM_MODEL: default qwen2.5:7b-instruct
- LLM_BASE_URL: default http://127.0.0.1:11434
- LLM_API_KEY: API key for selected provider
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
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data_sets"

MODEL_PATH = Path(os.getenv("MODEL_PATH", "")) if os.getenv("MODEL_PATH") else None
_dataset_path_env = os.getenv("DATASET_PATH", "").strip()
DATASET_PATH = Path(_dataset_path_env) if _dataset_path_env else (DATA_DIR / "aapl_training_dataset.csv")
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.55"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "0.45"))
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MERGE_ENABLED = os.getenv("LLM_MERGE_ENABLED", "false").lower() == "true"
LLM_MERGE_WEIGHT = float(os.getenv("LLM_MERGE_WEIGHT", "0.20"))
LLM_CONFIDENCE_FLOOR = float(os.getenv("LLM_CONFIDENCE_FLOOR", "0.55"))
LLM_CONFIDENCE_SOFT_GATE = os.getenv("LLM_CONFIDENCE_SOFT_GATE", "true").lower() == "true"
ADAPTIVE_THRESHOLD_ENABLED = os.getenv("ADAPTIVE_THRESHOLD_ENABLED", "true").lower() == "true"
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_BYPASS_ENV_PROXY = os.getenv("LLM_BYPASS_ENV_PROXY", "true").lower() == "true"
VOLATILITY_REGIME_ENABLED = os.getenv("VOLATILITY_REGIME_ENABLED", "true").lower() == "true"
VOLATILITY_LOOKBACK_ROWS = int(os.getenv("VOLATILITY_LOOKBACK_ROWS", "120"))
VOLATILITY_LOW_PCT = float(os.getenv("VOLATILITY_LOW_PCT", "1.20"))
VOLATILITY_HIGH_PCT = float(os.getenv("VOLATILITY_HIGH_PCT", "2.50"))
VOLATILITY_EXTREME_PCT = float(os.getenv("VOLATILITY_EXTREME_PCT", "4.00"))
LLM_MERGE_WEIGHT_HIGH = float(os.getenv("LLM_MERGE_WEIGHT_HIGH", "0.35"))
HIGH_REGIME_REQUIRE_LLM_CONFIRMATION = os.getenv("HIGH_REGIME_REQUIRE_LLM_CONFIRMATION", "true").lower() == "true"
HIGH_REGIME_MIN_LLM_CONFIDENCE = float(os.getenv("HIGH_REGIME_MIN_LLM_CONFIDENCE", "0.60"))
EXTREME_REGIME_FORCE_HOLD = os.getenv("EXTREME_REGIME_FORCE_HOLD", "true").lower() == "true"


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


def _llm_endpoint() -> str:
    if LLM_PROVIDER in {"deepseek", "openai_compatible"}:
        return f"{LLM_BASE_URL}/chat/completions"
    if LLM_PROVIDER == "ollama":
        return f"{LLM_BASE_URL}/api/chat"
    if LLM_PROVIDER == "huggingface_inference":
        base = LLM_BASE_URL if LLM_BASE_URL else "https://api-inference.huggingface.co"
        return f"{base}/models/{LLM_MODEL}"
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


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


def _post_llm_request(endpoint: str, headers: dict[str, str], payload: dict[str, Any]) -> requests.Response:
    try:
        return requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ProxyError:
        if not LLM_BYPASS_ENV_PROXY:
            raise

    with requests.Session() as session:
        session.trust_env = False
        return session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )


def generate_llm_overlay(
    probability_up: float,
    buy_threshold: float,
    sell_threshold: float,
    latest_row: pd.Series,
) -> dict[str, Any]:
    if not LLM_ENABLED:
        return {
            "enabled": False,
            "status": "disabled",
            "message": "LLM overlay disabled.",
        }

    if LLM_PROVIDER != "ollama" and not LLM_API_KEY:
        return {
            "enabled": True,
            "status": "missing_api_key",
            "provider": LLM_PROVIDER,
            "message": "LLM_API_KEY missing; cannot run LLM overlay.",
        }

    system_prompt = (
        "You are a risk-aware trading assistant. "
        "Given model probability and thresholds, output conservative recommendation. "
        "Return strict JSON only with keys: llm_signal, confidence, rationale, risk_flags."
    )
    user_payload = {
        "probability_up": probability_up,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "latest_row_preview": _row_preview(latest_row),
        "instructions": {
            "llm_signal": "BUY|SELL|HOLD",
            "confidence": "0.0-1.0",
            "rationale": "short plain-English reason",
            "risk_flags": ["array of short risk strings"],
        },
    }

    try:
        headers = {
            "Content-Type": "application/json",
        }
        if LLM_PROVIDER in {"deepseek", "openai_compatible", "huggingface_inference"}:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        if LLM_PROVIDER == "huggingface_inference":
            hf_prompt = (
                f"{system_prompt}\n"
                "Output only JSON object with keys llm_signal, confidence, rationale, risk_flags.\n"
                f"Input payload: {json.dumps(user_payload)}"
            )
            payload: dict[str, Any] = {
                "inputs": hf_prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.2,
                    "return_full_text": False,
                },
            }
        elif LLM_PROVIDER == "ollama":
            payload = {
                "model": LLM_MODEL,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            }
        else:
            payload = {
                "model": LLM_MODEL,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            }

        endpoint = _llm_endpoint()
        response = _post_llm_request(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
        )
        response.raise_for_status()
        response_payload = response.json()
        if LLM_PROVIDER == "huggingface_inference":
            if isinstance(response_payload, list) and response_payload:
                content = str(response_payload[0].get("generated_text", ""))
            elif isinstance(response_payload, dict):
                content = str(response_payload.get("generated_text", ""))
            else:
                content = str(response_payload)
        elif LLM_PROVIDER == "ollama":
            content = str(response_payload.get("message", {}).get("content", ""))
        else:
            content = response_payload["choices"][0]["message"]["content"]

        parsed = _extract_json_object(content)
        return {
            "enabled": True,
            "status": "ok",
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "endpoint": endpoint,
            "llm_signal": str(parsed.get("llm_signal", "HOLD")).upper(),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "rationale": str(parsed.get("rationale", "")),
            "risk_flags": parsed.get("risk_flags", []),
        }
    except requests.exceptions.ProxyError as exc:
        return {
            "enabled": True,
            "status": "proxy_error",
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "message": (
                "LLM request blocked by proxy tunnel. "
                "Set LLM_BYPASS_ENV_PROXY=true (default) and verify network allows direct HTTPS to provider endpoint. "
                f"Underlying error: {exc}"
            ),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "status": "error",
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "message": str(exc),
        }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
    if not VOLATILITY_REGIME_ENABLED:
        return {
            "enabled": False,
            "regime": "disabled",
            "volatility_pct": volatility_pct,
            "notes": "Volatility regime controller disabled.",
        }

    if volatility_pct >= VOLATILITY_EXTREME_PCT:
        regime = "extreme"
    elif volatility_pct >= VOLATILITY_HIGH_PCT:
        regime = "high"
    elif volatility_pct <= VOLATILITY_LOW_PCT:
        regime = "low"
    else:
        regime = "medium"

    return {
        "enabled": True,
        "regime": regime,
        "volatility_pct": volatility_pct,
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
        "reason": reason,
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
    features, full_row = load_latest_row(dataset_path)

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
    merge_result = merge_ml_llm_decision(
        ml_probability_up=probability,
        buy_threshold=effective_buy_threshold,
        sell_threshold=effective_sell_threshold,
        llm_overlay=llm_overlay,
        market_regime=market_regime,
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
