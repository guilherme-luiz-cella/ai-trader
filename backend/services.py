from __future__ import annotations

import json
import hashlib
import math
import os
import re
import socket
import ssl
import threading
import time
import uuid
import calendar
from collections import deque
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
load_dotenv(PROJECT_ROOT / ".env", override=False)

import sys

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from generate_trade_signal import (  # noqa: E402
    ADAPTIVE_THRESHOLD_ENABLED,
    BUY_THRESHOLD,
    DATASET_PATH,
    LLM_ENABLED,
    SELL_THRESHOLD,
    generate_trade_decision,
    load_latest_row,
    latest_file_with_suffix,
    resolve_model_path,
)
from backend.llm_client import ensure_llm_startup_logged, get_llm_status, llm_chat

ARTIFACTS_DIR = RESEARCH_DIR / "artifacts"
DATA_DIR = RESEARCH_DIR / "data_sets"
MEMORY_DIR = RESEARCH_DIR / "runtime_memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
TRADE_MEMORY_PATH = MEMORY_DIR / "trade_memory.jsonl"
AUTOPILOT_STATE_PATH = MEMORY_DIR / "autopilot_state.json"
AUTOPILOT_EXECUTION_JOURNAL_PATH = MEMORY_DIR / "autopilot_execution_journal.jsonl"
AUTOPILOT_RUN_SUMMARY_PATH = MEMORY_DIR / "autopilot_run_summaries.jsonl"

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
BINANCE_BYPASS_ENV_PROXY = os.getenv("BINANCE_BYPASS_ENV_PROXY", "true").lower() == "true"
BINANCE_TIMEOUT_SECONDS = int(os.getenv("BINANCE_TIMEOUT_SECONDS", "30"))
CHECK_SYMBOL = os.getenv("CHECK_SYMBOL", "BTC/USDT")
ACCOUNT_REFERENCE_USD = float(os.getenv("ACCOUNT_REFERENCE_USD", "6.93"))
SIZE_MIN_CONFIDENCE = float(os.getenv("SIZE_MIN_CONFIDENCE", "0.05"))
DECISION_MIN_CONFIDENCE = float(os.getenv("DECISION_MIN_CONFIDENCE", "0.55"))
TARGET_MONITOR_SYMBOLS_RAW = os.getenv("TARGET_MONITOR_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
TARGET_MONITOR_MAX_ASSETS = int(os.getenv("TARGET_MONITOR_MAX_ASSETS", "3"))
TARGET_PRICE_MAP_RAW = os.getenv("TARGET_PRICE_MAP", "")
TARGET_SURPASS_PCT = float(os.getenv("TARGET_SURPASS_PCT", "0.01"))
PROFIT_PARKING_STABLE_ASSET = os.getenv("PROFIT_PARKING_STABLE_ASSET", "USDT").strip().upper() or "USDT"
STABLE_RESERVE_MIN_PCT = float(os.getenv("STABLE_RESERVE_MIN_PCT", "0.35"))
SIGNAL_ENGINE_MODE = os.getenv("SIGNAL_ENGINE_MODE", "classic").strip().lower() or "classic"
AUTOPILOT_MULTI_SYMBOL_ENABLED = os.getenv("AUTOPILOT_MULTI_SYMBOL_ENABLED", "true").lower() == "true"
AUTOPILOT_UNIVERSE_QUOTES_RAW = os.getenv("AUTOPILOT_UNIVERSE_QUOTES", "USDT,FDUSD,USDC,BRL")
AUTOPILOT_SYMBOL_DENYLIST_RAW = os.getenv("AUTOPILOT_SYMBOL_DENYLIST", "")
AUTOPILOT_MAX_CANDIDATES = max(5, int(os.getenv("AUTOPILOT_MAX_CANDIDATES", "24")))
AUTOPILOT_SWITCH_SCORE_DELTA = max(0.0, float(os.getenv("AUTOPILOT_SWITCH_SCORE_DELTA", "0.08")))
PROD_ALPHA_AUTO_RUN = os.getenv("PROD_ALPHA_AUTO_RUN", "true").lower() == "true"
PROD_ALPHA_TARGET_SYMBOL = os.getenv("PROD_ALPHA_TARGET_SYMBOL", CHECK_SYMBOL)
PROD_ALPHA_BTC_PATH = Path(os.getenv("PROD_ALPHA_BTC_PATH", str(PROJECT_ROOT / "backtest" / "data" / "BTC-6h-1000wks-data.csv")))
PROD_ALPHA_ETH_PATH = Path(os.getenv("PROD_ALPHA_ETH_PATH", str(PROJECT_ROOT / "backtest" / "data" / "ETH-1d-1000wks-data.csv")))
PROD_ALPHA_SOL_PATH = Path(os.getenv("PROD_ALPHA_SOL_PATH", str(PROJECT_ROOT / "backtest" / "data" / "SOL-1d-1000wks-data.csv")))
FRED_API_KEY = os.getenv("FRED_API_KEY", os.getenv("FRED", "")).strip()
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred").rstrip("/")
FRED_TIMEOUT_SECONDS = int(os.getenv("FRED_TIMEOUT_SECONDS", "20"))
FRED_BYPASS_ENV_PROXY = os.getenv("FRED_BYPASS_ENV_PROXY", "true").lower() == "true"
AUTOPILOT_ALERTING_ENABLED = os.getenv("AUTOPILOT_ALERTING_ENABLED", "false").lower() == "true"
AUTOPILOT_ALERT_WEBHOOK_URL = os.getenv("AUTOPILOT_ALERT_WEBHOOK_URL", "").strip()
AUTOPILOT_PAGERDUTY_ROUTING_KEY = os.getenv("AUTOPILOT_PAGERDUTY_ROUTING_KEY", "").strip()
AUTOPILOT_PAGERDUTY_SEVERITY = os.getenv("AUTOPILOT_PAGERDUTY_SEVERITY", "critical").strip().lower() or "critical"
AUTOPILOT_TWILIO_ACCOUNT_SID = os.getenv("AUTOPILOT_TWILIO_ACCOUNT_SID", "").strip()
AUTOPILOT_TWILIO_AUTH_TOKEN = os.getenv("AUTOPILOT_TWILIO_AUTH_TOKEN", "").strip()
AUTOPILOT_TWILIO_FROM_NUMBER = os.getenv("AUTOPILOT_TWILIO_FROM_NUMBER", "").strip()
AUTOPILOT_TWILIO_TO_NUMBER = os.getenv("AUTOPILOT_TWILIO_TO_NUMBER", "").strip()
AUTOPILOT_TWILIO_CHANNEL = os.getenv("AUTOPILOT_TWILIO_CHANNEL", "whatsapp").strip().lower() or "whatsapp"
AUTOPILOT_UNATTENDED_MIN_BURNIN_RUNS = max(1, int(os.getenv("AUTOPILOT_UNATTENDED_MIN_BURNIN_RUNS", "8")))
AUTOPILOT_UNATTENDED_MAX_SKIP_RATE = max(0.0, float(os.getenv("AUTOPILOT_UNATTENDED_MAX_SKIP_RATE", "0.60")))
AUTOPILOT_UNATTENDED_MAX_RECONCILIATION_INCIDENTS = max(0, int(os.getenv("AUTOPILOT_UNATTENDED_MAX_RECONCILIATION_INCIDENTS", "0")))
AUTOPILOT_IDEMPOTENCY_WINDOW_SECONDS = max(60, int(os.getenv("AUTOPILOT_IDEMPOTENCY_WINDOW_SECONDS", "900")))
AUTOPILOT_BURNIN_MAX_ERROR_RATE = max(0.0, float(os.getenv("AUTOPILOT_BURNIN_MAX_ERROR_RATE", "0.10")))
AUTOPILOT_BURNIN_MIN_FINALIZATION_SUCCESS_RATE = max(0.0, float(os.getenv("AUTOPILOT_BURNIN_MIN_FINALIZATION_SUCCESS_RATE", "1.00")))


def parse_symbol_list(raw: str) -> list[str]:
    def _normalize(token: str) -> str:
        value = (token or "").strip().upper()
        if not value:
            return ""
        return value if "/" in value else f"{value}/USDT"

    symbols: list[str] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        normalized = _normalize(token)
        if normalized:
            symbols.append(normalized)
    deduped: list[str] = []
    for symbol in symbols:
        if symbol not in deduped:
            deduped.append(symbol)
    return deduped


def parse_target_price_map(raw: str) -> dict[str, float]:
    def _symbol_key(token: str) -> str:
        value = (token or "").strip().upper()
        if not value:
            return ""
        normalized = value if "/" in value else f"{value}/USDT"
        return normalized.replace("/", "")

    mapping: dict[str, float] = {}
    for item in str(raw or "").split(","):
        chunk = item.strip()
        if not chunk or ":" not in chunk:
            continue
        symbol_text, price_text = chunk.split(":", 1)
        symbol_key = _symbol_key(symbol_text.strip())
        try:
            target_price = float(price_text.strip())
        except ValueError:
            continue
        if target_price > 0:
            mapping[symbol_key] = target_price
    return mapping


TARGET_MONITOR_SYMBOLS = parse_symbol_list(TARGET_MONITOR_SYMBOLS_RAW)[: max(1, TARGET_MONITOR_MAX_ASSETS)]
TARGET_PRICE_MAP = parse_target_price_map(TARGET_PRICE_MAP_RAW)
AUTOPILOT_UNIVERSE_QUOTES = [token.strip().upper() for token in AUTOPILOT_UNIVERSE_QUOTES_RAW.split(",") if token.strip()]
AUTOPILOT_SYMBOL_DENYLIST = {token.strip().upper() for token in AUTOPILOT_SYMBOL_DENYLIST_RAW.split(",") if token.strip()}

PROD_ALPHA_LOCK = threading.Lock()
PROD_ALPHA_CACHE: dict[str, Any] = {
    "comparison_path": "",
    "predictions_path": "",
    "comparison": None,
    "predictions": None,
}

STATE_LOCK = threading.Lock()
RUNTIME_STATE: dict[str, Any] = {
    "last_trade_ts": 0.0,
    "latest_price_fallback": None,
    "live_history": [],
}

AUTOPILOT_LOCK = threading.Lock()
AUTOPILOT_THREAD: threading.Thread | None = None
AUTOPILOT_STOP_EVENT = threading.Event()
AUTOPILOT_STATE: dict[str, Any] = {
    "running": False,
    "status": "idle",
    "run_id": 0,
    "stop_requested": False,
    "interval_seconds": 0,
    "current_cycle": 0,
    "target_cycles": 0,
    "symbol": "",
    "started_at": None,
    "updated_at": None,
    "last_error": "",
    "logs": [],
    "latest_decision": {},
    "latest_trade_result": {},
    "cycle_plan": {},
    "execution_mode": "normal",
    "min_qty": 0.0,
    "step_size": 0.0,
    "min_notional": 0.0,
    "minimum_valid_quote": 0.0,
    "free_quote": 0.0,
    "free_base": 0.0,
    "computed_order_size_quote": 0.0,
    "computed_order_size_base": 0.0,
    "can_buy_minimum": False,
    "required_base_asset": "",
    "required_quote_asset": "",
    "direct_trade_possible": False,
    "funding_path": "none",
    "funding_skip_reason": "",
    "free_bnb": 0.0,
    "eligible_funding_assets": [],
    "funding_diagnostics": {},
    "wallet_summary": {},
    "conversion_plan": {},
    "skip_reason": "",
    "opportunity_rankings": [],
    "opportunity_winner": {},
    "opportunity_meta": {},
    "starting_value": 0.0,
    "current_value": 0.0,
    "goal_value": 0.0,
    "progress_pct": 0.0,
    "goal_reached": False,
    "continue_until_goal": True,
    "extra_cycles_used": 0,
    "failed_cycles_in_row": 0,
    "final_stable_target_asset": "",
    "finalization_status": "",
    "finalization_result": {},
    "final_stop_reason": "",
    "latest_execution_intent": {},
    "persistence_healthy": False,
    "persistence_path": str(AUTOPILOT_STATE_PATH),
    "execution_journal_path": str(AUTOPILOT_EXECUTION_JOURNAL_PATH),
    "reconciliation_state": "clean",
    "reconciliation_details": {},
    "reconciliation_checked_at": None,
    "requires_human_review": False,
    "alert_severity": "info",
    "alerts": [],
    "unattended_mode": False,
    "burn_in_report": {},
}

BINANCE_HTTP_LOCK = threading.Lock()
BINANCE_HTTP_SESSION: requests.Session | None = None

BINANCE_CLIENT_LOCK = threading.Lock()
BINANCE_CLIENT: Any | None = None

LATENCY_LOCK = threading.Lock()
LATENCY_HISTORY: deque[dict[str, Any]] = deque(maxlen=120)
LATENCY_CONSECUTIVE_BREACHES: int = 0

LATENCY_MONITOR_LOCK = threading.Lock()
LATENCY_MONITOR_THREAD: threading.Thread | None = None
LATENCY_MONITOR_STOP = threading.Event()
LATENCY_MONITOR_INTERVAL_SECONDS = max(1.0, float(os.getenv("LATENCY_MONITOR_INTERVAL_SECONDS", "2.0")))


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def execution_fingerprint(action: str, symbol: str, quantity: float, quote_amount: float, run_id: int = 0, cycle: int = 0, stage: str = "") -> str:
    raw = "|".join(
        [
            str(action or "").strip().lower(),
            normalize_symbol(symbol),
            f"{float(quantity or 0.0):.12f}",
            f"{float(quote_amount or 0.0):.12f}",
            str(int(run_id or 0)),
            str(int(cycle or 0)),
            str(stage or "").strip().lower(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def persist_autopilot_state() -> None:
    with AUTOPILOT_LOCK:
        snapshot = json.loads(json.dumps(AUTOPILOT_STATE))
    snapshot["persisted_at"] = utc_now_iso()
    snapshot["persistence_healthy"] = True
    _atomic_write_json(AUTOPILOT_STATE_PATH, snapshot)
    with AUTOPILOT_LOCK:
        AUTOPILOT_STATE["persistence_healthy"] = True


def append_execution_journal(entry: dict[str, Any]) -> None:
    _append_jsonl(AUTOPILOT_EXECUTION_JOURNAL_PATH, {"recorded_at": utc_now_iso(), **entry})


def recent_execution_records(limit: int = 200) -> list[dict[str, Any]]:
    return _load_jsonl(AUTOPILOT_EXECUTION_JOURNAL_PATH, limit=limit)


def has_recent_execution_fingerprint(fingerprint: str, window_seconds: int | None = None) -> dict[str, Any] | None:
    safe_window = max(1, int(window_seconds or AUTOPILOT_IDEMPOTENCY_WINDOW_SECONDS))
    now_ts = time.time()
    for row in reversed(recent_execution_records(limit=300)):
        if str(row.get("execution_fingerprint") or "") != str(fingerprint or ""):
            continue
        recorded_at = str(row.get("recorded_at") or "")
        try:
            recorded_ts = calendar.timegm(time.strptime(recorded_at, "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            recorded_ts = now_ts
        if now_ts - recorded_ts <= safe_window:
            return row
    return None


def append_run_summary(summary: dict[str, Any]) -> None:
    _append_jsonl(AUTOPILOT_RUN_SUMMARY_PATH, {"recorded_at": utc_now_iso(), **summary})


def build_burn_in_validation_report(min_runs: int | None = None) -> dict[str, Any]:
    safe_min_runs = max(1, int(min_runs or AUTOPILOT_UNATTENDED_MIN_BURNIN_RUNS))
    rows = _load_jsonl(AUTOPILOT_RUN_SUMMARY_PATH, limit=200)
    supervised_live = [row for row in rows if bool(row.get("allow_live")) and not bool(row.get("unattended_mode"))]
    total_runs = len(supervised_live)
    total_cycles = sum(int(row.get("cycles_completed") or 0) for row in supervised_live)
    total_skips = sum(int(row.get("skip_cycles") or 0) for row in supervised_live)
    total_exec = sum(int(row.get("executed_cycles") or 0) for row in supervised_live)
    total_reconciliation = sum(int(row.get("reconciliation_incidents") or 0) for row in supervised_live)
    total_conversion_success = sum(int(row.get("conversion_successes") or 0) for row in supervised_live)
    total_conversion_fail = sum(int(row.get("conversion_failures") or 0) for row in supervised_live)
    total_finalization_success = sum(1 for row in supervised_live if str(row.get("finalization_status") or "") in {"completed", "ok", "dust_only"})
    total_finalization_fail = sum(1 for row in supervised_live if str(row.get("finalization_status") or "") in {"error", "failed"})
    total_error_runs = sum(1 for row in supervised_live if str(row.get("stop_reason") or "") in {"error", "max_failed_cycles_reached"})
    total_duplicate_blocks = sum(int(row.get("duplicate_blocks") or 0) for row in supervised_live)
    skip_rate = (total_skips / total_cycles) if total_cycles > 0 else 0.0
    execution_success_rate = (total_exec / total_cycles) if total_cycles > 0 else 0.0
    error_rate = (total_error_runs / total_runs) if total_runs > 0 else 0.0
    finalization_success_rate = (total_finalization_success / max(total_finalization_success + total_finalization_fail, 1)) if (total_finalization_success + total_finalization_fail) > 0 else 1.0
    unattended_eligible = (
        total_runs >= safe_min_runs
        and skip_rate <= AUTOPILOT_UNATTENDED_MAX_SKIP_RATE
        and total_reconciliation <= AUTOPILOT_UNATTENDED_MAX_RECONCILIATION_INCIDENTS
        and total_finalization_fail == 0
        and total_duplicate_blocks == 0
        and error_rate <= AUTOPILOT_BURNIN_MAX_ERROR_RATE
        and finalization_success_rate >= AUTOPILOT_BURNIN_MIN_FINALIZATION_SUCCESS_RATE
    )
    return {
        "total_supervised_live_runs": total_runs,
        "minimum_required_runs": safe_min_runs,
        "skip_rate": skip_rate,
        "execution_success_rate": execution_success_rate,
        "error_rate": error_rate,
        "reconciliation_incidents": total_reconciliation,
        "conversion_successes": total_conversion_success,
        "conversion_failures": total_conversion_fail,
        "finalization_successes": total_finalization_success,
        "finalization_failures": total_finalization_fail,
        "finalization_success_rate": finalization_success_rate,
        "duplicate_blocks": total_duplicate_blocks,
        "restart_recovery_events": sum(1 for row in supervised_live if bool(row.get("restart_recovery_observed"))),
        "policy": {
            "max_skip_rate": AUTOPILOT_UNATTENDED_MAX_SKIP_RATE,
            "max_error_rate": AUTOPILOT_BURNIN_MAX_ERROR_RATE,
            "max_reconciliation_incidents": AUTOPILOT_UNATTENDED_MAX_RECONCILIATION_INCIDENTS,
            "min_finalization_success_rate": AUTOPILOT_BURNIN_MIN_FINALIZATION_SUCCESS_RATE,
            "duplicate_blocks_required": 0,
        },
        "unattended_eligible": unattended_eligible,
    }


def emit_autopilot_alert(
    code: str,
    severity: str,
    message: str,
    *,
    requires_human_review: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    alert = {
        "timestamp": utc_now_iso(),
        "code": str(code or "").strip() or "autopilot_alert",
        "severity": str(severity or "warn").strip().lower(),
        "message": str(message or "").strip(),
        "requires_human_review": bool(requires_human_review),
        "details": dict(details or {}),
    }
    with AUTOPILOT_LOCK:
        alerts = list(AUTOPILOT_STATE.get("alerts", []) or [])
        alerts.append(alert)
        AUTOPILOT_STATE["alerts"] = alerts[-100:]
        AUTOPILOT_STATE["alert_severity"] = alert["severity"]
        if requires_human_review:
            AUTOPILOT_STATE["requires_human_review"] = True
    append_autopilot_log({"event": "alert", **alert})
    if AUTOPILOT_ALERTING_ENABLED and AUTOPILOT_ALERT_WEBHOOK_URL:
        try:
            requests.post(AUTOPILOT_ALERT_WEBHOOK_URL, json=alert, timeout=5)
        except Exception:
            pass
    if alert["severity"] in {"critical", "error"}:
        paging_results: list[dict[str, Any]] = []
        pagerduty_result = send_pagerduty_alert(alert)
        if pagerduty_result:
            paging_results.append(pagerduty_result)
        twilio_result = send_twilio_alert(alert)
        if twilio_result:
            paging_results.append(twilio_result)
        if paging_results:
            alert["paging_results"] = paging_results
    return alert


def send_pagerduty_alert(alert: dict[str, Any]) -> dict[str, Any] | None:
    if not AUTOPILOT_PAGERDUTY_ROUTING_KEY:
        return None
    payload = {
        "routing_key": AUTOPILOT_PAGERDUTY_ROUTING_KEY,
        "event_action": "trigger",
        "payload": {
            "summary": str(alert.get("message") or "Autopilot critical alert"),
            "source": "ai-trader-autopilot",
            "severity": AUTOPILOT_PAGERDUTY_SEVERITY if str(alert.get("severity") or "") == "critical" else "error",
            "custom_details": dict(alert),
        },
    }
    try:
        response = requests.post("https://events.pagerduty.com/v2/enqueue", json=payload, timeout=5)
        return {"provider": "pagerduty", "status_code": response.status_code, "ok": bool(response.ok)}
    except Exception as exc:
        return {"provider": "pagerduty", "ok": False, "error": str(exc)}


def send_twilio_alert(alert: dict[str, Any]) -> dict[str, Any] | None:
    if not (AUTOPILOT_TWILIO_ACCOUNT_SID and AUTOPILOT_TWILIO_AUTH_TOKEN and AUTOPILOT_TWILIO_FROM_NUMBER and AUTOPILOT_TWILIO_TO_NUMBER):
        return None
    def _channel_address(number: str) -> str:
        text = str(number or "").strip()
        if not text:
            return text
        if AUTOPILOT_TWILIO_CHANNEL == "whatsapp" and not text.startswith("whatsapp:"):
            return f"whatsapp:{text}"
        return text
    body = f"[ai-trader] {alert.get('severity', 'warn').upper()}: {alert.get('message', '')}"
    try:
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{AUTOPILOT_TWILIO_ACCOUNT_SID}/Messages.json",
            data={
                "From": _channel_address(AUTOPILOT_TWILIO_FROM_NUMBER),
                "To": _channel_address(AUTOPILOT_TWILIO_TO_NUMBER),
                "Body": body[:1500],
            },
            auth=(AUTOPILOT_TWILIO_ACCOUNT_SID, AUTOPILOT_TWILIO_AUTH_TOKEN),
            timeout=5,
        )
        return {"provider": "twilio", "status_code": response.status_code, "ok": bool(response.ok)}
    except Exception as exc:
        return {"provider": "twilio", "ok": False, "error": str(exc)}


def get_notification_status() -> dict[str, Any]:
    twilio_configured = bool(
        AUTOPILOT_TWILIO_ACCOUNT_SID
        and AUTOPILOT_TWILIO_AUTH_TOKEN
        and AUTOPILOT_TWILIO_FROM_NUMBER
        and AUTOPILOT_TWILIO_TO_NUMBER
    )
    pagerduty_configured = bool(AUTOPILOT_PAGERDUTY_ROUTING_KEY)
    webhook_configured = bool(AUTOPILOT_ALERT_WEBHOOK_URL)
    return {
        "alerting_enabled": bool(AUTOPILOT_ALERTING_ENABLED),
        "webhook_configured": webhook_configured,
        "pagerduty_configured": pagerduty_configured,
        "twilio_configured": twilio_configured,
        "twilio_channel": AUTOPILOT_TWILIO_CHANNEL,
        "whatsapp_configured": twilio_configured and AUTOPILOT_TWILIO_CHANNEL == "whatsapp",
        "paging_ready": bool(pagerduty_configured or twilio_configured or webhook_configured),
    }


def summarize_wallet_for_persistence(wallet_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    snapshot = dict(wallet_snapshot or {})
    balances = list(snapshot.get("balances", []) or [])
    top_balances = sorted(
        [
            {
                "asset": str(row.get("asset") or "").upper().strip(),
                "free": float(row.get("free") or 0.0),
                "total": float(row.get("total") or 0.0),
                "est_usdt": float(row.get("est_usdt") or 0.0),
            }
            for row in balances
            if str(row.get("asset") or "").strip()
        ],
        key=lambda row: float(row.get("est_usdt") or 0.0),
        reverse=True,
    )[:10]
    return {
        "captured_at": snapshot.get("captured_at"),
        "estimated_total_usdt": float(snapshot.get("estimated_total_usdt") or 0.0),
        "asset_count": int(snapshot.get("asset_count") or len(balances)),
        "top_balances": top_balances,
    }


def fetch_binance_order_truth(symbol: str, *, client_order_id: str = "", order_id: str = "") -> dict[str, Any]:
    exchange = get_binance_client()
    normalized_symbol = normalize_symbol(symbol)
    exchange.load_markets()
    market = exchange.market(normalized_symbol)
    market_id = str(market.get("id") or "").upper()
    client_id = str(client_order_id or "").strip()
    native_order_id = str(order_id or "").strip()

    raw_order: dict[str, Any] | None = None
    errors: list[str] = []
    if client_id and hasattr(exchange, "privateGetOrder"):
        try:
            raw_order = exchange.privateGetOrder({"symbol": market_id, "origClientOrderId": client_id})
        except Exception as exc:
            errors.append(str(exc))
    if raw_order is None and native_order_id and hasattr(exchange, "privateGetOrder"):
        try:
            raw_order = exchange.privateGetOrder({"symbol": market_id, "orderId": native_order_id})
        except Exception as exc:
            errors.append(str(exc))
    parsed_order: dict[str, Any] | None = None
    if raw_order is not None:
        try:
            parsed_order = exchange.parse_order(raw_order, market)
        except Exception:
            parsed_order = raw_order if isinstance(raw_order, dict) else {"info": raw_order}
    if parsed_order is None:
        try:
            candidates = exchange.fetch_orders(normalized_symbol, limit=20)
        except Exception as exc:
            candidates = []
            errors.append(str(exc))
        for item in candidates:
            if client_id and str(item.get("clientOrderId") or item.get("info", {}).get("clientOrderId") or "") == client_id:
                parsed_order = item
                break
            if native_order_id and str(item.get("id") or "") == native_order_id:
                parsed_order = item
                break
    if parsed_order is None:
        return {
            "status": "not_found",
            "symbol": normalized_symbol,
            "client_order_id": client_id,
            "order_id": native_order_id,
            "errors": errors,
        }
    raw_status = str(parsed_order.get("status") or parsed_order.get("info", {}).get("status") or "").lower()
    filled = float(parsed_order.get("filled") or parsed_order.get("executedQty") or parsed_order.get("info", {}).get("executedQty") or 0.0)
    amount = float(parsed_order.get("amount") or parsed_order.get("origQty") or parsed_order.get("info", {}).get("origQty") or 0.0)
    remaining = float(parsed_order.get("remaining") or max(amount - filled, 0.0))
    truth_status = "unknown"
    if raw_status in {"closed", "filled"}:
        truth_status = "filled"
    elif raw_status in {"open", "new"}:
        truth_status = "open"
    elif raw_status in {"canceled", "cancelled", "expired"}:
        truth_status = "canceled"
    elif raw_status in {"partially_filled"}:
        truth_status = "partially_filled"
    elif raw_status in {"rejected"}:
        truth_status = "rejected"
    elif filled > 0 and remaining > 0:
        truth_status = "partially_filled"
    elif filled > 0 and remaining <= 0:
        truth_status = "filled"
    return {
        "status": truth_status,
        "raw_status": raw_status,
        "symbol": normalized_symbol,
        "client_order_id": client_id or str(parsed_order.get("clientOrderId") or parsed_order.get("info", {}).get("clientOrderId") or ""),
        "order_id": native_order_id or str(parsed_order.get("id") or parsed_order.get("info", {}).get("orderId") or ""),
        "filled": filled,
        "remaining": remaining,
        "amount": amount,
        "average": float(parsed_order.get("average") or parsed_order.get("price") or parsed_order.get("info", {}).get("price") or 0.0),
        "order": parsed_order,
        "errors": errors,
    }


def reconcile_execution_record_with_exchange(record: dict[str, Any]) -> dict[str, Any]:
    symbol = str(record.get("symbol") or CHECK_SYMBOL)
    stage = str(record.get("stage") or "").strip().lower()
    fingerprint = str(record.get("execution_fingerprint") or "")
    exchange_truth = fetch_binance_order_truth(
        symbol,
        client_order_id=str(record.get("exchange_client_order_id") or record.get("client_order_id") or fingerprint),
        order_id=str(record.get("exchange_order_id") or record.get("order_id") or ""),
    )
    try:
        wallet_snapshot = get_wallet_snapshot()
        wallet_summary = summarize_wallet_for_persistence(wallet_snapshot)
    except Exception as exc:
        wallet_summary = {"error": str(exc)}
    try:
        account_snapshot = get_account_snapshot(symbol)
    except Exception as exc:
        account_snapshot = {"error": str(exc), "symbol": symbol}
    truth_status = str(exchange_truth.get("status") or "unknown")
    if truth_status == "filled":
        reconciliation_state = {
            "conversion": "interrupted_after_conversion",
            "signal_buy": "interrupted_after_buy",
            "signal_sell": "interrupted_after_sell",
            "finalization": "interrupted_during_finalization",
        }.get(stage, "partial_execution_needs_review")
        safe_next_action = "manual_review_required_before_resume"
        requires_human_review = True
    elif truth_status == "partially_filled":
        reconciliation_state = "partial_execution_needs_review"
        safe_next_action = "manual_review_required_before_resume"
        requires_human_review = True
    elif truth_status in {"open", "unknown", "not_found"}:
        reconciliation_state = "partial_execution_needs_review"
        safe_next_action = "manual_review_required_before_resume"
        requires_human_review = True
    elif truth_status in {"canceled", "rejected"}:
        reconciliation_state = "clean"
        safe_next_action = "supervised_restart_allowed"
        requires_human_review = False
    else:
        reconciliation_state = "partial_execution_needs_review"
        safe_next_action = "manual_review_required_before_resume"
        requires_human_review = True
    return {
        "record": record,
        "exchange_truth": exchange_truth,
        "wallet_summary": wallet_summary,
        "account_snapshot": account_snapshot,
        "reconciliation_state": reconciliation_state,
        "requires_human_review": requires_human_review,
        "safe_next_action": safe_next_action,
    }


def reconcile_execution_candidates(persisted: dict[str, Any]) -> list[dict[str, Any]]:
    latest_intent = dict(persisted.get("latest_execution_intent") or {})
    recent_records = recent_execution_records(limit=50)
    candidates: list[dict[str, Any]] = []
    if latest_intent:
        candidates.append(latest_intent)
    for record in reversed(recent_records):
        status = str(record.get("status") or "")
        if status not in {"intent_recorded", "uncertain", "executed"}:
            continue
        fingerprint = str(record.get("execution_fingerprint") or "")
        if fingerprint and any(str(item.get("execution_fingerprint") or "") == fingerprint for item in candidates):
            continue
        candidates.append(record)
    return candidates[:5]


def set_reconciliation_state(state: str, *, details: dict[str, Any] | None = None, requires_human_review: bool = False) -> None:
    update_autopilot_state(
        reconciliation_state=str(state or "clean"),
        reconciliation_details=dict(details or {}),
        reconciliation_checked_at=utc_now_iso(),
        requires_human_review=bool(requires_human_review),
    )


def derive_interrupted_reconciliation_state(snapshot: dict[str, Any]) -> str:
    latest_intent = dict(snapshot.get("latest_execution_intent") or {})
    latest_trade = dict(snapshot.get("latest_trade_result") or {})
    finalization_status = str(snapshot.get("finalization_status") or "")
    stage = str(latest_intent.get("stage") or "").strip().lower()
    if finalization_status in {"pending", "running"}:
        return "interrupted_during_finalization"
    if stage == "conversion":
        return "interrupted_after_conversion"
    if stage == "signal_buy":
        return "interrupted_after_buy"
    if stage == "signal_sell":
        return "interrupted_after_sell"
    if str((latest_trade.get("signal_trade") or {}).get("status") or "") in {"executed", "submitted"}:
        return "partial_execution_needs_review"
    return "interrupted_run_needs_review"


def reconcile_interrupted_autopilot_state() -> dict[str, Any]:
    persisted = _read_json_file(AUTOPILOT_STATE_PATH, {})
    if not persisted:
        report = {
            "status": "clean",
            "reconciliation_state": "clean",
            "requires_human_review": False,
            "safe_next_action": "normal_start_allowed",
        }
        update_autopilot_state(
            persistence_healthy=True,
            reconciliation_state="clean",
            reconciliation_details=report,
            reconciliation_checked_at=utc_now_iso(),
            burn_in_report=build_burn_in_validation_report(),
        )
        return report
    previous_status = str(persisted.get("status") or "idle")
    interrupted = previous_status in {"running", "starting", "stopping"} or bool(persisted.get("running"))
    if interrupted:
        recon_state = derive_interrupted_reconciliation_state(persisted)
        active_symbol = str(persisted.get("symbol") or CHECK_SYMBOL)
        try:
            wallet_snapshot = get_wallet_snapshot()
            wallet_summary = summarize_wallet_for_persistence(wallet_snapshot)
        except Exception as exc:
            wallet_summary = {"error": str(exc)}
        try:
            account_snapshot = get_account_snapshot(active_symbol)
        except Exception as exc:
            account_snapshot = {"error": str(exc), "symbol": active_symbol}
        exchange_reconciliations: list[dict[str, Any]] = []
        for candidate in reconcile_execution_candidates(persisted):
            try:
                exchange_reconciliations.append(reconcile_execution_record_with_exchange(candidate))
            except Exception as exc:
                exchange_reconciliations.append(
                    {
                        "record": candidate,
                        "exchange_truth": {"status": "lookup_error", "error": str(exc)},
                        "reconciliation_state": "partial_execution_needs_review",
                        "requires_human_review": True,
                        "safe_next_action": "manual_review_required_before_resume",
                    }
                )
        if exchange_reconciliations:
            if any(str(item.get("reconciliation_state") or "") != "clean" for item in exchange_reconciliations):
                recon_state = next(
                    (str(item.get("reconciliation_state") or "partial_execution_needs_review") for item in exchange_reconciliations if str(item.get("reconciliation_state") or "") != "clean"),
                    recon_state,
                )
            else:
                recon_state = "clean"
        report = {
            "status": "interrupted",
            "reconciliation_state": recon_state,
            "requires_human_review": recon_state != "clean",
            "previous_status": previous_status,
            "active_symbol": active_symbol,
            "latest_execution_intent": persisted.get("latest_execution_intent", {}),
            "latest_trade_result": persisted.get("latest_trade_result", {}),
            "wallet_summary": wallet_summary,
            "account_snapshot": account_snapshot,
            "exchange_reconciliations": exchange_reconciliations,
            "safe_next_action": "manual_review_required_before_resume" if recon_state != "clean" else "supervised_restart_allowed",
        }
        with AUTOPILOT_LOCK:
            AUTOPILOT_STATE.update(
                {
                    **json.loads(json.dumps(persisted)),
                    "running": False,
                    "status": "interrupted",
                    "stop_requested": False,
                    "reconciliation_state": recon_state,
                    "reconciliation_details": report,
                    "reconciliation_checked_at": utc_now_iso(),
                    "requires_human_review": recon_state != "clean",
                    "alert_severity": "critical" if recon_state != "clean" else "warn",
                    "persistence_healthy": True,
                    "burn_in_report": build_burn_in_validation_report(),
                }
            )
        persist_autopilot_state()
        emit_autopilot_alert(
            "interrupted_run_detected",
            "critical" if recon_state != "clean" else "warn",
            f"Interrupted autopilot run detected for {active_symbol}. {'Manual reconciliation is required before restart.' if recon_state != 'clean' else 'Exchange reconciliation is clean; supervised restart can be acknowledged manually.'}",
            requires_human_review=recon_state != "clean",
            details=report,
        )
        return report
    report = {
        "status": "clean",
        "reconciliation_state": str(persisted.get("reconciliation_state") or "clean"),
        "requires_human_review": bool(persisted.get("requires_human_review", False)),
        "safe_next_action": "normal_start_allowed" if not bool(persisted.get("requires_human_review", False)) else "manual_review_required_before_resume",
    }
    with AUTOPILOT_LOCK:
        AUTOPILOT_STATE.update(
            {
                **json.loads(json.dumps(persisted)),
                "running": False,
                "stop_requested": False,
                "persistence_healthy": True,
                "reconciliation_checked_at": utc_now_iso(),
                "burn_in_report": build_burn_in_validation_report(),
            }
        )
    persist_autopilot_state()
    return report


def unattended_start_gate(config: dict[str, Any]) -> dict[str, Any]:
    unattended_mode = bool(config.get("unattended_mode", False))
    if not unattended_mode:
        return {"ok": True, "reason": "", "checks": {}}
    paging_configured = bool(
        (config.get("alert_webhook_url", AUTOPILOT_ALERT_WEBHOOK_URL) and config.get("alerting_enabled", AUTOPILOT_ALERTING_ENABLED))
        or AUTOPILOT_PAGERDUTY_ROUTING_KEY
        or (AUTOPILOT_TWILIO_ACCOUNT_SID and AUTOPILOT_TWILIO_AUTH_TOKEN and AUTOPILOT_TWILIO_FROM_NUMBER and AUTOPILOT_TWILIO_TO_NUMBER)
    )
    checks = {
        "persistence_healthy": AUTOPILOT_STATE_PATH.exists(),
        "reconciliation_clean": str(AUTOPILOT_STATE.get("reconciliation_state") or "clean") == "clean" and not bool(AUTOPILOT_STATE.get("requires_human_review", False)),
        "preview_gate_passed": bool(config.get("preview_gate_passed", False)),
        "risk_guards_present": all(
            config.get(field) is not None
            for field in ("max_api_latency_ms", "max_ticker_age_ms", "max_spread_bps", "max_failed_cycles_in_row", "max_runtime_minutes")
        ),
        "alerting_configured": paging_configured,
        "no_unresolved_partial_execution": str(AUTOPILOT_STATE.get("reconciliation_state") or "clean") == "clean",
        "burn_in_eligible": bool(build_burn_in_validation_report().get("unattended_eligible", False)),
    }
    failed = [key for key, value in checks.items() if not bool(value)]
    return {
        "ok": not failed,
        "reason": "" if not failed else f"Unattended mode blocked: {', '.join(failed)}",
        "checks": checks,
    }


def record_autopilot_run_summary(
    *,
    run_id: int,
    allow_live: bool,
    unattended_mode: bool,
    cycles_completed: int,
    skip_cycles: int,
    executed_cycles: int,
    conversion_successes: int,
    conversion_failures: int,
    reconciliation_incidents: int,
    finalization_status: str,
    stop_reason: str,
    restart_recovery_observed: bool = False,
    duplicate_blocks: int = 0,
) -> None:
    append_run_summary(
        {
            "run_id": run_id,
            "allow_live": bool(allow_live),
            "unattended_mode": bool(unattended_mode),
            "cycles_completed": int(cycles_completed),
            "skip_cycles": int(skip_cycles),
            "executed_cycles": int(executed_cycles),
            "conversion_successes": int(conversion_successes),
            "conversion_failures": int(conversion_failures),
            "reconciliation_incidents": int(reconciliation_incidents),
            "finalization_status": str(finalization_status or ""),
            "stop_reason": str(stop_reason or ""),
            "restart_recovery_observed": bool(restart_recovery_observed),
            "duplicate_blocks": int(duplicate_blocks),
        }
    )


def append_trade_memory(entry: dict[str, Any]) -> None:
    record = {"recorded_at": utc_now_iso(), **entry}
    with TRADE_MEMORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def log_trade_memory(
    *,
    source: str,
    symbol: str,
    action: str,
    payload: dict[str, Any],
    signal: str | None = None,
    probability_up: float | None = None,
    cycle: int | None = None,
    target_cycles: int | None = None,
    size_plan: dict[str, Any] | None = None,
    account_snapshot: dict[str, Any] | None = None,
) -> None:
    append_trade_memory(
        {
            "source": source,
            "exchange": payload.get("exchange", "binance"),
            "symbol": normalize_symbol(symbol),
            "action": action,
            "signal": signal,
            "probability_up": probability_up,
            "cycle": cycle,
            "target_cycles": target_cycles,
            "status": payload.get("status"),
            "message": payload.get("message") or payload.get("guard_message") or payload.get("minimum_message") or "",
            "dry_run": bool(payload.get("dry_run", False)),
            "guard_ok": payload.get("guard_ok"),
            "guard_message": payload.get("guard_message"),
            "minimum_ok": payload.get("minimum_ok"),
            "minimum_message": payload.get("minimum_message"),
            "quantity": float(payload.get("quantity") or 0.0),
            "quote_amount": float(payload.get("quote_amount") or 0.0),
            "market_price": float(payload.get("market_price") or 0.0),
            "api_latency_ms": payload.get("api_latency_ms"),
            "ticker_age_ms": payload.get("ticker_age_ms"),
            "spread_bps": payload.get("spread_bps"),
            "min_qty": payload.get("min_qty"),
            "min_notional": payload.get("min_notional"),
            "result": payload.get("result"),
            "account_value_quote": (account_snapshot or {}).get("account_value_quote"),
            "best_bid": (account_snapshot or {}).get("best_bid"),
            "best_ask": (account_snapshot or {}).get("best_ask"),
            "size_strength": (size_plan or {}).get("strength"),
            "allocation_pct": (size_plan or {}).get("allocation_pct"),
            "quote_size": (size_plan or {}).get("quote_size"),
            "base_size": (size_plan or {}).get("base_size"),
        }
    )


def set_runtime_value(key: str, value: Any) -> None:
    with STATE_LOCK:
        RUNTIME_STATE[key] = value


def get_runtime_value(key: str, default: Any = None) -> Any:
    with STATE_LOCK:
        return RUNTIME_STATE.get(key, default)


def _binance_base_url() -> str:
    return "https://testnet.binance.vision" if BINANCE_TESTNET else "https://api.binance.com"


def _binance_host() -> str:
    return "testnet.binance.vision" if BINANCE_TESTNET else "api.binance.com"


def get_binance_http_session() -> requests.Session:
    global BINANCE_HTTP_SESSION
    with BINANCE_HTTP_LOCK:
        if BINANCE_HTTP_SESSION is not None:
            return BINANCE_HTTP_SESSION
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=32, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        if BINANCE_BYPASS_ENV_PROXY:
            session.trust_env = False
        BINANCE_HTTP_SESSION = session
        return BINANCE_HTTP_SESSION


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = int(round((max(0.0, min(100.0, pct)) / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _measure_dns_ms(host: str) -> tuple[float, str]:
    start = time.perf_counter()
    try:
        records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ip = str(records[0][4][0]) if records else ""
        return float((time.perf_counter() - start) * 1000.0), ip
    except Exception:
        return float((time.perf_counter() - start) * 1000.0), ""


def _measure_connect_tls_ms(host: str) -> float:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, 443), timeout=max(1.0, float(BINANCE_TIMEOUT_SECONDS))):
            pass
    except Exception:
        return float((time.perf_counter() - start) * 1000.0)

    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=max(1.0, float(BINANCE_TIMEOUT_SECONDS))) as sock:
            with context.wrap_socket(sock, server_hostname=host):
                pass
    except Exception:
        return float((time.perf_counter() - start) * 1000.0)
    return float((time.perf_counter() - start) * 1000.0)


def _record_latency_sample(sample: dict[str, Any]) -> None:
    with LATENCY_LOCK:
        LATENCY_HISTORY.append(dict(sample))
        set_runtime_value("latency_probe_latest", dict(sample))
        set_runtime_value("latency_history_size", len(LATENCY_HISTORY))


def _rolling_latency_stats(window: int = 20) -> dict[str, float]:
    with LATENCY_LOCK:
        samples = list(LATENCY_HISTORY)[-max(1, int(window)) :]
    totals = [float(s.get("total_ms", 0.0) or 0.0) for s in samples]
    dns_vals = [float(s.get("dns_ms", 0.0) or 0.0) for s in samples]
    connect_vals = [float(s.get("connect_tls_ms", 0.0) or 0.0) for s in samples]
    server_vals = [float(s.get("server_ms", 0.0) or 0.0) for s in samples]
    return {
        "count": float(len(samples)),
        "latest_total_ms": float(totals[-1] if totals else 0.0),
        "avg_total_ms": float(sum(totals) / len(totals)) if totals else 0.0,
        "median_total_ms": _median(totals),
        "p95_total_ms": _percentile(totals, 95.0),
        "avg_dns_ms": float(sum(dns_vals) / len(dns_vals)) if dns_vals else 0.0,
        "avg_connect_tls_ms": float(sum(connect_vals) / len(connect_vals)) if connect_vals else 0.0,
        "avg_server_ms": float(sum(server_vals) / len(server_vals)) if server_vals else 0.0,
    }


def _infer_latency_root_cause(sample: dict[str, Any], rolling: dict[str, float], ticker_latency_ms: float) -> str:
    retries = int(sample.get("retries", 0) or 0)
    if sample.get("error"):
        return "network_or_timeout_error"
    if retries > 0:
        return "retries_or_transient_network"
    dns_ms = float(sample.get("dns_ms", 0.0) or 0.0)
    connect_tls_ms = float(sample.get("connect_tls_ms", 0.0) or 0.0)
    server_ms = float(sample.get("server_ms", 0.0) or 0.0)
    total_ms = float(sample.get("total_ms", 0.0) or 0.0)
    if total_ms <= 0:
        return "unknown"
    network_ratio = (dns_ms + connect_tls_ms) / total_ms
    server_ratio = server_ms / total_ms
    if network_ratio >= 0.55:
        return "network_latency"
    if server_ratio >= 0.55:
        return "server_latency"
    rolling_total = float(rolling.get("avg_total_ms", 0.0) or 0.0)
    if ticker_latency_ms > max(rolling_total * 1.7, total_ms * 1.7):
        return "local_app_overhead_or_exchange_client"
    return "mixed_latency"


def _latency_probe(symbol: str) -> dict[str, Any]:
    norm_symbol = normalize_symbol(symbol)
    market_id = to_binance_market_id(norm_symbol)
    endpoint_path = "/api/v3/ticker/bookTicker"
    endpoint = f"{_binance_base_url()}{endpoint_path}"
    host = _binance_host()

    dns_ms, resolved_ip = _measure_dns_ms(host)
    connect_tls_ms = _measure_connect_tls_ms(host)

    request_start = time.perf_counter()
    error_text = ""
    status_code = 0
    response_elapsed_ms = 0.0
    retries = 0
    try:
        session = get_binance_http_session()
        response = session.get(endpoint, params={"symbol": market_id}, timeout=BINANCE_TIMEOUT_SECONDS)
        status_code = int(response.status_code)
        response.raise_for_status()
        response_elapsed_ms = float(getattr(response, "elapsed", 0.0).total_seconds() * 1000.0) if getattr(response, "elapsed", None) else 0.0
        raw = getattr(response, "raw", None)
        retry_obj = getattr(raw, "retries", None) if raw is not None else None
        retries = int(len(getattr(retry_obj, "history", []) or [])) if retry_obj is not None else 0
    except Exception as exc:
        error_text = str(exc)

    total_ms = float((time.perf_counter() - request_start) * 1000.0)
    server_ms = max(0.0, total_ms - dns_ms - connect_tls_ms)

    return {
        "timestamp": utc_now_iso(),
        "symbol": norm_symbol,
        "endpoint": endpoint_path,
        "host": host,
        "resolved_ip": resolved_ip,
        "status_code": status_code,
        "dns_ms": float(dns_ms),
        "connect_tls_ms": float(connect_tls_ms),
        "server_ms": float(server_ms),
        "total_ms": float(total_ms),
        "response_elapsed_ms": float(response_elapsed_ms),
        "retries": int(retries),
        "error": error_text,
    }


def _latency_monitor_loop() -> None:
    while not LATENCY_MONITOR_STOP.is_set():
        symbol = str(get_runtime_value("latency_monitor_symbol", CHECK_SYMBOL) or CHECK_SYMBOL)
        sample = _latency_probe(symbol)
        _record_latency_sample(sample)
        LATENCY_MONITOR_STOP.wait(LATENCY_MONITOR_INTERVAL_SECONDS)


def ensure_latency_monitor(symbol: str | None = None) -> None:
    global LATENCY_MONITOR_THREAD
    if symbol:
        set_runtime_value("latency_monitor_symbol", normalize_symbol(symbol))
    with LATENCY_MONITOR_LOCK:
        if LATENCY_MONITOR_THREAD is not None and LATENCY_MONITOR_THREAD.is_alive():
            return
        LATENCY_MONITOR_STOP.clear()
        LATENCY_MONITOR_THREAD = threading.Thread(target=_latency_monitor_loop, daemon=True, name="latency-monitor")
        LATENCY_MONITOR_THREAD.start()


def normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return "BTC/USDT"
    if "/" in symbol:
        return symbol
    return f"{symbol}/USDT"


def split_symbol_assets(symbol: str) -> tuple[str, str]:
    base, quote = normalize_symbol(symbol).split("/", 1)
    return base, quote


def to_binance_market_id(symbol: str) -> str:
    return normalize_symbol(symbol).replace("/", "")


def target_key(symbol: str) -> str:
    return normalize_symbol(symbol).replace("/", "")


def resolve_target_price(symbol: str) -> float | None:
    return TARGET_PRICE_MAP.get(target_key(symbol))


def stable_assets_set(primary: str | None = None) -> set[str]:
    stable = {"USDT", "USDC", "BUSD", "FDUSD", "DAI", "TUSD"}
    if primary:
        stable.add(str(primary).upper())
    return stable


def apply_decision_confidence_gate(
    *,
    size_plan: dict[str, Any],
    decision_payload: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    gated = dict(size_plan)
    decision_confidence = float(decision_payload.get("decision_confidence") or 0.0)
    if min_confidence <= 0:
        gated["decision_confidence"] = decision_confidence
        gated["decision_min_confidence"] = min_confidence
        return gated

    if decision_confidence < min_confidence:
        gated.update(
            {
                "status": "blocked",
                "quote_size": 0.0,
                "base_size": 0.0,
                "allocation_pct": 0.0,
                "message": f"Decision confidence {decision_confidence:.2f} is below minimum {min_confidence:.2f}.",
                "decision_confidence": decision_confidence,
                "decision_min_confidence": min_confidence,
            }
        )
        return gated

    gated["decision_confidence"] = decision_confidence
    gated["decision_min_confidence"] = min_confidence
    return gated


def from_binance_market_id(symbol_id: str, quote_asset: str) -> str:
    quote = (quote_asset or "USDT").upper()
    if symbol_id.endswith(quote) and len(symbol_id) > len(quote):
        base = symbol_id[: -len(quote)]
        return f"{base}/{quote}"
    return symbol_id


def get_api_docs() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "TradingBackendAPI",
        "version": "1.1",
        "notes": [
            "All responses are JSON.",
            "Use FRED_API_KEY in .env to enable /macro/fred.",
        ],
        "env": {
            "required": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
            "optional": [
                "FRED_API_KEY",
                "FRED_BASE_URL",
                "FRED_TIMEOUT_SECONDS",
                "FRED_BYPASS_ENV_PROXY",
            ],
        },
        "routes": {
            "GET": {
                "/health": {"description": "Service health check."},
                "/signal": {
                    "description": "Generate ML/LLM trading signal.",
                    "query": ["buy_threshold", "sell_threshold", "adaptive_threshold_enabled"],
                },
                "/dashboard": {
                    "description": "Aggregate dashboard payload.",
                    "query": [
                        "decision_min_confidence",
                        "target_monitor_symbols",
                        "stable_asset",
                        "stable_reserve_min_pct",
                    ],
                },
                "/wallet": {"description": "Wallet balances and totals."},
                "/account": {
                    "description": "Symbol account snapshot.",
                    "query": ["symbol"],
                },
                "/market-scan": {
                    "description": "Scan symbols and rank opportunities.",
                    "query": ["max_symbols", "quote_asset"],
                },
                "/market/chart": {
                    "description": "Fetch recent OHLCV candles and autopilot markers for a symbol.",
                    "query": ["symbol", "interval", "limit"],
                },
                "/market/ticker": {
                    "description": "Fetch latest ticker metrics for a symbol.",
                    "query": ["symbol"],
                },
                "/autopilot/events": {
                    "description": "Return recent autopilot cycle events and latest trade state.",
                    "query": ["limit"],
                },
                "/live/history": {"description": "Recent captured live points."},
                "/autopilot/status": {"description": "Autopilot runtime status."},
                "/macro/fred": {
                    "description": "Fetch FRED macro series observations and latest value.",
                    "query": ["series_id", "limit", "start_date", "end_date"],
                    "example": "/macro/fred?series_id=DFF&limit=24",
                },
                "/docs": {"description": "API documentation payload."},
            },
            "POST": {
                "/autopilot/start": {"description": "Start autopilot loop."},
                "/autopilot/stop": {"description": "Stop autopilot loop."},
                "/trade/preview": {
                    "description": "Preview trade checks without execution.",
                    "latency_fields": ["warning_latency_ms", "degraded_latency_ms", "block_latency_ms", "consecutive_breach_limit"],
                },
                "/trade/action": {
                    "description": "Execute trade action (or dry-run).",
                    "latency_fields": ["warning_latency_ms", "degraded_latency_ms", "block_latency_ms", "consecutive_breach_limit"],
                },
                "/rebalance/execute": {"description": "Execute (or dry-run) recommended rebalance orders."},
                "/live/capture": {"description": "Append live signal point."},
                "/support/chat": {"description": "LLM support response for dashboard."},
                "/ai/command": {"description": "Natural language command parser."},
            },
        },
    }


def get_fred_series_observations(
    series_id: str,
    limit: int = 24,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is missing. Set it in .env to use /macro/fred.")

    norm_series = (series_id or "").strip().upper()
    if not norm_series:
        raise ValueError("series_id is required.")

    sample_limit = max(1, min(int(limit), 1000))
    params: dict[str, Any] = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": norm_series,
        "sort_order": "desc",
        "limit": sample_limit,
    }
    if start_date:
        params["observation_start"] = str(start_date)
    if end_date:
        params["observation_end"] = str(end_date)

    url = f"{FRED_BASE_URL}/series/observations"
    with requests.Session() as session:
        if FRED_BYPASS_ENV_PROXY:
            session.trust_env = False
        response = session.get(url, params=params, timeout=FRED_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()

    observations_raw = payload.get("observations", [])
    observations: list[dict[str, Any]] = []
    for item in observations_raw:
        value_text = str(item.get("value", "."))
        value_num: float | None = None
        if value_text not in {".", "", "nan", "NaN"}:
            try:
                value_num = float(value_text)
            except ValueError:
                value_num = None
        observations.append(
            {
                "date": item.get("date"),
                "value": value_num,
                "value_raw": value_text,
            }
        )

    latest = observations[0] if observations else {"date": None, "value": None, "value_raw": None}
    return {
        "series_id": norm_series,
        "count": len(observations),
        "latest": latest,
        "observations": observations,
    }


def binance_public_json(path: str, params: dict[str, Any] | None = None) -> Any:
    base = _binance_base_url()
    url = f"{base}{path}"
    session = get_binance_http_session()
    response = session.get(url, params=params or {}, timeout=BINANCE_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_binance_client():
    global BINANCE_CLIENT
    try:
        import ccxt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'ccxt'. Install requirements-api.txt.") from exc

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")

    with BINANCE_CLIENT_LOCK:
        if BINANCE_CLIENT is not None:
            return BINANCE_CLIENT

        exchange = ccxt.binance(
            {
                "apiKey": BINANCE_API_KEY,
                "secret": BINANCE_API_SECRET,
                "enableRateLimit": True,
                "options": {
                    "fetchCurrencies": False,
                },
            }
        )
        exchange.timeout = BINANCE_TIMEOUT_SECONDS * 1000
        if BINANCE_BYPASS_ENV_PROXY:
            session = getattr(exchange, "session", None)
            if session is not None:
                session.trust_env = False
        exchange.set_sandbox_mode(BINANCE_TESTNET)
        BINANCE_CLIENT = exchange
        return BINANCE_CLIENT


def format_binance_error(exc: Exception, action: str) -> str:
    raw = str(exc)
    lower = raw.lower()
    hints: list[str] = []
    if "exchangeinfo" in lower or "market metadata" in lower:
        hints.append("Binance market metadata request failed (exchangeInfo).")
        hints.append("This is usually IP whitelist, region/network, or temporary Binance connectivity restriction.")
    if "451" in lower or "restricted" in lower or "location" in lower:
        hints.append("Your region or endpoint may be restricted for this account/network.")
    if "invalid api-key" in lower or "signature" in lower or "-2015" in lower:
        hints.append("Check BINANCE_API_KEY/BINANCE_API_SECRET and account API permissions.")
    if "timeout" in lower or "network" in lower or "econn" in lower:
        hints.append("Network timeout/connectivity issue. Retry in a few seconds.")
    if "proxy" in lower:
        hints.append("Proxy tunnel appears to block Binance API. Set BINANCE_BYPASS_ENV_PROXY=true.")
    hints.append("If using Binance key restrictions, add your current public IP to API whitelist or disable whitelist for testing.")
    hint_text = " ".join(hints)
    return f"Binance {action} failed: {raw}. {hint_text}" if hint_text else f"Binance {action} failed: {raw}"


def wallet_permission_hint(error: Exception | str) -> str:
    raw = str(error)
    lowered = raw.lower()
    if "/api/v3/exchangeinfo" in lowered or "api/v3/exchangeinfo" in lowered:
        return (
            "Binance market metadata request failed (exchangeInfo). "
            "This is usually IP whitelist, region/network, or temporary Binance connectivity restriction."
        )
    if "/sapi/v1/capital/config/getall" in lowered or "capital/config/getall" in lowered:
        return (
            "Binance wallet permission missing for this API key. "
            "Enable API Read permissions (and allow SAPI wallet endpoints) in Binance API Management."
        )
    if "permission" in lowered or "forbidden" in lowered:
        return "Binance denied this wallet request. Check API key permissions and IP whitelist in Binance API Management."
    return raw


def get_ticker_with_metrics(symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    endpoint_path = "/api/v3/ticker/bookTicker"
    source = "ccxt"
    start = time.monotonic()
    try:
        ticker = exchange.fetch_ticker(norm_symbol)
    except Exception as exc:
        source = "binance_public_fallback"
        try:
            market_id = to_binance_market_id(norm_symbol)
            book = binance_public_json("/api/v3/ticker/bookTicker", {"symbol": market_id})
            price_data = binance_public_json("/api/v3/ticker/price", {"symbol": market_id})
            now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)
            ticker = {
                "symbol": norm_symbol,
                "bid": float(book.get("bidPrice") or 0.0),
                "ask": float(book.get("askPrice") or 0.0),
                "last": float(price_data.get("price") or 0.0),
                "timestamp": now_ms,
            }
        except Exception as fallback_exc:
            raise RuntimeError(
                format_binance_error(exc, "ticker fetch") + f" Fallback public ticker path failed: {fallback_exc}"
            ) from fallback_exc
    api_latency_ms = (time.monotonic() - start) * 1000
    now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)
    ticker_ts = ticker.get("timestamp")
    ticker_age_ms = float(max(0, now_ms - int(ticker_ts))) if ticker_ts else 0.0
    bid = float(ticker.get("bid") or ticker.get("last") or 0.0)
    ask = float(ticker.get("ask") or ticker.get("last") or 0.0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(ticker.get("last") or 0.0)
    spread_bps = ((ask - bid) / mid * 10000) if mid > 0 and ask >= bid else 0.0
    return ticker, {
        "api_latency_ms": float(api_latency_ms),
        "ticker_age_ms": float(ticker_age_ms),
        "spread_bps": float(spread_bps),
        "ticker_endpoint": endpoint_path,
        "ticker_source": source,
    }


def _resolve_latency_policy(
    *,
    max_api_latency_ms: int,
    warning_latency_ms: int | None = None,
    degraded_latency_ms: int | None = None,
    block_latency_ms: int | None = None,
    consecutive_breach_limit: int | None = None,
) -> dict[str, float | int]:
    block_ms = max(1000, int(block_latency_ms or max_api_latency_ms or 3000))
    warning_ms = int(warning_latency_ms) if warning_latency_ms is not None else max(400, int(block_ms * 0.50))
    degraded_ms = int(degraded_latency_ms) if degraded_latency_ms is not None else max(warning_ms + 200, int(block_ms * 0.75))
    if degraded_ms >= block_ms:
        degraded_ms = max(warning_ms + 100, block_ms - 300)
    if warning_ms >= degraded_ms:
        warning_ms = max(200, degraded_ms - 300)
    return {
        "warning_latency_ms": int(warning_ms),
        "degraded_latency_ms": int(degraded_ms),
        "block_latency_ms": int(block_ms),
        "consecutive_breach_limit": max(2, int(consecutive_breach_limit or 5)),
    }


def _latest_latency_sample(symbol: str) -> dict[str, Any]:
    ensure_latency_monitor(symbol)
    latest = get_runtime_value("latency_probe_latest", {})
    if isinstance(latest, dict) and latest:
        return dict(latest)
    return {
        "timestamp": utc_now_iso(),
        "symbol": normalize_symbol(symbol),
        "endpoint": "/api/v3/ticker/bookTicker",
        "dns_ms": 0.0,
        "connect_tls_ms": 0.0,
        "server_ms": 0.0,
        "total_ms": 0.0,
        "retries": 0,
        "error": "",
    }


def _latency_guard_state(
    *,
    symbol: str,
    action: str,
    ticker_latency_ms: float,
    policy: dict[str, float | int],
) -> dict[str, Any]:
    global LATENCY_CONSECUTIVE_BREACHES
    sample = _latest_latency_sample(symbol)
    total_ms = float(sample.get("total_ms", 0.0) or 0.0)

    if total_ms <= 0 and ticker_latency_ms > 0:
        total_ms = float(ticker_latency_ms)
        sample["total_ms"] = total_ms
    _record_latency_sample(sample)
    rolling = _rolling_latency_stats(window=20)

    warning_ms = int(policy["warning_latency_ms"])
    degraded_ms = int(policy["degraded_latency_ms"])
    block_ms = int(policy["block_latency_ms"])
    breach_limit = int(policy["consecutive_breach_limit"])

    sample_error = bool(sample.get("error"))
    effective_breach = bool(total_ms >= block_ms or (sample_error and total_ms >= degraded_ms))
    if effective_breach:
        with LATENCY_LOCK:
            LATENCY_CONSECUTIVE_BREACHES += 1
            streak = int(LATENCY_CONSECUTIVE_BREACHES)
    else:
        with LATENCY_LOCK:
            LATENCY_CONSECUTIVE_BREACHES = 0
            streak = 0

    p95_ms = float(rolling.get("p95_total_ms", 0.0) or 0.0)
    avg_ms = float(rolling.get("avg_total_ms", 0.0) or 0.0)
    sample_count = int(rolling.get("count", 0) or 0)

    mode = "normal"
    reason = "latency_within_limits"
    if total_ms >= (block_ms * 2.5):
        mode = "blocked"
        reason = "extreme_latency"
    elif streak >= breach_limit:
        mode = "blocked"
        reason = f"repeated_block_breaches_{streak}"
    elif total_ms >= block_ms or (sample_count >= 5 and p95_ms >= block_ms and avg_ms >= degraded_ms):
        mode = "exit_only"
        reason = "pre_block_exit_only"
    elif total_ms >= degraded_ms or avg_ms >= degraded_ms or (sample_error and sample_count >= 3):
        mode = "degraded"
        reason = "degraded_latency" if not sample_error else "degraded_probe_error"
    elif total_ms >= warning_ms:
        mode = "warning"
        reason = "warning_latency"

    root_cause = _infer_latency_root_cause(sample=sample, rolling=rolling, ticker_latency_ms=float(ticker_latency_ms))
    endpoint = str(sample.get("endpoint", "unknown"))

    allowed = True
    if mode == "degraded":
        allowed = action in {"market_sell", "cancel_all_orders"}
    elif mode == "exit_only":
        allowed = action in {"market_sell", "cancel_all_orders"}
    elif mode == "blocked":
        allowed = action == "cancel_all_orders"

    if mode in {"normal", "warning"}:
        message = (
            f"Market conditions accepted ({mode}): latency_total={total_ms:.0f}ms, "
            f"p95={p95_ms:.0f}ms, endpoint={endpoint}."
        )
    elif mode == "degraded":
        message = (
            f"Degraded latency mode (hold-only): endpoint={endpoint}, total={total_ms:.0f}ms, "
            f"avg={avg_ms:.0f}ms, threshold={degraded_ms}ms."
        )
    elif mode == "exit_only":
        message = (
            f"Exit-only latency mode: endpoint={endpoint}, total={total_ms:.0f}ms, "
            f"p95={p95_ms:.0f}ms near/over block={block_ms}ms."
        )
    else:
        message = (
            f"Blocked: API latency too high at endpoint={endpoint} "
            f"({total_ms:.0f}ms, block={block_ms}ms, streak={streak}/{breach_limit})."
        )

    return {
        "ok": bool(allowed),
        "mode": mode,
        "reason": reason,
        "message": message,
        "policy": policy,
        "rolling": rolling,
        "sample": sample,
        "consecutive_breaches": streak,
        "root_cause": root_cause,
    }


def validate_market_conditions(
    *,
    action: str,
    symbol: str,
    api_latency_ms: float,
    ticker_age_ms: float,
    spread_bps: float,
    max_api_latency_ms: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
    warning_latency_ms: int | None = None,
    degraded_latency_ms: int | None = None,
    block_latency_ms: int | None = None,
    consecutive_breach_limit: int | None = None,
) -> dict[str, Any]:
    last_trade_ts = float(get_runtime_value("last_trade_ts", 0.0) or 0.0)
    cooldown_left = max(0.0, min_trade_cooldown_seconds - (time.time() - last_trade_ts))
    policy = _resolve_latency_policy(
        max_api_latency_ms=max_api_latency_ms,
        warning_latency_ms=warning_latency_ms,
        degraded_latency_ms=degraded_latency_ms,
        block_latency_ms=block_latency_ms,
        consecutive_breach_limit=consecutive_breach_limit,
    )
    latency_state = _latency_guard_state(
        symbol=symbol,
        action=action,
        ticker_latency_ms=float(api_latency_ms),
        policy=policy,
    )
    if not bool(latency_state.get("ok", False)):
        return latency_state
    if ticker_age_ms > max_ticker_age_ms:
        return {
            **latency_state,
            "ok": False,
            "mode": "blocked",
            "reason": "stale_market_data",
            "message": f"Blocked: market data is stale ({ticker_age_ms:.0f}ms > {max_ticker_age_ms}ms).",
        }
    if spread_bps > max_spread_bps:
        return {
            **latency_state,
            "ok": False,
            "mode": "blocked",
            "reason": "spread_too_wide",
            "message": f"Blocked: spread too wide ({spread_bps:.1f} bps > {max_spread_bps:.1f} bps).",
        }
    if cooldown_left > 0:
        return {
            **latency_state,
            "ok": False,
            "mode": "blocked",
            "reason": "trade_cooldown",
            "message": f"Blocked: cooldown active ({cooldown_left:.1f}s remaining).",
        }
    return {
        **latency_state,
        "ok": True,
    }


def get_market_price(symbol: str) -> float:
    try:
        ticker, _ = get_ticker_with_metrics(symbol)
        last = ticker.get("last")
        if last is not None:
            return float(last)
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        raise RuntimeError("Ticker has no usable price")
    except Exception:
        fallback = get_runtime_value("latest_price_fallback")
        if fallback is not None:
            return float(fallback)
        raise RuntimeError("Unable to fetch market price from Binance and no fallback price is available.")


def get_market_chart(symbol: str, interval: str = "5m", limit: int = 200) -> dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    normalized_interval = str(interval or "5m").strip().lower()
    valid_intervals = {"1m", "5m", "15m", "1h"}
    if normalized_interval not in valid_intervals:
        raise ValueError(f"Unsupported interval '{interval}'. Use one of: {', '.join(sorted(valid_intervals))}.")

    safe_limit = max(50, min(int(limit or 200), 500))
    raw_candles = binance_public_json(
        "/api/v3/klines",
        params={
            "symbol": to_binance_market_id(normalized_symbol),
            "interval": normalized_interval,
            "limit": safe_limit,
        },
    )
    candles: list[dict[str, Any]] = []
    for row in raw_candles or []:
        if not isinstance(row, list) or len(row) < 7:
            continue
        candles.append(
            {
                "open_time_ms": int(row[0]),
                "open_time": pd.Timestamp(int(row[0]), unit="ms", tz="UTC").isoformat(),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time_ms": int(row[6]),
                "close_time": pd.Timestamp(int(row[6]), unit="ms", tz="UTC").isoformat(),
            }
        )

    current_price = float(candles[-1]["close"]) if candles else get_market_price(normalized_symbol)
    target_price = resolve_target_price(normalized_symbol)
    autopilot = autopilot_snapshot()
    latest_trade = autopilot.get("latest_trade_result", {}) or {}
    latest_decision = autopilot.get("latest_decision", {}) or {}
    logs = [entry for entry in autopilot.get("logs", []) if normalize_symbol(str(entry.get("symbol", normalized_symbol))) == normalized_symbol]
    markers: list[dict[str, Any]] = []
    for entry in logs[-100:]:
        skip_reason = str(entry.get("skip_reason") or "")
        final_action = str(entry.get("final_action") or entry.get("signal") or "HOLD").upper()
        marker_type = "trade" if final_action in {"BUY", "SELL"} else "state"
        if skip_reason:
            marker_type = "skip"
        markers.append(
            {
                "timestamp": str(entry.get("timestamp") or ""),
                "action": final_action,
                "type": marker_type,
                "price": float(entry.get("last_price") or entry.get("ask") or entry.get("bid") or current_price),
                "guard_mode": str(entry.get("guard_mode") or autopilot.get("execution_mode") or "normal"),
                "skip_reason": skip_reason,
                "probability_up": float(entry.get("probability_up") or 0.0),
                "confidence": float(entry.get("decision_confidence") or 0.0),
                "trade_status": str(entry.get("trade_status") or ""),
                "conversion_happened": bool((entry.get("conversion_result") or {}).get("status")),
                "conversion_result": entry.get("conversion_result") or {},
            }
        )

    latest_marker = None
    if latest_trade:
        latest_marker = {
            "timestamp": str((latest_trade.get("signal_trade") or {}).get("timestamp") or autopilot.get("updated_at") or utc_now_iso()),
            "action": str(latest_trade.get("final_action") or latest_trade.get("final_signal") or "HOLD").upper(),
            "type": "latest",
            "price": float((latest_trade.get("signal_trade") or {}).get("market_price") or current_price),
            "guard_mode": str((latest_trade.get("signal_trade") or {}).get("guard_mode") or autopilot.get("execution_mode") or "normal"),
            "skip_reason": str((latest_trade.get("signal_trade") or {}).get("skip_reason") or latest_trade.get("skip_reason") or ""),
            "probability_up": float(latest_trade.get("probability_up") or 0.0),
            "confidence": float(latest_decision.get("decision_confidence") or 0.0),
            "trade_status": str((latest_trade.get("signal_trade") or {}).get("status") or ""),
            "conversion_happened": bool((latest_trade.get("conversion_result") or {}).get("status")),
            "conversion_result": latest_trade.get("conversion_result") or {},
        }

    return {
        "symbol": normalized_symbol,
        "interval": normalized_interval,
        "limit": safe_limit,
        "candles": candles,
        "current_price": current_price,
        "target_price": float(target_price) if target_price is not None else None,
        "guard_mode": str(autopilot.get("execution_mode") or (latest_trade.get("signal_trade") or {}).get("guard_mode") or "normal"),
        "latest_decision": latest_decision,
        "latest_trade_result": latest_trade,
        "latest_marker": latest_marker,
        "markers": markers,
        "supported_intervals": ["1m", "5m", "15m", "1h"],
        "target_enabled": target_price is not None,
    }


def get_market_ticker(symbol: str) -> dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    ticker, metrics = get_ticker_with_metrics(normalized_symbol)
    return {
        "symbol": normalized_symbol,
        "last": float(ticker.get("last") or 0.0),
        "bid": float(ticker.get("bid") or 0.0),
        "ask": float(ticker.get("ask") or 0.0),
        "spread_bps": float(metrics.get("spread_bps") or 0.0),
        "api_latency_ms": float(metrics.get("api_latency_ms") or 0.0),
        "ticker_age_ms": int(metrics.get("ticker_age_ms") or 0),
        "captured_at": utc_now_iso(),
    }


def get_autopilot_events(limit: int = 100) -> dict[str, Any]:
    snapshot = autopilot_snapshot()
    safe_limit = max(1, min(int(limit or 100), 500))
    logs = list(snapshot.get("logs", []) or [])
    latest = snapshot.get("latest_trade_result", {}) or {}
    return {
        "symbol": str(snapshot.get("symbol") or CHECK_SYMBOL),
        "events": logs[-safe_limit:],
        "latest_trade_result": latest,
        "status": str(snapshot.get("status") or "idle"),
        "guard_mode": str(snapshot.get("execution_mode") or (latest.get("signal_trade") or {}).get("guard_mode") or "normal"),
        "reconciliation_state": str(snapshot.get("reconciliation_state") or "clean"),
        "requires_human_review": bool(snapshot.get("requires_human_review", False)),
        "alerts": list(snapshot.get("alerts", []) or [])[-20:],
        "updated_at": snapshot.get("updated_at"),
    }


def get_account_snapshot(symbol: str) -> dict[str, Any]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    base, quote = split_symbol_assets(norm_symbol)
    ticker, metrics = get_ticker_with_metrics(norm_symbol)
    bid = float(ticker.get("bid") or ticker.get("last") or 0.0)
    ask = float(ticker.get("ask") or ticker.get("last") or 0.0)
    mark = float(ticker.get("last") or ((bid + ask) / 2 if bid and ask else 0.0))

    try:
        balance = exchange.fetch_balance()
        try:
            open_orders = exchange.fetch_open_orders(norm_symbol)
        except Exception:
            open_orders = []
    except Exception as exc:
        raise RuntimeError(format_binance_error(exc, "account snapshot")) from exc

    base_total = float(balance.get("total", {}).get(base, 0) or 0)
    base_free = float(balance.get("free", {}).get(base, 0) or 0)
    base_used = float(balance.get("used", {}).get(base, 0) or 0)
    quote_total = float(balance.get("total", {}).get(quote, 0) or 0)
    quote_free = float(balance.get("free", {}).get(quote, 0) or 0)
    quote_used = float(balance.get("used", {}).get(quote, 0) or 0)
    account_value_quote = quote_total + (base_total * mark)

    positions = []
    if base_total > 0:
        positions.append({"coin": base, "size": base_free, "free": base_free, "used": base_used, "total": base_total, "entry_px": 0.0, "roe_pct": 0.0})

    return {
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "captured_at": utc_now_iso(),
        "symbol": norm_symbol,
        "best_ask": ask,
        "best_bid": bid,
        "account_value_quote": account_value_quote,
        "base_asset": base,
        "base_total": base_total,
        "base_free": base_free,
        "base_used": base_used,
        "quote_asset": quote,
        "quote_total": quote_total,
        "quote_free": quote_free,
        "quote_used": quote_used,
        "open_orders_count": len(open_orders),
        "api_latency_ms": metrics["api_latency_ms"],
        "ticker_age_ms": metrics["ticker_age_ms"],
        "spread_bps": metrics["spread_bps"],
        "positions": positions,
    }


def get_market_requirements(symbol: str) -> dict[str, float | int]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    try:
        exchange.load_markets()
    except Exception as exc:
        raise RuntimeError(format_binance_error(exc, "market metadata")) from exc
    market = exchange.market(norm_symbol)
    filters = list((market.get("info", {}) or {}).get("filters", []) or [])
    step_size = 0.0
    for item in filters:
        if str(item.get("filterType", "")).upper() == "LOT_SIZE":
            try:
                step_size = float(item.get("stepSize") or 0.0)
            except (TypeError, ValueError):
                step_size = 0.0
            break
    return {
        "min_qty": float((market.get("limits", {}).get("amount", {}).get("min") or 0) or 0),
        "min_notional": float((market.get("limits", {}).get("cost", {}).get("min") or 0) or 0),
        "qty_precision": int(market.get("precision", {}).get("amount") or 8),
        "price_precision": int(market.get("precision", {}).get("price") or 8),
        "step_size": float(step_size or 0.0),
    }


def validate_order_minimums(
    action: str,
    quantity: float,
    quote_amount: float,
    market_price: float,
    min_qty: float,
    min_notional: float,
) -> tuple[bool, str]:
    if action == "cancel_all_orders":
        return True, "No minimum required for cancel_all_orders."
    if action == "market_buy" and quote_amount > 0 and quantity <= 0:
        if min_notional > 0 and quote_amount < min_notional:
            return False, f"Blocked: quote amount too low ({quote_amount:.6f} < min notional {min_notional:.6f})."
        return True, "Minimum checks passed for quote-based market buy."
    if quantity <= 0:
        return False, "Blocked: quantity must be greater than zero."
    est_notional = quantity * market_price
    if min_qty > 0 and quantity < min_qty:
        return False, f"Blocked: quantity too low ({quantity:.10f} < min qty {min_qty:.10f})."
    if min_notional > 0 and est_notional < min_notional:
        return False, f"Blocked: notional too low ({est_notional:.6f} < min notional {min_notional:.6f})."
    return True, "Minimum checks passed."


def round_to_step(value: float, step_size: float, mode: str = "floor") -> float:
    numeric_value = max(0.0, float(value or 0.0))
    numeric_step = max(0.0, float(step_size or 0.0))
    if numeric_value <= 0:
        return 0.0
    if numeric_step <= 0:
        return numeric_value
    step = Decimal(str(numeric_step))
    raw = Decimal(str(numeric_value)) / step
    rounded = raw.to_integral_value(rounding=ROUND_CEILING if mode == "ceil" else ROUND_FLOOR)
    return float(rounded * step)


def build_buy_sizing_plan(
    *,
    symbol: str,
    market_price: float,
    market_reqs: dict[str, float | int],
    balance_snapshot: dict[str, Any],
    desired_quote: float,
    risk_cap_quote: float,
) -> dict[str, Any]:
    price = max(0.0, float(market_price or 0.0))
    min_qty = max(0.0, float(market_reqs.get("min_qty") or 0.0))
    min_notional = max(0.0, float(market_reqs.get("min_notional") or 0.0))
    step_size = max(0.0, float(market_reqs.get("step_size") or 0.0))
    free_quote = max(0.0, float(balance_snapshot.get("quote_free") or 0.0))
    free_base = max(0.0, float(balance_snapshot.get("base_free") or 0.0))
    allowed_quote = max(0.0, min(float(desired_quote or 0.0), float(risk_cap_quote or 0.0), free_quote))
    raw_min_qty_from_notional = (min_notional / price) if price > 0 and min_notional > 0 else 0.0
    minimum_base_required = max(min_qty, raw_min_qty_from_notional)
    if step_size > 0 and minimum_base_required > 0:
        minimum_base_required = round_to_step(minimum_base_required, step_size, mode="ceil")
    minimum_valid_quote = minimum_base_required * price if price > 0 else 0.0
    can_buy_minimum = bool(price > 0 and minimum_base_required > 0 and minimum_valid_quote <= min(float(risk_cap_quote or 0.0), free_quote))

    skip_reason = ""
    computed_quote = allowed_quote
    computed_base = round_to_step((computed_quote / price) if price > 0 else 0.0, step_size, mode="floor")
    if price <= 0:
        skip_reason = "invalid_market_price"
    elif free_quote <= 0:
        skip_reason = "insufficient_free_quote"
    elif allowed_quote <= 0:
        skip_reason = "risk_cap_zero"
    elif computed_base <= 0:
        skip_reason = "below_step_size_after_rounding"
    else:
        notional_after_round = computed_base * price
        if min_qty > 0 and computed_base < min_qty:
            if can_buy_minimum:
                computed_base = minimum_base_required
                computed_quote = minimum_valid_quote
            else:
                skip_reason = "below_min_qty"
        if not skip_reason and min_notional > 0 and notional_after_round < min_notional:
            if can_buy_minimum:
                computed_base = minimum_base_required
                computed_quote = minimum_valid_quote
            else:
                skip_reason = "below_min_notional"

    if not skip_reason and computed_base > 0:
        computed_base = max(0.0, round_to_step(computed_base, step_size, mode="floor"))
        computed_quote = computed_base * price
        if computed_base <= 0:
            skip_reason = "below_step_size_after_rounding"
        elif computed_quote > free_quote:
            skip_reason = "insufficient_free_quote"
        elif computed_quote > max(0.0, float(risk_cap_quote or 0.0)):
            skip_reason = "risk_cap_exceeded"

    return {
        "symbol": normalize_symbol(symbol),
        "min_qty": min_qty,
        "step_size": step_size,
        "min_notional": min_notional,
        "minimum_valid_quote": minimum_valid_quote,
        "minimum_valid_base": minimum_base_required,
        "free_quote": free_quote,
        "free_base": free_base,
        "computed_order_size_quote": max(0.0, computed_quote if not skip_reason else 0.0),
        "computed_order_size_base": max(0.0, computed_base if not skip_reason else 0.0),
        "can_buy_minimum": can_buy_minimum,
        "skip_reason": skip_reason,
    }


def resolve_effective_trade_request(
    *,
    exchange: Any,
    action: str,
    symbol: str,
    quantity: float,
    quote_amount: float,
    market_price: float,
    balance_snapshot: dict[str, Any],
    market_reqs: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    fee_buffer_pct = 0.0025
    effective_quantity = max(0.0, float(quantity or 0.0))
    effective_quote_amount = max(0.0, float(quote_amount or 0.0))
    base_free = max(0.0, float(balance_snapshot.get("base_free") or 0.0))
    quote_free = max(0.0, float(balance_snapshot.get("quote_free") or 0.0))
    size_cap_reason = ""
    if action == "market_buy":
        quote_cap = max(0.0, quote_free * (1.0 - fee_buffer_pct))
        if effective_quantity <= 0 and effective_quote_amount > 0:
            if effective_quote_amount > quote_cap:
                size_cap_reason = f"Capped quote buy amount to free {balance_snapshot.get('quote_asset')} balance."
            effective_quote_amount = min(effective_quote_amount, quote_cap)
        elif effective_quantity > 0 and market_price > 0:
            max_qty = quote_cap / market_price
            if effective_quantity > max_qty:
                size_cap_reason = f"Capped buy quantity to free {balance_snapshot.get('quote_asset')} balance."
            effective_quantity = min(effective_quantity, max_qty)
    elif action == "market_sell":
        if effective_quantity > base_free:
            size_cap_reason = f"Capped sell quantity to free {balance_snapshot.get('base_asset')} balance."
        effective_quantity = min(effective_quantity, base_free)
    if action in {"market_buy", "market_sell"} and effective_quantity > 0:
        effective_quantity = max(0.0, float(exchange.amount_to_precision(symbol, effective_quantity)))
        step_size = float((market_reqs or {}).get("step_size") or 0.0)
        if action == "market_buy" and step_size > 0:
            effective_quantity = max(0.0, round_to_step(effective_quantity, step_size, mode="floor"))
    return {
        "effective_quantity": effective_quantity,
        "effective_quote_amount": effective_quote_amount,
        "size_cap_reason": size_cap_reason,
    }


def dataframe_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    clean = df.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].astype(str)
    return clean.to_dict(orient="records")


def get_wallet_snapshot() -> dict[str, Any]:
    exchange = get_binance_client()
    try:
        balance = exchange.fetch_balance()
    except Exception as exc:
        raise RuntimeError(format_binance_error(exc, "wallet snapshot")) from exc
    totals = balance.get("total", {})
    free_map = balance.get("free", {})
    used_map = balance.get("used", {})
    non_zero_assets = [asset for asset, total in totals.items() if float(total or 0) > 0]

    markets_loaded = True
    try:
        exchange.load_markets()
    except Exception:
        markets_loaded = False

    rows: list[dict[str, Any]] = []
    total_estimated_usdt = 0.0
    for asset in non_zero_assets:
        total = float(totals.get(asset, 0) or 0)
        free = float(free_map.get(asset, 0) or 0)
        used = float(used_map.get(asset, 0) or 0)
        est_usdt = 0.0
        if asset in {"USDT", "USDC", "BUSD", "FDUSD", "DAI"}:
            est_usdt = total
        else:
            pair = f"{asset}/USDT"
            if markets_loaded and pair in exchange.markets:
                try:
                    ticker = exchange.fetch_ticker(pair)
                    est_usdt = total * float(ticker.get("last") or 0.0)
                except Exception:
                    est_usdt = 0.0
        total_estimated_usdt += est_usdt
        rows.append({"asset": asset, "free": free, "used": used, "total": total, "est_usdt": est_usdt})

    wallet_df = pd.DataFrame(rows)
    if not wallet_df.empty:
        wallet_df = wallet_df.sort_values("est_usdt", ascending=False)
    return {
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "captured_at": utc_now_iso(),
        "asset_count": len(non_zero_assets),
        "estimated_total_usdt": total_estimated_usdt,
        "balances": dataframe_to_records(wallet_df),
    }


def wallet_balance_index(wallet_snapshot: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    rows = list((wallet_snapshot or {}).get("balances", []) or [])
    index: dict[str, dict[str, float]] = {}
    for row in rows:
        asset = str(row.get("asset", "")).upper().strip()
        if not asset:
            continue
        index[asset] = {
            "free": float(row.get("free") or 0.0),
            "used": float(row.get("used") or 0.0),
            "total": float(row.get("total") or 0.0),
            "est_usdt": float(row.get("est_usdt") or 0.0),
        }
    return index


def wallet_free_summary(wallet_snapshot: dict[str, Any] | None, preferred_assets: list[str] | None = None) -> dict[str, float]:
    index = wallet_balance_index(wallet_snapshot)
    ordered_assets = preferred_assets or []
    summary: dict[str, float] = {}
    seen: set[str] = set()
    for asset in ordered_assets:
        key = str(asset).upper().strip()
        if not key:
            continue
        summary[key] = float(index.get(key, {}).get("free", 0.0))
        seen.add(key)
    for asset, values in sorted(index.items(), key=lambda item: float(item[1].get("est_usdt", 0.0)), reverse=True):
        if asset in seen or float(values.get("free", 0.0)) <= 0:
            continue
        summary[asset] = float(values.get("free", 0.0))
        if len(summary) >= 8:
            break
    return summary


def balance_snapshot_from_wallet(symbol: str, wallet_snapshot: dict[str, Any] | None, market_price: float = 0.0) -> dict[str, Any]:
    base_asset, quote_asset = split_symbol_assets(symbol)
    index = wallet_balance_index(wallet_snapshot)
    base_row = index.get(base_asset, {})
    quote_row = index.get(quote_asset, {})
    base_total = float(base_row.get("total", 0.0))
    quote_total = float(quote_row.get("total", 0.0))
    return {
        "symbol": normalize_symbol(symbol),
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "base_free": float(base_row.get("free", 0.0)),
        "base_used": float(base_row.get("used", 0.0)),
        "base_total": base_total,
        "quote_free": float(quote_row.get("free", 0.0)),
        "quote_used": float(quote_row.get("used", 0.0)),
        "quote_total": quote_total,
        "account_value_quote": quote_total + (base_total * max(0.0, float(market_price or 0.0))),
        "captured_at": str((wallet_snapshot or {}).get("captured_at") or utc_now_iso()),
        "open_orders_count": 0,
    }


def resolve_conversion_market(source_asset: str, target_asset: str) -> dict[str, Any] | None:
    exchange = get_binance_client()
    try:
        exchange.load_markets()
    except Exception:
        return None
    source = str(source_asset or "").upper().strip()
    target = str(target_asset or "").upper().strip()
    if not source or not target or source == target:
        return None
    sell_symbol = f"{source}/{target}"
    if sell_symbol in exchange.markets:
        return {"symbol": sell_symbol, "mode": "sell_base", "source_asset": source, "target_asset": target}
    buy_symbol = f"{target}/{source}"
    if buy_symbol in exchange.markets:
        return {"symbol": buy_symbol, "mode": "buy_base", "source_asset": source, "target_asset": target}
    return None


def build_conversion_plan(
    *,
    source_asset: str,
    target_asset: str,
    target_amount_needed: float,
    wallet_snapshot: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    market = resolve_conversion_market(source_asset, target_asset)
    if market is None:
        return None
    target_needed = max(0.0, float(target_amount_needed or 0.0))
    if target_needed <= 0:
        return None
    conversion_symbol = str(market["symbol"])
    conversion_price = float(get_market_price(conversion_symbol))
    conversion_reqs = get_market_requirements(conversion_symbol)
    conversion_buffer_pct = max(0.0, float(config.get("conversion_fee_buffer_pct", 0.003))) + (max(0.0, float(config.get("conversion_max_slippage_bps", 30.0))) / 10000.0)
    source_balance_snapshot = balance_snapshot_from_wallet(conversion_symbol, wallet_snapshot, conversion_price)
    if str(market["mode"]) == "sell_base":
        source_free = max(0.0, float(source_balance_snapshot.get("base_free") or 0.0))
        if source_free <= 0:
            return None
        source_qty_needed = target_needed / max(conversion_price * max(0.0001, 1.0 - conversion_buffer_pct), 1e-9)
        source_qty_needed = round_to_step(source_qty_needed, float(conversion_reqs.get("step_size") or 0.0), mode="ceil")
        if source_qty_needed <= 0 or source_qty_needed > source_free:
            return None
        sell_plan = build_sell_sizing_plan(
            symbol=conversion_symbol,
            market_price=conversion_price,
            market_reqs=conversion_reqs,
            balance_snapshot=source_balance_snapshot,
            desired_base=source_qty_needed,
        )
        if sell_plan.get("skip_reason"):
            return None
        converted_target_amount = float(sell_plan.get("computed_order_size_quote") or 0.0) * max(0.0001, 1.0 - conversion_buffer_pct)
        if converted_target_amount < target_needed:
            return None
        return {
            "path": "convert",
            "conversion_symbol": conversion_symbol,
            "conversion_mode": "sell_base",
            "source_asset": source_asset,
            "target_asset": target_asset,
            "action": "market_sell",
            "quantity": float(sell_plan.get("computed_order_size_base") or 0.0),
            "quote_amount": 0.0,
            "conversion_price": conversion_price,
            "conversion_buffer_pct": conversion_buffer_pct,
            "estimated_target_amount": converted_target_amount,
            "min_qty": float(conversion_reqs.get("min_qty") or 0.0),
            "step_size": float(conversion_reqs.get("step_size") or 0.0),
            "min_notional": float(conversion_reqs.get("min_notional") or 0.0),
        }
    source_free_quote = max(0.0, float(source_balance_snapshot.get("quote_free") or 0.0))
    if source_free_quote <= 0:
        return None
    source_quote_needed = target_needed * conversion_price * (1.0 + conversion_buffer_pct)
    buy_plan = build_buy_sizing_plan(
        symbol=conversion_symbol,
        market_price=conversion_price,
        market_reqs=conversion_reqs,
        balance_snapshot=source_balance_snapshot,
        desired_quote=source_quote_needed,
        risk_cap_quote=source_free_quote,
    )
    if buy_plan.get("skip_reason"):
        return None
    converted_target_amount = float(buy_plan.get("computed_order_size_base") or 0.0) * max(0.0001, 1.0 - conversion_buffer_pct)
    if converted_target_amount < target_needed:
        return None
    return {
        "path": "convert",
        "conversion_symbol": conversion_symbol,
        "conversion_mode": "buy_base",
        "source_asset": source_asset,
        "target_asset": target_asset,
        "action": "market_buy",
        "quantity": 0.0,
        "quote_amount": float(buy_plan.get("computed_order_size_quote") or 0.0),
        "conversion_price": conversion_price,
        "conversion_buffer_pct": conversion_buffer_pct,
        "estimated_target_amount": converted_target_amount,
        "min_qty": float(conversion_reqs.get("min_qty") or 0.0),
        "step_size": float(conversion_reqs.get("step_size") or 0.0),
        "min_notional": float(conversion_reqs.get("min_notional") or 0.0),
    }


def resolve_wallet_funding_path(
    *,
    symbol: str,
    desired_quote: float,
    minimum_quote_needed: float,
    wallet_snapshot: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _, quote_asset = split_symbol_assets(symbol)
    balance_index = wallet_balance_index(wallet_snapshot)
    free_quote = max(0.0, float(balance_index.get(quote_asset, {}).get("free", 0.0)))
    free_bnb = max(0.0, float(balance_index.get("BNB", {}).get("free", 0.0)))
    required_minimum_quote = max(0.0, float(minimum_quote_needed or 0.0))
    desired_quote_amount = max(0.0, float(desired_quote or 0.0))
    target_quote = max(required_minimum_quote, desired_quote_amount)
    minimum_shortfall = max(0.0, required_minimum_quote - free_quote)
    desired_shortfall = max(0.0, target_quote - free_quote)
    diagnostics: dict[str, Any] = {
        "required_quote_asset": quote_asset,
        "required_quote_amount": target_quote,
        "required_minimum_quote": required_minimum_quote,
        "minimum_quote_needed": required_minimum_quote,
        "desired_quote": desired_quote_amount,
        "free_direct_quote": free_quote,
        "free_bnb": free_bnb,
        "eligible_funding_assets": [],
        "attempted_conversion_pairs": [],
        "selected_conversion_pair": "",
        "estimated_quote_after_conversion": free_quote,
        "minimum_valid_target_buy_quote": required_minimum_quote,
        "minimum_shortfall": minimum_shortfall,
        "desired_shortfall": desired_shortfall,
    }
    if free_quote >= required_minimum_quote and required_minimum_quote > 0:
        return {
            "funding_path": "direct",
            "direct_trade_possible": True,
            "required_quote_asset": quote_asset,
            "required_quote_amount": target_quote,
            "available_quote_amount": free_quote,
            "free_direct_quote": free_quote,
            "free_bnb": free_bnb,
            "eligible_funding_assets": diagnostics["eligible_funding_assets"],
            "funding_diagnostics": diagnostics,
            "conversion_plan": None,
        }
    shortfall = minimum_shortfall
    if shortfall <= 0 and target_quote > 0:
        return {
            "funding_path": "direct",
            "direct_trade_possible": True,
            "required_quote_asset": quote_asset,
            "required_quote_amount": target_quote,
            "available_quote_amount": free_quote,
            "free_direct_quote": free_quote,
            "free_bnb": free_bnb,
            "eligible_funding_assets": diagnostics["eligible_funding_assets"],
            "funding_diagnostics": diagnostics,
            "conversion_plan": None,
        }
    source_candidates: list[str] = []
    preferred_assets = [quote_asset, "USDT", "FDUSD", "USDC", "BRL", "BNB"]
    for asset in preferred_assets:
        asset_key = str(asset).upper().strip()
        if asset_key == quote_asset:
            continue
        row = balance_index.get(asset_key)
        if row and float(row.get("free", 0.0)) > 0:
            source_candidates.append(asset_key)
    for asset, row in sorted(balance_index.items(), key=lambda item: float(item[1].get("est_usdt", 0.0)), reverse=True):
        if asset == quote_asset or asset in source_candidates or float(row.get("free", 0.0)) <= 0:
            continue
        source_candidates.append(asset)
    diagnostics["eligible_funding_assets"] = [
        {
            "asset": asset,
            "free": float(balance_index.get(asset, {}).get("free", 0.0)),
            "est_usdt": float(balance_index.get(asset, {}).get("est_usdt", 0.0)),
        }
        for asset in source_candidates
    ]
    for source_asset in source_candidates:
        attempted_pair = ""
        conversion_plan = None
        shortfall_targets: list[float] = []
        if desired_shortfall > 0:
            shortfall_targets.append(desired_shortfall)
        if shortfall > 0 and (not shortfall_targets or abs(shortfall_targets[-1] - shortfall) > 1e-9):
            shortfall_targets.append(shortfall)
        if not shortfall_targets:
            shortfall_targets.append(0.0)
        for shortfall_target in shortfall_targets:
            conversion_plan = build_conversion_plan(
                source_asset=source_asset,
                target_asset=quote_asset,
                target_amount_needed=shortfall_target,
                wallet_snapshot=wallet_snapshot,
                config=config,
            )
            if conversion_plan is not None:
                break
        if conversion_plan is not None:
            attempted_pair = str(conversion_plan.get("conversion_symbol") or "")
        else:
            market = resolve_conversion_market(source_asset, quote_asset)
            attempted_pair = str((market or {}).get("symbol") or "")
        diagnostics["attempted_conversion_pairs"].append(
            {
                "source_asset": source_asset,
                "conversion_symbol": attempted_pair,
                "status": "candidate_valid" if conversion_plan is not None else "candidate_rejected",
            }
        )
        if conversion_plan is None:
            continue
        total_available_after = free_quote + float(conversion_plan.get("estimated_target_amount") or 0.0)
        if total_available_after < required_minimum_quote:
            continue
        diagnostics["selected_conversion_pair"] = str(conversion_plan.get("conversion_symbol") or "")
        diagnostics["estimated_quote_after_conversion"] = total_available_after
        return {
            "funding_path": "convert",
            "direct_trade_possible": False,
            "required_quote_asset": quote_asset,
            "required_quote_amount": target_quote,
            "available_quote_amount": free_quote,
            "free_direct_quote": free_quote,
            "free_bnb": free_bnb,
            "eligible_funding_assets": diagnostics["eligible_funding_assets"],
            "funding_skip_reason": "",
            "funding_diagnostics": diagnostics,
            "conversion_plan": conversion_plan,
        }
    funding_skip_reason = "quote_asset_mismatch" if free_quote <= 0 and not source_candidates else "no_safe_funding_path"
    diagnostics["estimated_quote_after_conversion"] = free_quote
    diagnostics["funding_skip_reason"] = funding_skip_reason
    return {
        "funding_path": "none",
        "direct_trade_possible": False,
        "required_quote_asset": quote_asset,
        "required_quote_amount": target_quote,
        "available_quote_amount": free_quote,
        "free_direct_quote": free_quote,
        "free_bnb": free_bnb,
        "eligible_funding_assets": diagnostics["eligible_funding_assets"],
        "funding_skip_reason": funding_skip_reason,
        "funding_diagnostics": diagnostics,
        "conversion_plan": None,
    }


def is_leveraged_token_asset(asset: str) -> bool:
    token = str(asset or "").upper().strip()
    if not token:
        return False
    leveraged_suffixes = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")
    return any(token.endswith(suffix) for suffix in leveraged_suffixes)


def normalize_symbol_key(symbol: str) -> str:
    raw = str(symbol or "").upper().strip()
    if not raw:
        return ""
    return raw if "/" in raw else from_binance_market_id(raw)


def build_autopilot_candidate_universe(
    *,
    quote_assets: list[str],
    max_candidates: int,
    denylist: set[str],
) -> list[str]:
    exchange = get_binance_client()
    exchange.load_markets()
    quote_set = {str(item or "").upper().strip() for item in quote_assets if str(item or "").strip()}
    if not quote_set:
        quote_set = {"USDT"}
    normalized_denylist = {normalize_symbol_key(item) for item in (denylist or set()) if str(item).strip()}
    symbols: list[str] = []
    for symbol, market in exchange.markets.items():
        if not market.get("spot") or not market.get("active"):
            continue
        if ":" in symbol:
            continue
        base_asset = str(market.get("base") or "").upper().strip()
        quote_asset = str(market.get("quote") or "").upper().strip()
        norm_symbol = normalize_symbol(symbol)
        if quote_asset not in quote_set:
            continue
        if is_leveraged_token_asset(base_asset):
            continue
        if norm_symbol in normalized_denylist or norm_symbol.replace("/", "") in normalized_denylist:
            continue
        symbols.append(norm_symbol)
    symbols = sorted(set(symbols))
    return symbols[: max(5, int(max_candidates))]


def derive_candidate_signal(percentage_24h: float, decision_confidence: float, buy_threshold: float, sell_threshold: float) -> tuple[str, float, float]:
    pct = float(percentage_24h or 0.0)
    momentum_strength = min(abs(pct) / 3.0, 1.0)
    threshold_strength = max(0.05, min(1.0, (float(decision_confidence or 0.0) * 0.7) + (momentum_strength * 0.3)))
    buy_trigger = max(0.10, float(buy_threshold) * 0.9)
    sell_trigger = max(0.10, float(1.0 - sell_threshold) * 0.9)
    if pct >= buy_trigger:
        probability_up = 0.5 + min(0.45, momentum_strength * 0.45)
        return "BUY", probability_up, threshold_strength
    if pct <= -sell_trigger:
        probability_up = 0.5 - min(0.45, momentum_strength * 0.45)
        return "SELL", probability_up, threshold_strength
    return "HOLD", 0.5, min(0.55, threshold_strength)


def build_candidate_decision_payload(
    *,
    base_payload: dict[str, Any],
    symbol: str,
    signal: str,
    probability_up: float,
    confidence: float,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    decision = dict(payload.get("decision") or {})
    decision["signal"] = str(signal).upper()
    decision["probability_up"] = float(probability_up)
    decision["ml_signal"] = str(signal).upper()
    decision["ml_probability_up"] = float(probability_up)
    decision["decision_confidence"] = float(confidence)
    decision["symbol"] = normalize_symbol(symbol)
    payload["decision_confidence"] = float(confidence)
    payload["probability_up"] = float(probability_up)
    payload["signal"] = str(signal).upper()
    payload["decision"] = decision
    return payload


def rank_autopilot_opportunities(
    *,
    config: dict[str, Any],
    wallet_snapshot: dict[str, Any],
    base_decision_payload: dict[str, Any],
    preferred_symbol: str,
    previous_symbol: str,
) -> dict[str, Any]:
    quote_assets = config.get("opportunity_quote_assets", AUTOPILOT_UNIVERSE_QUOTES)
    if isinstance(quote_assets, str):
        quote_assets = [item.strip().upper() for item in str(quote_assets).split(",") if item.strip()]
    max_candidates = int(config.get("opportunity_max_candidates", AUTOPILOT_MAX_CANDIDATES))
    denylist = config.get("symbol_denylist", AUTOPILOT_SYMBOL_DENYLIST)
    if isinstance(denylist, str):
        denylist = {item.strip().upper() for item in denylist.split(",") if item.strip()}
    elif isinstance(denylist, list):
        denylist = {str(item).upper().strip() for item in denylist if str(item).strip()}
    else:
        denylist = set(denylist or set())
    switch_score_delta = float(config.get("switch_score_delta", AUTOPILOT_SWITCH_SCORE_DELTA))
    decision_confidence_base = float(base_decision_payload.get("decision_confidence") or 0.0)
    buy_threshold = float(base_decision_payload.get("buy_threshold") or BUY_THRESHOLD)
    sell_threshold = float(base_decision_payload.get("sell_threshold") or SELL_THRESHOLD)

    universe_symbols = build_autopilot_candidate_universe(
        quote_assets=list(quote_assets or AUTOPILOT_UNIVERSE_QUOTES),
        max_candidates=max_candidates,
        denylist=set(denylist),
    )
    if not universe_symbols:
        fallback = normalize_symbol(str(preferred_symbol or CHECK_SYMBOL))
        universe_symbols = [fallback]

    exchange = get_binance_client()
    tickers: dict[str, Any] = {}
    try:
        tickers = exchange.fetch_tickers(universe_symbols)
    except Exception:
        tickers = {}

    portfolio_value = float((wallet_snapshot or {}).get("estimated_total_usdt") or 0.0)
    ranked: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    winner_score = -1e9
    preferred_entry: dict[str, Any] | None = None

    for symbol in universe_symbols:
        ticker = tickers.get(symbol)
        if not ticker:
            try:
                ticker = exchange.fetch_ticker(symbol)
            except Exception:
                ticker = {}
        last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
        bid = float(ticker.get("bid") or last_price or 0.0)
        ask = float(ticker.get("ask") or last_price or 0.0)
        spread_bps = ((ask - bid) / last_price * 10000.0) if last_price > 0 and ask >= bid else 0.0
        quote_volume = float(ticker.get("quoteVolume") or 0.0)
        change_pct = float(ticker.get("percentage") or 0.0)
        raw_signal, probability_up, local_confidence = derive_candidate_signal(change_pct, decision_confidence_base, buy_threshold, sell_threshold)
        candidate_confidence = max(0.05, min(0.99, (decision_confidence_base * 0.55) + (local_confidence * 0.45)))
        candidate_payload = build_candidate_decision_payload(
            base_payload=base_decision_payload,
            symbol=symbol,
            signal=raw_signal,
            probability_up=probability_up,
            confidence=candidate_confidence,
        )
        try:
            account_snapshot = get_account_snapshot(symbol)
        except Exception as exc:
            ranked.append(
                {
                    "symbol": symbol,
                    "raw_signal": raw_signal,
                    "final_action": "SKIP",
                    "score": -1.0,
                    "confidence": candidate_confidence,
                    "expected_return": abs(change_pct) / 100.0,
                    "spread_bps": spread_bps,
                    "quote_volume": quote_volume,
                    "wallet_fundable": False,
                    "direct_trade_possible": False,
                    "funding_path": "none",
                    "skip_reason": "account_snapshot_error",
                    "rejection_reason": str(exc),
                }
            )
            continue

        signal_resolution = resolve_autopilot_signal(
            raw_signal=raw_signal,
            probability_up=probability_up,
            decision_confidence=candidate_confidence,
            decision_min_confidence=float(config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
            target_price=resolve_target_price(symbol),
            current_price=float(account_snapshot.get("best_bid") or account_snapshot.get("best_ask") or 0.0),
        )
        trade_plan = build_autopilot_trade_plan(
            symbol=symbol,
            signal=str(signal_resolution.get("final_signal") or raw_signal),
            probability_up=probability_up,
            decision_payload=candidate_payload,
            balance_snapshot=account_snapshot,
            wallet_snapshot=wallet_snapshot,
            config=config,
            forced_exit=bool(signal_resolution.get("forced_exit", False)),
        )
        final_action = str(trade_plan.get("final_action") or "HOLD").upper()
        execution_plan = dict(trade_plan.get("execution_plan") or {})
        skip_reason = str(trade_plan.get("skip_reason") or execution_plan.get("skip_reason") or "")
        expected_return = abs(change_pct) / 100.0
        liquidity_score = min(1.0, quote_volume / 2_000_000.0)
        spread_penalty = min(1.0, max(0.0, spread_bps) / 45.0)
        momentum_score = min(1.0, abs(change_pct) / 2.5)
        guard_action = "hold"
        if final_action == "BUY":
            guard_action = "market_buy"
        elif final_action == "SELL":
            guard_action = "market_sell"
        guard_state = validate_market_conditions(
            action=guard_action,
            symbol=symbol,
            api_latency_ms=float(account_snapshot.get("api_latency_ms") or 0.0),
            ticker_age_ms=int(account_snapshot.get("ticker_age_ms") or 0),
            spread_bps=float(account_snapshot.get("spread_bps") or spread_bps),
            max_api_latency_ms=int(config.get("max_api_latency_ms", 1200)),
            max_ticker_age_ms=int(config.get("max_ticker_age_ms", 3000)),
            max_spread_bps=float(config.get("max_spread_bps", 20.0)),
            min_trade_cooldown_seconds=int(config.get("min_trade_cooldown_seconds", 5)),
            warning_latency_ms=int(config.get("warning_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.55)),
            ),
            degraded_latency_ms=int(config.get("degraded_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.8)),
            ),
            block_latency_ms=int(config.get("block_latency_ms", int(config.get("max_api_latency_ms", 1200)))),
            consecutive_breach_limit=int(config.get("consecutive_breach_limit", 3)),
        )
        wallet_fundable = final_action in {"BUY", "SELL"} and not skip_reason and bool(guard_state.get("ok", False))
        actionable = bool(wallet_fundable)
        holding_value = 0.0
        if final_action == "SELL":
            holding_value = float(account_snapshot.get("base_free") or 0.0) * max(0.0, float(account_snapshot.get("best_bid") or last_price or 0.0))
        score = (candidate_confidence * 0.30) + (momentum_score * 0.28) + (liquidity_score * 0.18) + (expected_return * 0.14) - (spread_penalty * 0.22)
        if final_action == "SELL" and holding_value > 0 and portfolio_value > 0:
            score += min(0.25, (holding_value / portfolio_value) * 0.30)
        if symbol == normalize_symbol(str(previous_symbol or "")):
            score += 0.04
        if symbol == normalize_symbol(str(preferred_symbol or "")):
            score += 0.03
        if not wallet_fundable:
            score -= 0.45
        rejection_reason = ""
        if final_action == "SKIP":
            rejection_reason = skip_reason or "not_actionable"
        elif final_action == "HOLD":
            rejection_reason = signal_resolution.get("override_reason") or "hold_signal"
        elif not bool(guard_state.get("ok", False)):
            rejection_reason = str(guard_state.get("reason") or "guard_blocked")
        if rejection_reason:
            score -= 0.10

        row = {
            "symbol": symbol,
            "raw_signal": raw_signal,
            "resolved_signal": str(signal_resolution.get("final_signal") or raw_signal),
            "override_reason": str(signal_resolution.get("override_reason") or ""),
            "final_action": final_action,
            "score": float(score),
            "confidence": float(candidate_confidence),
            "expected_return": float(expected_return),
            "spread_bps": float(spread_bps),
            "quote_volume": float(quote_volume),
            "wallet_fundable": bool(wallet_fundable),
            "actionable": actionable,
            "direct_trade_possible": bool(trade_plan.get("direct_trade_possible", False)),
            "funding_path": str(trade_plan.get("funding_path") or "none"),
            "skip_reason": skip_reason,
            "guard_mode": str(guard_state.get("mode") or "normal"),
            "rejection_reason": rejection_reason,
            "trade_plan": trade_plan,
            "signal_resolution": signal_resolution,
            "account_snapshot": account_snapshot,
            "probability_up": float(probability_up),
            "decision_payload": candidate_payload,
        }
        ranked.append(row)
        if symbol == normalize_symbol(str(preferred_symbol or "")):
            preferred_entry = row
        if actionable and score > winner_score:
            winner = row
            winner_score = score

    ranked.sort(key=lambda item: float(item.get("score") or -1e9), reverse=True)
    if winner is None:
        winner = None

    if winner and preferred_entry and winner.get("symbol") != preferred_entry.get("symbol"):
        score_delta = float(winner.get("score") or 0.0) - float(preferred_entry.get("score") or 0.0)
        preferred_actionable = bool(preferred_entry.get("actionable"))
        if preferred_actionable and score_delta < switch_score_delta:
            winner = {**preferred_entry, "selection_reason": "hysteresis_sticky_symbol"}
        else:
            winner = {**winner, "selection_reason": "higher_opportunity_score"}
    elif winner:
        winner = {**winner, "selection_reason": "best_actionable_candidate"}

    return {
        "ranked": ranked,
        "winner": winner or {
            "symbol": "",
            "selection_reason": "no_actionable_candidate",
            "final_action": "SKIP",
            "wallet_fundable": False,
            "skip_reason": "no_actionable_candidate",
        },
        "meta": {
            "quote_assets": list(quote_assets or []),
            "universe_size": len(universe_symbols),
            "evaluated": len(ranked),
            "switch_score_delta": switch_score_delta,
            "multi_symbol_enabled": True,
            "actionable_candidates": sum(1 for row in ranked if bool(row.get("actionable"))),
        },
    }


def build_market_scan(max_symbols: int, quote_asset: str) -> list[dict[str, Any]]:
    exchange = get_binance_client()
    quote_asset = (quote_asset or "USDT").strip().upper()
    try:
        exchange.load_markets()
    except Exception as exc:
        try:
            tickers_24h = binance_public_json("/api/v3/ticker/24hr")
            if not isinstance(tickers_24h, list):
                return []
            rows = []
            for ticker in tickers_24h:
                symbol_id = str(ticker.get("symbol", "")).upper()
                if not symbol_id.endswith(quote_asset):
                    continue
                last = float(ticker.get("lastPrice") or 0.0)
                bid = float(ticker.get("bidPrice") or last or 0.0)
                ask = float(ticker.get("askPrice") or last or 0.0)
                pct = float(ticker.get("priceChangePercent") or 0.0)
                qv = float(ticker.get("quoteVolume") or 0.0)
                spread_bps = ((ask - bid) / last * 10000) if last > 0 and ask >= bid else 0.0
                ai_score = (abs(pct) * 1.2) + (min(qv / 1_000_000, 100.0) * 0.15) - (spread_bps * 0.05)
                rows.append(
                    {
                        "symbol": from_binance_market_id(symbol_id, quote_asset),
                        "last": last,
                        "change_pct": pct,
                        "quote_volume": qv,
                        "spread_bps": spread_bps,
                        "ai_bias": "BUY" if pct >= 0 else "SELL",
                        "ai_score": ai_score,
                    }
                )
            return rows[: max(1, int(max_symbols))]
        except Exception as fallback_exc:
            raise RuntimeError(
                format_binance_error(exc, "market scan metadata") + f" Fallback public scan path failed: {fallback_exc}"
            ) from fallback_exc

    universe = [
        symbol
        for symbol, market in exchange.markets.items()
        if market.get("spot") and market.get("active") and symbol.endswith(f"/{quote_asset}") and ":" not in symbol
    ]
    universe = sorted(universe)[: max(1, int(max_symbols))]
    if not universe:
        return []
    try:
        tickers = exchange.fetch_tickers(universe)
    except Exception:
        tickers = {}
        for symbol in universe:
            try:
                tickers[symbol] = exchange.fetch_ticker(symbol)
            except Exception:
                continue

    rows = []
    for symbol in universe:
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        last = float(ticker.get("last") or 0.0)
        bid = float(ticker.get("bid") or last or 0.0)
        ask = float(ticker.get("ask") or last or 0.0)
        pct = float(ticker.get("percentage") or 0.0)
        qv = float(ticker.get("quoteVolume") or 0.0)
        spread_bps = ((ask - bid) / last * 10000) if last > 0 and ask >= bid else 0.0
        ai_score = (abs(pct) * 1.2) + (min(qv / 1_000_000, 100.0) * 0.15) - (spread_bps * 0.05)
        rows.append(
            {
                "symbol": symbol,
                "last": last,
                "change_pct": pct,
                "quote_volume": qv,
                "spread_bps": spread_bps,
                "ai_bias": "BUY" if pct >= 0 else "SELL",
                "ai_score": ai_score,
            }
        )
    rows.sort(key=lambda item: item["ai_score"], reverse=True)
    return rows


def execute_account_action(
    action: str,
    symbol: str,
    quantity: float,
    quote_amount: float,
    dry_run: bool,
    max_api_latency_ms: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
    warning_latency_ms: int | None = None,
    degraded_latency_ms: int | None = None,
    block_latency_ms: int | None = None,
    consecutive_breach_limit: int | None = None,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    balance_snapshot = get_account_snapshot(norm_symbol)
    wallet_snapshot = get_wallet_snapshot()
    ensure_latency_monitor(norm_symbol)
    ticker, metrics = get_ticker_with_metrics(norm_symbol)
    market_price = float(ticker.get("last") or get_market_price(norm_symbol))
    market_reqs = get_market_requirements(norm_symbol)
    guard_state = validate_market_conditions(
        action=action,
        symbol=norm_symbol,
        api_latency_ms=metrics["api_latency_ms"],
        ticker_age_ms=metrics["ticker_age_ms"],
        spread_bps=metrics["spread_bps"],
        max_api_latency_ms=max_api_latency_ms,
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
        warning_latency_ms=warning_latency_ms,
        degraded_latency_ms=degraded_latency_ms,
        block_latency_ms=block_latency_ms,
        consecutive_breach_limit=consecutive_breach_limit,
    )
    guard_ok = bool(guard_state.get("ok", False))
    guard_message = str(guard_state.get("message", ""))
    payload = {
        "action": action,
        "symbol": norm_symbol,
        "quantity": quantity,
        "quote_amount": quote_amount,
        "market_price": market_price,
        "dry_run": dry_run,
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "api_latency_ms": metrics["api_latency_ms"],
        "ticker_age_ms": metrics["ticker_age_ms"],
        "spread_bps": metrics["spread_bps"],
        "ticker_endpoint": metrics.get("ticker_endpoint", ""),
        "ticker_source": metrics.get("ticker_source", ""),
        "latency_breakdown": guard_state.get("sample", {}),
        "latency_rolling": guard_state.get("rolling", {}),
        "latency_policy": guard_state.get("policy", {}),
        "latency_root_cause": guard_state.get("root_cause", "unknown"),
        "guard_mode": guard_state.get("mode", "normal"),
        "guard_reason": guard_state.get("reason", ""),
        "consecutive_latency_breaches": guard_state.get("consecutive_breaches", 0),
        "guard_ok": guard_ok,
        "guard_message": guard_message,
        "min_qty": market_reqs["min_qty"],
        "min_notional": market_reqs["min_notional"],
        "qty_precision": market_reqs["qty_precision"],
        "price_precision": market_reqs["price_precision"],
        "balance_snapshot": {
            "captured_at": balance_snapshot.get("captured_at"),
            "base_asset": balance_snapshot.get("base_asset"),
            "base_free": balance_snapshot.get("base_free"),
            "base_used": balance_snapshot.get("base_used"),
            "base_total": balance_snapshot.get("base_total"),
            "quote_asset": balance_snapshot.get("quote_asset"),
            "quote_free": balance_snapshot.get("quote_free"),
            "quote_used": balance_snapshot.get("quote_used"),
            "quote_total": balance_snapshot.get("quote_total"),
            "open_orders_count": balance_snapshot.get("open_orders_count"),
        },
        "wallet_summary": wallet_free_summary(wallet_snapshot, [balance_snapshot.get("base_asset", ""), balance_snapshot.get("quote_asset", ""), "BNB", "USDT", "FDUSD", "USDC", "BRL"]),
    }
    effective_trade = resolve_effective_trade_request(
        exchange=exchange,
        action=action,
        symbol=norm_symbol,
        quantity=quantity,
        quote_amount=quote_amount,
        market_price=market_price,
        balance_snapshot=balance_snapshot,
        market_reqs=market_reqs,
    )
    effective_quantity = float(effective_trade["effective_quantity"])
    effective_quote_amount = float(effective_trade["effective_quote_amount"])
    buy_sizing = build_buy_sizing_plan(
        symbol=norm_symbol,
        market_price=market_price,
        market_reqs=market_reqs,
        balance_snapshot=balance_snapshot,
        desired_quote=effective_quote_amount if effective_quote_amount > 0 else effective_quantity * market_price,
        risk_cap_quote=max(effective_quote_amount if effective_quote_amount > 0 else effective_quantity * market_price, 0.0),
    ) if action == "market_buy" else {
        "min_qty": float(market_reqs.get("min_qty") or 0.0),
        "step_size": float(market_reqs.get("step_size") or 0.0),
        "min_notional": float(market_reqs.get("min_notional") or 0.0),
        "minimum_valid_quote": 0.0,
        "free_quote": max(0.0, float(balance_snapshot.get("quote_free") or 0.0)),
        "free_base": max(0.0, float(balance_snapshot.get("base_free") or 0.0)),
        "computed_order_size_quote": effective_quantity * market_price,
        "computed_order_size_base": effective_quantity,
        "can_buy_minimum": False,
        "skip_reason": "",
    }
    funding_resolution = resolve_wallet_funding_path(
        symbol=norm_symbol,
        desired_quote=float(buy_sizing.get("computed_order_size_quote") or effective_quote_amount or effective_quantity * market_price),
        minimum_quote_needed=float(buy_sizing.get("minimum_valid_quote") or 0.0),
        wallet_snapshot=wallet_snapshot,
        config={},
    ) if action == "market_buy" else {
        "funding_path": "direct",
        "direct_trade_possible": action == "market_sell" and float(balance_snapshot.get("base_free") or 0.0) > 0,
        "required_quote_asset": balance_snapshot.get("quote_asset", ""),
        "required_quote_amount": float(buy_sizing.get("computed_order_size_quote") or 0.0),
        "available_quote_amount": float(balance_snapshot.get("quote_free") or 0.0),
        "conversion_plan": None,
    }
    payload["effective_quantity"] = effective_quantity
    payload["effective_quote_amount"] = effective_quote_amount
    payload["size_cap_reason"] = str(effective_trade["size_cap_reason"])
    payload["step_size"] = float(market_reqs.get("step_size") or 0.0)
    payload["minimum_valid_quote"] = float(buy_sizing.get("minimum_valid_quote") or 0.0)
    payload["free_quote"] = float(buy_sizing.get("free_quote") or 0.0)
    payload["free_base"] = float(buy_sizing.get("free_base") or 0.0)
    payload["computed_order_size_quote"] = float(buy_sizing.get("computed_order_size_quote") or 0.0)
    payload["computed_order_size_base"] = float(buy_sizing.get("computed_order_size_base") or 0.0)
    payload["can_buy_minimum"] = bool(buy_sizing.get("can_buy_minimum", False))
    payload["skip_reason"] = str(buy_sizing.get("skip_reason") or "")
    payload["required_base_asset"] = str(balance_snapshot.get("base_asset", ""))
    payload["required_quote_asset"] = str(funding_resolution.get("required_quote_asset", balance_snapshot.get("quote_asset", "")))
    payload["direct_trade_possible"] = bool(funding_resolution.get("direct_trade_possible", False))
    payload["funding_path"] = str(funding_resolution.get("funding_path", "none"))
    payload["funding_skip_reason"] = str(funding_resolution.get("funding_skip_reason", ""))
    payload["free_direct_quote"] = float(funding_resolution.get("free_direct_quote") or balance_snapshot.get("quote_free") or 0.0)
    payload["free_bnb"] = float(funding_resolution.get("free_bnb") or 0.0)
    payload["eligible_funding_assets"] = list(funding_resolution.get("eligible_funding_assets") or [])
    payload["funding_diagnostics"] = dict(funding_resolution.get("funding_diagnostics") or {})
    payload["conversion_plan"] = funding_resolution.get("conversion_plan") or {}
    ctx = dict(execution_context or {})
    fingerprint = str(ctx.get("execution_fingerprint") or execution_fingerprint(action, norm_symbol, effective_quantity, effective_quote_amount, int(ctx.get("run_id", 0) or 0), int(ctx.get("cycle", 0) or 0), str(ctx.get("stage") or "")))
    payload["execution_context"] = ctx
    payload["execution_fingerprint"] = fingerprint
    quantity = effective_quantity
    quote_amount = effective_quote_amount
    if not guard_ok:
        payload["status"] = "blocked"
        payload["message"] = guard_message
        payload["slow_endpoint"] = (guard_state.get("sample", {}) or {}).get("endpoint", "")
        log_trade_memory(source="trade_action", symbol=norm_symbol, action=action, payload=payload)
        return payload
    minimum_ok, minimum_message = validate_order_minimums(
        action=action,
        quantity=quantity,
        quote_amount=quote_amount,
        market_price=market_price,
        min_qty=float(market_reqs["min_qty"]),
        min_notional=float(market_reqs["min_notional"]),
    )
    payload["minimum_ok"] = minimum_ok
    payload["minimum_message"] = minimum_message
    if not minimum_ok:
        payload["status"] = "blocked"
        payload["message"] = minimum_message
        log_trade_memory(source="trade_action", symbol=norm_symbol, action=action, payload=payload)
        return payload
    if dry_run:
        payload["status"] = "dry_run_only"
        payload["message"] = "No order sent. Dry-run passed guard and minimum checks."
        log_trade_memory(source="trade_action", symbol=norm_symbol, action=action, payload=payload)
        return payload
    duplicate_record = has_recent_execution_fingerprint(fingerprint)
    if duplicate_record and str(duplicate_record.get("status") or "") in {"intent_recorded", "submitted", "executed", "uncertain"}:
        payload["status"] = "blocked"
        payload["message"] = "Blocked duplicate execution attempt during idempotency window."
        payload["skip_reason"] = "duplicate_execution_blocked"
        payload["duplicate_record"] = duplicate_record
        emit_autopilot_alert(
            "duplicate_execution_blocked",
            "critical" if bool(ctx.get("unattended_mode", False)) else "error",
            f"Blocked duplicate execution attempt for {norm_symbol}.",
            requires_human_review=bool(ctx.get("unattended_mode", False)),
            details={"execution_fingerprint": fingerprint, "duplicate_record": duplicate_record, "execution_context": ctx},
        )
        log_trade_memory(source="trade_action", symbol=norm_symbol, action=action, payload=payload)
        return payload
    intent_record = {
        "run_id": int(ctx.get("run_id", 0) or 0),
        "cycle": int(ctx.get("cycle", 0) or 0),
        "stage": str(ctx.get("stage") or ""),
        "symbol": norm_symbol,
        "action": action,
        "quantity": quantity,
        "quote_amount": quote_amount,
        "execution_fingerprint": fingerprint,
        "status": "intent_recorded",
        "dry_run": False,
        "unattended_mode": bool(ctx.get("unattended_mode", False)),
        "requires_confirmation": True,
    }
    append_execution_journal(intent_record)
    update_autopilot_state(latest_execution_intent=intent_record)
    if action == "market_buy":
        try:
            if quantity <= 0 and quote_amount > 0:
                try:
                    result = exchange.create_market_buy_order_with_cost(norm_symbol, quote_amount, {"newClientOrderId": fingerprint})
                except Exception:
                    result = exchange.create_order(norm_symbol, "market", "buy", None, None, {"quoteOrderQty": quote_amount, "newClientOrderId": fingerprint})
            else:
                if quantity <= 0:
                    raise ValueError("Provide quantity or quote amount for market_buy.")
                order_qty = float(exchange.amount_to_precision(norm_symbol, quantity))
                result = exchange.create_order(norm_symbol, "market", "buy", order_qty, None, {"newClientOrderId": fingerprint})
        except Exception as exc:
            uncertain_record = {**intent_record, "status": "uncertain", "error": str(exc)}
            append_execution_journal(uncertain_record)
            update_autopilot_state(latest_execution_intent=uncertain_record)
            raise
    elif action == "market_sell":
        try:
            if quantity <= 0:
                raise ValueError("Provide quantity for market_sell.")
            sell_qty = float(exchange.amount_to_precision(norm_symbol, quantity))
            result = exchange.create_order(norm_symbol, "market", "sell", sell_qty, None, {"newClientOrderId": fingerprint})
        except Exception as exc:
            uncertain_record = {**intent_record, "status": "uncertain", "error": str(exc)}
            append_execution_journal(uncertain_record)
            update_autopilot_state(latest_execution_intent=uncertain_record)
            raise
    elif action == "cancel_all_orders":
        try:
            result = exchange.cancel_all_orders(norm_symbol)
        except Exception as exc:
            uncertain_record = {**intent_record, "status": "uncertain", "error": str(exc)}
            append_execution_journal(uncertain_record)
            update_autopilot_state(latest_execution_intent=uncertain_record)
            raise
    else:
        raise ValueError(f"Unsupported action: {action}")
    payload["status"] = "executed"
    payload["result"] = result
    set_runtime_value("last_trade_ts", time.time())
    confirmed_record = {
        **intent_record,
        "status": "executed",
        "result": result,
        "exchange_order_id": str((result or {}).get("id") or ""),
        "exchange_client_order_id": str((result or {}).get("clientOrderId") or fingerprint),
    }
    append_execution_journal(confirmed_record)
    update_autopilot_state(latest_execution_intent=confirmed_record)
    log_trade_memory(source="trade_action", symbol=norm_symbol, action=action, payload=payload)
    return payload


def preview_trade_action(
    action: str,
    symbol: str,
    quantity: float,
    quote_amount: float,
    max_api_latency_ms: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
    warning_latency_ms: int | None = None,
    degraded_latency_ms: int | None = None,
    block_latency_ms: int | None = None,
    consecutive_breach_limit: int | None = None,
) -> dict[str, Any]:
    norm_symbol = normalize_symbol(symbol)
    exchange = get_binance_client()
    balance_snapshot = get_account_snapshot(norm_symbol)
    wallet_snapshot = get_wallet_snapshot()
    ensure_latency_monitor(norm_symbol)
    ticker, metrics = get_ticker_with_metrics(norm_symbol)
    market_price = float(ticker.get("last") or get_market_price(norm_symbol))
    market_reqs = get_market_requirements(norm_symbol)
    effective_trade = resolve_effective_trade_request(
        exchange=exchange,
        action=action,
        symbol=norm_symbol,
        quantity=quantity,
        quote_amount=quote_amount,
        market_price=market_price,
        balance_snapshot=balance_snapshot,
        market_reqs=market_reqs,
    )
    effective_quantity = float(effective_trade["effective_quantity"])
    effective_quote_amount = float(effective_trade["effective_quote_amount"])
    buy_sizing = build_buy_sizing_plan(
        symbol=norm_symbol,
        market_price=market_price,
        market_reqs=market_reqs,
        balance_snapshot=balance_snapshot,
        desired_quote=effective_quote_amount if effective_quote_amount > 0 else effective_quantity * market_price,
        risk_cap_quote=max(effective_quote_amount if effective_quote_amount > 0 else effective_quantity * market_price, 0.0),
    ) if action == "market_buy" else {
        "minimum_valid_quote": 0.0,
        "free_quote": max(0.0, float(balance_snapshot.get("quote_free") or 0.0)),
        "free_base": max(0.0, float(balance_snapshot.get("base_free") or 0.0)),
        "computed_order_size_quote": float((effective_quote_amount if effective_quote_amount > 0 else effective_quantity * market_price) or 0.0),
        "computed_order_size_base": effective_quantity,
        "can_buy_minimum": False,
        "skip_reason": "",
    }
    funding_resolution = resolve_wallet_funding_path(
        symbol=norm_symbol,
        desired_quote=float(buy_sizing.get("computed_order_size_quote") or effective_quote_amount or effective_quantity * market_price),
        minimum_quote_needed=float(buy_sizing.get("minimum_valid_quote") or 0.0),
        wallet_snapshot=wallet_snapshot,
        config={},
    ) if action == "market_buy" else {
        "funding_path": "direct",
        "direct_trade_possible": action == "market_sell" and float(balance_snapshot.get("base_free") or 0.0) > 0,
        "required_quote_asset": balance_snapshot.get("quote_asset", ""),
        "required_quote_amount": float(buy_sizing.get("computed_order_size_quote") or 0.0),
        "available_quote_amount": float(balance_snapshot.get("quote_free") or 0.0),
        "conversion_plan": None,
    }
    guard_state = validate_market_conditions(
        action=action,
        symbol=norm_symbol,
        api_latency_ms=metrics["api_latency_ms"],
        ticker_age_ms=metrics["ticker_age_ms"],
        spread_bps=metrics["spread_bps"],
        max_api_latency_ms=max_api_latency_ms,
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
        warning_latency_ms=warning_latency_ms,
        degraded_latency_ms=degraded_latency_ms,
        block_latency_ms=block_latency_ms,
        consecutive_breach_limit=consecutive_breach_limit,
    )
    guard_ok = bool(guard_state.get("ok", False))
    guard_message = str(guard_state.get("message", ""))
    minimum_ok, minimum_message = validate_order_minimums(
        action=action,
        quantity=effective_quantity,
        quote_amount=effective_quote_amount,
        market_price=market_price,
        min_qty=float(market_reqs["min_qty"]),
        min_notional=float(market_reqs["min_notional"]),
    )
    return {
        "status": "ok" if guard_ok and minimum_ok else "blocked",
        "previewed_at": utc_now_iso(),
        "action": action,
        "symbol": norm_symbol,
        "quantity": quantity,
        "quote_amount": quote_amount,
        "effective_quantity": effective_quantity,
        "effective_quote_amount": effective_quote_amount,
        "size_cap_reason": str(effective_trade["size_cap_reason"]),
        "step_size": float(market_reqs.get("step_size") or 0.0),
        "market_price": market_price,
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "api_latency_ms": metrics["api_latency_ms"],
        "ticker_age_ms": metrics["ticker_age_ms"],
        "spread_bps": metrics["spread_bps"],
        "ticker_endpoint": metrics.get("ticker_endpoint", ""),
        "ticker_source": metrics.get("ticker_source", ""),
        "latency_breakdown": guard_state.get("sample", {}),
        "latency_rolling": guard_state.get("rolling", {}),
        "latency_policy": guard_state.get("policy", {}),
        "latency_root_cause": guard_state.get("root_cause", "unknown"),
        "guard_mode": guard_state.get("mode", "normal"),
        "guard_reason": guard_state.get("reason", ""),
        "consecutive_latency_breaches": guard_state.get("consecutive_breaches", 0),
        "guard_ok": guard_ok,
        "guard_message": guard_message,
        "minimum_ok": minimum_ok,
        "minimum_message": minimum_message,
        "min_qty": market_reqs["min_qty"],
        "min_notional": market_reqs["min_notional"],
        "qty_precision": market_reqs["qty_precision"],
        "price_precision": market_reqs["price_precision"],
        "minimum_valid_quote": float(buy_sizing.get("minimum_valid_quote") or 0.0),
        "free_quote": float(buy_sizing.get("free_quote") or 0.0),
        "free_base": float(buy_sizing.get("free_base") or 0.0),
        "computed_order_size_quote": float(buy_sizing.get("computed_order_size_quote") or 0.0),
        "computed_order_size_base": float(buy_sizing.get("computed_order_size_base") or 0.0),
        "can_buy_minimum": bool(buy_sizing.get("can_buy_minimum", False)),
        "skip_reason": str(buy_sizing.get("skip_reason") or ""),
        "required_base_asset": balance_snapshot.get("base_asset", ""),
        "required_quote_asset": funding_resolution.get("required_quote_asset", balance_snapshot.get("quote_asset", "")),
        "wallet_summary": wallet_free_summary(wallet_snapshot, [balance_snapshot.get("base_asset", ""), balance_snapshot.get("quote_asset", ""), "BNB", "USDT", "FDUSD", "USDC", "BRL"]),
        "direct_trade_possible": bool(funding_resolution.get("direct_trade_possible", False)),
        "funding_path": str(funding_resolution.get("funding_path", "none")),
        "funding_skip_reason": str(funding_resolution.get("funding_skip_reason", "")),
        "free_direct_quote": float(funding_resolution.get("free_direct_quote") or balance_snapshot.get("quote_free") or 0.0),
        "free_bnb": float(funding_resolution.get("free_bnb") or 0.0),
        "eligible_funding_assets": list(funding_resolution.get("eligible_funding_assets") or []),
        "funding_diagnostics": dict(funding_resolution.get("funding_diagnostics") or {}),
        "conversion_plan": funding_resolution.get("conversion_plan") or {},
        "balance_snapshot": {
            "captured_at": balance_snapshot.get("captured_at"),
            "base_asset": balance_snapshot.get("base_asset"),
            "base_free": balance_snapshot.get("base_free"),
            "quote_asset": balance_snapshot.get("quote_asset"),
            "quote_free": balance_snapshot.get("quote_free"),
        },
    }


def append_live_history(price_symbol: str, signal: str, probability_up: float) -> dict[str, Any]:
    snapshot = get_account_snapshot(price_symbol)
    point = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "symbol": normalize_symbol(price_symbol),
        "best_bid": snapshot["best_bid"],
        "best_ask": snapshot["best_ask"],
        "signal": signal,
        "probability_up": probability_up,
        "account_value": snapshot["account_value_quote"],
    }
    with STATE_LOCK:
        history = list(RUNTIME_STATE.get("live_history", []))
        history.append(point)
        RUNTIME_STATE["live_history"] = history[-500:]
    return point


def get_live_history() -> list[dict[str, Any]]:
    with STATE_LOCK:
        return list(RUNTIME_STATE.get("live_history", []))


def resolve_latest_model() -> Path:
    latest = latest_file_with_suffix(ARTIFACTS_DIR, "_model.joblib")
    if latest is None:
        raise FileNotFoundError("No model artifact found in research/artifacts")
    return latest


def resolve_dataset_path() -> Path:
    multi_symbol = DATA_DIR / "multi_symbol_training_dataset.csv"
    if multi_symbol.exists():
        return multi_symbol
    preferred = DATA_DIR / "aapl_training_dataset.csv"
    if preferred.exists():
        return preferred
    latest_dataset = latest_file_with_suffix(DATA_DIR, "_training_dataset.csv")
    if latest_dataset is None:
        raise FileNotFoundError("No combined dataset found in research/data_sets")
    return latest_dataset


def risk_plan(
    deposit_amount: float,
    active_capital_pct: float,
    reserve_pct: float,
    max_trade_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_daily_loss_pct: float,
    max_drawdown_pct: float,
    withdrawal_target_pct: float,
) -> dict[str, float]:
    active_capital = deposit_amount * active_capital_pct
    reserve_cash = deposit_amount * reserve_pct
    max_trade_size = active_capital * max_trade_pct
    daily_loss_limit = deposit_amount * max_daily_loss_pct
    drawdown_limit = deposit_amount * max_drawdown_pct
    withdrawal_target = deposit_amount * (1 + withdrawal_target_pct)
    return {
        "deposit_amount": deposit_amount,
        "active_capital": active_capital,
        "reserve_cash": reserve_cash,
        "max_trade_size": max_trade_size,
        "daily_loss_limit": daily_loss_limit,
        "drawdown_limit": drawdown_limit,
        "withdrawal_target": withdrawal_target,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
    }


def clamp_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def confidence_strength(signal: str, probability_up: float) -> float:
    prob_up = clamp_value(float(probability_up), 0.0, 1.0)
    if str(signal).upper() == "BUY":
        edge = (prob_up - 0.5) * 2.0
    elif str(signal).upper() == "SELL":
        edge = ((1.0 - prob_up) - 0.5) * 2.0
    else:
        edge = 0.0
    return clamp_value(edge, 0.0, 1.0)


def auto_order_size_from_confidence(
    signal: str,
    probability_up: float,
    market_price: float,
    max_trade_size_quote: float,
    min_confidence: float = SIZE_MIN_CONFIDENCE,
) -> dict[str, float | str]:
    normalized_signal = str(signal).upper()
    if normalized_signal not in {"BUY", "SELL"}:
        return {
            "status": "hold_signal",
            "strength": 0.0,
            "quote_size": 0.0,
            "base_size": 0.0,
            "allocation_pct": 0.0,
            "message": "Signal is HOLD, so no position size is allocated.",
        }

    strength = confidence_strength(signal, probability_up)
    if strength < min_confidence or market_price <= 0 or max_trade_size_quote <= 0:
        return {
            "status": "blocked",
            "strength": strength,
            "quote_size": 0.0,
            "base_size": 0.0,
            "allocation_pct": 0.0,
            "message": "Confidence, price, or max trade size is too low for allocation.",
        }
    allocation_pct = 0.20 + (0.80 * strength)
    quote_size = max_trade_size_quote * allocation_pct
    base_size = quote_size / market_price
    return {
        "status": "ok",
        "strength": strength,
        "quote_size": quote_size,
        "base_size": base_size,
        "allocation_pct": allocation_pct,
    }


def build_sell_sizing_plan(
    *,
    symbol: str,
    market_price: float,
    market_reqs: dict[str, float | int],
    balance_snapshot: dict[str, Any],
    desired_base: float,
) -> dict[str, Any]:
    price = max(0.0, float(market_price or 0.0))
    min_qty = max(0.0, float(market_reqs.get("min_qty") or 0.0))
    min_notional = max(0.0, float(market_reqs.get("min_notional") or 0.0))
    step_size = max(0.0, float(market_reqs.get("step_size") or 0.0))
    free_quote = max(0.0, float(balance_snapshot.get("quote_free") or 0.0))
    free_base = max(0.0, float(balance_snapshot.get("base_free") or 0.0))
    computed_base = min(max(0.0, float(desired_base or 0.0)), free_base)
    computed_base = round_to_step(computed_base, step_size, mode="floor")
    computed_quote = computed_base * price
    minimum_base_required = max(min_qty, (min_notional / price) if price > 0 and min_notional > 0 else 0.0)
    if step_size > 0 and minimum_base_required > 0:
        minimum_base_required = round_to_step(minimum_base_required, step_size, mode="ceil")
    dust_base = max(0.0, free_base - computed_base)
    skip_reason = ""
    if free_base <= 0:
        skip_reason = "insufficient_free_base"
    elif computed_base <= 0:
        skip_reason = "below_step_size_after_rounding"
    elif min_qty > 0 and computed_base < min_qty:
        if free_base >= minimum_base_required > 0:
            computed_base = minimum_base_required
            computed_quote = computed_base * price
            dust_base = max(0.0, free_base - computed_base)
        else:
            skip_reason = "below_min_qty"
    elif min_notional > 0 and computed_quote < min_notional:
        if free_base >= minimum_base_required > 0:
            computed_base = minimum_base_required
            computed_quote = computed_base * price
            dust_base = max(0.0, free_base - computed_base)
        else:
            skip_reason = "below_min_notional"
    return {
        "symbol": normalize_symbol(symbol),
        "min_qty": min_qty,
        "step_size": step_size,
        "min_notional": min_notional,
        "minimum_valid_quote": min_notional,
        "minimum_valid_base": minimum_base_required,
        "free_quote": free_quote,
        "free_base": free_base,
        "dust_base": dust_base,
        "computed_order_size_quote": max(0.0, computed_quote if not skip_reason else 0.0),
        "computed_order_size_base": max(0.0, computed_base if not skip_reason else 0.0),
        "can_buy_minimum": False,
        "skip_reason": skip_reason,
    }


def resolve_autopilot_runtime_profile(
    *,
    signal: str,
    decision_confidence: float,
    cycle_plan: dict[str, Any],
) -> dict[str, int]:
    normalized_signal = str(signal or "HOLD").upper()
    confidence = max(0.0, float(decision_confidence or 0.0))
    recommended_cycles = int(cycle_plan.get("recommended_cycles") or 0)
    if normalized_signal in {"BUY", "SELL"}:
        default_cycles = 6 if confidence >= 0.7 else 4
        default_interval = 15 if confidence >= 0.75 else 20 if confidence >= 0.6 else 30
    else:
        default_cycles = 2
        default_interval = 45 if confidence < 0.6 else 30
    target_cycles = recommended_cycles if recommended_cycles > 0 else default_cycles
    return {
        "interval_seconds": max(10, default_interval),
        "target_cycles": max(1, min(12, target_cycles)),
    }


def build_autopilot_trade_plan(
    *,
    symbol: str,
    signal: str,
    probability_up: float,
    decision_payload: dict[str, Any],
    balance_snapshot: dict[str, Any],
    wallet_snapshot: dict[str, Any] | None,
    config: dict[str, Any],
    forced_exit: bool = False,
) -> dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    base_asset, quote_asset = split_symbol_assets(normalized_symbol)
    market_reqs = get_market_requirements(normalized_symbol)
    market_price = float(balance_snapshot.get("best_ask") or balance_snapshot.get("best_bid") or get_market_price(normalized_symbol) or 0.0)
    wallet_total_usdt = float((wallet_snapshot or {}).get("estimated_total_usdt") or 0.0)
    account_value = max(wallet_total_usdt, float(balance_snapshot.get("account_value_quote") or 0.0), ACCOUNT_REFERENCE_USD)
    plan = risk_plan(
        deposit_amount=account_value,
        active_capital_pct=float(config.get("active_capital_pct", 0.85 if account_value <= 25 else (0.50 if ACCOUNT_REFERENCE_USD <= 10 else 0.70))),
        reserve_pct=float(config.get("reserve_pct", 0.30)),
        max_trade_pct=float(config.get("max_trade_pct", 0.80 if account_value <= 25 else (0.20 if ACCOUNT_REFERENCE_USD <= 10 else 0.10))),
        stop_loss_pct=float(config.get("stop_loss_pct", 0.03)),
        take_profit_pct=float(config.get("take_profit_pct", 0.05)),
        max_daily_loss_pct=float(config.get("max_daily_loss_pct", 0.05)),
        max_drawdown_pct=float(config.get("max_drawdown_pct", 0.15)),
        withdrawal_target_pct=float(config.get("withdrawal_target_pct", 0.25)),
    )
    max_trade_size_quote = max(0.0, float(plan["max_trade_size"]))
    size_plan = auto_order_size_from_confidence(
        signal=signal,
        probability_up=probability_up,
        market_price=market_price,
        max_trade_size_quote=max_trade_size_quote,
        min_confidence=float(config.get("size_min_confidence", SIZE_MIN_CONFIDENCE)),
    )
    size_plan = apply_decision_confidence_gate(
        size_plan=size_plan,
        decision_payload=decision_payload,
        min_confidence=float(config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
    )
    base_size = float(size_plan.get("base_size") or 0.0)
    quote_size = float(size_plan.get("quote_size") or 0.0)
    skip_reason = ""
    final_action = str(signal or "HOLD").upper()
    funding_path = "none"
    direct_trade_possible = False
    funding_skip_reason = ""
    funding_diagnostics: dict[str, Any] = {}
    eligible_funding_assets: list[dict[str, Any]] = []
    free_bnb = float(wallet_balance_index(wallet_snapshot).get("BNB", {}).get("free", 0.0))
    conversion_plan: dict[str, Any] | None = None
    execution_plan: dict[str, Any]
    wallet_summary = wallet_free_summary(wallet_snapshot, [base_asset, quote_asset, "BNB", "USDT", "FDUSD", "USDC", "BRL"])

    if final_action == "BUY":
        if str(size_plan.get("status")) != "ok":
            skip_reason = "size_planner_blocked"
            execution_plan = build_buy_sizing_plan(
                symbol=normalized_symbol,
                market_price=market_price,
                market_reqs=market_reqs,
                balance_snapshot=balance_snapshot,
                desired_quote=0.0,
                risk_cap_quote=max_trade_size_quote,
            )
        else:
            direct_execution_plan = build_buy_sizing_plan(
                symbol=normalized_symbol,
                market_price=market_price,
                market_reqs=market_reqs,
                balance_snapshot=balance_snapshot,
                desired_quote=quote_size,
                risk_cap_quote=max_trade_size_quote,
            )
            direct_trade_possible = not bool(direct_execution_plan.get("skip_reason"))
            if direct_trade_possible:
                funding_path = "direct"
                execution_plan = direct_execution_plan
                skip_reason = ""
            else:
                funding_resolution = resolve_wallet_funding_path(
                    symbol=normalized_symbol,
                    desired_quote=max(quote_size, float(direct_execution_plan.get("minimum_valid_quote") or 0.0)),
                    minimum_quote_needed=float(direct_execution_plan.get("minimum_valid_quote") or 0.0),
                    wallet_snapshot=wallet_snapshot or {"balances": []},
                    config=config,
                )
                conversion_plan = funding_resolution.get("conversion_plan")
                funding_path = str(funding_resolution.get("funding_path") or "none")
                direct_trade_possible = bool(funding_resolution.get("direct_trade_possible", False))
                funding_skip_reason = str(funding_resolution.get("funding_skip_reason") or "")
                funding_diagnostics = dict(funding_resolution.get("funding_diagnostics") or {})
                eligible_funding_assets = list(funding_resolution.get("eligible_funding_assets") or [])
                free_bnb = float(funding_resolution.get("free_bnb") or free_bnb)
                if conversion_plan is not None:
                    conversion_quote_amount = float(conversion_plan.get("estimated_target_amount") or 0.0)
                    funded_snapshot = dict(balance_snapshot)
                    funded_snapshot["quote_free"] = max(0.0, float(balance_snapshot.get("quote_free") or 0.0)) + conversion_quote_amount
                    execution_plan = build_buy_sizing_plan(
                        symbol=normalized_symbol,
                        market_price=market_price,
                        market_reqs=market_reqs,
                        balance_snapshot=funded_snapshot,
                        desired_quote=quote_size,
                        risk_cap_quote=max_trade_size_quote,
                    )
                    if execution_plan.get("skip_reason"):
                        raw_conversion_reason = str(execution_plan.get("skip_reason") or "")
                        if raw_conversion_reason in {"risk_cap_exceeded", "risk_cap_zero"}:
                            skip_reason = "conversion_exceeds_risk_limit"
                        elif raw_conversion_reason in {"below_min_qty", "below_min_notional", "below_step_size_after_rounding"}:
                            skip_reason = "conversion_below_minimum"
                        else:
                            skip_reason = raw_conversion_reason or "conversion_below_minimum"
                    else:
                        skip_reason = ""
                else:
                    execution_plan = direct_execution_plan
                    skip_reason = str(direct_execution_plan.get("skip_reason") or "")
                    if skip_reason in {"insufficient_free_quote", "risk_cap_zero", "risk_cap_exceeded"}:
                        skip_reason = funding_skip_reason or "no_safe_funding_path"
                if skip_reason:
                    final_action = "SKIP"
    elif final_action == "SELL":
        desired_sell_base = max(0.0, float(balance_snapshot.get("base_free") or 0.0)) if forced_exit else base_size
        execution_plan = build_sell_sizing_plan(
            symbol=normalized_symbol,
            market_price=market_price,
            market_reqs=market_reqs,
            balance_snapshot=balance_snapshot,
            desired_base=desired_sell_base,
        )
        skip_reason = str(execution_plan.get("skip_reason") or "")
        if skip_reason:
            final_action = "SKIP"
        funding_path = "direct" if not skip_reason else "none"
        direct_trade_possible = not bool(skip_reason)
    else:
        execution_plan = {
            "symbol": normalized_symbol,
            "min_qty": float(market_reqs.get("min_qty") or 0.0),
            "step_size": float(market_reqs.get("step_size") or 0.0),
            "min_notional": float(market_reqs.get("min_notional") or 0.0),
            "minimum_valid_quote": 0.0,
            "free_quote": max(0.0, float(balance_snapshot.get("quote_free") or 0.0)),
            "free_base": max(0.0, float(balance_snapshot.get("base_free") or 0.0)),
            "computed_order_size_quote": 0.0,
            "computed_order_size_base": 0.0,
            "can_buy_minimum": False,
            "skip_reason": "",
        }

    if final_action == "BUY" and skip_reason:
        final_action = "SKIP"
    if final_action == "SELL" and skip_reason:
        final_action = "SKIP"

    return {
        "final_action": final_action,
        "skip_reason": skip_reason,
        "market_price": market_price,
        "market_requirements": market_reqs,
        "wallet_summary": wallet_summary,
        "required_base_asset": base_asset,
        "required_quote_asset": quote_asset,
        "free_base": float(balance_snapshot.get("base_free") or 0.0),
        "free_quote": float(balance_snapshot.get("quote_free") or 0.0),
        "direct_trade_possible": direct_trade_possible,
        "funding_path": funding_path,
        "funding_skip_reason": funding_skip_reason,
        "free_bnb": free_bnb,
        "eligible_funding_assets": eligible_funding_assets,
        "funding_diagnostics": funding_diagnostics,
        "conversion_plan": conversion_plan,
        "risk_plan": plan,
        "size_plan": {
            **size_plan,
            "quote_size": float(execution_plan.get("computed_order_size_quote") or quote_size),
            "base_size": float(execution_plan.get("computed_order_size_base") or base_size),
        },
        "execution_plan": execution_plan,
    }


def estimate_cycles_to_goal(
    current_value: float,
    goal_value: float,
    signal: str,
    probability_up: float,
    order_size: float,
    market_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    fee_drag_pct: float,
) -> dict[str, Any]:
    goal_profit = max(0.0, goal_value - current_value)
    if goal_profit <= 0:
        return {"status": "goal_already_reached", "goal_profit": 0.0, "expected_profit_per_cycle": 0.0, "expected_return_per_cycle": 0.0, "recommended_cycles": 0, "win_probability": 0.5}
    normalized_signal = str(signal).upper()
    if normalized_signal not in {"BUY", "SELL"}:
        return {
            "status": "hold_signal",
            "goal_profit": goal_profit,
            "expected_profit_per_cycle": 0.0,
            "expected_return_per_cycle": 0.0,
            "recommended_cycles": 0,
            "win_probability": 0.5,
            "message": "Signal is HOLD, so cycle planning is paused until BUY/SELL appears.",
        }
    notional = max(0.0, order_size * market_price)
    if notional <= 0:
        return {
            "status": "invalid_order_notional",
            "goal_profit": goal_profit,
            "expected_profit_per_cycle": 0.0,
            "expected_return_per_cycle": 0.0,
            "recommended_cycles": 0,
            "win_probability": 0.5,
            "message": "Order notional is zero; increase size or wait for stronger signal.",
        }
    prob_up = clamp_value(float(probability_up), 0.0, 1.0)
    if normalized_signal == "BUY":
        win_probability = prob_up
    elif normalized_signal == "SELL":
        win_probability = 1.0 - prob_up
    else:
        win_probability = 0.5
    expected_return = (win_probability * take_profit_pct) - ((1.0 - win_probability) * stop_loss_pct) - fee_drag_pct
    expected_profit_per_cycle = notional * expected_return
    if expected_profit_per_cycle <= 0:
        return {
            "status": "non_positive_expectancy",
            "goal_profit": goal_profit,
            "expected_profit_per_cycle": expected_profit_per_cycle,
            "expected_return_per_cycle": expected_return,
            "recommended_cycles": 0,
            "win_probability": win_probability,
        }
    recommended_cycles = int(math.ceil(goal_profit / expected_profit_per_cycle))
    return {
        "status": "ok",
        "goal_profit": goal_profit,
        "expected_profit_per_cycle": expected_profit_per_cycle,
        "expected_return_per_cycle": expected_return,
        "recommended_cycles": max(1, min(recommended_cycles, 1000)),
        "win_probability": win_probability,
    }


def compute_goal_progress(starting_value: float, current_value: float, goal_value: float) -> dict[str, Any]:
    start = max(0.0, float(starting_value or 0.0))
    current = max(0.0, float(current_value or 0.0))
    goal = max(0.0, float(goal_value or 0.0))
    if goal <= 0:
        return {
            "starting_value": start,
            "current_value": current,
            "goal_value": goal,
            "progress_pct": 0.0,
            "goal_reached": False,
        }
    if goal <= start:
        progress_pct = 1.0 if current >= goal else max(0.0, min(1.0, current / goal))
    else:
        progress_pct = max(0.0, min(1.0, (current - start) / max(goal - start, 1e-9)))
    return {
        "starting_value": start,
        "current_value": current,
        "goal_value": goal,
        "progress_pct": progress_pct,
        "goal_reached": current >= goal and goal > 0,
    }


def build_no_action_trade_plan(symbol: str, balance_snapshot: dict[str, Any], wallet_snapshot: dict[str, Any] | None, skip_reason: str) -> dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    base_asset, quote_asset = split_symbol_assets(normalized_symbol)
    market_reqs = get_market_requirements(normalized_symbol)
    return {
        "final_action": "SKIP" if skip_reason else "HOLD",
        "skip_reason": skip_reason,
        "market_price": float(balance_snapshot.get("best_ask") or balance_snapshot.get("best_bid") or 0.0),
        "market_requirements": market_reqs,
        "wallet_summary": wallet_free_summary(wallet_snapshot, [base_asset, quote_asset, "BNB", "USDT", "FDUSD", "USDC", "BRL"]),
        "required_base_asset": base_asset,
        "required_quote_asset": quote_asset,
        "free_base": float(balance_snapshot.get("base_free") or 0.0),
        "free_quote": float(balance_snapshot.get("quote_free") or 0.0),
        "direct_trade_possible": False,
        "funding_path": "none",
        "funding_skip_reason": skip_reason,
        "free_bnb": float(wallet_balance_index(wallet_snapshot).get("BNB", {}).get("free", 0.0)),
        "eligible_funding_assets": [],
        "funding_diagnostics": {},
        "conversion_plan": None,
        "risk_plan": {},
        "size_plan": {"status": "hold_signal", "strength": 0.0, "quote_size": 0.0, "base_size": 0.0, "allocation_pct": 0.0},
        "execution_plan": {
            "symbol": normalized_symbol,
            "min_qty": float(market_reqs.get("min_qty") or 0.0),
            "step_size": float(market_reqs.get("step_size") or 0.0),
            "min_notional": float(market_reqs.get("min_notional") or 0.0),
            "minimum_valid_quote": 0.0,
            "free_quote": max(0.0, float(balance_snapshot.get("quote_free") or 0.0)),
            "free_base": max(0.0, float(balance_snapshot.get("base_free") or 0.0)),
            "computed_order_size_quote": 0.0,
            "computed_order_size_base": 0.0,
            "can_buy_minimum": False,
            "skip_reason": skip_reason,
        },
    }


def execute_goal_finalization(config: dict[str, Any], touched_assets: set[str] | None = None, run_id: int = 0, cycle: int = 0, unattended_mode: bool = False) -> dict[str, Any]:
    stable_asset = str(config.get("stable_asset", PROFIT_PARKING_STABLE_ASSET)).upper()
    allow_live = bool(config.get("allow_live", False))
    dry_run = not allow_live
    wallet_snapshot = get_wallet_snapshot()
    stable_set = stable_assets_set(stable_asset)
    skip_assets = {str(item).upper().strip() for item in (touched_assets or set()) if str(item).strip()}
    parking_orders = build_quote_parking_orders(
        wallet_snapshot=wallet_snapshot,
        stable_asset=stable_asset,
        max_orders=max(3, int(config.get("finalization_max_orders", 20))),
        min_usdt_value=max(0.0, float(config.get("finalization_min_usdt_value", 5.0))),
    )
    parking_orders = [
        order
        for order in parking_orders
        if str(order.get("source_asset", "")).upper().strip() not in stable_set
        and str(order.get("source_asset", "")).upper().strip() not in skip_assets
    ]
    results: list[dict[str, Any]] = []
    for order in parking_orders:
        result = execute_account_action(
            action=str(order.get("action") or ""),
            symbol=str(order.get("symbol") or ""),
            quantity=float(order.get("quantity") or 0.0),
            quote_amount=float(order.get("quote_amount") or 0.0),
            dry_run=dry_run,
            max_api_latency_ms=int(config.get("max_api_latency_ms", 1200)),
            warning_latency_ms=int(config.get("warning_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.55))),
            degraded_latency_ms=int(config.get("degraded_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.8))),
            block_latency_ms=int(config.get("block_latency_ms", config.get("max_api_latency_ms", 1200))),
            consecutive_breach_limit=int(config.get("consecutive_breach_limit", 3)),
            max_ticker_age_ms=int(config.get("max_ticker_age_ms", 3000)),
            max_spread_bps=float(config.get("max_spread_bps", 20.0)),
            min_trade_cooldown_seconds=int(config.get("min_trade_cooldown_seconds", 0)),
            execution_context={
                "run_id": run_id,
                "cycle": cycle,
                "stage": "finalization",
                "unattended_mode": unattended_mode,
            },
        )
        result["order_type"] = "goal_finalization"
        results.append(result)
    wallet_after = get_wallet_snapshot()
    dust_assets = [
        row
        for row in list((wallet_after or {}).get("balances", []) or [])
        if str(row.get("asset", "")).upper().strip() not in stable_set and float(row.get("est_usdt") or 0.0) > 0
    ]
    return {
        "status": "ok" if results or not dust_assets else "dust_only",
        "stable_asset": stable_asset,
        "orders": parking_orders,
        "results": results,
        "dust_assets": dust_assets,
        "wallet_after": wallet_after,
    }


def llm_support_chat(message: str, context: dict[str, Any]) -> dict[str, Any]:
    ensure_llm_startup_logged()
    if not LLM_ENABLED:
        return {"status": "disabled", "answer": "LLM is disabled. Enable LLM_ENABLED=true in environment."}
    system_prompt = (
        "You are a trading support assistant for a live dashboard. "
        "Be concise, practical, and risk-aware. "
        "Explain whether ML and LLM are merged using provided context and suggest safe next actions."
    )
    user_prompt = f"User question: {message}\nCurrent engine context: {json.dumps(context, ensure_ascii=True)}"
    result = llm_chat(system_prompt=system_prompt, user_content=user_prompt, max_new_tokens=300)
    if result.get("status") != "ok":
        return {
            "status": "error",
            "answer": f"LLM support chat error: {result.get('error') or 'unknown error'}",
            "endpoint": result.get("endpoint", ""),
            "provider": result.get("provider", ""),
            "model": result.get("active_model") or result.get("active_model_path", ""),
            "fallback_active": bool(result.get("fallback_active", False)),
        }
    answer = str(result.get("content", "")).strip() or "No response content from LLM."
    return {
        "status": "ok",
        "answer": answer,
        "endpoint": result.get("endpoint", ""),
        "provider": result.get("provider", ""),
        "model": result.get("active_model") or result.get("active_model_path", ""),
        "is_trained_model": bool(result.get("is_trained_model", False)),
        "fallback_active": bool(result.get("fallback_active", False)),
    }


def _prod_alpha_artifact_paths() -> tuple[Path | None, Path | None]:
    folder = ARTIFACTS_DIR / "prod_alpha"
    if not folder.exists():
        return None, None
    comparison_files = sorted(folder.glob("comparison_*.json"), key=lambda path: path.stat().st_mtime)
    prediction_files = sorted(folder.glob("predictions_*.csv"), key=lambda path: path.stat().st_mtime)
    return (comparison_files[-1] if comparison_files else None, prediction_files[-1] if prediction_files else None)


def _run_prod_alpha_pipeline_once() -> bool:
    try:
        import importlib

        pipeline_module = importlib.import_module("prod_alpha.pipeline")
        AlphaConfig = getattr(pipeline_module, "AlphaConfig")
        run_production_pipeline = getattr(pipeline_module, "run_production_pipeline")

        cfg = AlphaConfig(
            asset_paths={
                "BTC": PROD_ALPHA_BTC_PATH,
                "ETH": PROD_ALPHA_ETH_PATH,
                "SOL": PROD_ALPHA_SOL_PATH,
            },
            benchmark_asset="BTC",
        )
        run_production_pipeline(cfg)
        return True
    except Exception:
        return False


def _load_prod_alpha_outputs() -> tuple[dict[str, Any] | None, pd.DataFrame | None, dict[str, str]]:
    comparison_path, predictions_path = _prod_alpha_artifact_paths()
    if (comparison_path is None or predictions_path is None) and PROD_ALPHA_AUTO_RUN:
        _run_prod_alpha_pipeline_once()
        comparison_path, predictions_path = _prod_alpha_artifact_paths()

    if comparison_path is None or predictions_path is None:
        return None, None, {"comparison": "", "predictions": ""}

    with PROD_ALPHA_LOCK:
        if PROD_ALPHA_CACHE.get("comparison_path") != str(comparison_path):
            with comparison_path.open("r", encoding="utf-8") as handle:
                PROD_ALPHA_CACHE["comparison"] = json.load(handle)
            PROD_ALPHA_CACHE["comparison_path"] = str(comparison_path)

        if PROD_ALPHA_CACHE.get("predictions_path") != str(predictions_path):
            PROD_ALPHA_CACHE["predictions"] = pd.read_csv(predictions_path)
            PROD_ALPHA_CACHE["predictions_path"] = str(predictions_path)

        comparison = PROD_ALPHA_CACHE.get("comparison")
        predictions = PROD_ALPHA_CACHE.get("predictions")

    return comparison, predictions, {"comparison": str(comparison_path), "predictions": str(predictions_path)}


def _signal_from_class(value: int) -> str:
    if int(value) == 2:
        return "BUY"
    if int(value) == 1:
        return "SELL"
    return "HOLD"


def _prod_symbol_candidates(symbol: str) -> list[str]:
    norm = normalize_symbol(symbol)
    base, quote = split_symbol_assets(norm)
    return [
        norm.upper(),
        base.upper(),
        f"{base}{quote}".upper(),
    ]


def _build_prod_alpha_decision(
    buy_threshold: float,
    sell_threshold: float,
    requested_symbol: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    comparison, predictions, artifacts = _load_prod_alpha_outputs()
    if comparison is None or predictions is None or predictions.empty:
        raise RuntimeError("prod_alpha artifacts not available; run research/prod_alpha/run_prod_alpha.py first.")

    deployed_strategy = str(comparison.get("deployed_strategy", "baseline_fallback"))
    signal_column = "upgraded_signal" if deployed_strategy == "upgraded" else "baseline_signal"
    if signal_column not in predictions.columns:
        signal_column = "baseline_signal" if "baseline_signal" in predictions.columns else "upgraded_signal"

    target_symbol = requested_symbol or PROD_ALPHA_TARGET_SYMBOL
    candidates = {item.upper() for item in _prod_symbol_candidates(target_symbol)}
    pred = predictions.copy()
    pred["symbol_norm"] = pred["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
    filtered = pred[pred["symbol_norm"].isin({item.replace("/", "") for item in candidates})]
    if filtered.empty:
        fallback_base = split_symbol_assets(PROD_ALPHA_TARGET_SYMBOL)[0]
        filtered = pred[pred["symbol"].astype(str).str.upper() == fallback_base.upper()]
    if filtered.empty:
        filtered = pred

    filtered = filtered.sort_values("timestamp")
    latest = filtered.iloc[-1]
    signal_class = int(latest.get(signal_column, 0))
    signal = _signal_from_class(signal_class)

    metrics_key = "prediction_metrics"
    pred_metrics = comparison.get(metrics_key, {}) if isinstance(comparison.get(metrics_key, {}), dict) else {}
    if deployed_strategy == "upgraded":
        model_quality = float(pred_metrics.get("upgraded_macro_f1", 0.45) or 0.45)
    else:
        model_quality = float(pred_metrics.get("baseline_macro_f1", 0.45) or 0.45)
    decision_confidence = clamp_value(0.45 + (0.45 * model_quality), 0.0, 1.0)

    if signal == "BUY":
        probability_up = clamp_value(0.5 + (0.4 * decision_confidence), 0.5, 0.95)
    elif signal == "SELL":
        probability_up = clamp_value(0.5 - (0.4 * decision_confidence), 0.05, 0.5)
    else:
        probability_up = 0.5

    decision = {
        "signal": signal,
        "probability_up": float(probability_up),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
        "threshold_context": {
            "mode": "prod_alpha_artifact",
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
            "deployed_strategy": deployed_strategy,
            "signal_source_column": signal_column,
        },
        "decision_engine": f"prod_alpha_{deployed_strategy}",
        "market_regime": {
            "enabled": False,
            "regime": "artifact",
            "notes": "Using latest prod_alpha artifact decision stream.",
        },
        "ml_signal": signal,
        "ml_probability_up": float(probability_up),
        "llm_overlay": {"enabled": False, "status": "not_used", "message": "prod_alpha engine mode bypasses LLM overlay."},
        "llm_merge": {"enabled": False, "status": "not_used", "merged_probability_up": float(probability_up), "merge_weight": 0.0},
        "decision_confidence": float(decision_confidence),
        "confidence_breakdown": {
            "enabled": True,
            "model_confidence": float(model_quality),
            "llm_confidence": 0.0,
            "data_confidence": float(model_quality),
            "weights": {"model": 1.0, "llm": 0.0, "data": 0.0},
            "final_signal": signal,
            "merged_confidence": float(decision_confidence),
            "engine_mode": "prod_alpha",
        },
        "safety_guard": {"triggered": False, "reason": ""},
        "latest_timestamp": str(latest.get("timestamp", "")),
        "prod_alpha": {
            "deployed_strategy": deployed_strategy,
            "signal_column": signal_column,
            "symbol": str(latest.get("symbol", "")),
            "artifacts": artifacts,
        },
    }
    return decision, artifacts


def get_decision_payload(
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    adaptive_threshold_enabled: bool = ADAPTIVE_THRESHOLD_ENABLED,
    engine_mode: str | None = None,
) -> dict[str, Any]:
    selected_engine = str(engine_mode or SIGNAL_ENGINE_MODE or "classic").strip().lower()
    model_path = resolve_model_path()
    dataset_path = DATASET_PATH
    artifacts: dict[str, str] = {}

    if selected_engine == "prod_alpha":
        try:
            decision, artifacts = _build_prod_alpha_decision(
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                requested_symbol=PROD_ALPHA_TARGET_SYMBOL,
            )
        except Exception:
            decision = generate_trade_decision(
                model_path=model_path,
                dataset_path=dataset_path,
                base_buy_threshold=buy_threshold,
                base_sell_threshold=sell_threshold,
                adaptive_threshold_enabled=adaptive_threshold_enabled,
            )
            selected_engine = "classic_fallback"
    else:
        decision = generate_trade_decision(
            model_path=model_path,
            dataset_path=dataset_path,
            base_buy_threshold=buy_threshold,
            base_sell_threshold=sell_threshold,
            adaptive_threshold_enabled=adaptive_threshold_enabled,
        )

    feature_row, latest_row = load_latest_row(dataset_path)
    set_runtime_value("latest_price_fallback", latest_row.get("close", None))
    return {
        "status": "ok",
        "generated_at": utc_now_iso(),
        "engine": decision.get("decision_engine", "ml_primary"),
        "engine_mode": selected_engine,
        "signal": decision.get("signal"),
        "probability_up": decision.get("probability_up"),
        "buy_threshold": decision.get("buy_threshold"),
        "sell_threshold": decision.get("sell_threshold"),
        "ml_signal": decision.get("ml_signal"),
        "ml_probability_up": decision.get("ml_probability_up"),
        "llm_overlay": decision.get("llm_overlay"),
        "llm_merge": decision.get("llm_merge"),
        "decision_confidence": decision.get("decision_confidence"),
        "confidence_breakdown": decision.get("confidence_breakdown"),
        "decision": decision,
        "feature_row": dataframe_to_records(feature_row),
        "latest_row": latest_row.to_dict(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "artifacts": artifacts,
    }


def build_target_rebalance_plan(
    decision_payload: dict[str, Any],
    wallet_snapshot: dict[str, Any] | None,
    monitor_symbols: list[str],
    stable_asset: str,
    stable_reserve_min_pct: float,
) -> dict[str, Any]:
    symbols = [normalize_symbol(item) for item in (monitor_symbols or TARGET_MONITOR_SYMBOLS)]
    if not symbols:
        symbols = TARGET_MONITOR_SYMBOLS

    stable_asset = (stable_asset or PROFIT_PARKING_STABLE_ASSET).upper()
    stable_set = stable_assets_set(stable_asset)
    stable_reserve_min_pct = clamp_value(float(stable_reserve_min_pct), 0.0, 0.95)

    wallet_rows = list((wallet_snapshot or {}).get("balances", []) or [])
    wallet_by_asset: dict[str, dict[str, Any]] = {}
    for row in wallet_rows:
        asset = str(row.get("asset", "")).upper().strip()
        if not asset:
            continue
        wallet_by_asset[asset] = row

    decision = decision_payload.get("decision", {})
    base_signal = str(decision.get("signal", "HOLD")).upper()
    decision_confidence = float(decision_payload.get("decision_confidence") or decision.get("decision_confidence") or 0.0)

    monitor_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        base_asset, quote_asset = split_symbol_assets(symbol)
        target = resolve_target_price(symbol)
        ticker_error = ""
        price = 0.0
        try:
            ticker, _ = get_ticker_with_metrics(symbol)
            price = float(ticker.get("last") or ticker.get("bid") or ticker.get("ask") or 0.0)
        except Exception as exc:
            ticker_error = str(exc)
            try:
                price = float(get_market_price(symbol))
            except Exception:
                price = 0.0

        wallet_row = wallet_by_asset.get(base_asset, {})
        holding_qty = float(wallet_row.get("total") or 0.0)
        holding_value_usdt = float(wallet_row.get("est_usdt") or (holding_qty * price))

        target_reached = bool(target and price > 0 and price >= target)
        target_surpassed = bool(target and price > 0 and price >= (target * (1.0 + TARGET_SURPASS_PCT)))

        action = "HOLD"
        reason = "Neutral conditions; hold."
        if target_reached:
            action = "SELL"
            reason = "Target reached or exceeded; lock in gains."
        elif base_signal == "BUY":
            action = "BUY"
            reason = "Model conditions favor BUY and target not reached yet."
        elif base_signal == "SELL":
            action = "SELL"
            reason = "Model conditions favor SELL for risk control."

        park_profit_to_stable = target_surpassed and holding_qty > 0 and quote_asset in stable_set
        parking_qty = 0.0
        parking_quote = 0.0
        if park_profit_to_stable and price > 0:
            exceed_ratio = max(0.0, (price - float(target or price)) / float(target or price))
            parking_qty = holding_qty * clamp_value(0.25 + exceed_ratio, 0.20, 1.00)
            parking_quote = parking_qty * price

        monitor_rows.append(
            {
                "symbol": symbol,
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "current_price": price,
                "target_price": float(target) if target else None,
                "target_reached": target_reached,
                "target_surpassed": target_surpassed,
                "action": action,
                "reason": reason,
                "holding_qty": holding_qty,
                "holding_value_usdt": holding_value_usdt,
                "park_profit_to_stable": park_profit_to_stable,
                "recommended_parking_qty": parking_qty,
                "recommended_parking_quote": parking_quote,
                "ticker_error": ticker_error,
            }
        )

    portfolio_value = float((wallet_snapshot or {}).get("estimated_total_usdt") or 0.0)
    stable_value = 0.0
    for row in wallet_rows:
        asset = str(row.get("asset", "")).upper().strip()
        if asset in stable_set:
            stable_value += float(row.get("est_usdt") or 0.0)

    target_stable_value = portfolio_value * stable_reserve_min_pct
    stable_shortfall = max(0.0, target_stable_value - stable_value)
    recommended_orders: list[dict[str, Any]] = []

    sell_candidates = [
        row
        for row in monitor_rows
        if row.get("action") == "SELL" and float(row.get("holding_qty") or 0.0) > 0 and float(row.get("current_price") or 0.0) > 0
    ]
    sell_candidates.sort(key=lambda row: (bool(row.get("target_surpassed")), float(row.get("holding_value_usdt") or 0.0)), reverse=True)

    remaining_shortfall = stable_shortfall
    for row in sell_candidates:
        if remaining_shortfall <= 0:
            break
        symbol = str(row.get("symbol"))
        current_price = float(row.get("current_price") or 0.0)
        holding_qty = float(row.get("holding_qty") or 0.0)
        holding_value = float(row.get("holding_value_usdt") or (holding_qty * current_price))
        if current_price <= 0 or holding_qty <= 0 or holding_value <= 0:
            continue
        sell_quote = min(holding_value, remaining_shortfall)
        sell_qty = min(holding_qty, sell_quote / current_price)
        if sell_qty <= 0:
            continue
        recommended_orders.append(
            {
                "action": "market_sell",
                "symbol": symbol,
                "quantity": sell_qty,
                "estimated_quote": sell_qty * current_price,
                "reason": f"Increase {stable_asset} reserve toward {stable_reserve_min_pct:.0%} portfolio allocation.",
            }
        )
        remaining_shortfall -= sell_qty * current_price

    stable_ratio = (stable_value / portfolio_value) if portfolio_value > 0 else 0.0
    strategy_checks = {
        "monitor_three_assets": len(symbols) >= 3,
        "buy_sell_hold_logic": True,
        "sell_on_target_reached": True,
        "hold_on_neutral": True,
        "park_profit_to_stable": True,
        "dynamic_rebalance": True,
    }

    return {
        "status": "ok",
        "strategy_signal": base_signal,
        "decision_confidence": decision_confidence,
        "monitor_symbols": symbols,
        "assets": monitor_rows,
        "stable_asset": stable_asset,
        "stable_ratio": stable_ratio,
        "stable_value": stable_value,
        "target_stable_ratio": stable_reserve_min_pct,
        "target_stable_value": target_stable_value,
        "stable_shortfall": stable_shortfall,
        "recommended_rebalance_orders": recommended_orders,
        "strategy_checks": strategy_checks,
    }


def build_quote_parking_orders(
    wallet_snapshot: dict[str, Any] | None,
    stable_asset: str,
    max_orders: int = 3,
    min_usdt_value: float = 10.0,
) -> list[dict[str, Any]]:
    stable_asset = (stable_asset or PROFIT_PARKING_STABLE_ASSET).upper()
    stable_set = stable_assets_set(stable_asset)
    wallet_rows = list((wallet_snapshot or {}).get("balances", []) or [])
    if not wallet_rows:
        return []

    candidates: list[dict[str, Any]] = []
    for row in wallet_rows:
        asset = str(row.get("asset", "")).upper().strip()
        if not asset or asset == stable_asset or asset in stable_set:
            continue
        total_qty = float(row.get("free") or row.get("total") or 0.0)
        est_usdt = float(row.get("est_usdt") or 0.0)
        if total_qty <= 0 or est_usdt < float(min_usdt_value):
            continue
        candidates.append(
            {
                "asset": asset,
                "total_qty": total_qty,
                "est_usdt": est_usdt,
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda item: float(item.get("est_usdt", 0.0)), reverse=True)
    exchange = get_binance_client()
    try:
        exchange.load_markets()
    except Exception:
        return []

    parking_orders: list[dict[str, Any]] = []
    for item in candidates:
        if len(parking_orders) >= max(0, int(max_orders)):
            break
        source_asset = str(item["asset"])
        total_qty = float(item["total_qty"])
        forward_symbol = f"{stable_asset}/{source_asset}"
        reverse_symbol = f"{source_asset}/{stable_asset}"

        if forward_symbol in exchange.markets:
            parking_orders.append(
                {
                    "action": "market_buy",
                    "symbol": forward_symbol,
                    "quantity": 0.0,
                    "quote_amount": total_qty,
                    "source_asset": source_asset,
                    "target_asset": stable_asset,
                    "estimated_quote": float(item["est_usdt"]),
                    "reason": f"Park {source_asset} balance into preferred stable asset {stable_asset}.",
                }
            )
            continue

        if reverse_symbol in exchange.markets:
            parking_orders.append(
                {
                    "action": "market_sell",
                    "symbol": reverse_symbol,
                    "quantity": total_qty,
                    "quote_amount": 0.0,
                    "source_asset": source_asset,
                    "target_asset": stable_asset,
                    "estimated_quote": float(item["est_usdt"]),
                    "reason": f"Park {source_asset} balance into preferred stable asset {stable_asset}.",
                }
            )

    return parking_orders


def execute_rebalance_orders(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    decision_payload = get_decision_payload(
        buy_threshold=float(cfg.get("buy_threshold", BUY_THRESHOLD)),
        sell_threshold=float(cfg.get("sell_threshold", SELL_THRESHOLD)),
        adaptive_threshold_enabled=bool(cfg.get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
    )

    decision_confidence = float(decision_payload.get("decision_confidence") or 0.0)
    decision_min_confidence = float(cfg.get("decision_min_confidence", DECISION_MIN_CONFIDENCE))
    if decision_min_confidence > 0 and decision_confidence < decision_min_confidence:
        return {
            "status": "blocked",
            "reason": f"Decision confidence {decision_confidence:.2f} is below minimum {decision_min_confidence:.2f}.",
            "decision_confidence": decision_confidence,
            "decision_min_confidence": decision_min_confidence,
            "orders": [],
            "results": [],
        }

    configured_symbols = cfg.get("target_monitor_symbols", TARGET_MONITOR_SYMBOLS)
    if isinstance(configured_symbols, str):
        monitor_symbols = parse_symbol_list(configured_symbols)
    elif isinstance(configured_symbols, list):
        monitor_symbols = [normalize_symbol(str(item)) for item in configured_symbols if str(item).strip()]
    else:
        monitor_symbols = TARGET_MONITOR_SYMBOLS
    monitor_symbols = monitor_symbols[: max(1, TARGET_MONITOR_MAX_ASSETS)]

    stable_asset = str(cfg.get("stable_asset", PROFIT_PARKING_STABLE_ASSET)).upper()
    stable_reserve_min_pct = float(cfg.get("stable_reserve_min_pct", STABLE_RESERVE_MIN_PCT))
    rebalance_min_delta_usdt = max(0.0, float(cfg.get("rebalance_min_delta_usdt", 25.0)))
    allow_live = bool(cfg.get("allow_live", False))
    dry_run = not allow_live
    max_orders = max(0, int(cfg.get("max_orders", 3)))
    enable_quote_parking = bool(cfg.get("enable_quote_parking", True))
    parking_max_orders = max(0, int(cfg.get("parking_max_orders", 2)))
    parking_min_usdt_value = max(0.0, float(cfg.get("parking_min_usdt_value", 10.0)))
    skip_symbols = {normalize_symbol(str(item)) for item in (cfg.get("skip_symbols") or []) if str(item).strip()}
    skip_assets = {str(item).upper().strip() for item in (cfg.get("skip_assets") or []) if str(item).strip()}

    wallet_snapshot = get_wallet_snapshot()
    strategy = build_target_rebalance_plan(
        decision_payload=decision_payload,
        wallet_snapshot=wallet_snapshot,
        monitor_symbols=monitor_symbols,
        stable_asset=stable_asset,
        stable_reserve_min_pct=stable_reserve_min_pct,
    )

    orders = []
    for order in list(strategy.get("recommended_rebalance_orders", [])):
        order_symbol = normalize_symbol(str(order.get("symbol", "")))
        if order_symbol in skip_symbols:
            continue
        base_asset, _ = split_symbol_assets(order_symbol)
        if base_asset in skip_assets:
            continue
        if float(order.get("estimated_quote") or 0.0) < rebalance_min_delta_usdt:
            continue
        orders.append(order)
    orders = orders[:max_orders]
    parking_plan_orders = build_quote_parking_orders(
        wallet_snapshot=wallet_snapshot,
        stable_asset=stable_asset,
        max_orders=parking_max_orders,
        min_usdt_value=parking_min_usdt_value,
    ) if enable_quote_parking else []
    parking_plan_orders = [
        order
        for order in parking_plan_orders
        if normalize_symbol(str(order.get("symbol", ""))) not in skip_symbols
        and str(order.get("source_asset", "")).upper().strip() not in skip_assets
        and str(order.get("target_asset", "")).upper().strip() not in skip_assets
    ]

    if not orders and not parking_plan_orders:
        return {
            "status": "no_orders",
            "reason": "No rebalance or quote-parking orders are currently recommended.",
            "decision_confidence": decision_confidence,
            "decision_min_confidence": decision_min_confidence,
            "orders": [],
            "parking_orders": [],
            "results": [],
            "strategy": strategy,
            "dry_run": dry_run,
            "enable_quote_parking": enable_quote_parking,
            "rebalance_min_delta_usdt": rebalance_min_delta_usdt,
            "skip_symbols": sorted(skip_symbols),
            "skip_assets": sorted(skip_assets),
        }

    results: list[dict[str, Any]] = []
    for order in orders:
        symbol = normalize_symbol(str(order.get("symbol", "")))
        quantity = float(order.get("quantity") or 0.0)
        if not symbol or quantity <= 0:
            results.append(
                {
                    "status": "invalid_order",
                    "order": order,
                    "message": "Invalid symbol or quantity for rebalance order.",
                }
            )
            continue

        trade_result = execute_account_action(
            action="market_sell",
            symbol=symbol,
            quantity=quantity,
            quote_amount=0.0,
            dry_run=dry_run,
            max_api_latency_ms=int(cfg.get("max_api_latency_ms", 1200)),
            warning_latency_ms=int(cfg.get("warning_latency_ms", cfg.get("max_api_latency_ms", 1200) * 0.55)),
            degraded_latency_ms=int(cfg.get("degraded_latency_ms", cfg.get("max_api_latency_ms", 1200) * 0.8)),
            block_latency_ms=int(cfg.get("block_latency_ms", cfg.get("max_api_latency_ms", 1200))),
            consecutive_breach_limit=int(cfg.get("consecutive_breach_limit", 3)),
            max_ticker_age_ms=int(cfg.get("max_ticker_age_ms", 3000)),
            max_spread_bps=float(cfg.get("max_spread_bps", 20.0)),
            min_trade_cooldown_seconds=int(cfg.get("min_trade_cooldown_seconds", 0)),
        )
        trade_result["rebalance_reason"] = order.get("reason", "")
        trade_result["estimated_quote"] = order.get("estimated_quote", 0.0)
        trade_result["order_type"] = "rebalance"
        results.append(trade_result)

    parking_orders = parking_plan_orders
    if enable_quote_parking and parking_orders:
        if allow_live:
            try:
                wallet_after_rebalance = get_wallet_snapshot()
                parking_orders = build_quote_parking_orders(
                    wallet_snapshot=wallet_after_rebalance,
                    stable_asset=stable_asset,
                    max_orders=parking_max_orders,
                    min_usdt_value=parking_min_usdt_value,
                )
            except Exception:
                parking_orders = parking_plan_orders

        for order in parking_orders:
            symbol = normalize_symbol(str(order.get("symbol", "")))
            action = str(order.get("action", "")).strip().lower()
            quantity = float(order.get("quantity") or 0.0)
            quote_amount = float(order.get("quote_amount") or 0.0)
            if not symbol or action not in {"market_buy", "market_sell"}:
                results.append(
                    {
                        "status": "invalid_order",
                        "order": order,
                        "message": "Invalid quote-parking order payload.",
                        "order_type": "quote_parking",
                    }
                )
                continue

            trade_result = execute_account_action(
                action=action,
                symbol=symbol,
                quantity=quantity,
                quote_amount=quote_amount,
                dry_run=dry_run,
                max_api_latency_ms=int(cfg.get("max_api_latency_ms", 1200)),
                warning_latency_ms=int(cfg.get("warning_latency_ms", cfg.get("max_api_latency_ms", 1200) * 0.55)),
                degraded_latency_ms=int(cfg.get("degraded_latency_ms", cfg.get("max_api_latency_ms", 1200) * 0.8)),
                block_latency_ms=int(cfg.get("block_latency_ms", cfg.get("max_api_latency_ms", 1200))),
                consecutive_breach_limit=int(cfg.get("consecutive_breach_limit", 3)),
                max_ticker_age_ms=int(cfg.get("max_ticker_age_ms", 3000)),
                max_spread_bps=float(cfg.get("max_spread_bps", 20.0)),
                min_trade_cooldown_seconds=int(cfg.get("min_trade_cooldown_seconds", 0)),
            )
            trade_result["rebalance_reason"] = order.get("reason", "")
            trade_result["estimated_quote"] = order.get("estimated_quote", 0.0)
            trade_result["order_type"] = "quote_parking"
            trade_result["source_asset"] = order.get("source_asset", "")
            trade_result["target_asset"] = order.get("target_asset", "")
            results.append(trade_result)

    status_counts: dict[str, int] = {}
    for item in results:
        key = str(item.get("status", "unknown"))
        status_counts[key] = status_counts.get(key, 0) + 1

    return {
        "status": "ok",
        "decision_confidence": decision_confidence,
        "decision_min_confidence": decision_min_confidence,
        "orders": orders,
        "parking_orders": parking_orders,
        "results": results,
        "result_counts": status_counts,
        "strategy": strategy,
        "dry_run": dry_run,
        "enable_quote_parking": enable_quote_parking,
        "parking_max_orders": parking_max_orders,
        "parking_min_usdt_value": parking_min_usdt_value,
        "rebalance_min_delta_usdt": rebalance_min_delta_usdt,
        "skip_symbols": sorted(skip_symbols),
        "skip_assets": sorted(skip_assets),
    }


def get_dashboard_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_llm_startup_logged()
    configured_symbols = (config or {}).get("target_monitor_symbols", TARGET_MONITOR_SYMBOLS)
    if isinstance(configured_symbols, str):
        configured_symbols = parse_symbol_list(configured_symbols)
    elif isinstance(configured_symbols, list):
        configured_symbols = [normalize_symbol(str(item)) for item in configured_symbols if str(item).strip()]
    else:
        configured_symbols = TARGET_MONITOR_SYMBOLS

    cfg = {
        "buy_threshold": float((config or {}).get("buy_threshold", BUY_THRESHOLD)),
        "sell_threshold": float((config or {}).get("sell_threshold", SELL_THRESHOLD)),
        "adaptive_threshold_enabled": bool((config or {}).get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
        "deposit_amount": float((config or {}).get("deposit_amount", ACCOUNT_REFERENCE_USD)),
        "active_capital_pct": float((config or {}).get("active_capital_pct", 0.50 if ACCOUNT_REFERENCE_USD <= 10 else 0.70)),
        "reserve_pct": float((config or {}).get("reserve_pct", 0.30)),
        "max_trade_pct": float((config or {}).get("max_trade_pct", 0.20 if ACCOUNT_REFERENCE_USD <= 10 else 0.10)),
        "stop_loss_pct": float((config or {}).get("stop_loss_pct", 0.03)),
        "take_profit_pct": float((config or {}).get("take_profit_pct", 0.05)),
        "max_daily_loss_pct": float((config or {}).get("max_daily_loss_pct", 0.05)),
        "max_drawdown_pct": float((config or {}).get("max_drawdown_pct", 0.15)),
        "withdrawal_target_pct": float((config or {}).get("withdrawal_target_pct", 0.25)),
        "live_symbol": normalize_symbol(str((config or {}).get("live_symbol", CHECK_SYMBOL))),
        "market_scan_enabled": bool((config or {}).get("market_scan_enabled", True)),
        "market_scan_max_symbols": int((config or {}).get("market_scan_max_symbols", 60)),
        "market_scan_quote_asset": str((config or {}).get("market_scan_quote_asset", "USDT")),
        "size_min_confidence": float((config or {}).get("size_min_confidence", SIZE_MIN_CONFIDENCE)),
        "decision_min_confidence": float((config or {}).get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
        "goal_value": float((config or {}).get("goal_value", 0.0) or 0.0),
        "target_monitor_symbols": configured_symbols[: max(1, TARGET_MONITOR_MAX_ASSETS)],
        "stable_asset": str((config or {}).get("stable_asset", PROFIT_PARKING_STABLE_ASSET)).upper(),
        "stable_reserve_min_pct": float((config or {}).get("stable_reserve_min_pct", STABLE_RESERVE_MIN_PCT)),
    }
    decision_payload = get_decision_payload(
        buy_threshold=cfg["buy_threshold"],
        sell_threshold=cfg["sell_threshold"],
        adaptive_threshold_enabled=cfg["adaptive_threshold_enabled"],
    )
    plan = risk_plan(
        deposit_amount=cfg["deposit_amount"],
        active_capital_pct=cfg["active_capital_pct"],
        reserve_pct=cfg["reserve_pct"],
        max_trade_pct=cfg["max_trade_pct"],
        stop_loss_pct=cfg["stop_loss_pct"],
        take_profit_pct=cfg["take_profit_pct"],
        max_daily_loss_pct=cfg["max_daily_loss_pct"],
        max_drawdown_pct=cfg["max_drawdown_pct"],
        withdrawal_target_pct=cfg["withdrawal_target_pct"],
    )
    wallet_snapshot: dict[str, Any] | None = None
    wallet_error = ""
    try:
        wallet_snapshot = get_wallet_snapshot()
    except Exception as exc:
        wallet_error = wallet_permission_hint(exc)
    account_snapshot: dict[str, Any] | None = None
    account_error = ""
    try:
        account_snapshot = get_account_snapshot(cfg["live_symbol"])
    except Exception as exc:
        account_error = wallet_permission_hint(exc)
    market_scan: list[dict[str, Any]] = []
    market_scan_error = ""
    if cfg["market_scan_enabled"]:
        try:
            market_scan = build_market_scan(cfg["market_scan_max_symbols"], cfg["market_scan_quote_asset"])
        except Exception as exc:
            market_scan_error = wallet_permission_hint(exc)
    decision = decision_payload["decision"]
    market_price = 0.0
    try:
        market_price = get_market_price(cfg["live_symbol"])
    except Exception:
        fallback = get_runtime_value("latest_price_fallback")
        market_price = float(fallback or 0.0)
    size_plan = auto_order_size_from_confidence(
        signal=str(decision.get("signal", "HOLD")),
        probability_up=float(decision.get("probability_up", 0.5)),
        market_price=float(market_price),
        max_trade_size_quote=float(plan["max_trade_size"]),
        min_confidence=float(cfg["size_min_confidence"]),
    )
    size_plan = apply_decision_confidence_gate(
        size_plan=size_plan,
        decision_payload=decision_payload,
        min_confidence=float(cfg["decision_min_confidence"]),
    )

    target_strategy = build_target_rebalance_plan(
        decision_payload=decision_payload,
        wallet_snapshot=wallet_snapshot,
        monitor_symbols=[normalize_symbol(symbol) for symbol in cfg["target_monitor_symbols"]],
        stable_asset=cfg["stable_asset"],
        stable_reserve_min_pct=float(cfg["stable_reserve_min_pct"]),
    )
    autopilot_preview: dict[str, Any] | None = None
    opportunity_panel: dict[str, Any] = {"ranked": [], "winner": {}, "meta": {}}
    if account_snapshot is not None:
        preview_signal = resolve_autopilot_signal(
            raw_signal=str(decision.get("signal", "HOLD")),
            probability_up=float(decision.get("probability_up", 0.5)),
            decision_confidence=float(decision_payload.get("decision_confidence") or decision.get("decision_confidence") or 0.0),
            decision_min_confidence=float(cfg["decision_min_confidence"]),
            target_price=resolve_target_price(cfg["live_symbol"]),
            current_price=float(account_snapshot.get("best_bid") or account_snapshot.get("best_ask") or 0.0),
        )
        autopilot_preview = build_autopilot_trade_plan(
            symbol=cfg["live_symbol"],
            signal=str(preview_signal.get("final_signal", "HOLD")),
            probability_up=float(decision.get("probability_up", 0.5)),
            decision_payload=decision_payload,
            balance_snapshot=account_snapshot,
            wallet_snapshot=wallet_snapshot,
            config=cfg,
            forced_exit=bool(preview_signal.get("forced_exit", False)),
        )
        autopilot_preview = {
            "raw_signal": preview_signal.get("raw_signal", ""),
            "override_reason": preview_signal.get("override_reason", ""),
            **autopilot_preview,
        }
    if wallet_snapshot is not None:
        try:
            opportunity_panel = rank_autopilot_opportunities(
                config={
                    **cfg,
                    "multi_symbol_enabled": True,
                    "opportunity_max_candidates": int((config or {}).get("opportunity_max_candidates", AUTOPILOT_MAX_CANDIDATES)),
                    "opportunity_quote_assets": (config or {}).get("opportunity_quote_assets", AUTOPILOT_UNIVERSE_QUOTES),
                    "symbol_denylist": (config or {}).get("symbol_denylist", list(AUTOPILOT_SYMBOL_DENYLIST)),
                    "switch_score_delta": float((config or {}).get("switch_score_delta", AUTOPILOT_SWITCH_SCORE_DELTA)),
                },
                wallet_snapshot=wallet_snapshot,
                base_decision_payload=decision_payload,
                preferred_symbol=cfg["live_symbol"],
                previous_symbol=cfg["live_symbol"],
            )
            opportunity_panel = {
                "ranked": [
                    {
                        "symbol": row.get("symbol"),
                        "raw_signal": row.get("raw_signal"),
                        "resolved_signal": row.get("resolved_signal"),
                        "final_action": row.get("final_action"),
                        "score": row.get("score"),
                        "confidence": row.get("confidence"),
                        "expected_return": row.get("expected_return"),
                        "wallet_fundable": row.get("wallet_fundable"),
                        "direct_trade_possible": row.get("direct_trade_possible"),
                        "funding_path": row.get("funding_path"),
                        "skip_reason": row.get("skip_reason"),
                        "rejection_reason": row.get("rejection_reason"),
                    }
                    for row in list(opportunity_panel.get("ranked") or [])[:12]
                ],
                "winner": dict(opportunity_panel.get("winner") or {}),
                "meta": dict(opportunity_panel.get("meta") or {}),
            }
        except Exception as exc:
            opportunity_panel = {
                "ranked": [],
                "winner": {},
                "meta": {"error": str(exc)},
            }
    dashboard_goal_value = float(cfg.get("goal_value") or 0.0)
    if dashboard_goal_value <= 0:
        dashboard_goal_value = float((wallet_snapshot or {}).get("estimated_total_usdt", plan["deposit_amount"])) * 1.25
    cycle_plan = estimate_cycles_to_goal(
        current_value=float((wallet_snapshot or {}).get("estimated_total_usdt", plan["deposit_amount"])),
        goal_value=dashboard_goal_value,
        signal=str(decision.get("signal", "HOLD")),
        probability_up=float(decision.get("probability_up", 0.5)),
        order_size=float(size_plan.get("base_size", 0.0)),
        market_price=float(market_price or 0.0),
        take_profit_pct=float(plan["take_profit_pct"]),
        stop_loss_pct=float(plan["stop_loss_pct"]),
        fee_drag_pct=0.003,
    )
    return {
        "status": "ok",
        "config": cfg,
        "llm_status": get_llm_status(),
        "notification_status": get_notification_status(),
        "decision_payload": decision_payload,
        "risk_plan": plan,
        "wallet_snapshot": wallet_snapshot,
        "wallet_error": wallet_error,
        "account_snapshot": account_snapshot,
        "account_error": account_error,
        "market_scan": market_scan,
        "market_scan_error": market_scan_error,
        "target_strategy": target_strategy,
        "autopilot_preview": autopilot_preview,
        "opportunity_panel": opportunity_panel,
        "goal_value": dashboard_goal_value,
        "size_plan": size_plan,
        "cycle_plan": cycle_plan,
        "autopilot": autopilot_snapshot(),
        "autopilot_recovery": {
            "persistence_healthy": bool(autopilot_snapshot().get("persistence_healthy", False)),
            "reconciliation_state": str(autopilot_snapshot().get("reconciliation_state") or "clean"),
            "requires_human_review": bool(autopilot_snapshot().get("requires_human_review", False)),
            "burn_in_report": build_burn_in_validation_report(),
        },
        "live_history": get_live_history(),
        "latest_price": market_price,
    }


def autopilot_snapshot() -> dict[str, Any]:
    with AUTOPILOT_LOCK:
        return json.loads(json.dumps(AUTOPILOT_STATE))


def update_autopilot_state(**updates: Any) -> None:
    with AUTOPILOT_LOCK:
        AUTOPILOT_STATE.update(updates)
        AUTOPILOT_STATE["updated_at"] = utc_now_iso()
    persist_autopilot_state()


def append_autopilot_log(entry: dict[str, Any]) -> None:
    with AUTOPILOT_LOCK:
        logs = list(AUTOPILOT_STATE.get("logs", []))
        logs.append(entry)
        AUTOPILOT_STATE["logs"] = logs[-200:]
        AUTOPILOT_STATE["updated_at"] = utc_now_iso()
    persist_autopilot_state()


def should_stop_autopilot() -> bool:
    return AUTOPILOT_STOP_EVENT.is_set()


def _autopilot_thread_alive_unlocked() -> bool:
    return AUTOPILOT_THREAD is not None and AUTOPILOT_THREAD.is_alive()


def wait_for_autopilot_interval(seconds: int) -> bool:
    deadline = time.time() + max(0, int(seconds))
    while time.time() < deadline:
        if AUTOPILOT_STOP_EVENT.wait(timeout=0.1):
            return True
    return AUTOPILOT_STOP_EVENT.is_set()


def resolve_autopilot_signal(
    *,
    raw_signal: str,
    probability_up: float,
    decision_confidence: float,
    decision_min_confidence: float,
    target_price: float | None,
    current_price: float,
) -> dict[str, Any]:
    normalized_signal = str(raw_signal or "HOLD").upper()
    if normalized_signal not in {"BUY", "SELL", "HOLD"}:
        normalized_signal = "HOLD"
    target_reached = bool(target_price and current_price > 0 and current_price >= float(target_price))
    if target_reached:
        return {
            "raw_signal": normalized_signal,
            "final_signal": "SELL",
            "override_reason": "target_reached",
            "forced_exit": True,
            "probability_up": probability_up,
            "decision_confidence": decision_confidence,
            "decision_min_confidence": decision_min_confidence,
            "target_price": target_price,
            "current_price": current_price,
        }
    if decision_min_confidence > 0 and decision_confidence < decision_min_confidence:
        return {
            "raw_signal": normalized_signal,
            "final_signal": "HOLD",
            "override_reason": "low_confidence",
            "forced_exit": False,
            "probability_up": probability_up,
            "decision_confidence": decision_confidence,
            "decision_min_confidence": decision_min_confidence,
            "target_price": target_price,
            "current_price": current_price,
        }
    return {
        "raw_signal": normalized_signal,
        "final_signal": normalized_signal,
        "override_reason": "",
        "forced_exit": False,
        "probability_up": probability_up,
        "decision_confidence": decision_confidence,
        "decision_min_confidence": decision_min_confidence,
        "target_price": target_price,
        "current_price": current_price,
    }


def maybe_execute_signal_trade(
    signal: str,
    symbol: str,
    order_size: float,
    allow_live: bool,
    max_api_latency_ms: int,
    warning_latency_ms: int,
    degraded_latency_ms: int,
    block_latency_ms: int,
    consecutive_breach_limit: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if signal not in {"BUY", "SELL"}:
        return {"status": "no_trade", "message": "Signal is HOLD, no trade action sent.", "signal": signal}
    action = "market_buy" if signal == "BUY" else "market_sell"
    return execute_account_action(
        action=action,
        symbol=symbol,
        quantity=order_size,
        quote_amount=0.0,
        dry_run=not allow_live,
        max_api_latency_ms=max_api_latency_ms,
        warning_latency_ms=warning_latency_ms,
        degraded_latency_ms=degraded_latency_ms,
        block_latency_ms=block_latency_ms,
        consecutive_breach_limit=consecutive_breach_limit,
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
        execution_context=execution_context,
    )


def resolve_autopilot_cycle_candidate(
    *,
    config: dict[str, Any],
    decision_payload: dict[str, Any],
    wallet_snapshot: dict[str, Any],
    preferred_symbol: str,
    previous_symbol: str,
) -> dict[str, Any]:
    preferred_symbol = normalize_symbol(preferred_symbol or CHECK_SYMBOL)
    if bool(config.get("multi_symbol_enabled", AUTOPILOT_MULTI_SYMBOL_ENABLED)):
        opportunity_bundle = rank_autopilot_opportunities(
            config=config,
            wallet_snapshot=wallet_snapshot,
            base_decision_payload=decision_payload,
            preferred_symbol=preferred_symbol,
            previous_symbol=previous_symbol,
        )
        winner = dict(opportunity_bundle.get("winner") or {})
        ranked = list(opportunity_bundle.get("ranked") or [])
        if not winner or not str(winner.get("symbol") or "").strip():
            selected_symbol = preferred_symbol
            account_snapshot = get_account_snapshot(selected_symbol)
            trade_plan = build_no_action_trade_plan(selected_symbol, account_snapshot, wallet_snapshot, "no_actionable_candidate")
            return {
                "symbol": selected_symbol,
                "account_snapshot": account_snapshot,
                "signal_resolution": {
                    "raw_signal": "HOLD",
                    "final_signal": "HOLD",
                    "override_reason": "no_actionable_candidate",
                    "forced_exit": False,
                    "probability_up": 0.5,
                    "decision_confidence": 0.0,
                    "decision_min_confidence": float(config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
                    "target_price": resolve_target_price(selected_symbol),
                    "current_price": float(account_snapshot.get("best_bid") or account_snapshot.get("best_ask") or 0.0),
                },
                "decision_payload": decision_payload,
                "trade_plan": trade_plan,
                "opportunity_rankings": [
                    {
                        "symbol": row.get("symbol"),
                        "raw_signal": row.get("raw_signal"),
                        "resolved_signal": row.get("resolved_signal"),
                        "final_action": row.get("final_action"),
                        "score": row.get("score"),
                        "confidence": row.get("confidence"),
                        "expected_return": row.get("expected_return"),
                        "wallet_fundable": row.get("wallet_fundable"),
                        "actionable": row.get("actionable"),
                        "direct_trade_possible": row.get("direct_trade_possible"),
                        "funding_path": row.get("funding_path"),
                        "skip_reason": row.get("skip_reason"),
                        "rejection_reason": row.get("rejection_reason"),
                        "guard_mode": row.get("guard_mode"),
                    }
                    for row in ranked[:12]
                ],
                "opportunity_winner": dict(winner),
                "opportunity_meta": dict(opportunity_bundle.get("meta") or {}),
            }
        selected_symbol = normalize_symbol(str(winner.get("symbol") or preferred_symbol))
        account_snapshot = dict(winner.get("account_snapshot") or {})
        if not account_snapshot:
            account_snapshot = get_account_snapshot(selected_symbol)
        signal_resolution = dict(winner.get("signal_resolution") or {})
        if not signal_resolution:
            raw_signal = str((decision_payload.get("decision") or {}).get("signal", "HOLD")).upper()
            probability_up = float((decision_payload.get("decision") or {}).get("probability_up", 0.5))
            decision_confidence = float(decision_payload.get("decision_confidence") or (decision_payload.get("decision") or {}).get("decision_confidence") or 0.0)
            signal_resolution = resolve_autopilot_signal(
                raw_signal=raw_signal,
                probability_up=probability_up,
                decision_confidence=decision_confidence,
                decision_min_confidence=float(config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
                target_price=resolve_target_price(selected_symbol),
                current_price=float(account_snapshot.get("best_bid") or account_snapshot.get("best_ask") or 0.0),
            )
        candidate_decision_payload = dict(winner.get("decision_payload") or decision_payload)
        trade_plan = dict(winner.get("trade_plan") or {})
        if not trade_plan:
            trade_plan = build_autopilot_trade_plan(
                symbol=selected_symbol,
                signal=str(signal_resolution.get("final_signal") or "HOLD"),
                probability_up=float(signal_resolution.get("probability_up") or 0.5),
                decision_payload=candidate_decision_payload,
                balance_snapshot=account_snapshot,
                wallet_snapshot=wallet_snapshot,
                config=config,
                forced_exit=bool(signal_resolution.get("forced_exit", False)),
            )
        return {
            "symbol": selected_symbol,
            "account_snapshot": account_snapshot,
            "signal_resolution": signal_resolution,
            "decision_payload": candidate_decision_payload,
            "trade_plan": trade_plan,
            "opportunity_rankings": [
                {
                    "symbol": row.get("symbol"),
                    "raw_signal": row.get("raw_signal"),
                    "resolved_signal": row.get("resolved_signal"),
                    "final_action": row.get("final_action"),
                    "score": row.get("score"),
                    "confidence": row.get("confidence"),
                    "expected_return": row.get("expected_return"),
                    "wallet_fundable": row.get("wallet_fundable"),
                    "actionable": row.get("actionable"),
                    "direct_trade_possible": row.get("direct_trade_possible"),
                    "funding_path": row.get("funding_path"),
                    "skip_reason": row.get("skip_reason"),
                    "rejection_reason": row.get("rejection_reason"),
                    "guard_mode": row.get("guard_mode"),
                }
                for row in ranked[:12]
            ],
            "opportunity_winner": {
                "symbol": winner.get("symbol", selected_symbol),
                "selection_reason": winner.get("selection_reason", ""),
                "score": winner.get("score"),
                "final_action": winner.get("final_action"),
                "confidence": winner.get("confidence"),
                "expected_return": winner.get("expected_return"),
                "wallet_fundable": winner.get("wallet_fundable"),
                "funding_path": winner.get("funding_path"),
                "skip_reason": winner.get("skip_reason"),
            },
            "opportunity_meta": dict(opportunity_bundle.get("meta") or {}),
        }

    selected_symbol = preferred_symbol
    account_snapshot = get_account_snapshot(selected_symbol)
    raw_signal = str((decision_payload.get("decision") or {}).get("signal", "HOLD")).upper()
    probability_up = float((decision_payload.get("decision") or {}).get("probability_up", 0.5))
    decision_confidence = float(decision_payload.get("decision_confidence") or (decision_payload.get("decision") or {}).get("decision_confidence") or 0.0)
    signal_resolution = resolve_autopilot_signal(
        raw_signal=raw_signal,
        probability_up=probability_up,
        decision_confidence=decision_confidence,
        decision_min_confidence=float(config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
        target_price=resolve_target_price(selected_symbol),
        current_price=float(account_snapshot.get("best_bid") or account_snapshot.get("best_ask") or 0.0),
    )
    trade_plan = build_autopilot_trade_plan(
        symbol=selected_symbol,
        signal=str(signal_resolution.get("final_signal") or raw_signal),
        probability_up=probability_up,
        decision_payload=decision_payload,
        balance_snapshot=account_snapshot,
        wallet_snapshot=wallet_snapshot,
        config=config,
        forced_exit=bool(signal_resolution.get("forced_exit", False)),
    )
    return {
        "symbol": selected_symbol,
        "account_snapshot": account_snapshot,
        "signal_resolution": signal_resolution,
        "decision_payload": decision_payload,
        "trade_plan": trade_plan,
        "opportunity_rankings": [],
        "opportunity_winner": {},
        "opportunity_meta": {"multi_symbol_enabled": False},
    }


def run_autopilot(config: dict[str, Any], run_id: int) -> None:
    try:
        config = normalize_autopilot_config(config)
        preferred_symbol = normalize_symbol(str(config.get("symbol", CHECK_SYMBOL)))
        previous_symbol = preferred_symbol
        continue_until_goal = bool(config.get("continue_until_goal", True))
        max_extra_cycles = max(0, int(config.get("max_extra_cycles", 24)))
        max_failed_cycles_in_row = max(1, int(config.get("max_failed_cycles_in_row", 8)))
        max_runtime_minutes = max(1.0, float(config.get("max_runtime_minutes", 240.0)))
        max_drawdown_pct = clamp_value(float(config.get("max_drawdown_pct", 0.15)), 0.0, 0.95)
        stable_asset = str(config.get("stable_asset", PROFIT_PARKING_STABLE_ASSET)).upper()
        unattended_mode = bool(config.get("unattended_mode", False))
        started_monotonic = time.time()
        start_gate = unattended_start_gate(config)
        if not bool(start_gate.get("ok", False)):
            raise RuntimeError(str(start_gate.get("reason") or "Unattended start gate failed."))
        set_reconciliation_state("clean", details={"startup_gate": start_gate}, requires_human_review=False)
        decision_payload = get_decision_payload(
            buy_threshold=float(config.get("buy_threshold", BUY_THRESHOLD)),
            sell_threshold=float(config.get("sell_threshold", SELL_THRESHOLD)),
            adaptive_threshold_enabled=bool(config.get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
        )
        initial_wallet_snapshot = get_wallet_snapshot()
        initial_selection = resolve_autopilot_cycle_candidate(
            config=config,
            decision_payload=decision_payload,
            wallet_snapshot=initial_wallet_snapshot,
            preferred_symbol=preferred_symbol,
            previous_symbol=previous_symbol,
        )
        symbol = normalize_symbol(str(initial_selection.get("symbol") or preferred_symbol))
        initial_snapshot = dict(initial_selection.get("account_snapshot") or {})
        initial_trade_plan = dict(initial_selection.get("trade_plan") or {})
        initial_signal_resolution = dict(initial_selection.get("signal_resolution") or {})
        initial_decision_payload = dict(initial_selection.get("decision_payload") or decision_payload)
        initial_decision = dict(initial_decision_payload.get("decision") or decision_payload.get("decision") or {})
        initial_opportunity_rankings = list(initial_selection.get("opportunity_rankings") or [])
        initial_opportunity_winner = dict(initial_selection.get("opportunity_winner") or {})
        initial_opportunity_meta = dict(initial_selection.get("opportunity_meta") or {})
        starting_value = max(
            0.0,
            float(initial_wallet_snapshot.get("estimated_total_usdt") or 0.0),
            float(initial_snapshot.get("account_value_quote", 0.0) or 0.0),
        )
        current_value = starting_value
        if not initial_trade_plan:
            initial_trade_plan = build_autopilot_trade_plan(
                symbol=symbol,
                signal=str(initial_signal_resolution.get("final_signal") or initial_decision.get("signal", "HOLD")).upper(),
                probability_up=float(initial_signal_resolution.get("probability_up") or initial_decision.get("probability_up", 0.5)),
                decision_payload=initial_decision_payload,
                balance_snapshot=initial_snapshot,
                wallet_snapshot=initial_wallet_snapshot,
                config=config,
                forced_exit=bool(initial_signal_resolution.get("forced_exit", False)),
            )
        market_price = float(initial_trade_plan.get("market_price") or get_market_price(symbol))
        goal_value = float(config.get("goal_value", 0.0) or 0.0)
        if goal_value <= 0:
            goal_value = max(current_value * 1.25, current_value)
        goal_progress = compute_goal_progress(starting_value, current_value, goal_value)
        cycle_plan = estimate_cycles_to_goal(
            current_value=current_value,
            goal_value=goal_value,
            signal=str(initial_decision.get("signal", "HOLD")),
            probability_up=float(initial_decision.get("probability_up", 0.5)),
            order_size=float((initial_trade_plan.get("size_plan") or {}).get("base_size", 0.0)),
            market_price=market_price,
            take_profit_pct=float(config.get("take_profit_pct", 0.05)),
            stop_loss_pct=float(config.get("stop_loss_pct", 0.03)),
            fee_drag_pct=float(config.get("fee_drag_pct", 0.0)),
        )
        runtime_profile = resolve_autopilot_runtime_profile(
            signal=str(initial_decision.get("signal", "HOLD")),
            decision_confidence=float(initial_decision_payload.get("decision_confidence") or initial_decision.get("decision_confidence") or 0.0),
            cycle_plan=cycle_plan,
        )
        interval_seconds = int(runtime_profile["interval_seconds"])
        planned_cycles = 0 if cycle_plan.get("status") == "goal_already_reached" else int(runtime_profile["target_cycles"])
        update_autopilot_state(
            running=True,
            status="running",
            run_id=run_id,
            stop_requested=False,
            symbol=symbol,
            interval_seconds=interval_seconds,
            current_cycle=0,
            target_cycles=planned_cycles,
            cycle_plan=cycle_plan,
            latest_decision=initial_decision,
            latest_trade_result={"signal_trade": initial_trade_plan.get("execution_plan", {})},
            execution_mode="normal",
            min_qty=float((initial_trade_plan.get("execution_plan") or {}).get("min_qty") or 0.0),
            step_size=float((initial_trade_plan.get("execution_plan") or {}).get("step_size") or 0.0),
            min_notional=float((initial_trade_plan.get("execution_plan") or {}).get("min_notional") or 0.0),
            minimum_valid_quote=float((initial_trade_plan.get("execution_plan") or {}).get("minimum_valid_quote") or 0.0),
            free_quote=float((initial_trade_plan.get("execution_plan") or {}).get("free_quote") or 0.0),
            free_base=float((initial_trade_plan.get("execution_plan") or {}).get("free_base") or 0.0),
            computed_order_size_quote=float((initial_trade_plan.get("execution_plan") or {}).get("computed_order_size_quote") or 0.0),
            computed_order_size_base=float((initial_trade_plan.get("execution_plan") or {}).get("computed_order_size_base") or 0.0),
            can_buy_minimum=bool((initial_trade_plan.get("execution_plan") or {}).get("can_buy_minimum", False)),
            required_base_asset=str(initial_trade_plan.get("required_base_asset") or ""),
            required_quote_asset=str(initial_trade_plan.get("required_quote_asset") or ""),
            direct_trade_possible=bool(initial_trade_plan.get("direct_trade_possible", False)),
            funding_path=str(initial_trade_plan.get("funding_path") or "none"),
            funding_skip_reason=str(initial_trade_plan.get("funding_skip_reason") or ""),
            free_bnb=float(initial_trade_plan.get("free_bnb") or 0.0),
            eligible_funding_assets=list(initial_trade_plan.get("eligible_funding_assets") or []),
            funding_diagnostics=dict(initial_trade_plan.get("funding_diagnostics") or {}),
            wallet_summary=initial_trade_plan.get("wallet_summary") or {},
            conversion_plan=initial_trade_plan.get("conversion_plan") or {},
            skip_reason=str((initial_trade_plan.get("execution_plan") or {}).get("skip_reason") or ""),
            opportunity_rankings=initial_opportunity_rankings,
            opportunity_winner=initial_opportunity_winner,
            opportunity_meta=initial_opportunity_meta,
            starting_value=float(goal_progress.get("starting_value") or 0.0),
            current_value=float(goal_progress.get("current_value") or 0.0),
            goal_value=float(goal_progress.get("goal_value") or 0.0),
            progress_pct=float(goal_progress.get("progress_pct") or 0.0),
            goal_reached=bool(goal_progress.get("goal_reached", False)),
            continue_until_goal=continue_until_goal,
            extra_cycles_used=0,
            failed_cycles_in_row=0,
            final_stable_target_asset=stable_asset,
            finalization_status="pending" if bool(goal_progress.get("goal_reached", False)) else "",
            finalization_result={},
            final_stop_reason="",
            latest_execution_intent={},
            unattended_mode=unattended_mode,
            burn_in_report=build_burn_in_validation_report(),
            last_error="",
        )
        if bool(goal_progress.get("goal_reached", False)):
            finalization_result = execute_goal_finalization(config, run_id=run_id, unattended_mode=unattended_mode)
            finalization_status = "completed" if str(finalization_result.get("status") or "") in {"ok", "dust_only"} else str(finalization_result.get("status") or "error")
            append_autopilot_log(
                {
                    "cycle": 0,
                    "timestamp": utc_now_iso(),
                    "event": "goal_finalization",
                    "starting_value": goal_progress.get("starting_value"),
                    "current_value": goal_progress.get("current_value"),
                    "goal_value": goal_progress.get("goal_value"),
                    "progress_pct": goal_progress.get("progress_pct"),
                    "goal_reached": True,
                    "finalization_status": finalization_status,
                    "finalization_result": finalization_result,
                    "final_stop_reason": "goal_already_reached_finalized",
                }
            )
            update_autopilot_state(
                running=False,
                status="completed",
                stop_requested=False,
                finalization_status=finalization_status,
                finalization_result=finalization_result,
                final_stop_reason="goal_already_reached_finalized",
            )
            record_autopilot_run_summary(
                run_id=run_id,
                allow_live=bool(config.get("allow_live", False)),
                unattended_mode=unattended_mode,
                cycles_completed=0,
                skip_cycles=0,
                executed_cycles=0,
                conversion_successes=0,
                conversion_failures=0,
                reconciliation_incidents=0,
                finalization_status=finalization_status,
                stop_reason="goal_already_reached_finalized",
            )
            return
        if planned_cycles <= 0 and not continue_until_goal:
            update_autopilot_state(running=False, status="completed", stop_requested=False, final_stop_reason="planned_cycles_completed")
            record_autopilot_run_summary(
                run_id=run_id,
                allow_live=bool(config.get("allow_live", False)),
                unattended_mode=unattended_mode,
                cycles_completed=0,
                skip_cycles=0,
                executed_cycles=0,
                conversion_successes=0,
                conversion_failures=0,
                reconciliation_incidents=0,
                finalization_status="",
                stop_reason="planned_cycles_completed",
            )
            return

        cycle_index = 0
        extra_cycles_used = 0
        failed_cycles_in_row = 0
        current_target_cycles = planned_cycles
        executed_cycles = 0
        skip_cycles = 0
        conversion_successes = 0
        conversion_failures = 0
        reconciliation_incidents = 0
        while True:
            if should_stop_autopilot():
                update_autopilot_state(running=False, status="cancelled", stop_requested=False)
                record_autopilot_run_summary(
                    run_id=run_id,
                    allow_live=bool(config.get("allow_live", False)),
                    unattended_mode=unattended_mode,
                    cycles_completed=cycle_index,
                    skip_cycles=skip_cycles,
                    executed_cycles=executed_cycles,
                    conversion_successes=conversion_successes,
                    conversion_failures=conversion_failures,
                    reconciliation_incidents=reconciliation_incidents,
                    finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                    stop_reason="cancelled",
                )
                return
            elapsed_minutes = max(0.0, (time.time() - started_monotonic) / 60.0)
            if elapsed_minutes >= max_runtime_minutes:
                emit_autopilot_alert("goal_timeout", "error", "Autopilot stopped after reaching max runtime.", requires_human_review=unattended_mode, details={"run_id": run_id, "elapsed_minutes": elapsed_minutes})
                update_autopilot_state(
                    running=False,
                    status="completed",
                    stop_requested=False,
                    current_value=float(goal_progress.get("current_value") or 0.0),
                    progress_pct=float(goal_progress.get("progress_pct") or 0.0),
                    goal_reached=bool(goal_progress.get("goal_reached", False)),
                    final_stop_reason="max_runtime_reached",
                )
                record_autopilot_run_summary(
                    run_id=run_id,
                    allow_live=bool(config.get("allow_live", False)),
                    unattended_mode=unattended_mode,
                    cycles_completed=cycle_index,
                    skip_cycles=skip_cycles,
                    executed_cycles=executed_cycles,
                    conversion_successes=conversion_successes,
                    conversion_failures=conversion_failures,
                    reconciliation_incidents=reconciliation_incidents,
                    finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                    stop_reason="max_runtime_reached",
                )
                return
            drawdown_pct = 0.0
            if starting_value > 0:
                drawdown_pct = max(0.0, (starting_value - current_value) / starting_value)
            if drawdown_pct >= max_drawdown_pct:
                emit_autopilot_alert("max_drawdown_breach", "critical", "Autopilot stopped after drawdown kill switch breached.", requires_human_review=True, details={"run_id": run_id, "drawdown_pct": drawdown_pct})
                update_autopilot_state(
                    running=False,
                    status="completed",
                    stop_requested=False,
                    current_value=float(goal_progress.get("current_value") or 0.0),
                    progress_pct=float(goal_progress.get("progress_pct") or 0.0),
                    goal_reached=bool(goal_progress.get("goal_reached", False)),
                    final_stop_reason="max_drawdown_kill_switch",
                )
                record_autopilot_run_summary(
                    run_id=run_id,
                    allow_live=bool(config.get("allow_live", False)),
                    unattended_mode=unattended_mode,
                    cycles_completed=cycle_index,
                    skip_cycles=skip_cycles,
                    executed_cycles=executed_cycles,
                    conversion_successes=conversion_successes,
                    conversion_failures=conversion_failures,
                    reconciliation_incidents=reconciliation_incidents,
                    finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                    stop_reason="max_drawdown_kill_switch",
                )
                return
            if cycle_index >= planned_cycles:
                if continue_until_goal and not bool(goal_progress.get("goal_reached", False)):
                    if extra_cycles_used >= max_extra_cycles:
                        emit_autopilot_alert("goal_timeout", "error", "Autopilot exhausted extra cycles before reaching goal.", requires_human_review=unattended_mode, details={"run_id": run_id, "extra_cycles_used": extra_cycles_used})
                        update_autopilot_state(
                            running=False,
                            status="completed",
                            stop_requested=False,
                            current_value=float(goal_progress.get("current_value") or 0.0),
                            progress_pct=float(goal_progress.get("progress_pct") or 0.0),
                            goal_reached=bool(goal_progress.get("goal_reached", False)),
                            extra_cycles_used=extra_cycles_used,
                            failed_cycles_in_row=failed_cycles_in_row,
                            final_stop_reason="max_extra_cycles_reached",
                        )
                        record_autopilot_run_summary(
                            run_id=run_id,
                            allow_live=bool(config.get("allow_live", False)),
                            unattended_mode=unattended_mode,
                            cycles_completed=cycle_index,
                            skip_cycles=skip_cycles,
                            executed_cycles=executed_cycles,
                            conversion_successes=conversion_successes,
                            conversion_failures=conversion_failures,
                            reconciliation_incidents=reconciliation_incidents,
                            finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                            stop_reason="max_extra_cycles_reached",
                        )
                        return
                    extra_cycles_used += 1
                    current_target_cycles = planned_cycles + extra_cycles_used
                    update_autopilot_state(target_cycles=current_target_cycles, extra_cycles_used=extra_cycles_used)
                else:
                    update_autopilot_state(
                        running=False,
                        status="completed",
                        stop_requested=False,
                        current_value=float(goal_progress.get("current_value") or 0.0),
                        progress_pct=float(goal_progress.get("progress_pct") or 0.0),
                        goal_reached=bool(goal_progress.get("goal_reached", False)),
                        extra_cycles_used=extra_cycles_used,
                        failed_cycles_in_row=failed_cycles_in_row,
                        final_stop_reason="planned_cycles_completed",
                    )
                    record_autopilot_run_summary(
                        run_id=run_id,
                        allow_live=bool(config.get("allow_live", False)),
                        unattended_mode=unattended_mode,
                        cycles_completed=cycle_index,
                        skip_cycles=skip_cycles,
                        executed_cycles=executed_cycles,
                        conversion_successes=conversion_successes,
                        conversion_failures=conversion_failures,
                        reconciliation_incidents=reconciliation_incidents,
                        finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                        stop_reason="planned_cycles_completed",
                    )
                    return

            cycle_number = cycle_index + 1
            decision_payload = get_decision_payload(
                buy_threshold=float(config.get("buy_threshold", BUY_THRESHOLD)),
                sell_threshold=float(config.get("sell_threshold", SELL_THRESHOLD)),
                adaptive_threshold_enabled=bool(config.get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
            )
            wallet_snapshot = get_wallet_snapshot()
            cycle_selection = resolve_autopilot_cycle_candidate(
                config=config,
                decision_payload=decision_payload,
                wallet_snapshot=wallet_snapshot,
                preferred_symbol=symbol,
                previous_symbol=previous_symbol,
            )
            symbol = normalize_symbol(str(cycle_selection.get("symbol") or symbol))
            previous_symbol = symbol
            snapshot = dict(cycle_selection.get("account_snapshot") or get_account_snapshot(symbol))
            signal_resolution = dict(cycle_selection.get("signal_resolution") or {})
            candidate_decision_payload = dict(cycle_selection.get("decision_payload") or decision_payload)
            decision = dict(candidate_decision_payload.get("decision") or decision_payload.get("decision") or {})
            raw_signal = str(signal_resolution.get("raw_signal") or decision.get("signal", "HOLD")).upper()
            probability_up = float(signal_resolution.get("probability_up") or decision.get("probability_up", 0.5))
            signal = str(signal_resolution.get("final_signal") or raw_signal)
            decision_confidence = float(signal_resolution.get("decision_confidence") or candidate_decision_payload.get("decision_confidence") or decision.get("decision_confidence") or 0.0)
            decision_min_confidence = float(signal_resolution.get("decision_min_confidence") or config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE))
            trade_plan = dict(cycle_selection.get("trade_plan") or {})
            if not trade_plan:
                trade_plan = build_autopilot_trade_plan(
                    symbol=symbol,
                    signal=signal,
                    probability_up=probability_up,
                    decision_payload=candidate_decision_payload,
                    balance_snapshot=snapshot,
                    wallet_snapshot=wallet_snapshot,
                    config=config,
                    forced_exit=bool(signal_resolution.get("forced_exit", False)),
                )
            opportunity_rankings = list(cycle_selection.get("opportunity_rankings") or [])
            opportunity_winner = dict(cycle_selection.get("opportunity_winner") or {})
            opportunity_meta = dict(cycle_selection.get("opportunity_meta") or {})
            size_plan = dict(trade_plan.get("size_plan") or {})
            execution_plan = dict(trade_plan.get("execution_plan") or {})
            conversion_plan = dict(trade_plan.get("conversion_plan") or {}) if trade_plan.get("conversion_plan") else None
            order_size = float(execution_plan.get("computed_order_size_base") or size_plan.get("base_size") or 0.0)
            append_live_history(symbol, signal, probability_up)
            cycle_outcome = str(trade_plan.get("final_action") or signal or "HOLD").upper()
            conversion_result: dict[str, Any] | None = None
            if cycle_outcome == "BUY" and conversion_plan is not None:
                set_reconciliation_state(
                    "conversion_intent_recorded",
                    details={"symbol": symbol, "cycle": cycle_number, "conversion_plan": conversion_plan},
                    requires_human_review=False,
                )
                conversion_result = execute_account_action(
                    action=str(conversion_plan.get("action") or ""),
                    symbol=str(conversion_plan.get("conversion_symbol") or ""),
                    quantity=float(conversion_plan.get("quantity") or 0.0),
                    quote_amount=float(conversion_plan.get("quote_amount") or 0.0),
                    dry_run=not bool(config.get("allow_live", False)),
                    max_api_latency_ms=int(config.get("max_api_latency_ms", 1200)),
                    warning_latency_ms=int(config.get("warning_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.55))),
                    degraded_latency_ms=int(config.get("degraded_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.8))),
                    block_latency_ms=int(config.get("block_latency_ms", config.get("max_api_latency_ms", 1200))),
                    consecutive_breach_limit=int(config.get("consecutive_breach_limit", 3)),
                    max_ticker_age_ms=int(config.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(config.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(config.get("min_trade_cooldown_seconds", 5)),
                    execution_context={
                        "run_id": run_id,
                        "cycle": cycle_number,
                        "stage": "conversion",
                        "unattended_mode": unattended_mode,
                    },
                )
                conversion_status = str(conversion_result.get("status") or "")
                if conversion_status in {"blocked", "error"}:
                    cycle_outcome = "SKIP"
                    execution_plan["skip_reason"] = conversion_result.get("skip_reason") or conversion_result.get("guard_reason") or "conversion_failed"
                    set_reconciliation_state(
                        "partial_execution_needs_review" if conversion_status == "error" else "clean",
                        details={"conversion_result": conversion_result, "cycle": cycle_number, "symbol": symbol},
                        requires_human_review=conversion_status == "error",
                    )
                elif bool(config.get("allow_live", False)):
                    set_reconciliation_state(
                        "interrupted_after_conversion",
                        details={"conversion_result": conversion_result, "cycle": cycle_number, "symbol": symbol},
                        requires_human_review=False,
                    )
                    refreshed_snapshot = get_account_snapshot(symbol)
                    refreshed_wallet_snapshot = get_wallet_snapshot()
                    trade_plan = build_autopilot_trade_plan(
                        symbol=symbol,
                        signal=signal,
                        probability_up=probability_up,
                        decision_payload=candidate_decision_payload,
                        balance_snapshot=refreshed_snapshot,
                        wallet_snapshot=refreshed_wallet_snapshot,
                        config=config,
                        forced_exit=bool(signal_resolution.get("forced_exit", False)),
                    )
                    size_plan = dict(trade_plan.get("size_plan") or {})
                    execution_plan = dict(trade_plan.get("execution_plan") or {})
                    order_size = float(execution_plan.get("computed_order_size_base") or size_plan.get("base_size") or 0.0)
                    cycle_outcome = str(trade_plan.get("final_action") or "SKIP").upper()
            if cycle_outcome == "BUY":
                set_reconciliation_state(
                    "buy_intent_recorded",
                    details={"symbol": symbol, "cycle": cycle_number, "order_size": order_size},
                    requires_human_review=False,
                )
                trade_result = maybe_execute_signal_trade(
                    signal="BUY",
                    symbol=symbol,
                    order_size=order_size,
                    allow_live=bool(config.get("allow_live", False)),
                    max_api_latency_ms=int(config.get("max_api_latency_ms", 1200)),
                    warning_latency_ms=int(config.get("warning_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.55))),
                    degraded_latency_ms=int(config.get("degraded_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.8))),
                    block_latency_ms=int(config.get("block_latency_ms", config.get("max_api_latency_ms", 1200))),
                    consecutive_breach_limit=int(config.get("consecutive_breach_limit", 3)),
                    max_ticker_age_ms=int(config.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(config.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(config.get("min_trade_cooldown_seconds", 5)),
                    execution_context={
                        "run_id": run_id,
                        "cycle": cycle_number,
                        "stage": "signal_buy",
                        "unattended_mode": unattended_mode,
                    },
                )
            elif cycle_outcome == "SELL":
                set_reconciliation_state(
                    "sell_intent_recorded",
                    details={"symbol": symbol, "cycle": cycle_number, "order_size": order_size},
                    requires_human_review=False,
                )
                trade_result = maybe_execute_signal_trade(
                    signal="SELL",
                    symbol=symbol,
                    order_size=order_size,
                    allow_live=bool(config.get("allow_live", False)),
                    max_api_latency_ms=int(config.get("max_api_latency_ms", 1200)),
                    warning_latency_ms=int(config.get("warning_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.55))),
                    degraded_latency_ms=int(config.get("degraded_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.8))),
                    block_latency_ms=int(config.get("block_latency_ms", config.get("max_api_latency_ms", 1200))),
                    consecutive_breach_limit=int(config.get("consecutive_breach_limit", 3)),
                    max_ticker_age_ms=int(config.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(config.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(config.get("min_trade_cooldown_seconds", 5)),
                    execution_context={
                        "run_id": run_id,
                        "cycle": cycle_number,
                        "stage": "signal_sell",
                        "unattended_mode": unattended_mode,
                    },
                )
            elif cycle_outcome == "SKIP":
                trade_result = {
                    "status": "skip",
                    "message": str(execution_plan.get("skip_reason") or "Autopilot skipped this cycle."),
                    "skip_reason": str(execution_plan.get("skip_reason") or "no_safe_funding_path"),
                    "symbol": symbol,
                    "effective_quantity": 0.0,
                    "effective_quote_amount": 0.0,
                    **execution_plan,
                }
            else:
                trade_result = {
                    "status": "no_trade",
                    "message": "Signal resolved to HOLD, no trade action sent.",
                    "skip_reason": "",
                    "symbol": symbol,
                    "effective_quantity": 0.0,
                    "effective_quote_amount": 0.0,
                    **execution_plan,
                }
            if trade_result.get("status") == "blocked":
                cycle_outcome = "SKIP"
                trade_result["skip_reason"] = str(trade_result.get("skip_reason") or trade_result.get("guard_reason") or "guard_blocked")
            if str(trade_result.get("status") or "") == "executed":
                if cycle_outcome == "BUY":
                    set_reconciliation_state("interrupted_after_buy", details={"trade_result": trade_result, "cycle": cycle_number, "symbol": symbol}, requires_human_review=False)
                elif cycle_outcome == "SELL":
                    set_reconciliation_state("interrupted_after_sell", details={"trade_result": trade_result, "cycle": cycle_number, "symbol": symbol}, requires_human_review=False)
            elif str(trade_result.get("status") or "") in {"error"}:
                set_reconciliation_state("partial_execution_needs_review", details={"trade_result": trade_result, "cycle": cycle_number, "symbol": symbol}, requires_human_review=True)
            final_action = cycle_outcome
            if str(trade_result.get("status") or "") == "blocked":
                emit_autopilot_alert(
                    "guard_blocked",
                    "error",
                    f"Guard blocked {symbol} {cycle_outcome}.",
                    requires_human_review=False,
                    details={"cycle": cycle_number, "symbol": symbol, "guard_reason": trade_result.get("guard_reason"), "skip_reason": trade_result.get("skip_reason")},
                )
            if str((conversion_result or {}).get("status") or "") in {"error", "blocked"}:
                emit_autopilot_alert(
                    "conversion_failed",
                    "error",
                    f"Conversion failed before trading {symbol}.",
                    requires_human_review=str((conversion_result or {}).get("status") or "") == "error",
                    details={"cycle": cycle_number, "symbol": symbol, "conversion_result": conversion_result},
                )
            rebalance_result: dict[str, Any] = {
                "status": "disabled",
                "reason": "Auto rebalance disabled for this run.",
            }
            touched_assets: set[str] = set()
            if final_action in {"BUY", "SELL"} and order_size > 0:
                base_asset, _ = split_symbol_assets(symbol)
                touched_assets.add(base_asset)
            if bool(config.get("auto_rebalance_enabled", False)):
                try:
                    rebalance_result = execute_rebalance_orders(
                        {
                            "allow_live": bool(config.get("allow_live", False)),
                            "max_orders": int(config.get("rebalance_max_orders", 3)),
                            "decision_min_confidence": float(config.get("decision_min_confidence", DECISION_MIN_CONFIDENCE)),
                            "target_monitor_symbols": config.get("target_monitor_symbols", TARGET_MONITOR_SYMBOLS),
                            "stable_asset": str(config.get("stable_asset", PROFIT_PARKING_STABLE_ASSET)),
                            "stable_reserve_min_pct": float(config.get("stable_reserve_min_pct", STABLE_RESERVE_MIN_PCT)),
                            "enable_quote_parking": bool(config.get("enable_quote_parking", True)),
                            "parking_max_orders": int(config.get("parking_max_orders", 2)),
                            "parking_min_usdt_value": float(config.get("parking_min_usdt_value", 10.0)),
                            "rebalance_min_delta_usdt": float(config.get("rebalance_min_delta_usdt", 25.0)),
                            "skip_symbols": [symbol] if touched_assets else [],
                            "skip_assets": sorted(touched_assets),
                            "max_api_latency_ms": int(config.get("max_api_latency_ms", 1200)),
                            "warning_latency_ms": int(config.get("warning_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.55))),
                            "degraded_latency_ms": int(config.get("degraded_latency_ms", int(config.get("max_api_latency_ms", 1200) * 0.8))),
                            "block_latency_ms": int(config.get("block_latency_ms", config.get("max_api_latency_ms", 1200))),
                            "consecutive_breach_limit": int(config.get("consecutive_breach_limit", 3)),
                            "max_ticker_age_ms": int(config.get("max_ticker_age_ms", 3000)),
                            "max_spread_bps": float(config.get("max_spread_bps", 20.0)),
                            "min_trade_cooldown_seconds": int(config.get("rebalance_min_trade_cooldown_seconds", 0)),
                            "buy_threshold": float(config.get("buy_threshold", BUY_THRESHOLD)),
                            "sell_threshold": float(config.get("sell_threshold", SELL_THRESHOLD)),
                            "adaptive_threshold_enabled": bool(config.get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
                        }
                    )
                except Exception as rebalance_exc:
                    rebalance_result = {
                        "status": "error",
                        "reason": str(rebalance_exc),
                    }
            trade_action = "market_buy" if final_action == "BUY" else "market_sell" if final_action == "SELL" else "hold"
            log_trade_memory(
                source="autopilot",
                symbol=symbol,
                action=trade_action,
                payload=trade_result,
                signal=signal,
                probability_up=probability_up,
                cycle=cycle_number,
                target_cycles=current_target_cycles,
                size_plan=size_plan,
                account_snapshot=snapshot,
            )
            refreshed_wallet_snapshot = wallet_snapshot
            refreshed_snapshot = snapshot
            try:
                refreshed_wallet_snapshot = get_wallet_snapshot()
                current_value = max(
                    0.0,
                    float(refreshed_wallet_snapshot.get("estimated_total_usdt") or 0.0),
                    float(snapshot.get("account_value_quote") or 0.0),
                )
            except Exception:
                current_value = max(0.0, float(snapshot.get("account_value_quote") or 0.0), current_value)
            try:
                refreshed_snapshot = get_account_snapshot(symbol)
            except Exception:
                refreshed_snapshot = snapshot
            goal_progress = compute_goal_progress(starting_value, current_value, goal_value)
            skip_reason = str(trade_result.get("skip_reason", execution_plan.get("skip_reason", "")) or "")
            if str((conversion_result or {}).get("status") or "") == "executed":
                conversion_successes += 1
            elif str((conversion_result or {}).get("status") or "") in {"blocked", "error"}:
                conversion_failures += 1
            if final_action in {"BUY", "SELL"} and str(trade_result.get("status") or "") == "executed":
                executed_cycles += 1
            elif final_action == "SKIP":
                skip_cycles += 1
            if final_action == "SKIP" and skip_reason:
                emit_autopilot_alert(
                    "repeated_skip" if failed_cycles_in_row > 0 else "skip_cycle",
                    "warn",
                    f"Autopilot skipped {symbol}: {skip_reason}",
                    requires_human_review=False,
                    details={"cycle": cycle_number, "symbol": symbol, "skip_reason": skip_reason},
                )
            failure_reasons = {
                "conversion_failed",
                "guard_blocked",
                "no_safe_funding_path",
                "conversion_below_minimum",
                "conversion_exceeds_risk_limit",
                "insufficient_free_quote",
                "insufficient_free_base",
                "below_min_qty",
                "below_min_notional",
                "below_step_size_after_rounding",
            }
            cycle_failed = False
            if str(trade_result.get("status") or "") in {"error", "blocked"}:
                cycle_failed = True
            elif str((conversion_result or {}).get("status") or "") in {"error", "blocked"}:
                cycle_failed = True
            elif str(rebalance_result.get("status") or "") == "error":
                cycle_failed = True
            elif final_action == "SKIP" and skip_reason in failure_reasons:
                cycle_failed = True
            if cycle_failed:
                failed_cycles_in_row += 1
                if str(AUTOPILOT_STATE.get("reconciliation_state") or "clean") != "clean":
                    reconciliation_incidents += 1
            else:
                failed_cycles_in_row = 0
                set_reconciliation_state(
                    "clean",
                    details={"cycle": cycle_number, "symbol": symbol, "final_action": final_action, "trade_status": trade_result.get("status")},
                    requires_human_review=False,
                )
            append_autopilot_log(
                {
                    "cycle": cycle_number,
                    "target_cycles": current_target_cycles,
                    "timestamp": utc_now_iso(),
                    "symbol": symbol,
                    "raw_signal": raw_signal,
                    "signal": signal,
                    "override_reason": signal_resolution.get("override_reason", ""),
                    "decision_confidence": decision_confidence,
                    "decision_min_confidence": decision_min_confidence,
                    "probability_up": probability_up,
                    "bid": refreshed_snapshot.get("best_bid"),
                    "ask": refreshed_snapshot.get("best_ask"),
                    "account_value": refreshed_snapshot.get("account_value_quote"),
                    "min_qty": execution_plan.get("min_qty", 0.0),
                    "step_size": execution_plan.get("step_size", 0.0),
                    "min_notional": execution_plan.get("min_notional", 0.0),
                    "minimum_valid_quote": execution_plan.get("minimum_valid_quote", 0.0),
                    "free_quote": execution_plan.get("free_quote", 0.0),
                    "free_base": execution_plan.get("free_base", 0.0),
                    "required_base_asset": trade_plan.get("required_base_asset", ""),
                    "required_quote_asset": trade_plan.get("required_quote_asset", ""),
                    "wallet_summary": trade_plan.get("wallet_summary", {}),
                    "direct_trade_possible": trade_plan.get("direct_trade_possible", False),
                    "funding_path": trade_plan.get("funding_path", "none"),
                    "funding_skip_reason": trade_plan.get("funding_skip_reason", ""),
                    "free_bnb": trade_plan.get("free_bnb", 0.0),
                    "eligible_funding_assets": trade_plan.get("eligible_funding_assets", []),
                    "funding_diagnostics": trade_plan.get("funding_diagnostics", {}),
                    "conversion_plan": conversion_plan or {},
                    "opportunity_rankings": opportunity_rankings,
                    "opportunity_winner": opportunity_winner,
                    "opportunity_meta": opportunity_meta,
                    "final_action": final_action,
                    "order_size": order_size,
                    "final_order_size": trade_result.get("effective_quantity", order_size),
                    "final_quote_amount": trade_result.get("effective_quote_amount", execution_plan.get("computed_order_size_quote", 0.0)),
                    "computed_order_size_quote": execution_plan.get("computed_order_size_quote", 0.0),
                    "computed_order_size_base": execution_plan.get("computed_order_size_base", 0.0),
                    "can_buy_minimum": execution_plan.get("can_buy_minimum", False),
                    "skip_reason": skip_reason,
                    "size_strength": size_plan.get("strength"),
                    "quote_size": size_plan.get("quote_size"),
                    "allocation_pct": size_plan.get("allocation_pct"),
                    "trade_status": trade_result.get("status"),
                    "trade_message": trade_result.get("message", ""),
                    "trade_reason_code": trade_result.get("guard_reason") or trade_result.get("minimum_message") or trade_result.get("message", ""),
                    "execution_mode": trade_result.get("guard_mode", "normal"),
                    "conversion_result": conversion_result or {},
                    "rebalance_status": rebalance_result.get("status"),
                    "rebalance_result_counts": rebalance_result.get("result_counts", {}),
                    "rebalance_orders": rebalance_result.get("orders", []),
                    "rebalance_actions": rebalance_result.get("results", []),
                    "starting_value": goal_progress.get("starting_value"),
                    "current_value": goal_progress.get("current_value"),
                    "goal_value": goal_progress.get("goal_value"),
                    "progress_pct": goal_progress.get("progress_pct"),
                    "goal_reached": goal_progress.get("goal_reached"),
                    "continue_until_goal": continue_until_goal,
                    "extra_cycles_used": extra_cycles_used,
                    "failed_cycles_in_row": failed_cycles_in_row,
                    "final_stable_target_asset": stable_asset,
                    "dust_base": execution_plan.get("dust_base", 0.0),
                    "api_latency_ms": trade_result.get("api_latency_ms", refreshed_snapshot.get("api_latency_ms")),
                    "ticker_age_ms": trade_result.get("ticker_age_ms", refreshed_snapshot.get("ticker_age_ms")),
                    "spread_bps": trade_result.get("spread_bps", refreshed_snapshot.get("spread_bps")),
                    "guard": trade_result.get("guard_message", ""),
                }
            )
            update_autopilot_state(
                current_cycle=cycle_number,
                target_cycles=current_target_cycles,
                symbol=symbol,
                latest_decision=decision,
                execution_mode=trade_result.get("guard_mode", "normal"),
                min_qty=float(execution_plan.get("min_qty") or 0.0),
                step_size=float(execution_plan.get("step_size") or 0.0),
                min_notional=float(execution_plan.get("min_notional") or 0.0),
                minimum_valid_quote=float(execution_plan.get("minimum_valid_quote") or 0.0),
                free_quote=float(execution_plan.get("free_quote") or 0.0),
                free_base=float(execution_plan.get("free_base") or 0.0),
                computed_order_size_quote=float(execution_plan.get("computed_order_size_quote") or 0.0),
                computed_order_size_base=float(execution_plan.get("computed_order_size_base") or 0.0),
                can_buy_minimum=bool(execution_plan.get("can_buy_minimum", False)),
                required_base_asset=str(trade_plan.get("required_base_asset") or ""),
                required_quote_asset=str(trade_plan.get("required_quote_asset") or ""),
                direct_trade_possible=bool(trade_plan.get("direct_trade_possible", False)),
                funding_path=str(trade_plan.get("funding_path") or "none"),
                funding_skip_reason=str(trade_plan.get("funding_skip_reason") or ""),
                free_bnb=float(trade_plan.get("free_bnb") or 0.0),
                eligible_funding_assets=list(trade_plan.get("eligible_funding_assets") or []),
                funding_diagnostics=dict(trade_plan.get("funding_diagnostics") or {}),
                wallet_summary=trade_plan.get("wallet_summary") or {},
                conversion_plan=conversion_plan or {},
                opportunity_rankings=opportunity_rankings,
                opportunity_winner=opportunity_winner,
                opportunity_meta=opportunity_meta,
                skip_reason=skip_reason,
                starting_value=float(goal_progress.get("starting_value") or 0.0),
                current_value=float(goal_progress.get("current_value") or 0.0),
                goal_value=float(goal_progress.get("goal_value") or 0.0),
                progress_pct=float(goal_progress.get("progress_pct") or 0.0),
                goal_reached=bool(goal_progress.get("goal_reached", False)),
                continue_until_goal=continue_until_goal,
                extra_cycles_used=extra_cycles_used,
                failed_cycles_in_row=failed_cycles_in_row,
                final_stable_target_asset=stable_asset,
                latest_trade_result={
                    "raw_signal": raw_signal,
                    "override_reason": signal_resolution.get("override_reason", ""),
                    "final_signal": signal,
                    "final_action": final_action,
                    "latest_decision": decision,
                    "wallet_summary": trade_plan.get("wallet_summary", {}),
                    "required_base_asset": trade_plan.get("required_base_asset", ""),
                    "required_quote_asset": trade_plan.get("required_quote_asset", ""),
                    "direct_trade_possible": trade_plan.get("direct_trade_possible", False),
                    "funding_path": trade_plan.get("funding_path", "none"),
                    "funding_skip_reason": trade_plan.get("funding_skip_reason", ""),
                    "free_bnb": trade_plan.get("free_bnb", 0.0),
                    "eligible_funding_assets": trade_plan.get("eligible_funding_assets", []),
                    "funding_diagnostics": trade_plan.get("funding_diagnostics", {}),
                    "conversion_plan": conversion_plan or {},
                    "conversion_result": conversion_result or {},
                    "opportunity_rankings": opportunity_rankings,
                    "opportunity_winner": opportunity_winner,
                    "opportunity_meta": opportunity_meta,
                    "size_plan": size_plan,
                    "execution_plan": execution_plan,
                    "signal_trade": trade_result,
                    "rebalance": rebalance_result,
                },
            )
            if bool(goal_progress.get("goal_reached", False)):
                update_autopilot_state(finalization_status="pending")
                set_reconciliation_state("interrupted_during_finalization", details={"cycle": cycle_number, "symbol": symbol}, requires_human_review=False)
                finalization_result = execute_goal_finalization(config, run_id=run_id, cycle=cycle_number, unattended_mode=unattended_mode)
                finalization_status = "completed" if str(finalization_result.get("status") or "") in {"ok", "dust_only"} else str(finalization_result.get("status") or "error")
                if finalization_status != "completed":
                    emit_autopilot_alert(
                        "failed_finalization",
                        "critical",
                        "Goal finalization did not complete cleanly.",
                        requires_human_review=True,
                        details={"cycle": cycle_number, "symbol": symbol, "finalization_result": finalization_result},
                    )
                append_autopilot_log(
                    {
                        "cycle": cycle_number,
                        "timestamp": utc_now_iso(),
                        "event": "goal_finalization",
                        "starting_value": goal_progress.get("starting_value"),
                        "current_value": goal_progress.get("current_value"),
                        "goal_value": goal_progress.get("goal_value"),
                        "progress_pct": goal_progress.get("progress_pct"),
                        "goal_reached": True,
                        "finalization_status": finalization_status,
                        "finalization_result": finalization_result,
                        "final_stop_reason": "goal_reached_finalized",
                    }
                )
                update_autopilot_state(
                    running=False,
                    status="completed",
                    stop_requested=False,
                    finalization_status=finalization_status,
                    finalization_result=finalization_result,
                    final_stop_reason="goal_reached_finalized",
                )
                set_reconciliation_state("clean" if finalization_status == "completed" else "interrupted_during_finalization", details=finalization_result, requires_human_review=finalization_status != "completed")
                record_autopilot_run_summary(
                    run_id=run_id,
                    allow_live=bool(config.get("allow_live", False)),
                    unattended_mode=unattended_mode,
                    cycles_completed=cycle_number,
                    skip_cycles=skip_cycles,
                    executed_cycles=executed_cycles,
                    conversion_successes=conversion_successes,
                    conversion_failures=conversion_failures,
                    reconciliation_incidents=reconciliation_incidents,
                    finalization_status=finalization_status,
                    stop_reason="goal_reached_finalized",
                )
                return
            if failed_cycles_in_row >= max_failed_cycles_in_row:
                emit_autopilot_alert("excessive_failed_cycles", "critical", "Autopilot stopped after excessive failed cycles.", requires_human_review=True, details={"run_id": run_id, "failed_cycles_in_row": failed_cycles_in_row})
                update_autopilot_state(
                    running=False,
                    status="completed",
                    stop_requested=False,
                    final_stop_reason="max_failed_cycles_reached",
                )
                record_autopilot_run_summary(
                    run_id=run_id,
                    allow_live=bool(config.get("allow_live", False)),
                    unattended_mode=unattended_mode,
                    cycles_completed=cycle_number,
                    skip_cycles=skip_cycles,
                    executed_cycles=executed_cycles,
                    conversion_successes=conversion_successes,
                    conversion_failures=conversion_failures,
                    reconciliation_incidents=reconciliation_incidents,
                    finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                    stop_reason="max_failed_cycles_reached",
                )
                return
            cycle_index += 1
            if wait_for_autopilot_interval(interval_seconds):
                update_autopilot_state(running=False, status="cancelled", stop_requested=False)
                record_autopilot_run_summary(
                    run_id=run_id,
                    allow_live=bool(config.get("allow_live", False)),
                    unattended_mode=unattended_mode,
                    cycles_completed=cycle_index,
                    skip_cycles=skip_cycles,
                    executed_cycles=executed_cycles,
                    conversion_successes=conversion_successes,
                    conversion_failures=conversion_failures,
                    reconciliation_incidents=reconciliation_incidents,
                    finalization_status=str(AUTOPILOT_STATE.get("finalization_status") or ""),
                    stop_reason="cancelled",
                )
                return
    except Exception as exc:
        append_autopilot_log({"cycle": int(autopilot_snapshot().get("current_cycle", 0)) + 1, "timestamp": utc_now_iso(), "error": str(exc)})
        emit_autopilot_alert("autopilot_exception", "critical", f"Autopilot crashed: {exc}", requires_human_review=True, details={"run_id": run_id})
        set_reconciliation_state("partial_execution_needs_review", details={"error": str(exc), "run_id": run_id}, requires_human_review=True)
        update_autopilot_state(running=False, status="error", stop_requested=False, last_error=str(exc))
        record_autopilot_run_summary(
            run_id=run_id,
            allow_live=bool(config.get("allow_live", False)),
            unattended_mode=bool(config.get("unattended_mode", False)),
            cycles_completed=int(autopilot_snapshot().get("current_cycle", 0) or 0),
            skip_cycles=0,
            executed_cycles=0,
            conversion_successes=0,
            conversion_failures=0,
            reconciliation_incidents=1,
            finalization_status=str(autopilot_snapshot().get("finalization_status") or ""),
            stop_reason="error",
            restart_recovery_observed=False,
        )


def normalize_autopilot_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    normalized: dict[str, Any] = {
        "symbol": normalize_symbol(str(raw.get("symbol", CHECK_SYMBOL))),
        "allow_live": bool(raw.get("allow_live", False)),
        "unattended_mode": bool(raw.get("unattended_mode", False)),
        "auto_rebalance_enabled": bool(raw.get("auto_rebalance_enabled", False)),
        "multi_symbol_enabled": bool(raw.get("multi_symbol_enabled", AUTOPILOT_MULTI_SYMBOL_ENABLED)),
        "continue_until_goal": bool(raw.get("continue_until_goal", True)),
        "opportunity_max_candidates": int(raw.get("opportunity_max_candidates", AUTOPILOT_MAX_CANDIDATES)),
        "switch_score_delta": float(raw.get("switch_score_delta", AUTOPILOT_SWITCH_SCORE_DELTA)),
        "opportunity_quote_assets": raw.get("opportunity_quote_assets", AUTOPILOT_UNIVERSE_QUOTES),
        "symbol_denylist": raw.get("symbol_denylist", list(AUTOPILOT_SYMBOL_DENYLIST)),
        "max_extra_cycles": int(raw.get("max_extra_cycles", 24)),
        "max_failed_cycles_in_row": int(raw.get("max_failed_cycles_in_row", 8)),
        "max_runtime_minutes": float(raw.get("max_runtime_minutes", 240)),
        "finalization_max_orders": int(raw.get("finalization_max_orders", 20)),
        "finalization_min_usdt_value": float(raw.get("finalization_min_usdt_value", 5.0)),
        "preview_gate_passed": bool(raw.get("preview_gate_passed", False)),
        "alerting_enabled": bool(raw.get("alerting_enabled", AUTOPILOT_ALERTING_ENABLED)),
        "alert_webhook_url": str(raw.get("alert_webhook_url", AUTOPILOT_ALERT_WEBHOOK_URL or "") or ""),
        "manual_reconciliation_ack": bool(raw.get("manual_reconciliation_ack", False)),
    }
    if float(raw.get("goal_value", 0.0) or 0.0) > 0:
        normalized["goal_value"] = float(raw.get("goal_value", 0.0))

    optional_float_fields = {
        "decision_min_confidence",
        "size_min_confidence",
        "take_profit_pct",
        "stop_loss_pct",
        "fee_drag_pct",
        "max_api_latency_ms",
        "warning_latency_ms",
        "degraded_latency_ms",
        "block_latency_ms",
        "consecutive_breach_limit",
        "max_ticker_age_ms",
        "max_spread_bps",
        "min_trade_cooldown_seconds",
        "rebalance_min_trade_cooldown_seconds",
        "rebalance_min_delta_usdt",
        "stable_reserve_min_pct",
    }
    for field in optional_float_fields:
        if raw.get(field) is not None:
            normalized[field] = float(raw.get(field))

    optional_string_fields = {
        "stable_asset",
    }
    for field in optional_string_fields:
        value = raw.get(field)
        if value not in {None, ""}:
            normalized[field] = str(value)

    return normalized


def start_autopilot(config: dict[str, Any]) -> dict[str, Any]:
    global AUTOPILOT_THREAD
    normalized_config = normalize_autopilot_config(config)
    reconcile_interrupted_autopilot_state()
    start_gate = unattended_start_gate(normalized_config)
    if not bool(start_gate.get("ok", False)):
        raise RuntimeError(str(start_gate.get("reason") or "Unattended mode gate failed."))
    with AUTOPILOT_LOCK:
        if _autopilot_thread_alive_unlocked() or bool(AUTOPILOT_STATE.get("running")) or str(AUTOPILOT_STATE.get("status", "")) in {"starting", "stopping"}:
            raise RuntimeError("Autopilot is already running.")
        if str(AUTOPILOT_STATE.get("reconciliation_state") or "clean") != "clean" or bool(AUTOPILOT_STATE.get("requires_human_review", False)):
            if bool(normalized_config.get("manual_reconciliation_ack", False)) and not bool(normalized_config.get("unattended_mode", False)):
                AUTOPILOT_STATE["reconciliation_state"] = "clean"
                AUTOPILOT_STATE["reconciliation_details"] = {"acknowledged_at": utc_now_iso(), "acknowledged_by_start_request": True}
                AUTOPILOT_STATE["reconciliation_checked_at"] = utc_now_iso()
                AUTOPILOT_STATE["requires_human_review"] = False
            else:
                raise RuntimeError("Autopilot cannot start until reconciliation is clean and no human review is required.")
        AUTOPILOT_STOP_EVENT.clear()
        run_id = int(AUTOPILOT_STATE.get("run_id", 0)) + 1
        AUTOPILOT_STATE.update(
            {
                "running": True,
                "status": "starting",
                "run_id": run_id,
                "stop_requested": False,
                "current_cycle": 0,
                "target_cycles": 0,
                "symbol": normalize_symbol(str(normalized_config.get("symbol", CHECK_SYMBOL))),
                "started_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "logs": [],
                "latest_decision": {},
                "latest_trade_result": {},
                "cycle_plan": {},
                "execution_mode": "normal",
                "min_qty": 0.0,
                "step_size": 0.0,
                "min_notional": 0.0,
                "minimum_valid_quote": 0.0,
                "free_quote": 0.0,
                "free_base": 0.0,
                "computed_order_size_quote": 0.0,
                "computed_order_size_base": 0.0,
                "can_buy_minimum": False,
                "required_base_asset": "",
                "required_quote_asset": "",
                "direct_trade_possible": False,
                "funding_path": "none",
                "funding_skip_reason": "",
                "free_bnb": 0.0,
                "eligible_funding_assets": [],
                "funding_diagnostics": {},
                "wallet_summary": {},
                "conversion_plan": {},
                "skip_reason": "",
                "opportunity_rankings": [],
                "opportunity_winner": {},
                "opportunity_meta": {},
                "starting_value": 0.0,
                "current_value": 0.0,
                "goal_value": 0.0,
                "progress_pct": 0.0,
                "goal_reached": False,
                "continue_until_goal": bool(normalized_config.get("continue_until_goal", True)),
                "extra_cycles_used": 0,
                "failed_cycles_in_row": 0,
                "final_stable_target_asset": str(normalized_config.get("stable_asset", PROFIT_PARKING_STABLE_ASSET)),
                "finalization_status": "",
                "finalization_result": {},
                "final_stop_reason": "",
                "latest_execution_intent": {},
                "persistence_healthy": AUTOPILOT_STATE_PATH.exists(),
                "persistence_path": str(AUTOPILOT_STATE_PATH),
                "execution_journal_path": str(AUTOPILOT_EXECUTION_JOURNAL_PATH),
                "reconciliation_state": "clean",
                "reconciliation_details": {},
                "reconciliation_checked_at": utc_now_iso(),
                "requires_human_review": False,
                "alert_severity": "info",
                "alerts": [],
                "unattended_mode": bool(normalized_config.get("unattended_mode", False)),
                "burn_in_report": build_burn_in_validation_report(),
                "last_error": "",
            }
        )
        AUTOPILOT_THREAD = threading.Thread(target=run_autopilot, args=(normalized_config, run_id), daemon=True)
        AUTOPILOT_THREAD.start()
    persist_autopilot_state()
    return autopilot_snapshot()


def stop_autopilot() -> dict[str, Any]:
    AUTOPILOT_STOP_EVENT.set()
    update_autopilot_state(stop_requested=True, status="stopping")
    return autopilot_snapshot()


def run_ai_command(command: str) -> dict[str, Any]:
    cmd = command.strip().lower()
    if not cmd:
        return {"status": "empty", "message": "No command entered."}
    if cmd in {"recommend", "recommend action", "ai recommend"}:
        decision_payload = get_decision_payload()
        signal = decision_payload["decision"]["signal"]
        probability_up = float(decision_payload["decision"]["probability_up"])
        if signal == "BUY":
            message = f"AI recommendation: BUY (probability_up={probability_up:.1%}) with current risk limits."
        elif signal == "SELL":
            message = f"AI recommendation: SELL or reduce exposure (probability_up={probability_up:.1%})."
        else:
            message = f"AI recommendation: HOLD and wait for a stronger edge (probability_up={probability_up:.1%})."
        return {"status": "ok", "message": message}
    deposit_match = re.match(r"set\s+deposit\s+(\d+(?:\.\d+)?)", cmd)
    if deposit_match:
        return {"status": "ok", "message": f"Suggested deposit amount updated to {float(deposit_match.group(1)):.2f}.", "patch": {"deposit_amount": float(deposit_match.group(1))}}
    buy_match = re.match(r"set\s+buy\s+threshold\s+(0(?:\.\d+)?|1(?:\.0+)?)", cmd)
    if buy_match:
        return {"status": "ok", "message": f"Buy threshold updated to {float(buy_match.group(1)):.2f}.", "patch": {"buy_threshold": float(buy_match.group(1))}}
    sell_match = re.match(r"set\s+sell\s+threshold\s+(0(?:\.\d+)?|1(?:\.0+)?)", cmd)
    if sell_match:
        return {"status": "ok", "message": f"Sell threshold updated to {float(sell_match.group(1)):.2f}.", "patch": {"sell_threshold": float(sell_match.group(1))}}
    return {"status": "unknown", "message": "Command not recognized. Try: set deposit 1000, set buy threshold 0.60, set sell threshold 0.40, recommend"}
