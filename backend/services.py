from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
load_dotenv(PROJECT_ROOT / ".env", override=True)

import sys

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from generate_trade_signal import (  # noqa: E402
    ADAPTIVE_THRESHOLD_ENABLED,
    BUY_THRESHOLD,
    DATASET_PATH,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_BYPASS_ENV_PROXY,
    LLM_ENABLED,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    SELL_THRESHOLD,
    generate_trade_decision,
    load_latest_row,
    latest_file_with_suffix,
    resolve_model_path,
)

ARTIFACTS_DIR = RESEARCH_DIR / "artifacts"
DATA_DIR = RESEARCH_DIR / "data_sets"
MEMORY_DIR = RESEARCH_DIR / "runtime_memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
TRADE_MEMORY_PATH = MEMORY_DIR / "trade_memory.jsonl"

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
BINANCE_BYPASS_ENV_PROXY = os.getenv("BINANCE_BYPASS_ENV_PROXY", "true").lower() == "true"
BINANCE_TIMEOUT_SECONDS = int(os.getenv("BINANCE_TIMEOUT_SECONDS", "30"))
CHECK_SYMBOL = os.getenv("CHECK_SYMBOL", "BTC/USDT")
ACCOUNT_REFERENCE_USD = float(os.getenv("ACCOUNT_REFERENCE_USD", "6.93"))

STATE_LOCK = threading.Lock()
RUNTIME_STATE: dict[str, Any] = {
    "last_trade_ts": 0.0,
    "latest_price_fallback": None,
    "live_history": [],
}

AUTOPILOT_LOCK = threading.Lock()
AUTOPILOT_THREAD: threading.Thread | None = None
AUTOPILOT_STATE: dict[str, Any] = {
    "running": False,
    "status": "idle",
    "stop_requested": False,
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
}


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def from_binance_market_id(symbol_id: str, quote_asset: str) -> str:
    quote = (quote_asset or "USDT").upper()
    if symbol_id.endswith(quote) and len(symbol_id) > len(quote):
        base = symbol_id[: -len(quote)]
        return f"{base}/{quote}"
    return symbol_id


def binance_public_json(path: str, params: dict[str, Any] | None = None) -> Any:
    base = "https://testnet.binance.vision" if BINANCE_TESTNET else "https://api.binance.com"
    url = f"{base}{path}"
    with requests.Session() as session:
        if BINANCE_BYPASS_ENV_PROXY:
            session.trust_env = False
        response = session.get(url, params=params or {}, timeout=BINANCE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()


def get_binance_client():
    try:
        import ccxt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'ccxt'. Install requirements-api.txt.") from exc

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")

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
    return exchange


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


def get_ticker_with_metrics(symbol: str) -> tuple[dict[str, Any], dict[str, float]]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    start = time.monotonic()
    try:
        ticker = exchange.fetch_ticker(norm_symbol)
    except Exception as exc:
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
    }


def validate_market_conditions(
    api_latency_ms: float,
    ticker_age_ms: float,
    spread_bps: float,
    max_api_latency_ms: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
) -> tuple[bool, str]:
    last_trade_ts = float(get_runtime_value("last_trade_ts", 0.0) or 0.0)
    cooldown_left = max(0.0, min_trade_cooldown_seconds - (time.time() - last_trade_ts))
    if api_latency_ms > max_api_latency_ms:
        return False, f"Blocked: API latency too high ({api_latency_ms:.0f}ms > {max_api_latency_ms}ms)."
    if ticker_age_ms > max_ticker_age_ms:
        return False, f"Blocked: market data is stale ({ticker_age_ms:.0f}ms > {max_ticker_age_ms}ms)."
    if spread_bps > max_spread_bps:
        return False, f"Blocked: spread too wide ({spread_bps:.1f} bps > {max_spread_bps:.1f} bps)."
    if cooldown_left > 0:
        return False, f"Blocked: cooldown active ({cooldown_left:.1f}s remaining)."
    return True, "Market conditions accepted."


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
    quote_total = float(balance.get("total", {}).get(quote, 0) or 0)
    quote_free = float(balance.get("free", {}).get(quote, 0) or 0)
    quote_used = float(balance.get("used", {}).get(quote, 0) or 0)
    account_value_quote = quote_total + (base_total * mark)

    positions = []
    if base_total > 0:
        positions.append({"coin": base, "size": base_total, "entry_px": 0.0, "roe_pct": 0.0})

    return {
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "symbol": norm_symbol,
        "best_ask": ask,
        "best_bid": bid,
        "account_value_quote": account_value_quote,
        "quote_asset": quote,
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
    return {
        "min_qty": float((market.get("limits", {}).get("amount", {}).get("min") or 0) or 0),
        "min_notional": float((market.get("limits", {}).get("cost", {}).get("min") or 0) or 0),
        "qty_precision": int(market.get("precision", {}).get("amount") or 8),
        "price_precision": int(market.get("precision", {}).get("price") or 8),
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
        "asset_count": len(non_zero_assets),
        "estimated_total_usdt": total_estimated_usdt,
        "balances": dataframe_to_records(wallet_df),
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
) -> dict[str, Any]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    ticker, metrics = get_ticker_with_metrics(norm_symbol)
    market_price = float(ticker.get("last") or get_market_price(norm_symbol))
    market_reqs = get_market_requirements(norm_symbol)
    guard_ok, guard_message = validate_market_conditions(
        api_latency_ms=metrics["api_latency_ms"],
        ticker_age_ms=metrics["ticker_age_ms"],
        spread_bps=metrics["spread_bps"],
        max_api_latency_ms=max_api_latency_ms,
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
    )
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
        "guard_ok": guard_ok,
        "guard_message": guard_message,
        "min_qty": market_reqs["min_qty"],
        "min_notional": market_reqs["min_notional"],
        "qty_precision": market_reqs["qty_precision"],
    }
    if dry_run:
        payload["status"] = "dry_run_only"
        payload["message"] = "No order sent. Disable dry-run to execute on Binance."
        log_trade_memory(source="trade_action", symbol=norm_symbol, action=action, payload=payload)
        return payload
    if not guard_ok:
        payload["status"] = "blocked"
        payload["message"] = guard_message
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
    if action == "market_buy":
        if quantity <= 0 and quote_amount > 0:
            try:
                result = exchange.create_market_buy_order_with_cost(norm_symbol, quote_amount)
            except Exception:
                result = exchange.create_order(norm_symbol, "market", "buy", None, None, {"quoteOrderQty": quote_amount})
        else:
            if quantity <= 0:
                raise ValueError("Provide quantity or quote amount for market_buy.")
            order_qty = float(exchange.amount_to_precision(norm_symbol, quantity))
            result = exchange.create_order(norm_symbol, "market", "buy", order_qty)
    elif action == "market_sell":
        if quantity <= 0:
            raise ValueError("Provide quantity for market_sell.")
        sell_qty = float(exchange.amount_to_precision(norm_symbol, quantity))
        result = exchange.create_order(norm_symbol, "market", "sell", sell_qty)
    elif action == "cancel_all_orders":
        result = exchange.cancel_all_orders(norm_symbol)
    else:
        raise ValueError(f"Unsupported action: {action}")
    payload["status"] = "executed"
    payload["result"] = result
    set_runtime_value("last_trade_ts", time.time())
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
) -> dict[str, Any]:
    norm_symbol = normalize_symbol(symbol)
    ticker, metrics = get_ticker_with_metrics(norm_symbol)
    market_price = float(ticker.get("last") or get_market_price(norm_symbol))
    market_reqs = get_market_requirements(norm_symbol)
    guard_ok, guard_message = validate_market_conditions(
        api_latency_ms=metrics["api_latency_ms"],
        ticker_age_ms=metrics["ticker_age_ms"],
        spread_bps=metrics["spread_bps"],
        max_api_latency_ms=max_api_latency_ms,
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
    )
    minimum_ok, minimum_message = validate_order_minimums(
        action=action,
        quantity=quantity,
        quote_amount=quote_amount,
        market_price=market_price,
        min_qty=float(market_reqs["min_qty"]),
        min_notional=float(market_reqs["min_notional"]),
    )
    return {
        "status": "ok" if guard_ok and minimum_ok else "blocked",
        "action": action,
        "symbol": norm_symbol,
        "quantity": quantity,
        "quote_amount": quote_amount,
        "market_price": market_price,
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "api_latency_ms": metrics["api_latency_ms"],
        "ticker_age_ms": metrics["ticker_age_ms"],
        "spread_bps": metrics["spread_bps"],
        "guard_ok": guard_ok,
        "guard_message": guard_message,
        "minimum_ok": minimum_ok,
        "minimum_message": minimum_message,
        "min_qty": market_reqs["min_qty"],
        "min_notional": market_reqs["min_notional"],
        "qty_precision": market_reqs["qty_precision"],
        "price_precision": market_reqs["price_precision"],
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
    min_confidence: float = 0.10,
) -> dict[str, float | str]:
    strength = confidence_strength(signal, probability_up)
    if strength < min_confidence or market_price <= 0 or max_trade_size_quote <= 0:
        return {"status": "blocked", "strength": strength, "quote_size": 0.0, "base_size": 0.0, "allocation_pct": 0.0}
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
    notional = max(0.0, order_size * market_price)
    if notional <= 0:
        return {"status": "invalid_order_notional", "goal_profit": goal_profit, "expected_profit_per_cycle": 0.0, "expected_return_per_cycle": 0.0, "recommended_cycles": 0, "win_probability": 0.5}
    prob_up = clamp_value(float(probability_up), 0.0, 1.0)
    if str(signal).upper() == "BUY":
        win_probability = prob_up
    elif str(signal).upper() == "SELL":
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


def llm_support_chat(message: str, context: dict[str, Any]) -> dict[str, Any]:
    if not LLM_ENABLED:
        return {"status": "disabled", "answer": "LLM is disabled. Enable LLM_ENABLED=true in environment."}
    if not LLM_API_KEY:
        return {"status": "missing_api_key", "answer": "LLM_API_KEY is missing."}
    if LLM_PROVIDER in {"deepseek", "openai_compatible"}:
        endpoint = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    elif LLM_PROVIDER == "huggingface_inference":
        endpoint = f"{LLM_BASE_URL.rstrip('/')}/models/{LLM_MODEL}"
    else:
        return {"status": "unsupported_provider", "answer": f"Unsupported LLM provider: {LLM_PROVIDER}"}
    system_prompt = (
        "You are a trading support assistant for a live dashboard. "
        "Be concise, practical, and risk-aware. "
        "Explain whether ML and LLM are merged using provided context and suggest safe next actions."
    )
    user_prompt = f"User question: {message}\nCurrent engine context: {json.dumps(context, ensure_ascii=True)}"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    try:
        if LLM_PROVIDER == "huggingface_inference":
            payload: dict[str, Any] = {
                "inputs": f"{system_prompt}\n{user_prompt}",
                "parameters": {"max_new_tokens": 350, "temperature": 0.2, "return_full_text": False},
            }
        else:
            payload = {
                "model": LLM_MODEL,
                "temperature": 0.2,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            }
        with requests.Session() as session:
            if LLM_BYPASS_ENV_PROXY:
                session.trust_env = False
            response = session.post(endpoint, headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
        if LLM_PROVIDER == "huggingface_inference":
            if isinstance(data, list) and data:
                answer = str(data[0].get("generated_text", ""))
            elif isinstance(data, dict):
                answer = str(data.get("generated_text", ""))
            else:
                answer = str(data)
        else:
            answer = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        if not answer.strip():
            answer = "No response content from LLM."
        return {"status": "ok", "answer": answer, "endpoint": endpoint, "provider": LLM_PROVIDER, "model": LLM_MODEL}
    except Exception as exc:
        return {"status": "error", "answer": f"LLM support chat error: {exc}", "endpoint": endpoint, "provider": LLM_PROVIDER, "model": LLM_MODEL}


def get_decision_payload(
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    adaptive_threshold_enabled: bool = ADAPTIVE_THRESHOLD_ENABLED,
) -> dict[str, Any]:
    model_path = resolve_model_path()
    dataset_path = DATASET_PATH
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
        "engine": decision.get("decision_engine", "ml_primary"),
        "signal": decision.get("signal"),
        "probability_up": decision.get("probability_up"),
        "buy_threshold": decision.get("buy_threshold"),
        "sell_threshold": decision.get("sell_threshold"),
        "ml_signal": decision.get("ml_signal"),
        "ml_probability_up": decision.get("ml_probability_up"),
        "llm_overlay": decision.get("llm_overlay"),
        "llm_merge": decision.get("llm_merge"),
        "decision": decision,
        "feature_row": dataframe_to_records(feature_row),
        "latest_row": latest_row.to_dict(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
    }


def get_dashboard_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
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
    )
    cycle_plan = estimate_cycles_to_goal(
        current_value=float((wallet_snapshot or {}).get("estimated_total_usdt", plan["deposit_amount"])),
        goal_value=float((wallet_snapshot or {}).get("estimated_total_usdt", plan["deposit_amount"])) * 1.25,
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
        "decision_payload": decision_payload,
        "risk_plan": plan,
        "wallet_snapshot": wallet_snapshot,
        "wallet_error": wallet_error,
        "account_snapshot": account_snapshot,
        "account_error": account_error,
        "market_scan": market_scan,
        "market_scan_error": market_scan_error,
        "size_plan": size_plan,
        "cycle_plan": cycle_plan,
        "autopilot": autopilot_snapshot(),
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


def append_autopilot_log(entry: dict[str, Any]) -> None:
    with AUTOPILOT_LOCK:
        logs = list(AUTOPILOT_STATE.get("logs", []))
        logs.append(entry)
        AUTOPILOT_STATE["logs"] = logs[-200:]
        AUTOPILOT_STATE["updated_at"] = utc_now_iso()


def should_stop_autopilot() -> bool:
    with AUTOPILOT_LOCK:
        return bool(AUTOPILOT_STATE.get("stop_requested", False))


def maybe_execute_signal_trade(
    signal: str,
    symbol: str,
    order_size: float,
    allow_live: bool,
    max_api_latency_ms: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
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
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
    )


def run_autopilot(config: dict[str, Any]) -> None:
    try:
        symbol = normalize_symbol(str(config.get("symbol", CHECK_SYMBOL)))
        interval_seconds = max(1, int(config.get("interval_seconds", 15)))
        requested_cycles = max(0, int(config.get("cycles", 1)))
        decision_payload = get_decision_payload(
            buy_threshold=float(config.get("buy_threshold", BUY_THRESHOLD)),
            sell_threshold=float(config.get("sell_threshold", SELL_THRESHOLD)),
            adaptive_threshold_enabled=bool(config.get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
        )
        initial_decision = decision_payload["decision"]
        current_value = float(config.get("current_value", 0.0) or 0.0)
        if current_value <= 0:
            current_value = float(get_account_snapshot(symbol).get("account_value_quote", 0.0) or 0.0)
        market_price = float(get_market_price(symbol))
        cycle_plan = estimate_cycles_to_goal(
            current_value=current_value,
            goal_value=float(config.get("goal_value", current_value)),
            signal=str(initial_decision.get("signal", "HOLD")),
            probability_up=float(initial_decision.get("probability_up", 0.5)),
            order_size=float(config.get("order_size", 0.0)),
            market_price=market_price,
            take_profit_pct=float(config.get("take_profit_pct", 0.05)),
            stop_loss_pct=float(config.get("stop_loss_pct", 0.03)),
            fee_drag_pct=float(config.get("fee_drag_pct", 0.0)),
        )
        effective_cycles = requested_cycles
        if bool(config.get("auto_cycles_enabled", False)):
            if cycle_plan.get("status") == "ok":
                effective_cycles = int(cycle_plan.get("recommended_cycles", requested_cycles))
            elif cycle_plan.get("status") == "goal_already_reached":
                effective_cycles = 0
        update_autopilot_state(
            running=True,
            status="running",
            stop_requested=False,
            symbol=symbol,
            current_cycle=0,
            target_cycles=effective_cycles,
            cycle_plan=cycle_plan,
            latest_decision=initial_decision,
            latest_trade_result={},
            last_error="",
        )
        if effective_cycles <= 0:
            update_autopilot_state(running=False, status="completed")
            return
        for cycle_index in range(effective_cycles):
            if should_stop_autopilot():
                update_autopilot_state(running=False, status="cancelled", stop_requested=False)
                return
            decision_payload = get_decision_payload(
                buy_threshold=float(config.get("buy_threshold", BUY_THRESHOLD)),
                sell_threshold=float(config.get("sell_threshold", SELL_THRESHOLD)),
                adaptive_threshold_enabled=bool(config.get("adaptive_threshold_enabled", ADAPTIVE_THRESHOLD_ENABLED)),
            )
            decision = decision_payload["decision"]
            signal = str(decision.get("signal", "HOLD"))
            probability_up = float(decision.get("probability_up", 0.5))
            snapshot = get_account_snapshot(symbol)
            order_size = float(config.get("order_size", 0.0))
            size_plan = {
                "status": "manual",
                "strength": confidence_strength(signal, probability_up),
                "quote_size": order_size * float(get_market_price(symbol)),
                "base_size": order_size,
                "allocation_pct": 0.0,
            }
            if bool(config.get("auto_size_enabled", False)):
                size_plan = auto_order_size_from_confidence(
                    signal=signal,
                    probability_up=probability_up,
                    market_price=float(snapshot.get("best_ask") or snapshot.get("best_bid") or get_market_price(symbol)),
                    max_trade_size_quote=float(config.get("max_trade_size_quote", 0.0)),
                )
                order_size = float(size_plan.get("base_size", 0.0))
            append_live_history(symbol, signal, probability_up)
            trade_result = maybe_execute_signal_trade(
                signal=signal,
                symbol=symbol,
                order_size=order_size,
                allow_live=bool(config.get("allow_live", False)),
                max_api_latency_ms=int(config.get("max_api_latency_ms", 1200)),
                max_ticker_age_ms=int(config.get("max_ticker_age_ms", 3000)),
                max_spread_bps=float(config.get("max_spread_bps", 20.0)),
                min_trade_cooldown_seconds=int(config.get("min_trade_cooldown_seconds", 5)),
            )
            trade_action = "market_buy" if signal == "BUY" else "market_sell" if signal == "SELL" else "hold"
            log_trade_memory(
                source="autopilot",
                symbol=symbol,
                action=trade_action,
                payload=trade_result,
                signal=signal,
                probability_up=probability_up,
                cycle=cycle_index + 1,
                target_cycles=effective_cycles,
                size_plan=size_plan,
                account_snapshot=snapshot,
            )
            append_autopilot_log(
                {
                    "cycle": cycle_index + 1,
                    "target_cycles": effective_cycles,
                    "timestamp": utc_now_iso(),
                    "symbol": symbol,
                    "signal": signal,
                    "probability_up": probability_up,
                    "bid": snapshot.get("best_bid"),
                    "ask": snapshot.get("best_ask"),
                    "account_value": snapshot.get("account_value_quote"),
                    "order_size": order_size,
                    "size_strength": size_plan.get("strength"),
                    "quote_size": size_plan.get("quote_size"),
                    "allocation_pct": size_plan.get("allocation_pct"),
                    "trade_status": trade_result.get("status"),
                    "trade_message": trade_result.get("message", ""),
                    "api_latency_ms": trade_result.get("api_latency_ms", snapshot.get("api_latency_ms")),
                    "ticker_age_ms": trade_result.get("ticker_age_ms", snapshot.get("ticker_age_ms")),
                    "spread_bps": trade_result.get("spread_bps", snapshot.get("spread_bps")),
                    "guard": trade_result.get("guard_message", ""),
                }
            )
            update_autopilot_state(current_cycle=cycle_index + 1, latest_decision=decision, latest_trade_result=trade_result)
            if cycle_index + 1 < effective_cycles:
                for _ in range(interval_seconds * 10):
                    if should_stop_autopilot():
                        update_autopilot_state(running=False, status="cancelled", stop_requested=False)
                        return
                    time.sleep(0.1)
        update_autopilot_state(running=False, status="completed", stop_requested=False)
    except Exception as exc:
        append_autopilot_log({"cycle": int(autopilot_snapshot().get("current_cycle", 0)) + 1, "timestamp": utc_now_iso(), "error": str(exc)})
        update_autopilot_state(running=False, status="error", stop_requested=False, last_error=str(exc))


def start_autopilot(config: dict[str, Any]) -> dict[str, Any]:
    global AUTOPILOT_THREAD
    if bool(autopilot_snapshot().get("running")):
        raise RuntimeError("Autopilot is already running.")
    update_autopilot_state(
        running=True,
        status="starting",
        stop_requested=False,
        current_cycle=0,
        target_cycles=int(config.get("cycles", 1)),
        symbol=normalize_symbol(str(config.get("symbol", CHECK_SYMBOL))),
        started_at=utc_now_iso(),
        updated_at=utc_now_iso(),
        logs=[],
        latest_decision={},
        latest_trade_result={},
        cycle_plan={},
        last_error="",
    )
    AUTOPILOT_THREAD = threading.Thread(target=run_autopilot, args=(config,), daemon=True)
    AUTOPILOT_THREAD.start()
    return autopilot_snapshot()


def stop_autopilot() -> dict[str, Any]:
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
