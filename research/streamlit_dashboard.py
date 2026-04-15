"""Streamlit dashboard for model-driven trading planning.

This dashboard helps you:
- inspect the latest trained model and dataset
- set a deposit amount and risk limits
- see the current BUY/HOLD/SELL signal
- estimate position size and cash reserves
- define when to stop trading or withdraw funds
- execute Binance spot orders with safeguards
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(PROJECT_ROOT / ".env")

from generate_trade_signal import load_latest_row, latest_file_with_suffix, load_model  # noqa: E402

ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data_sets"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
CHECK_SYMBOL = os.getenv("CHECK_SYMBOL", "BTC/USDT")

st.set_page_config(page_title="Trading Plan Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
    :root {
        --binance-yellow: #f0b90b;
        --binance-dark: #0b0e11;
        --binance-panel: #181a20;
        --binance-border: #2b3139;
        --binance-text: #eaecef;
        --binance-soft: #848e9c;
        --binance-green: #0ecb81;
        --binance-red: #f6465d;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, #1b2028 0%, #0b0e11 55%);
        color: var(--binance-text);
    }

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--binance-border);
        border-radius: 12px;
        padding: 10px 12px;
        background: linear-gradient(180deg, #1f232a, #171b21);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--binance-soft);
    }

    div[data-testid="stMetricValue"] {
        color: var(--binance-text);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: 600;
        border: 1px solid var(--binance-border);
        background: #11161c;
        color: var(--binance-soft);
    }

    .stTabs [aria-selected="true"] {
        color: #111 !important;
        background: var(--binance-yellow) !important;
        border-color: var(--binance-yellow) !important;
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 10px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        margin-right: 8px;
        border: 1px solid transparent;
    }

    .pill-buy {
        color: #052e22;
        background: rgba(14, 203, 129, 0.88);
    }

    .pill-sell {
        color: #3c0b13;
        background: rgba(246, 70, 93, 0.88);
    }

    .pill-hold {
        color: #3a2b06;
        background: rgba(240, 185, 11, 0.9);
    }

    .market-strip {
        border: 1px solid var(--binance-border);
        background: linear-gradient(180deg, #1b2027, #151920);
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    .market-strip strong {
        color: var(--binance-yellow);
    }

    div[data-testid="stSidebar"] {
        background: #0f1319;
        border-right: 1px solid var(--binance-border);
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.6rem;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
            padding-bottom: 2rem;
        }

        h1 {
            font-size: 1.45rem !important;
            line-height: 1.25;
        }

        h2 {
            font-size: 1.05rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            flex-wrap: nowrap;
            white-space: nowrap;
            padding-bottom: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px;
            font-size: 0.82rem;
        }

        div[data-testid="stMetric"] {
            border-radius: 10px;
            padding: 8px 10px;
        }

        .market-strip {
            padding: 8px 10px;
            font-size: 0.9rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


def normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return "BTC/USDT"
    if "/" in symbol:
        return symbol
    return f"{symbol}/USDT"


def split_symbol_assets(symbol: str) -> tuple[str, str]:
    symbol = normalize_symbol(symbol)
    base, quote = symbol.split("/", 1)
    return base, quote


def get_binance_client():
    try:
        import ccxt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'ccxt'. Install requirements in your project venv.") from exc

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")

    exchange = ccxt.binance(
        {
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "enableRateLimit": True,
        }
    )
    exchange.set_sandbox_mode(BINANCE_TESTNET)
    return exchange


def get_ticker_with_metrics(symbol: str) -> tuple[dict[str, Any], dict[str, float]]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    start = time.monotonic()
    ticker = exchange.fetch_ticker(norm_symbol)
    api_latency_ms = (time.monotonic() - start) * 1000

    now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)
    ticker_ts = ticker.get("timestamp")
    ticker_age_ms = float(max(0, now_ms - int(ticker_ts))) if ticker_ts else 0.0

    bid = float(ticker.get("bid") or ticker.get("last") or 0)
    ask = float(ticker.get("ask") or ticker.get("last") or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(ticker.get("last") or 0)
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
    last_trade_ts = float(st.session_state.get("last_trade_ts", 0.0) or 0.0)
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
        bid = float(ticker.get("bid") or 0)
        ask = float(ticker.get("ask") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        raise RuntimeError("Ticker has no usable price")
    except Exception:
        fallback = st.session_state.get("latest_price_fallback")
        if fallback is not None:
            return float(fallback)
        raise RuntimeError("Unable to fetch market price from Binance and no fallback price is available.")


def get_account_snapshot(symbol: str) -> dict[str, Any]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    base, quote = split_symbol_assets(norm_symbol)
    ticker, metrics = get_ticker_with_metrics(norm_symbol)
    bid = float(ticker.get("bid") or ticker.get("last") or 0)
    ask = float(ticker.get("ask") or ticker.get("last") or 0)
    mark = float(ticker.get("last") or ((bid + ask) / 2 if bid and ask else 0))

    balance = exchange.fetch_balance()
    open_orders = exchange.fetch_open_orders(norm_symbol)

    base_total = float(balance.get("total", {}).get(base, 0) or 0)
    quote_total = float(balance.get("total", {}).get(quote, 0) or 0)
    quote_free = float(balance.get("free", {}).get(quote, 0) or 0)
    quote_used = float(balance.get("used", {}).get(quote, 0) or 0)
    account_value_quote = quote_total + (base_total * mark)

    positions = []
    if base_total > 0:
        positions.append(
            {
                "coin": base,
                "size": base_total,
                "entry_px": 0.0,
                "roe_pct": 0.0,
            }
        )

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


def get_market_requirements(symbol: str) -> dict[str, float]:
    exchange = get_binance_client()
    norm_symbol = normalize_symbol(symbol)
    exchange.load_markets()
    market = exchange.market(norm_symbol)

    min_qty = float((market.get("limits", {}).get("amount", {}).get("min") or 0) or 0)
    min_notional = float((market.get("limits", {}).get("cost", {}).get("min") or 0) or 0)
    qty_precision = int(market.get("precision", {}).get("amount") or 8)
    price_precision = int(market.get("precision", {}).get("price") or 8)

    return {
        "min_qty": min_qty,
        "min_notional": min_notional,
        "qty_precision": qty_precision,
        "price_precision": price_precision,
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


def get_wallet_snapshot() -> dict[str, Any]:
    exchange = get_binance_client()
    balance = exchange.fetch_balance()
    totals = balance.get("total", {})
    free_map = balance.get("free", {})
    used_map = balance.get("used", {})

    non_zero_assets = [asset for asset, total in totals.items() if float(total or 0) > 0]

    # Load markets once so we can estimate values in USDT where pairs exist.
    exchange.load_markets()

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
            if pair in exchange.markets:
                try:
                    t = exchange.fetch_ticker(pair)
                    px = float(t.get("last") or 0)
                    est_usdt = total * px
                except Exception:
                    est_usdt = 0.0

        total_estimated_usdt += est_usdt
        rows.append(
            {
                "asset": asset,
                "free": free,
                "used": used,
                "total": total,
                "est_usdt": est_usdt,
            }
        )

    wallet_df = pd.DataFrame(rows)
    if not wallet_df.empty:
        wallet_df = wallet_df.sort_values("est_usdt", ascending=False)

    return {
        "exchange": "binance",
        "testnet": BINANCE_TESTNET,
        "asset_count": len(non_zero_assets),
        "estimated_total_usdt": total_estimated_usdt,
        "balances": wallet_df,
    }


def build_market_scan(
    max_symbols: int,
    quote_asset: str,
) -> pd.DataFrame:
    exchange = get_binance_client()
    exchange.load_markets()
    quote_asset = (quote_asset or "USDT").strip().upper()

    universe = [
        symbol
        for symbol, market in exchange.markets.items()
        if market.get("spot")
        and market.get("active")
        and symbol.endswith(f"/{quote_asset}")
        and ":" not in symbol
    ]
    universe = sorted(universe)[: max(1, int(max_symbols))]
    if not universe:
        return pd.DataFrame()

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
        t = tickers.get(symbol)
        if not t:
            continue
        last = float(t.get("last") or 0)
        bid = float(t.get("bid") or last or 0)
        ask = float(t.get("ask") or last or 0)
        pct = float(t.get("percentage") or 0)
        qv = float(t.get("quoteVolume") or 0)
        spread_bps = ((ask - bid) / last * 10000) if last > 0 and ask >= bid else 0.0
        ai_score = (abs(pct) * 1.2) + (min(qv / 1_000_000, 100.0) * 0.15) - (spread_bps * 0.05)
        ai_bias = "BUY" if pct >= 0 else "SELL"
        rows.append(
            {
                "symbol": symbol,
                "last": last,
                "change_pct": pct,
                "quote_volume": qv,
                "spread_bps": spread_bps,
                "ai_bias": ai_bias,
                "ai_score": ai_score,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("ai_score", ascending=False).reset_index(drop=True)


def refresh_market_scan(
    enabled: bool,
    max_symbols: int,
    quote_asset: str,
    min_interval_seconds: int,
) -> Optional[pd.DataFrame]:
    if not enabled:
        return None
    now = time.time()
    last_scan = float(st.session_state.get("last_market_scan_ts", 0.0) or 0.0)
    if now - last_scan < max(1, int(min_interval_seconds)):
        return st.session_state.get("market_scan_df")

    df = build_market_scan(max_symbols=max_symbols, quote_asset=quote_asset)
    st.session_state["market_scan_df"] = df
    st.session_state["last_market_scan_ts"] = now
    return df


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
        payload["message"] = "No order sent. Disable dry-run and confirm to execute on Binance."
        return payload

    if not guard_ok:
        payload["status"] = "blocked"
        payload["message"] = guard_message
        return payload

    minimum_ok, minimum_message = validate_order_minimums(
        action=action,
        quantity=quantity,
        quote_amount=quote_amount,
        market_price=market_price,
        min_qty=market_reqs["min_qty"],
        min_notional=market_reqs["min_notional"],
    )
    payload["minimum_ok"] = minimum_ok
    payload["minimum_message"] = minimum_message
    if not minimum_ok:
        payload["status"] = "blocked"
        payload["message"] = minimum_message
        return payload

    if action == "market_buy":
        order_qty = quantity
        if order_qty <= 0 and quote_amount > 0:
            # Prefer quote-based buy to spend an exact quote amount (useful for tiny balance tests like 10 BRL).
            try:
                result = exchange.create_market_buy_order_with_cost(norm_symbol, quote_amount)
            except Exception:
                result = exchange.create_order(
                    norm_symbol,
                    "market",
                    "buy",
                    None,
                    None,
                    {"quoteOrderQty": quote_amount},
                )
        else:
            if order_qty <= 0:
                raise ValueError("Provide quantity or quote amount for market_buy.")
            order_qty = float(exchange.amount_to_precision(norm_symbol, order_qty))
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
    st.session_state["last_trade_ts"] = time.time()
    return payload


def apply_ai_command(command: str) -> str:
    cmd = command.strip().lower()

    if not cmd:
        return "No command entered."

    if cmd in {"status", "account status", "show account"}:
        return "Use the Refresh Account Snapshot button in the account section."

    deposit_match = re.match(r"set\s+deposit\s+(\d+(?:\.\d+)?)", cmd)
    if deposit_match:
        st.session_state["_pending_widget_updates"] = {
            "deposit_amount": float(deposit_match.group(1))
        }
        st.session_state["_ai_command_feedback"] = f"Deposit amount set to {float(deposit_match.group(1)):.2f}."
        st.rerun()

    buy_match = re.match(r"set\s+buy\s+threshold\s+(0(?:\.\d+)?|1(?:\.0+)?)", cmd)
    if buy_match:
        st.session_state["_pending_widget_updates"] = {
            "buy_threshold": float(buy_match.group(1))
        }
        st.session_state["_ai_command_feedback"] = f"Buy threshold set to {float(buy_match.group(1)):.2f}."
        st.rerun()

    sell_match = re.match(r"set\s+sell\s+threshold\s+(0(?:\.\d+)?|1(?:\.0+)?)", cmd)
    if sell_match:
        st.session_state["_pending_widget_updates"] = {
            "sell_threshold": float(sell_match.group(1))
        }
        st.session_state["_ai_command_feedback"] = f"Sell threshold set to {float(sell_match.group(1)):.2f}."
        st.rerun()

    if cmd in {"recommend", "recommend action", "ai recommend"}:
        signal = st.session_state.get("signal", "HOLD")
        probability_up = st.session_state.get("probability_up", 0.5)
        if signal == "BUY":
            return f"AI recommendation: BUY (probability_up={probability_up:.1%}) with your configured risk limits."
        if signal == "SELL":
            return f"AI recommendation: SELL or reduce exposure (probability_up={probability_up:.1%})."
        return f"AI recommendation: HOLD and wait for a stronger edge (probability_up={probability_up:.1%})."

    return "Command not recognized. Try: 'set deposit 1000', 'set buy threshold 0.60', 'set sell threshold 0.40', 'recommend'."


def apply_pending_widget_updates() -> None:
    pending = st.session_state.pop("_pending_widget_updates", None)
    if not pending:
        return
    for key, value in pending.items():
        st.session_state[key] = value


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
    history = st.session_state.setdefault("live_history", [])
    history.append(point)
    st.session_state["live_history"] = history[-500:]
    return point


def refresh_live_api_state(
    symbol: str,
    signal: str,
    probability_up: float,
    min_interval_seconds: int,
) -> Optional[dict[str, Any]]:
    now = time.time()
    last_refresh = float(st.session_state.get("last_live_refresh_ts", 0.0) or 0.0)
    if now - last_refresh < max(1, int(min_interval_seconds)):
        return None

    point = append_live_history(symbol, signal, probability_up)
    st.session_state["latest_live_point"] = point
    st.session_state["last_live_refresh_ts"] = now
    return point


def maybe_execute_signal_trade(
    signal: str,
    symbol: str,
    order_size: float,
    allow_live: bool,
    confirmation_text: str,
    max_api_latency_ms: int,
    max_ticker_age_ms: int,
    max_spread_bps: float,
    min_trade_cooldown_seconds: int,
) -> dict[str, Any]:
    if signal not in {"BUY", "SELL"}:
        return {
            "status": "no_trade",
            "message": "Signal is HOLD, no trade action sent.",
            "signal": signal,
        }

    if signal == "BUY":
        action = "market_buy"
    else:
        action = "market_sell"

    dry_run = True
    if allow_live and confirmation_text.strip().upper() == "AUTO":
        dry_run = False

    return execute_account_action(
        action=action,
        symbol=symbol,
        quantity=order_size,
        quote_amount=0.0,
        dry_run=dry_run,
        max_api_latency_ms=max_api_latency_ms,
        max_ticker_age_ms=max_ticker_age_ms,
        max_spread_bps=max_spread_bps,
        min_trade_cooldown_seconds=min_trade_cooldown_seconds,
    )


def resolve_latest_model() -> Path:
    latest = latest_file_with_suffix(ARTIFACTS_DIR, "_model.joblib")
    if latest is None:
        raise FileNotFoundError("No model artifact found in research/artifacts")
    return latest


def resolve_dataset_path() -> Path:
    preferred = DATA_DIR / "aapl_training_dataset.csv"
    if preferred.exists():
        return preferred

    latest_dataset = latest_file_with_suffix(DATA_DIR, "_training_dataset.csv")
    if latest_dataset is None:
        raise FileNotFoundError("No combined dataset found in research/data_sets")
    return latest_dataset


def compute_signal(model_path: Path, dataset_path: Path, buy_threshold: float, sell_threshold: float) -> Tuple[str, float, pd.Series, pd.DataFrame]:
    model = load_model(model_path)
    features, full_row = load_latest_row(dataset_path)
    probability = float(model.predict_proba(features)[0][1])
    if probability >= buy_threshold:
        signal = "BUY"
    elif probability <= sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, probability, full_row, features


def risk_plan(deposit_amount: float, active_capital_pct: float, reserve_pct: float, max_trade_pct: float, stop_loss_pct: float, take_profit_pct: float, max_daily_loss_pct: float, max_drawdown_pct: float, withdrawal_target_pct: float) -> dict:
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


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


for key, default in {
    "buy_threshold": 0.55,
    "sell_threshold": 0.45,
    "deposit_amount": 1000.0,
    "active_capital_pct": 0.70,
    "reserve_pct": 0.30,
    "max_trade_pct": 0.10,
    "stop_loss_pct": 0.03,
    "take_profit_pct": 0.05,
    "max_daily_loss_pct": 0.05,
    "max_drawdown_pct": 0.15,
    "withdrawal_target_pct": 0.25,
    "live_symbol": CHECK_SYMBOL,
    "autopilot_interval_seconds": 15,
    "autopilot_cycles": 5,
    "autopilot_order_size": 0.001,
    "autopilot_live_enabled": False,
    "autopilot_confirmation": "",
    "max_api_latency_ms": 1200,
    "max_ticker_age_ms": 3000,
    "max_spread_bps": 20.0,
    "min_trade_cooldown_seconds": 5,
    "auto_refresh_enabled": True,
    "auto_refresh_interval_seconds": 15,
    "market_scan_enabled": True,
    "market_scan_max_symbols": 60,
    "market_scan_quote_asset": "USDT",
    "market_scan_interval_seconds": 30,
    "market_scan_auto_pick_symbol": True,
}.items():
    st.session_state.setdefault(key, default)

apply_pending_widget_updates()


st.title("Trading Plan Dashboard")
st.write("Use this dashboard to manage risk, monitor live AI trading signals, and run an autopilot cycle with safety checks.")

try:
    import joblib  # noqa: F401
except ModuleNotFoundError:
    st.error(
        "Missing dependency 'joblib' in this Streamlit environment. "
        "Launch with project venv: ./.venv/bin/python -m streamlit run research/streamlit_dashboard.py"
    )
    st.stop()

with st.sidebar:
    st.header("Model Inputs")
    model_path = resolve_latest_model()
    dataset_path = resolve_dataset_path()

    st.caption(f"Execution exchange: Binance ({'testnet' if BINANCE_TESTNET else 'live'})")

    buy_threshold = st.slider("Buy threshold", 0.50, 0.90, step=0.01, key="buy_threshold")
    sell_threshold = st.slider("Sell threshold", 0.10, 0.50, step=0.01, key="sell_threshold")

    st.divider()
    st.header("Delay Safeguards")
    max_api_latency_ms = st.number_input("Max API latency (ms)", min_value=100, max_value=10000, step=100, key="max_api_latency_ms")
    max_ticker_age_ms = st.number_input("Max ticker age (ms)", min_value=100, max_value=15000, step=100, key="max_ticker_age_ms")
    max_spread_bps = st.number_input("Max spread (bps)", min_value=1.0, max_value=300.0, step=1.0, key="max_spread_bps")
    min_trade_cooldown_seconds = st.number_input("Min cooldown between trades (s)", min_value=0, max_value=120, step=1, key="min_trade_cooldown_seconds")

    st.divider()
    st.header("Live API Refresh")
    auto_refresh_enabled = st.checkbox("Always live refresh", key="auto_refresh_enabled")
    auto_refresh_interval_seconds = st.number_input(
        "Refresh interval (seconds)",
        min_value=3,
        max_value=120,
        step=1,
        key="auto_refresh_interval_seconds",
    )

    st.divider()
    st.header("Market Capture")
    market_scan_enabled = st.checkbox("Scan whole market", key="market_scan_enabled")
    market_scan_quote_asset = st.text_input("Quote asset", key="market_scan_quote_asset")
    market_scan_max_symbols = st.number_input(
        "Max symbols in scan",
        min_value=10,
        max_value=400,
        step=10,
        key="market_scan_max_symbols",
    )
    market_scan_interval_seconds = st.number_input(
        "Scan refresh interval (seconds)",
        min_value=5,
        max_value=300,
        step=5,
        key="market_scan_interval_seconds",
    )
    market_scan_auto_pick_symbol = st.checkbox(
        "Auto-pick top symbol for live/autopilot",
        key="market_scan_auto_pick_symbol",
    )

    st.divider()
    st.header("Capital Planning")
    deposit_amount = st.number_input("Deposit amount", min_value=0.0, step=100.0, key="deposit_amount")
    active_capital_pct = st.slider("Capital deployed into trading", 0.10, 1.00, step=0.05, key="active_capital_pct")
    reserve_pct = st.slider("Cash reserve", 0.00, 0.90, step=0.05, key="reserve_pct")
    max_trade_pct = st.slider("Max size per trade", 0.01, 0.50, step=0.01, key="max_trade_pct")
    stop_loss_pct = st.slider("Stop loss per trade", 0.01, 0.20, step=0.01, key="stop_loss_pct")
    take_profit_pct = st.slider("Take profit per trade", 0.01, 0.50, step=0.01, key="take_profit_pct")
    max_daily_loss_pct = st.slider("Max daily loss", 0.01, 0.20, step=0.01, key="max_daily_loss_pct")
    max_drawdown_pct = st.slider("Max drawdown", 0.05, 0.50, step=0.01, key="max_drawdown_pct")
    withdrawal_target_pct = st.slider("Withdraw when account grows by", 0.05, 1.00, step=0.05, key="withdrawal_target_pct")

signal, probability_up, latest_row, feature_row = compute_signal(model_path, dataset_path, buy_threshold, sell_threshold)
st.session_state["signal"] = signal
st.session_state["probability_up"] = probability_up
st.session_state["latest_price_fallback"] = latest_row.get("close", None)
plan = risk_plan(
    deposit_amount=deposit_amount,
    active_capital_pct=active_capital_pct,
    reserve_pct=reserve_pct,
    max_trade_pct=max_trade_pct,
    stop_loss_pct=stop_loss_pct,
    take_profit_pct=take_profit_pct,
    max_daily_loss_pct=max_daily_loss_pct,
    max_drawdown_pct=max_drawdown_pct,
    withdrawal_target_pct=withdrawal_target_pct,
)

wallet_total_usdt: float | None = None
wallet_error: str | None = None
try:
    wallet_snapshot_head = get_wallet_snapshot()
    wallet_total_usdt = float(wallet_snapshot_head.get("estimated_total_usdt", 0.0) or 0.0)
except Exception as exc:
    wallet_error = str(exc)

primary_snapshot: dict[str, Any] | None = None
try:
    primary_snapshot = get_account_snapshot(st.session_state.get("live_symbol", CHECK_SYMBOL))
except Exception:
    primary_snapshot = None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Signal", signal)
col2.metric("Up probability", f"{probability_up:.1%}")
col3.metric("Deposit", format_currency(plan["deposit_amount"]))
col4.metric("Trade budget", format_currency(plan["active_capital"]))
col5.metric("Binance Money", format_currency(wallet_total_usdt or 0.0))

if signal == "BUY":
    signal_pill_class = "pill-buy"
elif signal == "SELL":
    signal_pill_class = "pill-sell"
else:
    signal_pill_class = "pill-hold"

strip_parts = [
    f"<span class='status-pill {signal_pill_class}'>{signal}</span>",
    f"<span><strong>AI Up Prob:</strong> {probability_up:.1%}</span>",
]

if primary_snapshot:
    strip_parts.extend(
        [
            f"<span style='margin-left:12px;'><strong>Pair:</strong> {primary_snapshot.get('symbol', '-')}</span>",
            f"<span style='margin-left:12px;'><strong>Bid/Ask:</strong> {primary_snapshot.get('best_bid', 0):,.4f} / {primary_snapshot.get('best_ask', 0):,.4f}</span>",
            f"<span style='margin-left:12px;'><strong>Spread:</strong> {primary_snapshot.get('spread_bps', 0):.2f} bps</span>",
            f"<span style='margin-left:12px;'><strong>Latency:</strong> {primary_snapshot.get('api_latency_ms', 0):.0f} ms</span>",
        ]
    )

st.markdown(f"<div class='market-strip'>{' '.join(strip_parts)}</div>", unsafe_allow_html=True)

if wallet_error:
    st.caption(f"Wallet estimation unavailable: {wallet_error}")

st.subheader("Current Decision")
if signal == "BUY":
    st.success("Model suggests taking risk only if your limits are acceptable.")
elif signal == "SELL":
    st.error("Model suggests staying defensive or reducing exposure.")
else:
    st.warning("Model is neutral. Waiting is usually the lower-risk choice.")

overview_tab, live_tab, wallet_tab, account_tab, ai_tab = st.tabs([
    "Market Overview",
    "Live Terminal",
    "My Wallet",
    "Trade Ticket",
    "AI Commands",
])

with overview_tab:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Capital and Risk Limits")
        st.write(
            {
                "active_capital": format_currency(plan["active_capital"]),
                "reserve_cash": format_currency(plan["reserve_cash"]),
                "max_trade_size": format_currency(plan["max_trade_size"]),
                "daily_loss_limit": format_currency(plan["daily_loss_limit"]),
                "drawdown_limit": format_currency(plan["drawdown_limit"]),
                "withdrawal_target": format_currency(plan["withdrawal_target"]),
                "stop_loss_pct": f"{plan['stop_loss_pct']:.0%}",
                "take_profit_pct": f"{plan['take_profit_pct']:.0%}",
            }
        )

        st.subheader("Trading Rules")
        rules = [
            f"Do not exceed {format_currency(plan['max_trade_size'])} on one position.",
            f"Stop trading for the day if losses reach {format_currency(plan['daily_loss_limit'])}.",
            f"Stop trading entirely if drawdown reaches {format_currency(plan['drawdown_limit'])}.",
            f"Consider withdrawing profit once equity reaches {format_currency(plan['withdrawal_target'])}.",
            f"Use a stop loss of {plan['stop_loss_pct']:.0%} and take profit of {plan['take_profit_pct']:.0%}.",
        ]
        for rule in rules:
            st.markdown(f"- {rule}")

    with right:
        st.subheader("Latest Model Row")
        st.dataframe(latest_row.to_frame().T, use_container_width=True)

        st.subheader("Latest Feature Snapshot")
        st.dataframe(feature_row, use_container_width=True)

st.subheader("Model / Dataset Info")
st.write(
    {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "latest_timestamp": str(latest_row.get("timestamp", "")),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
    }
)

st.info(
    "This dashboard is configured for Binance spot trading with dry-run protection."
)
st.caption(
    f"Current delay guardrails: latency <= {int(max_api_latency_ms)}ms, ticker age <= {int(max_ticker_age_ms)}ms, spread <= {float(max_spread_bps):.1f} bps, cooldown >= {int(min_trade_cooldown_seconds)}s"
)

live_refresh_status = None
if auto_refresh_enabled:
    try:
        live_refresh_status = refresh_live_api_state(
            symbol=st.session_state.get("live_symbol", CHECK_SYMBOL),
            signal=st.session_state.get("signal", signal),
            probability_up=st.session_state.get("probability_up", probability_up),
            min_interval_seconds=int(auto_refresh_interval_seconds),
        )
        if live_refresh_status is not None:
            st.caption(
                f"Live API refreshed at {live_refresh_status['timestamp']} for {live_refresh_status['symbol']}"
            )
    except Exception as refresh_exc:
        st.caption(f"Live refresh error: {refresh_exc}")

market_scan_df = None
if market_scan_enabled:
    try:
        market_scan_df = refresh_market_scan(
            enabled=market_scan_enabled,
            max_symbols=int(market_scan_max_symbols),
            quote_asset=market_scan_quote_asset,
            min_interval_seconds=int(market_scan_interval_seconds),
        )
    except Exception as scan_exc:
        st.caption(f"Market scan error: {scan_exc}")

if market_scan_auto_pick_symbol and isinstance(market_scan_df, pd.DataFrame) and not market_scan_df.empty:
    st.session_state["live_symbol"] = str(market_scan_df.iloc[0]["symbol"])

with live_tab:
    st.subheader("Live AI Trading Graph")
    st.write("Track price, account value, and AI probability in real time.")

    if isinstance(market_scan_df, pd.DataFrame) and not market_scan_df.empty:
        top_pick = market_scan_df.iloc[0]
        st.caption(
            f"Top market pick: {top_pick['symbol']} | score={top_pick['ai_score']:.2f} | bias={top_pick['ai_bias']} | change={top_pick['change_pct']:.2f}%"
        )

    live_col1, live_col2, live_col3 = st.columns(3)
    with live_col1:
        live_symbol = st.text_input("Live symbol", key="live_symbol")
    with live_col2:
        autopilot_interval_seconds = st.number_input("Autopilot interval (seconds)", min_value=5, max_value=300, key="autopilot_interval_seconds")
    with live_col3:
        autopilot_cycles = st.number_input("Run alone cycles", min_value=1, max_value=100, key="autopilot_cycles")

    if st.button("Capture Live Point"):
        try:
            capture_symbol = live_symbol
            if market_scan_auto_pick_symbol and isinstance(market_scan_df, pd.DataFrame) and not market_scan_df.empty:
                capture_symbol = str(market_scan_df.iloc[0]["symbol"])
            point = append_live_history(capture_symbol, signal, probability_up)
            st.success(f"Captured: bid={point['best_bid']}, ask={point['best_ask']}, signal={point['signal']}")
        except Exception as exc:
            st.error(str(exc))

    history = st.session_state.get("live_history", [])
    if history:
        history_df = pd.DataFrame(history)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
        history_df = history_df.sort_values("timestamp")
        plot_df = history_df.set_index("timestamp")[["best_bid", "best_ask", "probability_up", "account_value"]]
        st.line_chart(plot_df, use_container_width=True)
        st.dataframe(history_df.tail(20), use_container_width=True)
    else:
        st.info("No live points yet. Click 'Capture Live Point' to start the chart.")

    st.subheader("Run Alone (Autopilot)")
    st.write("Runs AI trading logic for multiple cycles without manual clicks.")
    autopilot_order_size = st.number_input("Autopilot order size", min_value=0.0, step=0.001, format="%.6f", key="autopilot_order_size")
    autopilot_live_enabled = st.checkbox("Allow live orders in autopilot", key="autopilot_live_enabled")
    autopilot_confirmation = st.text_input("Type AUTO to allow live autopilot trades", key="autopilot_confirmation")

    if st.button("Start Run Alone"):
        progress = st.progress(0)
        status_box = st.empty()
        logs = []
        for cycle in range(int(autopilot_cycles)):
            try:
                point = append_live_history(
                    live_symbol,
                    st.session_state.get("signal", signal),
                    st.session_state.get("probability_up", probability_up),
                )
                trade_result = maybe_execute_signal_trade(
                    signal=st.session_state.get("signal", signal),
                    symbol=live_symbol,
                    order_size=float(autopilot_order_size),
                    allow_live=autopilot_live_enabled,
                    confirmation_text=autopilot_confirmation,
                    max_api_latency_ms=int(max_api_latency_ms),
                    max_ticker_age_ms=int(max_ticker_age_ms),
                    max_spread_bps=float(max_spread_bps),
                    min_trade_cooldown_seconds=int(min_trade_cooldown_seconds),
                )
                logs.append({
                    "cycle": cycle + 1,
                    "timestamp": point["timestamp"],
                    "signal": point["signal"],
                    "bid": point["best_bid"],
                    "ask": point["best_ask"],
                    "trade_status": trade_result.get("status"),
                    "exchange": "binance",
                    "api_latency_ms": trade_result.get("api_latency_ms"),
                    "ticker_age_ms": trade_result.get("ticker_age_ms"),
                    "spread_bps": trade_result.get("spread_bps"),
                    "guard": trade_result.get("guard_message"),
                })
                status_box.info(f"Cycle {cycle + 1}/{int(autopilot_cycles)} complete.")
            except Exception as exc:
                logs.append({"cycle": cycle + 1, "error": str(exc)})
                status_box.error(f"Cycle {cycle + 1} failed: {exc}")
            progress.progress((cycle + 1) / int(autopilot_cycles))
            if cycle + 1 < int(autopilot_cycles):
                time.sleep(int(autopilot_interval_seconds))

        st.success("Autopilot run completed.")
        st.dataframe(pd.DataFrame(logs), use_container_width=True)

    if isinstance(market_scan_df, pd.DataFrame) and not market_scan_df.empty:
        st.subheader("Market Scanner")
        st.dataframe(market_scan_df.head(25), use_container_width=True)

with wallet_tab:
    st.subheader("My Wallet")
    st.caption(f"Binance wallet ({'testnet' if BINANCE_TESTNET else 'live'})")

    if st.button("Refresh My Wallet"):
        try:
            st.session_state["wallet_snapshot"] = get_wallet_snapshot()
            st.success("Wallet updated.")
        except Exception as exc:
            st.error(str(exc))

    if "wallet_snapshot" not in st.session_state:
        try:
            st.session_state["wallet_snapshot"] = get_wallet_snapshot()
        except Exception as exc:
            st.error(str(exc))

    wallet_snapshot = st.session_state.get("wallet_snapshot")
    if wallet_snapshot:
        w1, w2, w3 = st.columns(3)
        w1.metric("Estimated Wallet (USDT)", format_currency(wallet_snapshot.get("estimated_total_usdt", 0.0)))
        w2.metric("Assets", int(wallet_snapshot.get("asset_count", 0)))
        w3.metric("Exchange", wallet_snapshot.get("exchange", "binance").upper())

        balances_df = wallet_snapshot.get("balances")
        if isinstance(balances_df, pd.DataFrame) and not balances_df.empty:
            st.dataframe(balances_df, use_container_width=True)
        else:
            st.info("No non-zero balances found in this wallet.")

with account_tab:
    st.subheader("Trade Ticket")
    st.caption(f"Current trading tool: Binance ({'testnet' if BINANCE_TESTNET else 'live'})")
    if wallet_total_usdt is not None:
        st.metric("Estimated Total Wallet (USDT)", format_currency(wallet_total_usdt))
    snapshot_col, action_col = st.columns([1, 1])

    with snapshot_col:
        st.markdown("### Account Snapshot")
        snapshot_symbol = st.text_input("Snapshot symbol", value=CHECK_SYMBOL)
        if st.button("Refresh Account Snapshot"):
            try:
                snapshot = get_account_snapshot(snapshot_symbol)
                st.json(snapshot)
            except Exception as exc:
                st.error(str(exc))

    with action_col:
        st.markdown("### Execute Action")
        with st.form("trade_ticket_form", clear_on_submit=False):
            dry_run = st.checkbox("Dry-run mode", value=True)
            action = st.selectbox(
                "Action",
                ["market_buy", "market_sell", "cancel_all_orders"],
                help="market_buy can use quantity or quote amount. market_sell uses quantity.",
            )
            action_symbol = st.text_input("Trade symbol", value=CHECK_SYMBOL)
            quantity = st.number_input("Quantity (base asset)", min_value=0.0, value=0.001, step=0.001, format="%.6f")
            quote_amount = st.number_input("Quote amount (optional, for market_buy)", min_value=0.0, value=0.0, step=1.0)
            confirmation = st.text_input("Type EXECUTE to confirm live action")
            run_action_submit = st.form_submit_button("Run Action", use_container_width=True)

        try:
            _preview_price = get_market_price(action_symbol.strip())
            _preview_reqs = get_market_requirements(action_symbol.strip())
            _preview_ok, _preview_message = validate_order_minimums(
                action=action,
                quantity=float(quantity),
                quote_amount=float(quote_amount),
                market_price=float(_preview_price),
                min_qty=float(_preview_reqs["min_qty"]),
                min_notional=float(_preview_reqs["min_notional"]),
            )
            st.caption(
                f"Pair minimums: min_qty={_preview_reqs['min_qty']}, min_notional={_preview_reqs['min_notional']}. Check: {_preview_message}"
            )
            if not _preview_ok:
                st.warning("Current input may be rejected by Binance minimum filters.")
        except Exception as preview_exc:
            st.caption(f"Minimum check unavailable: {preview_exc}")

        if run_action_submit:
            try:
                if not dry_run and confirmation.strip().upper() != "EXECUTE":
                    st.error("Live execution blocked: type EXECUTE in the confirmation box.")
                else:
                    result = execute_account_action(
                        action=action,
                        symbol=action_symbol.strip(),
                        quantity=float(quantity),
                        quote_amount=float(quote_amount),
                        dry_run=dry_run,
                        max_api_latency_ms=int(max_api_latency_ms),
                        max_ticker_age_ms=int(max_ticker_age_ms),
                        max_spread_bps=float(max_spread_bps),
                        min_trade_cooldown_seconds=int(min_trade_cooldown_seconds),
                    )
                    st.json(result)
            except Exception as exc:
                st.error(str(exc))

with ai_tab:
    st.subheader("AI Command Center")
    st.write("Use short commands to control dashboard inputs and get recommendations.")
    st.caption("Examples: set deposit 2000 | set buy threshold 0.62 | set sell threshold 0.38 | recommend")

    ai_feedback = st.session_state.pop("_ai_command_feedback", None)
    if ai_feedback:
        st.success(ai_feedback)

    ai_command = st.text_input("AI command")
    if st.button("Run AI Command"):
        message = apply_ai_command(ai_command)
        st.success(message)

# Keep the panel continuously live by triggering timed reruns.
if auto_refresh_enabled:
    time.sleep(int(auto_refresh_interval_seconds))
    st.rerun()
