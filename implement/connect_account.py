"""Hyperliquid account connectivity and readiness check.

This script is intentionally read-only. It does not place orders.
Use it to verify:
- API key loading
- account address
- account value and margin summary
- open positions and resting orders
- current top-of-book for a symbol
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import eth_account
import requests
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.utils import constants

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

HYPER_LIQUID_KEY = os.getenv("HYPER_LIQUID_KEY")
CHECK_SYMBOL = os.getenv("CHECK_SYMBOL", "BTC")


def ask_bid(symbol: str) -> tuple[float, float]:
    response = requests.post(
        "https://api.hyperliquid.xyz/info",
        headers={"Content-Type": "application/json"},
        json={"type": "l2Book", "coin": symbol},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    levels = payload["levels"]
    bid = float(levels[0][0]["px"])
    ask = float(levels[1][0]["px"])
    return ask, bid


def summarize_positions(asset_positions: list[dict], symbol: str) -> list[dict]:
    positions = []
    for item in asset_positions:
        position = item.get("position", {})
        size = float(position.get("szi", 0))
        if size == 0:
            continue

        coin = position.get("coin")
        if symbol and coin != symbol:
            continue

        positions.append(
            {
                "symbol": coin,
                "size": size,
                "entry_px": float(position.get("entryPx", 0) or 0),
                "roe_pct": float(position.get("returnOnEquity", 0) or 0) * 100,
                "position_value": float(position.get("positionValue", 0) or 0),
            }
        )

    return positions


def main() -> None:
    if not HYPER_LIQUID_KEY:
        raise RuntimeError("HYPER_LIQUID_KEY is not set in .env")

    account = eth_account.Account.from_key(HYPER_LIQUID_KEY)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    user_state = info.user_state(account.address)
    open_orders = info.open_orders(account.address)

    margin_summary = user_state.get("marginSummary", {})
    account_value = float(margin_summary.get("accountValue", 0) or 0)
    total_margin_used = float(margin_summary.get("totalMarginUsed", 0) or 0)

    ask, bid = ask_bid(CHECK_SYMBOL)
    positions = summarize_positions(user_state.get("assetPositions", []), CHECK_SYMBOL)

    result = {
        "status": "connected",
        "address": account.address,
        "symbol_checked": CHECK_SYMBOL,
        "account_value": account_value,
        "total_margin_used": total_margin_used,
        "best_ask": ask,
        "best_bid": bid,
        "open_orders_count": len(open_orders),
        "open_positions": positions,
        "next_step": "Connection verified. Keep execution scripts in dry-run mode until limits are configured.",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
