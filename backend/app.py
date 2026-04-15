from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from backend import services

HOST = os.getenv("SIGNAL_API_HOST", "0.0.0.0")
PORT = int(os.getenv("SIGNAL_API_PORT", "8765"))
CORS_ALLOW_ORIGIN = os.getenv("SIGNAL_API_CORS_ALLOW_ORIGIN", "*")


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "TradingBackendAPI/1.0"

    def _apply_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", CORS_ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status.value)
        self._apply_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT.value)
        self._apply_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route == "/health":
                self._send_json({"status": "ok"})
                return
            if route == "/docs":
                self._send_json(services.get_api_docs())
                return
            if route == "/signal":
                payload = services.get_decision_payload(
                    buy_threshold=float(query.get("buy_threshold", [services.BUY_THRESHOLD])[0]),
                    sell_threshold=float(query.get("sell_threshold", [services.SELL_THRESHOLD])[0]),
                    adaptive_threshold_enabled=str(query.get("adaptive_threshold_enabled", [services.ADAPTIVE_THRESHOLD_ENABLED])[0]).lower() in {"1", "true", "yes", "on"},
                )
                self._send_json(payload)
                return
            if route == "/macro/fred":
                series_id = str(query.get("series_id", ["DFF"])[0])
                limit = int(query.get("limit", [24])[0])
                start_date = query.get("start_date", [None])[0]
                end_date = query.get("end_date", [None])[0]
                self._send_json(
                    {
                        "status": "ok",
                        "fred": services.get_fred_series_observations(
                            series_id=series_id,
                            limit=limit,
                            start_date=start_date,
                            end_date=end_date,
                        ),
                    }
                )
                return
            if route == "/autopilot/status":
                self._send_json({"status": "ok", "autopilot": services.autopilot_snapshot()})
                return
            if route == "/dashboard":
                config = {
                    "buy_threshold": float(query.get("buy_threshold", [services.BUY_THRESHOLD])[0]),
                    "sell_threshold": float(query.get("sell_threshold", [services.SELL_THRESHOLD])[0]),
                    "adaptive_threshold_enabled": str(query.get("adaptive_threshold_enabled", [services.ADAPTIVE_THRESHOLD_ENABLED])[0]).lower() in {"1", "true", "yes", "on"},
                    "deposit_amount": float(query.get("deposit_amount", [services.ACCOUNT_REFERENCE_USD])[0]),
                    "active_capital_pct": float(query.get("active_capital_pct", [0.50 if services.ACCOUNT_REFERENCE_USD <= 10 else 0.70])[0]),
                    "reserve_pct": float(query.get("reserve_pct", [0.30])[0]),
                    "max_trade_pct": float(query.get("max_trade_pct", [0.20 if services.ACCOUNT_REFERENCE_USD <= 10 else 0.10])[0]),
                    "stop_loss_pct": float(query.get("stop_loss_pct", [0.03])[0]),
                    "take_profit_pct": float(query.get("take_profit_pct", [0.05])[0]),
                    "max_daily_loss_pct": float(query.get("max_daily_loss_pct", [0.05])[0]),
                    "max_drawdown_pct": float(query.get("max_drawdown_pct", [0.15])[0]),
                    "withdrawal_target_pct": float(query.get("withdrawal_target_pct", [0.25])[0]),
                    "live_symbol": str(query.get("live_symbol", [services.CHECK_SYMBOL])[0]),
                    "market_scan_enabled": str(query.get("market_scan_enabled", ["true"])[0]).lower() in {"1", "true", "yes", "on"},
                    "market_scan_max_symbols": int(query.get("market_scan_max_symbols", [60])[0]),
                    "market_scan_quote_asset": str(query.get("market_scan_quote_asset", ["USDT"])[0]),
                }
                self._send_json(services.get_dashboard_payload(config))
                return
            if route == "/wallet":
                self._send_json({"status": "ok", "wallet": services.get_wallet_snapshot()})
                return
            if route == "/account":
                symbol = str(query.get("symbol", [services.CHECK_SYMBOL])[0])
                self._send_json({"status": "ok", "account": services.get_account_snapshot(symbol)})
                return
            if route == "/market-scan":
                max_symbols = int(query.get("max_symbols", [60])[0])
                quote_asset = str(query.get("quote_asset", ["USDT"])[0])
                self._send_json({"status": "ok", "rows": services.build_market_scan(max_symbols, quote_asset)})
                return
            if route == "/live/history":
                self._send_json({"status": "ok", "history": services.get_live_history()})
                return
            self._send_json({"status": "not_found", "message": "Unknown route"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            payload = self._read_json_body()
            if route == "/autopilot/start":
                self._send_json({"status": "ok", "autopilot": services.start_autopilot(payload)})
                return
            if route == "/autopilot/stop":
                self._send_json({"status": "ok", "autopilot": services.stop_autopilot()})
                return
            if route == "/live/capture":
                symbol = str(payload.get("symbol", services.CHECK_SYMBOL))
                signal = str(payload.get("signal", "HOLD"))
                probability_up = float(payload.get("probability_up", 0.5))
                self._send_json({"status": "ok", "point": services.append_live_history(symbol, signal, probability_up), "history": services.get_live_history()})
                return
            if route == "/trade/action":
                result = services.execute_account_action(
                    action=str(payload.get("action", "")),
                    symbol=str(payload.get("symbol", services.CHECK_SYMBOL)),
                    quantity=float(payload.get("quantity", 0.0)),
                    quote_amount=float(payload.get("quote_amount", 0.0)),
                    dry_run=bool(payload.get("dry_run", True)),
                    max_api_latency_ms=int(payload.get("max_api_latency_ms", 1200)),
                    max_ticker_age_ms=int(payload.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(payload.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(payload.get("min_trade_cooldown_seconds", 5)),
                )
                self._send_json({"status": "ok", "result": result})
                return
            if route == "/trade/preview":
                result = services.preview_trade_action(
                    action=str(payload.get("action", "")),
                    symbol=str(payload.get("symbol", services.CHECK_SYMBOL)),
                    quantity=float(payload.get("quantity", 0.0)),
                    quote_amount=float(payload.get("quote_amount", 0.0)),
                    max_api_latency_ms=int(payload.get("max_api_latency_ms", 1200)),
                    max_ticker_age_ms=int(payload.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(payload.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(payload.get("min_trade_cooldown_seconds", 5)),
                )
                self._send_json({"status": "ok", "result": result})
                return
            if route == "/support/chat":
                self._send_json({"status": "ok", "result": services.llm_support_chat(str(payload.get("message", "")), payload.get("context", {}))})
                return
            if route == "/ai/command":
                self._send_json({"status": "ok", "result": services.run_ai_command(str(payload.get("command", "")))})
                return
            self._send_json({"status": "not_found", "message": "Unknown route"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def main() -> None:
    with ThreadingHTTPServer((HOST, PORT), ApiHandler) as server:
        print(f"Backend API running at http://{HOST}:{PORT}")
        server.serve_forever()


if __name__ == "__main__":
    main()
