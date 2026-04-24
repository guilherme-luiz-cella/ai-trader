from __future__ import annotations

import json
import os
import secrets
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from backend import services

HOST = os.getenv("SIGNAL_API_HOST", "0.0.0.0")
PORT = int(os.getenv("SIGNAL_API_PORT", "8765"))
CORS_ALLOW_ORIGIN = os.getenv("SIGNAL_API_CORS_ALLOW_ORIGIN", "*")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
APP_LOGIN_EMAIL = os.getenv("APP_LOGIN_EMAIL", "").strip().lower()
APP_PASSWORD_HASH = os.getenv("APP_PASSWORD_HASH", "").strip()
APP_SESSION_TTL_SECONDS = max(300, int(os.getenv("APP_SESSION_TTL_SECONDS", "43200")))
ALLOWED_ACCESS_EMAILS = {
    email.strip().lower()
    for email in os.getenv("ALLOWED_ACCESS_EMAILS", "").split(",")
    if email.strip()
}
OPEN_ROUTES = {"/health", "/auth/login"}
PASSWORD_HASHER = PasswordHasher()
APP_SESSIONS: dict[str, dict[str, float | str]] = {}


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "TradingBackendAPI/1.0"

    def _apply_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", CORS_ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _auth_configured(self) -> bool:
        return bool(API_AUTH_TOKEN or ALLOWED_ACCESS_EMAILS or APP_LOGIN_EMAIL or APP_PASSWORD_HASH)

    def _request_email(self) -> str:
        return str(self.headers.get("CF-Access-Authenticated-User-Email", "")).strip().lower()

    def _request_bearer_token(self) -> str:
        value = str(self.headers.get("Authorization", "")).strip()
        if not value.startswith("Bearer "):
            return ""
        return value[len("Bearer ") :].strip()

    def _purge_expired_sessions(self) -> None:
        now = time.time()
        expired_tokens = [
            token
            for token, session in APP_SESSIONS.items()
            if float(session.get("expires_at", 0.0)) <= now
        ]
        for token in expired_tokens:
            APP_SESSIONS.pop(token, None)

    def _session_payload(self, token: str) -> dict[str, float | str] | None:
        self._purge_expired_sessions()
        if not token:
            return None
        return APP_SESSIONS.get(token)

    def _create_session_token(self, email: str) -> str:
        token = secrets.token_urlsafe(32)
        APP_SESSIONS[token] = {
            "email": email,
            "issued_at": time.time(),
            "expires_at": time.time() + APP_SESSION_TTL_SECONDS,
        }
        return token

    def _verify_login_password(self, password: str) -> bool:
        if not APP_PASSWORD_HASH:
            return False
        try:
            return PASSWORD_HASHER.verify(APP_PASSWORD_HASH, password)
        except (VerifyMismatchError, InvalidHashError):
            return False

    def _request_is_local(self) -> bool:
        host, *_rest = self.client_address
        return host in {"127.0.0.1", "::1"}

    def _is_authorized(self, route: str) -> bool:
        if route in OPEN_ROUTES:
            return True
        if not self._auth_configured():
            return True
        if self._request_is_local():
            return True
        request_email = self._request_email()
        if request_email and request_email in ALLOWED_ACCESS_EMAILS:
            return True
        request_token = self._request_bearer_token()
        if API_AUTH_TOKEN and request_token and request_token == API_AUTH_TOKEN:
            return True
        session = self._session_payload(request_token)
        if session is not None:
            return True
        return False

    def _require_authorization(self, route: str) -> bool:
        if self._is_authorized(route):
            return True
        self._send_json(
            {
                "status": "unauthorized",
                "message": "Authentication required.",
            },
            status=HTTPStatus.UNAUTHORIZED,
        )
        return False

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
            if not self._require_authorization(route):
                return
            if route == "/health":
                self._send_json(
                    {
                        "status": "ok",
                        "service": "signal-api",
                        "service_health": "healthy",
                        "llm_status": services.get_llm_status(),
                        "notification_status": services.get_notification_status(),
                        "autopilot": services.autopilot_snapshot(),
                        "burn_in_report": services.build_burn_in_validation_report(),
                        "auth": {
                            "login_enabled": bool(APP_LOGIN_EMAIL and APP_PASSWORD_HASH),
                            "cloudflare_access_enabled": bool(ALLOWED_ACCESS_EMAILS),
                            "api_token_enabled": bool(API_AUTH_TOKEN),
                        },
                    }
                )
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
            if route == "/autopilot/reconcile":
                self._send_json({"status": "ok", "reconciliation": services.reconcile_interrupted_autopilot_state(), "autopilot": services.autopilot_snapshot()})
                return
            if route == "/autopilot/readiness":
                self._send_json({"status": "ok", "burn_in_report": services.build_burn_in_validation_report(), "autopilot": services.autopilot_snapshot()})
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
                    "size_min_confidence": float(query.get("size_min_confidence", [services.SIZE_MIN_CONFIDENCE])[0]),
                    "decision_min_confidence": float(query.get("decision_min_confidence", [services.DECISION_MIN_CONFIDENCE])[0]),
                    "live_symbol": str(query.get("live_symbol", [services.CHECK_SYMBOL])[0]),
                    "market_scan_enabled": str(query.get("market_scan_enabled", ["true"])[0]).lower() in {"1", "true", "yes", "on"},
                    "market_scan_max_symbols": int(query.get("market_scan_max_symbols", [60])[0]),
                    "market_scan_quote_asset": str(query.get("market_scan_quote_asset", ["USDT"])[0]),
                    "target_monitor_symbols": str(query.get("target_monitor_symbols", [",".join(services.TARGET_MONITOR_SYMBOLS)])[0]),
                    "stable_asset": str(query.get("stable_asset", [services.PROFIT_PARKING_STABLE_ASSET])[0]),
                    "stable_reserve_min_pct": float(query.get("stable_reserve_min_pct", [services.STABLE_RESERVE_MIN_PCT])[0]),
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
            if route == "/market/chart":
                symbol = str(query.get("symbol", [services.CHECK_SYMBOL])[0])
                interval = str(query.get("interval", ["5m"])[0])
                limit = int(query.get("limit", [200])[0])
                self._send_json({"status": "ok", "chart": services.get_market_chart(symbol, interval=interval, limit=limit)})
                return
            if route == "/market/ticker":
                symbol = str(query.get("symbol", [services.CHECK_SYMBOL])[0])
                self._send_json({"status": "ok", "ticker": services.get_market_ticker(symbol)})
                return
            if route == "/autopilot/events":
                limit = int(query.get("limit", [100])[0])
                self._send_json({"status": "ok", "autopilot_events": services.get_autopilot_events(limit)})
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
            if route == "/auth/login":
                submitted_email = str(payload.get("email", "")).strip().lower()
                submitted_password = str(payload.get("password", ""))
                if not APP_LOGIN_EMAIL or not APP_PASSWORD_HASH:
                    self._send_json({"status": "error", "message": "App login is not configured."}, status=HTTPStatus.BAD_REQUEST)
                    return
                if submitted_email != APP_LOGIN_EMAIL or not self._verify_login_password(submitted_password):
                    self._send_json({"status": "unauthorized", "message": "Invalid email or password."}, status=HTTPStatus.UNAUTHORIZED)
                    return
                session_token = self._create_session_token(submitted_email)
                self._send_json(
                    {
                        "status": "ok",
                        "auth": {
                            "token": session_token,
                            "email": submitted_email,
                            "expires_in_seconds": APP_SESSION_TTL_SECONDS,
                        },
                    }
                )
                return
            if not self._require_authorization(route):
                return
            if route == "/autopilot/start":
                self._send_json({"status": "ok", "autopilot": services.start_autopilot(payload)})
                return
            if route == "/autopilot/stop":
                self._send_json({"status": "ok", "autopilot": services.stop_autopilot()})
                return
            if route == "/auth/logout":
                request_token = self._request_bearer_token()
                if request_token:
                    APP_SESSIONS.pop(request_token, None)
                self._send_json({"status": "ok"})
                return
            if route == "/live/capture":
                symbol = str(payload.get("symbol", services.CHECK_SYMBOL))
                signal = str(payload.get("signal", "HOLD"))
                probability_up = float(payload.get("probability_up", 0.5))
                self._send_json({"status": "ok", "point": services.append_live_history(symbol, signal, probability_up), "history": services.get_live_history()})
                return
            if route == "/trade/action":
                block_latency_ms = int(payload.get("block_latency_ms", payload.get("max_api_latency_ms", 3000)))
                result = services.execute_account_action(
                    action=str(payload.get("action", "")),
                    symbol=str(payload.get("symbol", services.CHECK_SYMBOL)),
                    quantity=float(payload.get("quantity", 0.0)),
                    quote_amount=float(payload.get("quote_amount", 0.0)),
                    dry_run=bool(payload.get("dry_run", True)),
                    max_api_latency_ms=block_latency_ms,
                    warning_latency_ms=int(payload.get("warning_latency_ms", int(block_latency_ms * 0.50))),
                    degraded_latency_ms=int(payload.get("degraded_latency_ms", int(block_latency_ms * 0.75))),
                    block_latency_ms=block_latency_ms,
                    consecutive_breach_limit=int(payload.get("consecutive_breach_limit", 5)),
                    max_ticker_age_ms=int(payload.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(payload.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(payload.get("min_trade_cooldown_seconds", 5)),
                )
                self._send_json({"status": "ok", "result": result})
                return
            if route == "/trade/preview":
                block_latency_ms = int(payload.get("block_latency_ms", payload.get("max_api_latency_ms", 3000)))
                result = services.preview_trade_action(
                    action=str(payload.get("action", "")),
                    symbol=str(payload.get("symbol", services.CHECK_SYMBOL)),
                    quantity=float(payload.get("quantity", 0.0)),
                    quote_amount=float(payload.get("quote_amount", 0.0)),
                    max_api_latency_ms=block_latency_ms,
                    warning_latency_ms=int(payload.get("warning_latency_ms", int(block_latency_ms * 0.50))),
                    degraded_latency_ms=int(payload.get("degraded_latency_ms", int(block_latency_ms * 0.75))),
                    block_latency_ms=block_latency_ms,
                    consecutive_breach_limit=int(payload.get("consecutive_breach_limit", 5)),
                    max_ticker_age_ms=int(payload.get("max_ticker_age_ms", 3000)),
                    max_spread_bps=float(payload.get("max_spread_bps", 20.0)),
                    min_trade_cooldown_seconds=int(payload.get("min_trade_cooldown_seconds", 5)),
                )
                self._send_json({"status": "ok", "result": result})
                return
            if route == "/rebalance/execute":
                result = services.execute_rebalance_orders(payload)
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
    services.ensure_llm_startup_logged()
    services.reconcile_interrupted_autopilot_state()
    with ThreadingHTTPServer((HOST, PORT), ApiHandler) as server:
        print(f"Backend API running at http://{HOST}:{PORT}")
        server.serve_forever()


if __name__ == "__main__":
    main()
