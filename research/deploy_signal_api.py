"""Simple HTTP API for live trading decisions.

Run:
    python research/deploy_signal_api.py

Endpoints:
- GET /health
- GET /signal
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

from generate_trade_signal import (  # noqa: E402
    ADAPTIVE_THRESHOLD_ENABLED,
    BUY_THRESHOLD,
    DATASET_PATH,
    SELL_THRESHOLD,
    generate_trade_decision,
    resolve_model_path,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("SIGNAL_API_HOST", "0.0.0.0")
PORT = int(os.getenv("SIGNAL_API_PORT", "8765"))


class SignalHandler(BaseHTTPRequestHandler):
    server_version = "TradingSignalAPI/1.0"

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return

        if self.path == "/signal":
            try:
                decision = generate_trade_decision(
                    model_path=resolve_model_path(),
                    dataset_path=DATASET_PATH,
                    base_buy_threshold=BUY_THRESHOLD,
                    base_sell_threshold=SELL_THRESHOLD,
                    adaptive_threshold_enabled=ADAPTIVE_THRESHOLD_ENABLED,
                )
                self._send_json(
                    {
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
                    }
                )
                return
            except Exception as exc:
                self._send_json(
                    {"status": "error", "message": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

        self._send_json(
            {"status": "not_found", "message": "Use /health or /signal"},
            status=HTTPStatus.NOT_FOUND,
        )


def main() -> None:
    with ThreadingHTTPServer((HOST, PORT), SignalHandler) as server:
        print(f"Signal API running at http://{HOST}:{PORT}")
        print("GET /health")
        print("GET /signal")
        server.serve_forever()


if __name__ == "__main__":
    main()
