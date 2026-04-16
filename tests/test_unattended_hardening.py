from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend import services


class UnattendedHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        memory_dir = Path(self.tempdir.name)
        services.AUTOPILOT_STATE_PATH = memory_dir / "autopilot_state.json"
        services.AUTOPILOT_EXECUTION_JOURNAL_PATH = memory_dir / "autopilot_execution_journal.jsonl"
        services.AUTOPILOT_RUN_SUMMARY_PATH = memory_dir / "autopilot_run_summaries.jsonl"
        services.TRADE_MEMORY_PATH = memory_dir / "trade_memory.jsonl"
        services.AUTOPILOT_STATE.clear()
        services.AUTOPILOT_STATE.update(
            {
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
                "persistence_path": str(services.AUTOPILOT_STATE_PATH),
                "execution_journal_path": str(services.AUTOPILOT_EXECUTION_JOURNAL_PATH),
                "reconciliation_state": "clean",
                "reconciliation_details": {},
                "reconciliation_checked_at": None,
                "requires_human_review": False,
                "alert_severity": "info",
                "alerts": [],
                "unattended_mode": False,
                "burn_in_report": {},
            }
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _persist_running_state(self, *, stage: str, symbol: str = "BTC/USDT", finalization_status: str = "") -> None:
        payload = {
            **services.autopilot_snapshot(),
            "running": True,
            "status": "running",
            "symbol": symbol,
            "finalization_status": finalization_status,
            "latest_execution_intent": {
                "run_id": 7,
                "cycle": 3,
                "stage": stage,
                "symbol": symbol,
                "execution_fingerprint": f"{stage}-fp",
                "exchange_client_order_id": f"{stage}-fp",
                "status": "uncertain",
            },
        }
        services._atomic_write_json(services.AUTOPILOT_STATE_PATH, payload)
        services.append_execution_journal(payload["latest_execution_intent"])

    @patch("backend.services.emit_autopilot_alert")
    @patch("backend.services.get_account_snapshot")
    @patch("backend.services.get_wallet_snapshot")
    @patch("backend.services.fetch_binance_order_truth")
    def test_restart_during_conversion_to_buy_requires_review(self, mock_truth, mock_wallet, mock_account, _mock_alert) -> None:
        self._persist_running_state(stage="conversion", symbol="BTC/USDT")
        mock_truth.return_value = {"status": "filled", "client_order_id": "conversion-fp", "order_id": "123"}
        mock_wallet.return_value = {"captured_at": "2026-04-16T00:00:00Z", "estimated_total_usdt": 100.0, "asset_count": 2, "balances": []}
        mock_account.return_value = {"symbol": "BTC/USDT", "base_free": 0.001, "quote_free": 20.0}

        report = services.reconcile_interrupted_autopilot_state()

        self.assertEqual(report["reconciliation_state"], "interrupted_after_conversion")
        self.assertTrue(report["requires_human_review"])
        self.assertEqual(report["exchange_reconciliations"][0]["exchange_truth"]["status"], "filled")

    @patch("backend.services.emit_autopilot_alert")
    @patch("backend.services.get_account_snapshot")
    @patch("backend.services.get_wallet_snapshot")
    @patch("backend.services.fetch_binance_order_truth")
    def test_restart_during_sell_with_partial_fill_needs_review(self, mock_truth, mock_wallet, mock_account, _mock_alert) -> None:
        self._persist_running_state(stage="signal_sell", symbol="ETH/USDT")
        mock_truth.return_value = {"status": "partially_filled", "client_order_id": "signal_sell-fp", "order_id": "456", "filled": 0.1, "remaining": 0.02}
        mock_wallet.return_value = {"captured_at": "2026-04-16T00:00:00Z", "estimated_total_usdt": 90.0, "asset_count": 2, "balances": []}
        mock_account.return_value = {"symbol": "ETH/USDT", "base_free": 0.02, "quote_free": 15.0}

        report = services.reconcile_interrupted_autopilot_state()

        self.assertEqual(report["reconciliation_state"], "partial_execution_needs_review")
        self.assertTrue(report["requires_human_review"])

    @patch("backend.services.emit_autopilot_alert")
    @patch("backend.services.get_account_snapshot")
    @patch("backend.services.get_wallet_snapshot")
    @patch("backend.services.fetch_binance_order_truth")
    def test_restart_during_finalization_keeps_finalization_reconciliation(self, mock_truth, mock_wallet, mock_account, _mock_alert) -> None:
        self._persist_running_state(stage="finalization", symbol="SOL/USDT", finalization_status="pending")
        mock_truth.return_value = {"status": "filled", "client_order_id": "finalization-fp", "order_id": "789"}
        mock_wallet.return_value = {"captured_at": "2026-04-16T00:00:00Z", "estimated_total_usdt": 110.0, "asset_count": 1, "balances": []}
        mock_account.return_value = {"symbol": "SOL/USDT", "base_free": 0.0, "quote_free": 110.0}

        report = services.reconcile_interrupted_autopilot_state()

        self.assertEqual(report["reconciliation_state"], "interrupted_during_finalization")
        self.assertTrue(report["requires_human_review"])

    @patch("backend.services.emit_autopilot_alert")
    @patch("backend.services.get_binance_client")
    @patch("backend.services.get_account_snapshot")
    @patch("backend.services.get_wallet_snapshot")
    @patch("backend.services.ensure_latency_monitor")
    @patch("backend.services.get_ticker_with_metrics")
    @patch("backend.services.get_market_price")
    @patch("backend.services.get_market_requirements")
    @patch("backend.services.validate_market_conditions")
    @patch("backend.services.resolve_effective_trade_request")
    @patch("backend.services.build_buy_sizing_plan")
    @patch("backend.services.resolve_wallet_funding_path")
    def test_duplicate_retry_after_restart_is_blocked(
        self,
        mock_funding,
        mock_buy_sizing,
        mock_effective,
        mock_guard,
        mock_market_reqs,
        mock_price,
        mock_ticker,
        _mock_monitor,
        mock_wallet,
        mock_account,
        mock_exchange,
        _mock_alert,
    ) -> None:
        fingerprint = services.execution_fingerprint("market_buy", "BTC/USDT", 0.001, 0.0, 7, 3, "signal_buy")
        services.append_execution_journal(
            {
                "run_id": 7,
                "cycle": 3,
                "stage": "signal_buy",
                "symbol": "BTC/USDT",
                "action": "market_buy",
                "quantity": 0.001,
                "quote_amount": 0.0,
                "execution_fingerprint": fingerprint,
                "status": "executed",
            }
        )
        mock_exchange.return_value = MagicMock()
        mock_account.return_value = {"captured_at": "2026-04-16T00:00:00Z", "base_asset": "BTC", "quote_asset": "USDT", "base_free": 0.0, "quote_free": 100.0, "base_used": 0.0, "base_total": 0.0, "quote_used": 0.0, "quote_total": 100.0, "open_orders_count": 0}
        mock_wallet.return_value = {"captured_at": "2026-04-16T00:00:00Z", "estimated_total_usdt": 100.0, "asset_count": 1, "balances": []}
        mock_ticker.return_value = ({"last": 50000.0}, {"api_latency_ms": 10.0, "ticker_age_ms": 100, "spread_bps": 2.0})
        mock_price.return_value = 50000.0
        mock_market_reqs.return_value = {"min_qty": 0.0001, "min_notional": 5.0, "qty_precision": 6, "price_precision": 2, "step_size": 0.0001}
        mock_guard.return_value = {"ok": True, "message": "", "sample": {}, "rolling": {}, "policy": {}, "root_cause": "none", "mode": "normal", "reason": "", "consecutive_breaches": 0}
        mock_effective.return_value = {"effective_quantity": 0.001, "effective_quote_amount": 0.0, "size_cap_reason": ""}
        mock_buy_sizing.return_value = {"minimum_valid_quote": 5.0, "free_quote": 100.0, "free_base": 0.0, "computed_order_size_quote": 50.0, "computed_order_size_base": 0.001, "can_buy_minimum": True, "skip_reason": ""}
        mock_funding.return_value = {"required_quote_asset": "USDT", "direct_trade_possible": True, "funding_path": "direct", "funding_skip_reason": "", "free_direct_quote": 100.0, "free_bnb": 0.0, "eligible_funding_assets": [], "funding_diagnostics": {}, "conversion_plan": None}

        result = services.execute_account_action(
            action="market_buy",
            symbol="BTC/USDT",
            quantity=0.001,
            quote_amount=0.0,
            dry_run=False,
            max_api_latency_ms=1200,
            max_ticker_age_ms=3000,
            max_spread_bps=20.0,
            min_trade_cooldown_seconds=0,
            execution_context={"run_id": 7, "cycle": 3, "stage": "signal_buy"},
        )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["skip_reason"], "duplicate_execution_blocked")

    @patch("backend.services.get_binance_client")
    def test_uncertain_order_state_recovery_via_binance_lookup(self, mock_exchange_factory) -> None:
        exchange = MagicMock()
        exchange.market.return_value = {"id": "BTCUSDT"}
        exchange.load_markets.return_value = None
        exchange.privateGetOrder.return_value = {
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "clientOrderId": "abc123",
            "orderId": 999,
            "executedQty": "0.001",
            "origQty": "0.001",
            "price": "50000",
        }
        exchange.parse_order.return_value = {
            "id": "999",
            "clientOrderId": "abc123",
            "status": "closed",
            "filled": 0.001,
            "remaining": 0.0,
            "amount": 0.001,
            "average": 50000.0,
            "info": {"status": "FILLED", "clientOrderId": "abc123", "orderId": 999},
        }
        mock_exchange_factory.return_value = exchange

        truth = services.fetch_binance_order_truth("BTC/USDT", client_order_id="abc123")

        self.assertEqual(truth["status"], "filled")
        self.assertEqual(truth["client_order_id"], "abc123")


if __name__ == "__main__":
    unittest.main()
