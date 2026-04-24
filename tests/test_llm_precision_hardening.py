from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend import llm_client
from research import generate_trade_signal as gts


class LlmOverlayPrecisionTests(unittest.TestCase):
    @patch("research.generate_trade_signal.ensure_llm_startup_logged")
    @patch("research.generate_trade_signal.llm_chat")
    def test_generate_llm_overlay_accepts_valid_structured_response(self, mock_llm_chat, _mock_startup) -> None:
        latest_row = MagicMock()
        latest_row.items.return_value = [("close", 100.0), ("volume", 10.0)]
        mock_llm_chat.return_value = {
            "status": "ok",
            "content": '{"llm_signal":"BUY","confidence":0.63,"rationale":"Momentum confirms the ML edge.","risk_flags":["volatility"]}',
            "provider": "groq",
            "active_model": "llama-3.3-70b-versatile",
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "is_trained_model": False,
            "fallback_active": False,
            "rate_limit": {"remaining_tokens": "100"},
        }

        with patch.object(gts, "LLM_ENABLED", True):
            overlay = gts.generate_llm_overlay(0.7, 0.55, 0.45, latest_row)

        self.assertEqual(overlay["status"], "ok")
        self.assertEqual(overlay["llm_signal"], "BUY")
        self.assertEqual(overlay["confidence"], 0.63)
        self.assertEqual(overlay["rate_limit"]["remaining_tokens"], "100")

    @patch("research.generate_trade_signal.ensure_llm_startup_logged")
    @patch("research.generate_trade_signal.llm_chat")
    def test_generate_llm_overlay_rejects_non_numeric_confidence(self, mock_llm_chat, _mock_startup) -> None:
        latest_row = MagicMock()
        latest_row.items.return_value = [("close", 100.0)]
        mock_llm_chat.return_value = {
            "status": "ok",
            "content": '{"llm_signal":"BUY","confidence":"0.40-0.70","rationale":"Maybe buy.","risk_flags":["uncertain"]}',
            "provider": "groq",
            "active_model": "llama-3.3-70b-versatile",
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "is_trained_model": False,
            "fallback_active": False,
            "rate_limit": {},
        }

        with patch.object(gts, "LLM_ENABLED", True):
            overlay = gts.generate_llm_overlay(0.7, 0.55, 0.45, latest_row)

        self.assertEqual(overlay["status"], "invalid_response")
        self.assertEqual(overlay["llm_signal"], "HOLD")
        self.assertEqual(overlay["confidence"], 0.0)

    @patch("research.generate_trade_signal.ensure_llm_startup_logged")
    @patch("research.generate_trade_signal.llm_chat")
    def test_generate_llm_overlay_rejects_invalid_signal(self, mock_llm_chat, _mock_startup) -> None:
        latest_row = MagicMock()
        latest_row.items.return_value = [("close", 100.0)]
        mock_llm_chat.return_value = {
            "status": "ok",
            "content": '{"llm_signal":"STRONG_BUY","confidence":0.91,"rationale":"Conviction.","risk_flags":[]}',
            "provider": "groq",
            "active_model": "llama-3.3-70b-versatile",
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "is_trained_model": False,
            "fallback_active": False,
            "rate_limit": {},
        }

        with patch.object(gts, "LLM_ENABLED", True):
            overlay = gts.generate_llm_overlay(0.7, 0.55, 0.45, latest_row)

        self.assertEqual(overlay["status"], "invalid_response")
        self.assertEqual(overlay["llm_signal"], "HOLD")

    def test_merge_ml_llm_decision_blocks_low_confidence_overlay(self) -> None:
        llm_overlay = {
            "status": "ok",
            "llm_signal": "BUY",
            "confidence": 0.20,
        }

        with patch.object(gts, "LLM_MERGE_ENABLED", True), patch.object(gts, "LLM_CONFIDENCE_SOFT_GATE", False):
            merged = gts.merge_ml_llm_decision(
                ml_probability_up=0.58,
                buy_threshold=0.55,
                sell_threshold=0.45,
                llm_overlay=llm_overlay,
                market_regime={"enabled": False, "regime": "disabled"},
                data_confidence=0.8,
            )

        self.assertEqual(merged["status"], "confidence_too_low")
        self.assertEqual(merged["merge_weight"], 0.0)
        self.assertEqual(merged["merged_probability_up"], 0.58)


class LlmClientPrecisionTests(unittest.TestCase):
    @patch("backend.llm_client._log_startup_status_once")
    @patch("backend.llm_client.get_llm_status")
    @patch("backend.llm_client._resolved_model_target")
    @patch("backend.llm_client.get_llm_runtime_config")
    @patch("backend.llm_client._get_session")
    def test_llm_chat_exposes_rate_limit_headers(
        self,
        mock_get_session,
        mock_get_runtime_config,
        mock_resolved_model_target,
        mock_get_status,
        _mock_startup,
    ) -> None:
        mock_get_runtime_config.return_value = {
            "enabled": True,
            "provider": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "secret",
            "timeout_seconds": 30,
            "temperature": 0.0,
        }
        mock_resolved_model_target.return_value = {"model": "llama-3.3-70b-versatile"}
        mock_get_status.return_value = {"provider": "groq", "active_model": "llama-3.3-70b-versatile"}

        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": '{"llm_signal":"HOLD"}'}}]}
        response.headers = {
            "retry-after": "2",
            "x-ratelimit-remaining-tokens": "11950",
            "x-ratelimit-reset-tokens": "7.66s",
        }
        response.raise_for_status.return_value = None

        session = MagicMock()
        session.post.return_value = response
        mock_get_session.return_value = session

        result = llm_client.llm_chat("system", "user", response_format={"type": "json_object"})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rate_limit"]["retry_after"], "2")
        self.assertEqual(result["rate_limit"]["remaining_tokens"], "11950")
        self.assertEqual(result["rate_limit"]["reset_tokens"], "7.66s")


if __name__ == "__main__":
    unittest.main()
