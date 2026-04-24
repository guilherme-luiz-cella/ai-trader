import { describe, expect, it } from "vitest";

import type { HttpClient, HttpRequestOptions } from "../../src/infrastructure/http/HttpClient";
import { HttpTradingControlRepository } from "../../src/infrastructure/repositories/HttpTradingControlRepository";

class FakeHttpClient implements HttpClient {
  public calls: Array<{ url: string; options?: HttpRequestOptions }> = [];

  constructor(private readonly responses: unknown[]) {}

  async request<TResponse>(url: string, options?: HttpRequestOptions): Promise<TResponse> {
    this.calls.push({ url, options });
    return this.responses.shift() as TResponse;
  }
}

describe("HttpTradingControlRepository", () => {
  it("maps backend payloads into domain entities", async () => {
    const httpClient = new FakeHttpClient([
      {
        auth: {
          token: "session-token",
          email: "operator@example.com",
          expires_in_seconds: 43200,
        },
      },
      {
        status: "ok",
        service: "signal-api",
        service_health: "healthy",
        auth: {
          login_enabled: true,
          cloudflare_access_enabled: false,
          api_token_enabled: false,
        },
      },
      {
        config: { live_symbol: "BTC/USDT" },
        decision_payload: {
          decision: {
            signal: "BUY",
            probability_up: 0.74,
            ml_probability_up: 0.69,
            decision_confidence: 0.66,
            decision_engine: "ml",
            safe_next_action: "manual_review_required_before_resume",
          },
        },
        wallet_snapshot: {
          estimated_total_usdt: 1200,
          asset_count: 5,
        },
        risk_plan: {
          max_trade_size: 120,
          reserve_cash: 300,
        },
        live_start_gate: {
          reason: "Preview gate passed.",
          checks: {
            preview_gate_passed: true,
          },
        },
      },
      {
        autopilot: {
          status: "idle",
          running: false,
          current_cycle: 0,
          target_cycles: 0,
          symbol: "BTC/USDT",
          updated_at: "2026-04-24T00:00:00Z",
          goal_value: 0,
          current_value: 1200,
          progress_pct: 0,
          safe_next_action: "normal_start_allowed",
          latest_trade_result: {
            status: "skipped",
          },
        },
      },
      {
        result: {
          status: "preview_ready",
          symbol: "BTC/USDT",
          action: "market_buy",
          guard_message: "Dry preview only.",
          minimum_message: "Minimums satisfied.",
          effective_quantity: 0.001,
          size_cap_reason: "",
          min_qty: 0.0001,
          min_notional: 5,
          market_price: 50000,
        },
      },
    ]);
    const repository = new HttpTradingControlRepository(httpClient);
    const config = { baseUrl: "http://127.0.0.1:8765" };

    const session = await repository.signIn(config, {
      email: "operator@example.com",
      password: "secret",
    });
    const health = await repository.getHealth(config, session.token);
    const dashboard = await repository.getDashboard(config, session.token);
    const autopilot = await repository.getAutopilot(config, session.token);
    const preview = await repository.previewTrade(config, session.token, {
      action: "market_buy",
      symbol: "BTC/USDT",
      quantity: 0.001,
      quoteAmount: 0,
      dryRun: true,
      maxApiLatencyMs: 1200,
      maxTickerAgeMs: 3000,
      maxSpreadBps: 20,
      minTradeCooldownSeconds: 5,
    });

    expect(session.email).toBe("operator@example.com");
    expect(health.serviceHealth).toBe("healthy");
    expect(dashboard.decision.signal).toBe("BUY");
    expect(autopilot.safeNextAction).toBe("normal_start_allowed");
    expect(preview.guardMessage).toBe("Dry preview only.");
    expect(httpClient.calls[4]?.options?.body).toMatchObject({
      dry_run: true,
      quote_amount: 0,
    });
  });

  it("fails when login response does not include a token", async () => {
    const httpClient = new FakeHttpClient([{ auth: {} }]);
    const repository = new HttpTradingControlRepository(httpClient);

    await expect(
      repository.signIn(
        { baseUrl: "http://127.0.0.1:8765" },
        {
          email: "operator@example.com",
          password: "secret",
        },
      ),
    ).rejects.toThrow("Backend login response did not include a session token.");
  });
});
