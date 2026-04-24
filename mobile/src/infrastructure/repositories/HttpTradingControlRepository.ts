import type { AuthSession } from "../../domain/entities/AuthSession";
import type { AutopilotSnapshot } from "../../domain/entities/AutopilotSnapshot";
import type { DashboardSnapshot } from "../../domain/entities/DashboardSnapshot";
import type { OperatorHealth } from "../../domain/entities/OperatorHealth";
import type { SignalDecision } from "../../domain/entities/SignalDecision";
import type { TradePreview, TradePreviewRequest } from "../../domain/entities/TradePreview";
import type { BackendConfig } from "../../domain/value-objects/BackendConfig";
import type { TradingControlRepository, LoginCommand } from "../../application/ports/TradingControlRepository";
import type { HttpClient } from "../http/HttpClient";

type JsonObject = Record<string, unknown>;

function asObject(value: unknown): JsonObject {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as JsonObject;
  }

  return {};
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function authHeaders(sessionToken: string): Record<string, string> {
  return sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {};
}

function readSignalDecision(value: unknown): SignalDecision {
  const source = asObject(value);

  return {
    signal: asString(source.signal, "HOLD"),
    probabilityUp: asNumber(source.probability_up),
    mlProbabilityUp: asNumber(source.ml_probability_up),
    decisionConfidence: asNumber(source.decision_confidence),
    decisionEngine: asString(source.decision_engine, "ml"),
    safeNextAction: asString(source.safe_next_action, "manual_review_required_before_resume"),
  };
}

function readHealth(value: unknown): OperatorHealth {
  const source = asObject(value);
  const auth = asObject(source.auth);

  return {
    status: asString(source.status, "unknown"),
    service: asString(source.service, "signal-api"),
    serviceHealth: asString(source.service_health, "unknown"),
    loginEnabled: asBoolean(auth.login_enabled),
    cloudflareAccessEnabled: asBoolean(auth.cloudflare_access_enabled),
    apiTokenEnabled: asBoolean(auth.api_token_enabled),
  };
}

function readDashboard(value: unknown): DashboardSnapshot {
  const source = asObject(value);
  const config = asObject(source.config);
  const decisionPayload = asObject(source.decision_payload);
  const decision = readSignalDecision(decisionPayload.decision);
  const wallet = asObject(source.wallet_snapshot);
  const riskPlan = asObject(source.risk_plan);
  const readiness = asObject(source.live_start_gate);
  const checks = asObject(readiness.checks);

  return {
    liveSymbol: asString(config.live_symbol, "BTC/USDT"),
    decision,
    estimatedWalletUsd: asNumber(wallet.estimated_total_usdt),
    walletAssetCount: asNumber(wallet.asset_count),
    maxTradeSizeUsd: asNumber(riskPlan.max_trade_size),
    reserveCashUsd: asNumber(riskPlan.reserve_cash),
    previewGatePassed: asBoolean(checks.preview_gate_passed),
    readinessReason: asString(readiness.reason),
  };
}

function readAutopilot(value: unknown): AutopilotSnapshot {
  const source = asObject(value);
  const latestTrade = asObject(source.latest_trade_result);
  const signalTrade = asObject(latestTrade.signal_trade);

  return {
    status: asString(source.status, "idle"),
    running: asBoolean(source.running),
    currentCycle: asNumber(source.current_cycle),
    targetCycles: asNumber(source.target_cycles),
    symbol: asString(source.symbol, "BTC/USDT"),
    updatedAt: asString(source.updated_at),
    goalValue: asNumber(source.goal_value),
    currentValue: asNumber(source.current_value),
    progressPct: asNumber(source.progress_pct),
    latestTradeStatus: asString(signalTrade.status, asString(latestTrade.status, "--")),
    safeNextAction: asString(source.safe_next_action, "manual_review_required_before_resume"),
    lastError: asString(source.last_error),
  };
}

function readTradePreview(value: unknown): TradePreview {
  const source = asObject(value);

  return {
    status: asString(source.status, "unknown"),
    symbol: asString(source.symbol, "BTC/USDT"),
    action: asString(source.action, "hold"),
    guardMessage: asString(source.guard_message),
    minimumMessage: asString(source.minimum_message),
    effectiveQuantity: asNumber(source.effective_quantity),
    sizeCapReason: asString(source.size_cap_reason),
    minQty: asNumber(source.min_qty),
    minNotional: asNumber(source.min_notional),
    marketPrice: asNumber(source.market_price),
  };
}

export class HttpTradingControlRepository implements TradingControlRepository {
  constructor(private readonly httpClient: HttpClient) {}

  async signIn(config: BackendConfig, command: LoginCommand): Promise<AuthSession> {
    const payload = await this.httpClient.request<JsonObject>(`${config.baseUrl}/auth/login`, {
      method: "POST",
      body: command,
    });
    const auth = asObject(payload.auth);
    const token = asString(auth.token);

    if (!token) {
      throw new Error("Backend login response did not include a session token.");
    }

    return {
      token,
      email: asString(auth.email, command.email),
      expiresInSeconds: asNumber(auth.expires_in_seconds),
    };
  }

  async getHealth(config: BackendConfig, sessionToken: string): Promise<OperatorHealth> {
    const payload = await this.httpClient.request<JsonObject>(`${config.baseUrl}/health`, {
      headers: authHeaders(sessionToken),
    });

    return readHealth(payload);
  }

  async getDashboard(config: BackendConfig, sessionToken: string): Promise<DashboardSnapshot> {
    const payload = await this.httpClient.request<JsonObject>(`${config.baseUrl}/dashboard`, {
      headers: authHeaders(sessionToken),
    });

    return readDashboard(payload);
  }

  async getAutopilot(config: BackendConfig, sessionToken: string): Promise<AutopilotSnapshot> {
    const payload = await this.httpClient.request<JsonObject>(`${config.baseUrl}/autopilot/status`, {
      headers: authHeaders(sessionToken),
    });

    return readAutopilot(payload.autopilot);
  }

  async previewTrade(config: BackendConfig, sessionToken: string, request: TradePreviewRequest): Promise<TradePreview> {
    const payload = await this.httpClient.request<JsonObject>(`${config.baseUrl}/trade/preview`, {
      method: "POST",
      headers: authHeaders(sessionToken),
      body: {
        action: request.action,
        symbol: request.symbol,
        quantity: request.quantity,
        quote_amount: request.quoteAmount,
        dry_run: request.dryRun,
        max_api_latency_ms: request.maxApiLatencyMs,
        max_ticker_age_ms: request.maxTickerAgeMs,
        max_spread_bps: request.maxSpreadBps,
        min_trade_cooldown_seconds: request.minTradeCooldownSeconds,
      },
    });

    const result = asObject(payload.result);
    if (Object.keys(result).length === 0) {
      throw new Error("Backend preview response did not include a preview result.");
    }

    return readTradePreview(result);
  }
}
