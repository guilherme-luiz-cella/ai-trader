import { describe, expect, it } from "vitest";

import type { AuthSession } from "../../src/domain/entities/AuthSession";
import type { AutopilotSnapshot } from "../../src/domain/entities/AutopilotSnapshot";
import type { DashboardSnapshot } from "../../src/domain/entities/DashboardSnapshot";
import type { OperatorHealth } from "../../src/domain/entities/OperatorHealth";
import type { TradePreview, TradePreviewRequest, TradeRequest, TradeResult } from "../../src/domain/entities/TradePreview";
import type { BackendConfig } from "../../src/domain/value-objects/BackendConfig";
import { ExecuteTradeUseCase } from "../../src/application/use-cases/ExecuteTradeUseCase";
import type { TradingControlRepository, LoginCommand } from "../../src/application/ports/TradingControlRepository";

class ExecuteRepository implements TradingControlRepository {
  public latestRequest: TradeRequest | null = null;

  async signIn(_config: BackendConfig, _command: LoginCommand): Promise<AuthSession> {
    throw new Error("unused");
  }

  async getHealth(_config: BackendConfig, _sessionToken: string): Promise<OperatorHealth> {
    throw new Error("unused");
  }

  async getDashboard(_config: BackendConfig, _sessionToken: string): Promise<DashboardSnapshot> {
    throw new Error("unused");
  }

  async getAutopilot(_config: BackendConfig, _sessionToken: string): Promise<AutopilotSnapshot> {
    throw new Error("unused");
  }

  async previewTrade(_config: BackendConfig, _sessionToken: string, _request: TradePreviewRequest): Promise<TradePreview> {
    throw new Error("unused");
  }

  async executeTrade(_config: BackendConfig, _sessionToken: string, request: TradeRequest): Promise<TradeResult> {
    this.latestRequest = request;

    return {
      status: "submitted",
      symbol: request.symbol,
      action: request.action,
      guardMessage: "",
      minimumMessage: "",
      effectiveQuantity: request.quantity,
      sizeCapReason: "",
      minQty: 0.0001,
      minNotional: 5,
      marketPrice: 50000,
      message: request.dryRun ? "Dry run." : "Live order submitted.",
    };
  }
}

describe("ExecuteTradeUseCase", () => {
  it("executes the requested live trade without forcing dry mode", async () => {
    const repository = new ExecuteRepository();
    const useCase = new ExecuteTradeUseCase(repository);

    const result = await useCase.execute({
      backendUrl: "http://127.0.0.1:8765",
      sessionToken: "session-token",
      request: {
        action: "market_buy",
        symbol: "btc/usdt",
        quantity: 0.001,
        quoteAmount: 0,
        dryRun: false,
        maxApiLatencyMs: 1200,
        maxTickerAgeMs: 3000,
        maxSpreadBps: 20,
        minTradeCooldownSeconds: 5,
      },
    });

    expect(result.status).toBe("submitted");
    expect(repository.latestRequest?.dryRun).toBe(false);
    expect(repository.latestRequest?.symbol).toBe("BTC/USDT");
  });

  it("rejects unsupported trade actions", async () => {
    const repository = new ExecuteRepository();
    const useCase = new ExecuteTradeUseCase(repository);

    await expect(
      useCase.execute({
        backendUrl: "http://127.0.0.1:8765",
        sessionToken: "session-token",
        request: {
          action: "hold" as TradeRequest["action"],
          symbol: "BTC/USDT",
          quantity: 0.001,
          quoteAmount: 0,
          dryRun: false,
          maxApiLatencyMs: 1200,
          maxTickerAgeMs: 3000,
          maxSpreadBps: 20,
          minTradeCooldownSeconds: 5,
        },
      }),
    ).rejects.toThrow("Unsupported trade action.");
  });
});
