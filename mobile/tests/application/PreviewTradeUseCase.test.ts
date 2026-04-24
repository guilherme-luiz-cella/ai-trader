import { describe, expect, it } from "vitest";

import type { AuthSession } from "../../src/domain/entities/AuthSession";
import type { AutopilotSnapshot } from "../../src/domain/entities/AutopilotSnapshot";
import type { DashboardSnapshot } from "../../src/domain/entities/DashboardSnapshot";
import type { OperatorHealth } from "../../src/domain/entities/OperatorHealth";
import type { TradePreview, TradePreviewRequest } from "../../src/domain/entities/TradePreview";
import type { BackendConfig } from "../../src/domain/value-objects/BackendConfig";
import { PreviewTradeUseCase } from "../../src/application/use-cases/PreviewTradeUseCase";
import type { TradingControlRepository, LoginCommand } from "../../src/application/ports/TradingControlRepository";

class PreviewRepository implements TradingControlRepository {
  public latestRequest: TradePreviewRequest | null = null;

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

  async previewTrade(_config: BackendConfig, _sessionToken: string, request: TradePreviewRequest): Promise<TradePreview> {
    this.latestRequest = request;

    return {
      status: "preview_ready",
      symbol: request.symbol,
      action: request.action,
      guardMessage: "Dry-run preview only.",
      minimumMessage: "",
      effectiveQuantity: request.quantity,
      sizeCapReason: "",
      minQty: 0.0001,
      minNotional: 5,
      marketPrice: 50000,
    };
  }
}

describe("PreviewTradeUseCase", () => {
  it("forces dry-run mode before calling the repository", async () => {
    const repository = new PreviewRepository();
    const useCase = new PreviewTradeUseCase(repository);

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

    expect(result.status).toBe("preview_ready");
    expect(repository.latestRequest?.dryRun).toBe(true);
    expect(repository.latestRequest?.symbol).toBe("BTC/USDT");
  });

  it("rejects unsupported actions", async () => {
    const repository = new PreviewRepository();
    const useCase = new PreviewTradeUseCase(repository);

    await expect(
      useCase.execute({
        backendUrl: "http://127.0.0.1:8765",
        sessionToken: "session-token",
        request: {
          action: "hold" as TradePreviewRequest["action"],
          symbol: "BTC/USDT",
          quantity: 0.001,
          quoteAmount: 0,
          dryRun: true,
          maxApiLatencyMs: 1200,
          maxTickerAgeMs: 3000,
          maxSpreadBps: 20,
          minTradeCooldownSeconds: 5,
        },
      }),
    ).rejects.toThrow("Unsupported preview action.");
  });
});
