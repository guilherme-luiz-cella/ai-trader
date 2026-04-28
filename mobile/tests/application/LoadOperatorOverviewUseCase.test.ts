import { describe, expect, it } from "vitest";

import type { AuthSession } from "../../src/domain/entities/AuthSession";
import type { AutopilotSnapshot } from "../../src/domain/entities/AutopilotSnapshot";
import type { DashboardSnapshot } from "../../src/domain/entities/DashboardSnapshot";
import type { OperatorHealth } from "../../src/domain/entities/OperatorHealth";
import type { TradePreview, TradePreviewRequest, TradeRequest, TradeResult } from "../../src/domain/entities/TradePreview";
import type { BackendConfig } from "../../src/domain/value-objects/BackendConfig";
import { LoadOperatorOverviewUseCase } from "../../src/application/use-cases/LoadOperatorOverviewUseCase";
import type { TradingControlRepository, LoginCommand } from "../../src/application/ports/TradingControlRepository";

class OverviewRepository implements TradingControlRepository {
  async signIn(_config: BackendConfig, _command: LoginCommand): Promise<AuthSession> {
    throw new Error("unused");
  }

  async getHealth(_config: BackendConfig, _sessionToken: string): Promise<OperatorHealth> {
    return {
      status: "ok",
      service: "signal-api",
      serviceHealth: "healthy",
      loginEnabled: true,
      cloudflareAccessEnabled: false,
      apiTokenEnabled: false,
    };
  }

  async getDashboard(_config: BackendConfig, _sessionToken: string): Promise<DashboardSnapshot> {
    return {
      liveSymbol: "BTC/USDT",
      decision: {
        signal: "BUY",
        probabilityUp: 0.76,
        mlProbabilityUp: 0.73,
        decisionConfidence: 0.68,
        decisionEngine: "ml",
        safeNextAction: "manual_review_required_before_resume",
      },
      estimatedWalletUsd: 1250,
      walletAssetCount: 4,
      maxTradeSizeUsd: 100,
      reserveCashUsd: 300,
      previewGatePassed: true,
      readinessReason: "",
    };
  }

  async getAutopilot(_config: BackendConfig, _sessionToken: string): Promise<AutopilotSnapshot> {
    return {
      status: "idle",
      running: false,
      currentCycle: 0,
      targetCycles: 0,
      symbol: "BTC/USDT",
      updatedAt: "2026-04-24T00:00:00Z",
      goalValue: 0,
      currentValue: 1250,
      progressPct: 0,
      latestTradeStatus: "--",
      safeNextAction: "normal_start_allowed",
      lastError: "",
    };
  }

  async previewTrade(
    _config: BackendConfig,
    _sessionToken: string,
    _request: TradePreviewRequest,
  ): Promise<TradePreview> {
    throw new Error("unused");
  }

  async executeTrade(
    _config: BackendConfig,
    _sessionToken: string,
    _request: TradeRequest,
  ): Promise<TradeResult> {
    throw new Error("unused");
  }
}

describe("LoadOperatorOverviewUseCase", () => {
  it("loads the combined operator overview", async () => {
    const repository = new OverviewRepository();
    const useCase = new LoadOperatorOverviewUseCase(repository);

    const overview = await useCase.execute({
      backendUrl: "http://127.0.0.1:8765/",
      sessionToken: "session-token",
    });

    expect(overview.health.status).toBe("ok");
    expect(overview.dashboard.decision.signal).toBe("BUY");
    expect(overview.autopilot.safeNextAction).toBe("normal_start_allowed");
  });

  it("fails fast when the session token is blank", async () => {
    const repository = new OverviewRepository();
    const useCase = new LoadOperatorOverviewUseCase(repository);

    await expect(
      useCase.execute({
        backendUrl: "http://127.0.0.1:8765",
        sessionToken: " ",
      }),
    ).rejects.toThrow("Session token is required.");
  });
});
