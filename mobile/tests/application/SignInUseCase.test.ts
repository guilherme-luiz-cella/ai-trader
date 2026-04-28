import { describe, expect, it } from "vitest";

import type { AuthSession } from "../../src/domain/entities/AuthSession";
import type { AutopilotSnapshot } from "../../src/domain/entities/AutopilotSnapshot";
import type { DashboardSnapshot } from "../../src/domain/entities/DashboardSnapshot";
import type { OperatorHealth } from "../../src/domain/entities/OperatorHealth";
import type { TradePreview, TradePreviewRequest, TradeRequest, TradeResult } from "../../src/domain/entities/TradePreview";
import type { BackendConfig } from "../../src/domain/value-objects/BackendConfig";
import { SignInUseCase } from "../../src/application/use-cases/SignInUseCase";
import type { TradingControlRepository, LoginCommand } from "../../src/application/ports/TradingControlRepository";

class StubRepository implements TradingControlRepository {
  public receivedConfig: BackendConfig | null = null;
  public receivedCommand: LoginCommand | null = null;

  async signIn(config: BackendConfig, command: LoginCommand): Promise<AuthSession> {
    this.receivedConfig = config;
    this.receivedCommand = command;

    return {
      token: "session-token",
      email: command.email,
      expiresInSeconds: 43200,
    };
  }

  async getHealth(): Promise<OperatorHealth> {
    throw new Error("unused");
  }

  async getDashboard(): Promise<DashboardSnapshot> {
    throw new Error("unused");
  }

  async getAutopilot(): Promise<AutopilotSnapshot> {
    throw new Error("unused");
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

describe("SignInUseCase", () => {
  it("signs in with normalized backend and lowercased email", async () => {
    const repository = new StubRepository();
    const useCase = new SignInUseCase(repository);

    const session = await useCase.execute({
      backendUrl: "http://127.0.0.1:8765/",
      email: "Operator@Example.com ",
      password: "secret",
    });

    expect(session.token).toBe("session-token");
    expect(repository.receivedConfig).toEqual({ baseUrl: "http://127.0.0.1:8765" });
    expect(repository.receivedCommand).toEqual({
      email: "operator@example.com",
      password: "secret",
    });
  });

  it("rejects empty email before calling the repository", async () => {
    const repository = new StubRepository();
    const useCase = new SignInUseCase(repository);

    await expect(
      useCase.execute({
        backendUrl: "http://127.0.0.1:8765",
        email: " ",
        password: "secret",
      }),
    ).rejects.toThrow("Email is required.");
  });
});
