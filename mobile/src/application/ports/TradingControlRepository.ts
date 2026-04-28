import type { AuthSession } from "../../domain/entities/AuthSession";
import type { AutopilotSnapshot } from "../../domain/entities/AutopilotSnapshot";
import type { DashboardSnapshot } from "../../domain/entities/DashboardSnapshot";
import type { OperatorHealth } from "../../domain/entities/OperatorHealth";
import type { TradePreview, TradePreviewRequest, TradeRequest, TradeResult } from "../../domain/entities/TradePreview";
import type { BackendConfig } from "../../domain/value-objects/BackendConfig";

export type LoginCommand = {
  email: string;
  password: string;
};

export interface TradingControlRepository {
  signIn(config: BackendConfig, command: LoginCommand): Promise<AuthSession>;
  getHealth(config: BackendConfig, sessionToken: string): Promise<OperatorHealth>;
  getDashboard(config: BackendConfig, sessionToken: string): Promise<DashboardSnapshot>;
  getAutopilot(config: BackendConfig, sessionToken: string): Promise<AutopilotSnapshot>;
  previewTrade(config: BackendConfig, sessionToken: string, request: TradePreviewRequest): Promise<TradePreview>;
  executeTrade(config: BackendConfig, sessionToken: string, request: TradeRequest): Promise<TradeResult>;
}
