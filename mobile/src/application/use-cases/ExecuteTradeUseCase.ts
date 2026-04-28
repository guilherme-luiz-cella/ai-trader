import type { TradeRequest, TradeResult } from "../../domain/entities/TradePreview";
import { normalizeBackendBaseUrl } from "../../domain/value-objects/BackendConfig";
import type { TradingControlRepository } from "../ports/TradingControlRepository";

type Input = {
  backendUrl: string;
  sessionToken: string;
  request: TradeRequest;
};

const SUPPORTED_ACTIONS = new Set<TradeRequest["action"]>([
  "market_buy",
  "market_sell",
  "cancel_all_orders",
]);

export class ExecuteTradeUseCase {
  constructor(private readonly repository: TradingControlRepository) {}

  async execute(input: Input): Promise<TradeResult> {
    const config = normalizeBackendBaseUrl(input.backendUrl);
    const sessionToken = input.sessionToken.trim();

    if (!sessionToken) {
      throw new Error("Session token is required.");
    }

    if (!SUPPORTED_ACTIONS.has(input.request.action)) {
      throw new Error("Unsupported trade action.");
    }

    if (!input.request.symbol.trim()) {
      throw new Error("Symbol is required.");
    }

    return this.repository.executeTrade(config, sessionToken, {
      ...input.request,
      symbol: input.request.symbol.trim().toUpperCase(),
    });
  }
}
