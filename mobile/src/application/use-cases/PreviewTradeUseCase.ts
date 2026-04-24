import type { TradePreview, TradePreviewRequest } from "../../domain/entities/TradePreview";
import { normalizeBackendBaseUrl } from "../../domain/value-objects/BackendConfig";
import type { TradingControlRepository } from "../ports/TradingControlRepository";

type Input = {
  backendUrl: string;
  sessionToken: string;
  request: TradePreviewRequest;
};

const SUPPORTED_ACTIONS = new Set<TradePreviewRequest["action"]>([
  "market_buy",
  "market_sell",
  "cancel_all_orders",
]);

export class PreviewTradeUseCase {
  constructor(private readonly repository: TradingControlRepository) {}

  async execute(input: Input): Promise<TradePreview> {
    const config = normalizeBackendBaseUrl(input.backendUrl);
    const sessionToken = input.sessionToken.trim();

    if (!sessionToken) {
      throw new Error("Session token is required.");
    }

    if (!SUPPORTED_ACTIONS.has(input.request.action)) {
      throw new Error("Unsupported preview action.");
    }

    if (!input.request.symbol.trim()) {
      throw new Error("Symbol is required.");
    }

    return this.repository.previewTrade(config, sessionToken, {
      ...input.request,
      symbol: input.request.symbol.trim().toUpperCase(),
      dryRun: true,
    });
  }
}
