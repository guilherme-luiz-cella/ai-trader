import type { OperatorOverview } from "../../domain/entities/OperatorOverview";
import { normalizeBackendBaseUrl } from "../../domain/value-objects/BackendConfig";
import type { TradingControlRepository } from "../ports/TradingControlRepository";

type Input = {
  backendUrl: string;
  sessionToken: string;
};

export class LoadOperatorOverviewUseCase {
  constructor(private readonly repository: TradingControlRepository) {}

  async execute(input: Input): Promise<OperatorOverview> {
    const config = normalizeBackendBaseUrl(input.backendUrl);
    const sessionToken = input.sessionToken.trim();

    if (!sessionToken) {
      throw new Error("Session token is required.");
    }

    const [health, dashboard, autopilot] = await Promise.all([
      this.repository.getHealth(config, sessionToken),
      this.repository.getDashboard(config, sessionToken),
      this.repository.getAutopilot(config, sessionToken),
    ]);

    return {
      health,
      dashboard,
      autopilot,
    };
  }
}
