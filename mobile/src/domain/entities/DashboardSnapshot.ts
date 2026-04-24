import type { SignalDecision } from "./SignalDecision";

export type DashboardSnapshot = {
  liveSymbol: string;
  decision: SignalDecision;
  estimatedWalletUsd: number;
  walletAssetCount: number;
  maxTradeSizeUsd: number;
  reserveCashUsd: number;
  previewGatePassed: boolean;
  readinessReason: string;
};
