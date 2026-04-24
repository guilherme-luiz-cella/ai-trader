export type AutopilotSnapshot = {
  status: string;
  running: boolean;
  currentCycle: number;
  targetCycles: number;
  symbol: string;
  updatedAt: string;
  goalValue: number;
  currentValue: number;
  progressPct: number;
  latestTradeStatus: string;
  safeNextAction: string;
  lastError: string;
};
