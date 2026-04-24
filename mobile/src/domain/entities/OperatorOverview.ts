import type { AutopilotSnapshot } from "./AutopilotSnapshot";
import type { DashboardSnapshot } from "./DashboardSnapshot";
import type { OperatorHealth } from "./OperatorHealth";

export type OperatorOverview = {
  health: OperatorHealth;
  dashboard: DashboardSnapshot;
  autopilot: AutopilotSnapshot;
};
