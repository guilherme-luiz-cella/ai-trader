export type SignalDecision = {
  signal: string;
  probabilityUp: number;
  mlProbabilityUp: number;
  decisionConfidence: number;
  decisionEngine: string;
  safeNextAction: string;
};
