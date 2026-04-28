export type TradeRequest = {
  action: "market_buy" | "market_sell" | "cancel_all_orders";
  symbol: string;
  quantity: number;
  quoteAmount: number;
  maxApiLatencyMs: number;
  maxTickerAgeMs: number;
  maxSpreadBps: number;
  minTradeCooldownSeconds: number;
  dryRun: boolean;
};

export type TradeResult = {
  status: string;
  symbol: string;
  action: string;
  guardMessage: string;
  minimumMessage: string;
  effectiveQuantity: number;
  sizeCapReason: string;
  minQty: number;
  minNotional: number;
  marketPrice: number;
  message?: string;
  spreadBps?: number;
  tickerAgeMs?: number;
  guardMode?: string;
};

export type TradePreviewRequest = TradeRequest;
export type TradePreview = TradeResult;
