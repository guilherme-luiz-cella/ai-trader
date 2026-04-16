import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

const initialConfig = {
  live_symbol: "BTC/USDT",
};

const initialAutopilotForm = {
  goal_value: 0,
};

const initialTradeForm = {
  action: "market_buy",
  symbol: "BTC/USDT",
  quantity: 0.001,
  quote_amount: 0,
  dry_run: true,
  max_api_latency_ms: 1200,
  max_ticker_age_ms: 3000,
  max_spread_bps: 20,
  min_trade_cooldown_seconds: 5,
};

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.status === "error") {
    throw new Error(payload.message || `Request failed for ${path}`);
  }
  return payload;
}

function formatNumber(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatCurrency(value) {
  return `$${Number(value || 0).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatDate(value) {
  if (!value) return "--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
}

function formatActionLabel(action) {
  const normalized = String(action || "").trim().toLowerCase();
  if (normalized === "buy") return "Buying";
  if (normalized === "sell") return "Selling";
  if (normalized === "skip") return "Skipping";
  if (normalized === "market_buy") return "Buying";
  if (normalized === "market_sell") return "Selling";
  if (normalized === "cancel_all_orders") return "Cancelling";
  if (normalized === "hold") return "Holding";
  return normalized ? normalized.replace(/_/g, " ") : "Waiting";
}

function toneForSignal(signal) {
  if (signal === "BUY") return "good";
  if (signal === "SELL") return "bad";
  return "warn";
}

function MetricCard({ label, value, tone = "default", detail }) {
  return (
    <div className={`metric-card metric-${tone}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      {detail ? <span className="metric-detail">{detail}</span> : null}
    </div>
  );
}

function StatusBadge({ tone = "default", children }) {
  return <span className={`status-badge ${tone}`}>{children}</span>;
}

function toneForAiRuntime(status) {
  if (status?.health_status === "error") return "bad";
  if (status?.fallback_active) return "warn";
  if (status?.enabled && status?.is_trained_model) return "good";
  return "default";
}

function IntentCard({ title, symbol, action, detail, tone = "default", meta = [] }) {
  return (
    <article className={`intent-card tone-${tone}`}>
      <div className="intent-head">
        <span className="intent-title">{title}</span>
        <StatusBadge tone={tone}>{action || "Waiting"}</StatusBadge>
      </div>
      <strong className="intent-symbol">{symbol || "--"}</strong>
      <p className="intent-detail">{detail || "No active instruction from the backend yet."}</p>
      {meta.length ? (
        <div className="intent-meta">
          {meta.map((item) => (
            <span key={`${item.label}-${item.value}`}>
              <small>{item.label}</small>
              <strong>{item.value}</strong>
            </span>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function DataTable({ columns, rows, emptyMessage = "No rows yet." }) {
  if (!rows || rows.length === 0) {
    return <p className="empty-state">{emptyMessage}</p>;
  }
  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${rowIndex}-${row[columns[0].key] ?? "row"}`}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row[column.key], row) : String(row[column.key] ?? "--")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LineChart({ title, rows, lines, yFormatter }) {
  const width = 640;
  const height = 240;
  const padding = 28;
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  const seriesValues = [];

  rows.forEach((row) => {
    lines.forEach((line) => {
      const value = Number(row[line.key]);
      if (Number.isFinite(value)) {
        seriesValues.push(value);
      }
    });
  });

  if (rows.length < 2 || seriesValues.length === 0) {
    return (
      <div className="chart-shell">
        <div className="panel-header">
          <h3>{title}</h3>
        </div>
        <p className="empty-state">Not enough live data yet for a chart.</p>
      </div>
    );
  }

  let minValue = Math.min(...seriesValues);
  let maxValue = Math.max(...seriesValues);
  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }

  const paths = lines.map((line) => {
    const points = rows
      .map((row, index) => {
        const value = Number(row[line.key]);
        if (!Number.isFinite(value)) {
          return null;
        }
        const x = padding + (usableWidth * index) / Math.max(rows.length - 1, 1);
        const y = padding + usableHeight - ((value - minValue) / (maxValue - minValue)) * usableHeight;
        return `${x},${y}`;
      })
      .filter(Boolean)
      .join(" ");
    return { ...line, points };
  });

  return (
    <div className="chart-shell">
      <div className="panel-header">
        <h3>{title}</h3>
        <span className="panel-caption">
          Range {yFormatter ? yFormatter(minValue) : formatNumber(minValue)} to {yFormatter ? yFormatter(maxValue) : formatNumber(maxValue)}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img" aria-label={title}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="chart-axis" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="chart-axis" />
        {paths.map((line) => (
          <polyline key={line.key} fill="none" stroke={line.color} strokeWidth="3" points={line.points} strokeLinecap="round" strokeLinejoin="round" />
        ))}
      </svg>
      <div className="chart-legend">
        {lines.map((line) => (
          <span key={line.key}>
            <i style={{ background: line.color }} />
            {line.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function ChartToggle({ active, onClick, children }) {
  return (
    <button type="button" className={`chart-toggle ${active ? "active" : ""}`} onClick={onClick}>
      {children}
    </button>
  );
}

function MarketMonitorChart({ chart, timeframe, onTimeframeChange, overlays, onOverlayChange, loading }) {
  const [hoverIndex, setHoverIndex] = useState(null);
  const [visibleCount, setVisibleCount] = useState(120);
  const [offsetFromRight, setOffsetFromRight] = useState(0);
  const dragStateRef = useRef({ active: false, x: 0 });
  const width = 980;
  const height = 420;
  const padding = { top: 28, right: 72, bottom: 72, left: 20 };
  const candleAreaHeight = 280;
  const volumeTop = padding.top + candleAreaHeight + 22;
  const volumeHeight = 52;
  const candles = chart?.candles || [];
  const markers = chart?.markers || [];
  const currentPrice = Number(chart?.current_price || 0);
  const targetPrice = Number(chart?.target_price || 0);
  const latestMarker = chart?.latest_marker || null;
  const maxVisible = Math.max(30, candles.length || 30);
  const effectiveVisibleCount = Math.max(30, Math.min(maxVisible, visibleCount));
  const maxOffset = Math.max(0, candles.length - effectiveVisibleCount);
  const effectiveOffset = Math.max(0, Math.min(maxOffset, offsetFromRight));
  const startIndex = Math.max(0, candles.length - effectiveVisibleCount - effectiveOffset);
  const visibleCandles = candles.slice(startIndex, startIndex + effectiveVisibleCount);
  const latestClose = Number(visibleCandles[visibleCandles.length - 1]?.close || currentPrice || 0);
  const latestVolume = Number(visibleCandles[visibleCandles.length - 1]?.volume || 0);
  const allPrices = visibleCandles.flatMap((candle) => [Number(candle.high), Number(candle.low)]);
  if (currentPrice > 0) allPrices.push(currentPrice);
  if (overlays.showTarget && targetPrice > 0) allPrices.push(targetPrice);

  if (candles.length < 2 || allPrices.length === 0) {
    return (
      <article className="chart-shell">
        <div className="panel-header">
          <h3>Live Market Graph</h3>
          <span className="panel-caption">{chart?.symbol || "--"} · {timeframe}</span>
        </div>
        <p className="empty-state">{loading ? "Loading market candles..." : "Not enough candle data yet for the active symbol."}</p>
      </article>
    );
  }

  let minPrice = Math.min(...allPrices);
  let maxPrice = Math.max(...allPrices);
  if (minPrice === maxPrice) {
    minPrice -= 1;
    maxPrice += 1;
  }
  const paddedRange = (maxPrice - minPrice) * 0.08;
  minPrice -= paddedRange;
  maxPrice += paddedRange;
  const usableWidth = width - padding.left - padding.right;
  const candleSlot = usableWidth / visibleCandles.length;
  const candleBodyWidth = Math.max(3, candleSlot * 0.58);
  const maxVolume = Math.max(...visibleCandles.map((candle) => Number(candle.volume || 0)), 1);
  const priceToY = (price) => padding.top + ((maxPrice - price) / (maxPrice - minPrice)) * candleAreaHeight;
  const volumeToHeight = (volume) => (Number(volume || 0) / maxVolume) * volumeHeight;
  const xForIndex = (index) => padding.left + candleSlot * index + candleSlot / 2;
  const markerToColor = (marker) => {
    if (marker.type === "skip") return "#ffcb57";
    if (marker.action === "BUY") return "#2fe38d";
    if (marker.action === "SELL") return "#ff6b7d";
    if (marker.action === "HOLD") return "#84a8ff";
    return "#b9c6dc";
  };
  const markerToGlyph = (marker) => {
    if (marker.type === "skip") return "S";
    if (marker.action === "BUY") return "B";
    if (marker.action === "SELL") return "S";
    if (marker.action === "HOLD") return "H";
    return "•";
  };
  const markerNodes = markers
    .map((marker, index) => {
      const markerTime = Date.parse(marker.timestamp || "");
      if (!Number.isFinite(markerTime)) return null;
      let nearestIndex = 0;
      let nearestDistance = Number.POSITIVE_INFINITY;
      visibleCandles.forEach((candle, candleIndex) => {
        const candleTime = Number(candle.open_time_ms || 0);
        const distance = Math.abs(candleTime - markerTime);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestIndex = candleIndex;
        }
      });
      const x = xForIndex(nearestIndex);
      const price = Number(marker.price || visibleCandles[nearestIndex]?.close || latestClose);
      const y = priceToY(price);
      const color = markerToColor(marker);
      return { marker, index, x, y, color };
    })
    .filter(Boolean);
  const hoveredCandle = hoverIndex !== null ? visibleCandles[Math.max(0, Math.min(visibleCandles.length - 1, hoverIndex))] : null;
  const hoveredX = hoverIndex !== null ? xForIndex(Math.max(0, Math.min(visibleCandles.length - 1, hoverIndex))) : null;

  function clampOffset(nextValue, nextVisibleCount = effectiveVisibleCount) {
    const maxNextOffset = Math.max(0, candles.length - nextVisibleCount);
    return Math.max(0, Math.min(maxNextOffset, nextValue));
  }

  function handleWheel(event) {
    event.preventDefault();
    const zoomOut = event.deltaY > 0;
    const step = zoomOut ? 12 : -12;
    const nextVisibleCount = Math.max(30, Math.min(maxVisible, effectiveVisibleCount + step));
    setVisibleCount(nextVisibleCount);
    setOffsetFromRight((current) => clampOffset(current, nextVisibleCount));
  }

  function handleMouseMove(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const localX = ((event.clientX - rect.left) / rect.width) * width;
    const relativeX = localX - padding.left;
    const idx = Math.max(0, Math.min(visibleCandles.length - 1, Math.floor(relativeX / candleSlot)));
    setHoverIndex(idx);
    if (!dragStateRef.current.active) {
      return;
    }
    const deltaPx = event.clientX - dragStateRef.current.x;
    const deltaCandles = Math.trunc(deltaPx / Math.max(1, candleSlot));
    if (deltaCandles !== 0) {
      setOffsetFromRight((current) => clampOffset(current - deltaCandles));
      dragStateRef.current.x = event.clientX;
    }
  }

  function handleMouseDown(event) {
    dragStateRef.current = { active: true, x: event.clientX };
  }

  function handleMouseUp() {
    dragStateRef.current = { active: false, x: 0 };
  }

  return (
    <article className="chart-shell market-monitor-card">
      <div className="panel-header">
        <div>
          <h3>Live Market Graph</h3>
          <span className="panel-caption">{chart?.symbol || "--"} · {timeframe} candles · {candles.length} bars</span>
        </div>
        <div className="market-chart-meta">
          <StatusBadge tone={chart?.guard_mode === "blocked" ? "bad" : chart?.guard_mode === "degraded" || chart?.guard_mode === "exit_only" ? "warn" : "good"}>
            {chart?.guard_mode || "normal"}
          </StatusBadge>
          <strong>{formatNumber(currentPrice || latestClose, 4)}</strong>
        </div>
      </div>

      <div className="chart-controls">
        <div className="chart-timeframes">
          {(chart?.supported_intervals || ["1m", "5m", "15m", "1h"]).map((interval) => (
            <ChartToggle key={interval} active={interval === timeframe} onClick={() => onTimeframeChange(interval)}>
              {interval}
            </ChartToggle>
          ))}
        </div>
        <div className="chart-overlays">
          <ChartToggle active={overlays.showTrades} onClick={() => onOverlayChange("showTrades")}>Trades</ChartToggle>
          <ChartToggle active={overlays.showTarget} onClick={() => onOverlayChange("showTarget")}>Target</ChartToggle>
          <ChartToggle active={overlays.showConversions} onClick={() => onOverlayChange("showConversions")}>Conversions</ChartToggle>
          <ChartToggle active={overlays.showVolume} onClick={() => onOverlayChange("showVolume")}>Volume</ChartToggle>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="market-chart"
        role="img"
        aria-label={`Live chart for ${chart?.symbol || "market symbol"}`}
        onWheel={handleWheel}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => {
          setHoverIndex(null);
          handleMouseUp();
        }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
      >
        <line x1={padding.left} y1={padding.top + candleAreaHeight} x2={width - padding.right} y2={padding.top + candleAreaHeight} className="chart-axis" />
        <line x1={width - padding.right} y1={padding.top} x2={width - padding.right} y2={padding.top + candleAreaHeight} className="chart-axis" />

        {[0.2, 0.5, 0.8].map((ratio) => {
          const price = maxPrice - (maxPrice - minPrice) * ratio;
          const y = priceToY(price);
          return (
            <g key={ratio}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} className="chart-grid-line" />
              <text x={width - padding.right + 10} y={y + 4} className="chart-label">{formatNumber(price, 4)}</text>
            </g>
          );
        })}

        {visibleCandles.map((candle, index) => {
          const x = xForIndex(index);
          const open = Number(candle.open || 0);
          const close = Number(candle.close || 0);
          const high = Number(candle.high || 0);
          const low = Number(candle.low || 0);
          const bullish = close >= open;
          const bodyTop = priceToY(Math.max(open, close));
          const bodyBottom = priceToY(Math.min(open, close));
          const bodyHeight = Math.max(2, bodyBottom - bodyTop);
          const volumeHeightValue = volumeToHeight(candle.volume);
          return (
            <g key={`${candle.open_time_ms}-${index}`}>
              <line x1={x} y1={priceToY(high)} x2={x} y2={priceToY(low)} className={`candle-wick ${bullish ? "bull" : "bear"}`} />
              <rect
                x={x - candleBodyWidth / 2}
                y={bodyTop}
                width={candleBodyWidth}
                height={bodyHeight}
                rx="2"
                className={`candle-body ${bullish ? "bull" : "bear"}`}
              />
              {overlays.showVolume ? (
                <rect
                  x={x - candleBodyWidth / 2}
                  y={volumeTop + volumeHeight - volumeHeightValue}
                  width={candleBodyWidth}
                  height={Math.max(2, volumeHeightValue)}
                  className={`volume-bar ${bullish ? "bull" : "bear"}`}
                />
              ) : null}
            </g>
          );
        })}

        <line x1={padding.left} y1={priceToY(currentPrice || latestClose)} x2={width - padding.right} y2={priceToY(currentPrice || latestClose)} className="price-line" />
        <text x={width - padding.right + 10} y={priceToY(currentPrice || latestClose) - 6} className="chart-label accent">Live {formatNumber(currentPrice || latestClose, 4)}</text>

        {overlays.showTarget && targetPrice > 0 ? (
          <>
            <line x1={padding.left} y1={priceToY(targetPrice)} x2={width - padding.right} y2={priceToY(targetPrice)} className="target-line" />
            <text x={width - padding.right + 10} y={priceToY(targetPrice) - 6} className="chart-label target">Target {formatNumber(targetPrice, 4)}</text>
          </>
        ) : null}

        {overlays.showTrades
          ? markerNodes
              .filter(({ marker }) => (marker.type !== "skip" || overlays.showTrades) && (!marker.conversion_happened || overlays.showConversions || marker.type === "skip"))
              .map(({ marker, index, x, y, color }) => (
                <g key={`${marker.timestamp}-${index}`}>
                  <circle cx={x} cy={y} r="10" fill={color} className="trade-marker-dot" />
                  <text x={x} y={y + 4} textAnchor="middle" className="trade-marker-label">{markerToGlyph(marker)}</text>
                  <title>
                    {`${marker.action} · ${marker.trade_status || marker.type}${marker.skip_reason ? ` · ${marker.skip_reason}` : ""}${marker.guard_mode ? ` · ${marker.guard_mode}` : ""}`}
                  </title>
                </g>
              ))
          : null}

        {latestMarker ? (
          <g>
            <line x1={padding.left} y1={18} x2={padding.left + 24} y2={18} className="latest-marker-line" />
            <text x={padding.left + 30} y={22} className="chart-label accent">
              Last {latestMarker.action}{latestMarker.skip_reason ? ` · ${latestMarker.skip_reason}` : ""}
            </text>
          </g>
        ) : null}

        {hoveredCandle && hoveredX !== null ? (
          <>
            <line x1={hoveredX} y1={padding.top} x2={hoveredX} y2={padding.top + candleAreaHeight + (overlays.showVolume ? volumeHeight + 24 : 0)} className="crosshair-line" />
            <text x={hoveredX + 8} y={padding.top + 14} className="chart-label accent">
              O {formatNumber(hoveredCandle.open, 4)} H {formatNumber(hoveredCandle.high, 4)} L {formatNumber(hoveredCandle.low, 4)} C {formatNumber(hoveredCandle.close, 4)}
            </text>
          </>
        ) : null}

        <text x={padding.left} y={height - 18} className="chart-label">{visibleCandles[0]?.open_time ? formatDate(visibleCandles[0].open_time) : "--"}</text>
        <text x={width - padding.right - 12} y={height - 18} textAnchor="end" className="chart-label">{visibleCandles[visibleCandles.length - 1]?.open_time ? formatDate(visibleCandles[visibleCandles.length - 1].open_time) : "--"}</text>
      </svg>

      <div className="chart-legend market-chart-legend">
        <span><i className="legend-line live" /> Price</span>
        <span><i className="legend-line target" /> Target</span>
        <span><i className="legend-dot buy" /> Trade markers</span>
        <span><i className="legend-dot skip" /> Skip markers</span>
      </div>

      <div className="chart-summary-grid">
        <div><span>Latest close</span><strong>{formatNumber(latestClose, 4)}</strong></div>
        <div><span>Latest volume</span><strong>{formatNumber(latestVolume, 2)}</strong></div>
        <div><span>Confidence</span><strong>{formatPercent(chart?.latest_decision?.decision_confidence || chart?.latest_decision?.probability_up || 0)}</strong></div>
        <div><span>Last bot action</span><strong>{latestMarker?.action || "--"}</strong></div>
      </div>
      <p className="panel-caption">Wheel to zoom, drag to pan, hover for candle crosshair.</p>
    </article>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [health, setHealth] = useState("checking");
  const [config, setConfig] = useState(initialConfig);
  const [dashboard, setDashboard] = useState(null);
  const [autopilotState, setAutopilotState] = useState(null);
  const [autopilotForm, setAutopilotForm] = useState(initialAutopilotForm);
  const [tradeForm, setTradeForm] = useState(initialTradeForm);
  const [tradePreview, setTradePreview] = useState(null);
  const [tradeResult, setTradeResult] = useState(null);
  const [accountSnapshot, setAccountSnapshot] = useState(null);
  const [supportPrompt, setSupportPrompt] = useState("");
  const [supportMessages, setSupportMessages] = useState([]);
  const [aiCommand, setAiCommand] = useState("");
  const [feedback, setFeedback] = useState({ kind: "", message: "" });
  const [loadingDashboard, setLoadingDashboard] = useState(false);
  const [submittingAutopilot, setSubmittingAutopilot] = useState(false);
  const [submittingTrade, setSubmittingTrade] = useState(false);
  const [submittingChat, setSubmittingChat] = useState(false);
  const [chartInterval, setChartInterval] = useState("5m");
  const [chartData, setChartData] = useState(null);
  const [loadingChart, setLoadingChart] = useState(false);
  const [liveTicker, setLiveTicker] = useState(null);
  const [loadingTicker, setLoadingTicker] = useState(false);
  const [autopilotEvents, setAutopilotEvents] = useState(null);
  const [chartOverlays, setChartOverlays] = useState({
    showTrades: true,
    showTarget: true,
    showConversions: true,
    showVolume: true,
  });
  const activeChartSymbol = autopilotState?.symbol || config.live_symbol || "BTC/USDT";

  async function refreshHealth() {
    try {
      const payload = await request("/health");
      setHealth(payload.status || "ok");
    } catch (error) {
      setHealth("offline");
      setFeedback({ kind: "error", message: error.message });
    }
  }

  async function refreshDashboard(options = {}) {
    const { silent = false } = options;
    if (!silent) {
      setLoadingDashboard(true);
      setFeedback({ kind: "", message: "" });
    }

    try {
      const payload = await request("/dashboard");
      const nextConfig = payload.config || initialConfig;
      setDashboard(payload);
      setAutopilotState(payload.autopilot || null);
      setConfig(nextConfig);
      setAccountSnapshot(payload.account_snapshot || null);

      setTradeForm((current) => ({
        ...current,
        symbol: nextConfig.live_symbol,
      }));
      setAutopilotForm((current) => ({
        ...current,
        goal_value: current.goal_value > 0 ? current.goal_value : Number(payload.goal_value || 0),
      }));
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      if (!silent) {
        setLoadingDashboard(false);
      }
    }
  }

  async function refreshAutopilotStatus() {
    try {
      const payload = await request("/autopilot/status");
      setAutopilotState(payload.autopilot || null);
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    }
  }

  async function refreshMarketChart(symbol = activeChartSymbol, interval = chartInterval, options = {}) {
    const { silent = false } = options;
    if (!silent) {
      setLoadingChart(true);
    }
    try {
      const payload = await request(`/market/chart?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}&limit=200`);
      setChartData(payload.chart || null);
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      if (!silent) {
        setLoadingChart(false);
      }
    }
  }

  async function refreshMarketTicker(symbol = activeChartSymbol, options = {}) {
    const { silent = false } = options;
    if (!silent) {
      setLoadingTicker(true);
    }
    try {
      const payload = await request(`/market/ticker?symbol=${encodeURIComponent(symbol)}`);
      setLiveTicker(payload.ticker || null);
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      if (!silent) {
        setLoadingTicker(false);
      }
    }
  }

  async function refreshAutopilotEvents(options = {}) {
    const { silent = false } = options;
    try {
      const payload = await request("/autopilot/events?limit=100");
      setAutopilotEvents(payload.autopilot_events || null);
    } catch (error) {
      if (!silent) {
        setFeedback({ kind: "error", message: error.message });
      }
    }
  }

  useEffect(() => {
    refreshHealth();
    refreshDashboard();
  }, []);

  useEffect(() => {
    if (!activeChartSymbol) {
      return undefined;
    }
    refreshMarketChart(activeChartSymbol, chartInterval);
    refreshMarketTicker(activeChartSymbol);
    refreshAutopilotEvents({ silent: true });
    const intervalHandle = window.setInterval(() => {
      refreshMarketChart(activeChartSymbol, chartInterval, { silent: true });
      refreshMarketTicker(activeChartSymbol, { silent: true });
      refreshAutopilotEvents({ silent: true });
    }, 5000);
    return () => window.clearInterval(intervalHandle);
  }, [activeChartSymbol, chartInterval]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      refreshHealth();
      refreshAutopilotStatus();
    }, 5000);
    return () => window.clearInterval(interval);
  }, []);

  function updateTradeField(name, value) {
    setTradeForm((current) => ({ ...current, [name]: value }));
  }

  function updateAutopilotField(name, value) {
    setAutopilotForm((current) => ({ ...current, [name]: value }));
  }

  function updateChartOverlay(name) {
    setChartOverlays((current) => ({ ...current, [name]: !current[name] }));
  }

  async function handleStartAutopilot(allowLive) {
    setSubmittingAutopilot(true);
    setFeedback({ kind: "", message: "" });
    const body = {
      allow_live: allowLive,
      symbol: config.live_symbol,
      auto_rebalance_enabled: false,
    };
    if (Number(autopilotForm.goal_value) > 0) {
      body.goal_value = Number(autopilotForm.goal_value);
    }
    try {
      const payload = await request("/autopilot/start", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setAutopilotState(payload.autopilot || null);
      setFeedback({
        kind: "success",
        message: allowLive ? "Live autopilot is running from the Python backend." : "Dry-run autopilot started on the backend.",
      });
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      setSubmittingAutopilot(false);
    }
  }

  async function handleStopAutopilot() {
    setSubmittingAutopilot(true);
    try {
      const payload = await request("/autopilot/stop", {
        method: "POST",
        body: JSON.stringify({}),
      });
      setAutopilotState(payload.autopilot || null);
      setFeedback({ kind: "success", message: "Autopilot stop request sent to the backend." });
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      setSubmittingAutopilot(false);
    }
  }

  async function handleRefreshAccount() {
    try {
      const payload = await request(`/account?symbol=${encodeURIComponent(tradeForm.symbol)}`);
      setAccountSnapshot(payload.account || null);
      setFeedback({ kind: "success", message: "Account snapshot refreshed." });
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    }
  }

  async function handlePreviewTrade() {
    setSubmittingTrade(true);
    try {
      const payload = await request("/trade/preview", {
        method: "POST",
        body: JSON.stringify(tradeForm),
      });
      setTradePreview(payload.result || null);
      setTradeResult(null);
      setFeedback({ kind: "success", message: "Trade preview updated from the backend guardrails." });
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      setSubmittingTrade(false);
    }
  }

  async function handleRunTrade() {
    setSubmittingTrade(true);
    try {
      const payload = await request("/trade/action", {
        method: "POST",
        body: JSON.stringify(tradeForm),
      });
      setTradeResult(payload.result || null);
      setFeedback({
        kind: "success",
        message: tradeForm.dry_run ? "Dry-run trade sent to the backend." : "Live trade request sent to Binance through the backend.",
      });
      refreshDashboard({ silent: true });
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    } finally {
      setSubmittingTrade(false);
    }
  }

  async function handleRunAiCommand() {
    if (!aiCommand.trim()) {
      return;
    }
    try {
      const payload = await request("/ai/command", {
        method: "POST",
        body: JSON.stringify({ command: aiCommand }),
      });
      const result = payload.result || {};
      await refreshDashboard();
      setFeedback({ kind: "success", message: result.message || "AI command applied." });
      setAiCommand("");
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    }
  }

  async function handleSupportChat(event) {
    event.preventDefault();
    if (!supportPrompt.trim()) {
      return;
    }
    const prompt = supportPrompt;
    const mergeContext = {
      decision_engine: dashboard?.decision_payload?.decision?.decision_engine,
      signal: dashboard?.decision_payload?.decision?.signal,
      probability_up: dashboard?.decision_payload?.decision?.probability_up,
      ml_probability_up: dashboard?.decision_payload?.decision?.ml_probability_up,
      llm_overlay_status: dashboard?.decision_payload?.decision?.llm_overlay?.status,
      llm_merge_status: dashboard?.decision_payload?.decision?.llm_merge?.status,
    };

    setSupportMessages((current) => [...current, { role: "user", content: prompt }]);
    setSupportPrompt("");
    setSubmittingChat(true);

    try {
      const payload = await request("/support/chat", {
        method: "POST",
        body: JSON.stringify({
          message: prompt,
          context: mergeContext,
        }),
      });
      const result = payload.result || {};
      setSupportMessages((current) => [...current, { role: "assistant", content: result.answer || "No answer." }]);
    } catch (error) {
      setSupportMessages((current) => [...current, { role: "assistant", content: error.message }]);
    } finally {
      setSubmittingChat(false);
    }
  }

  const decision = dashboard?.decision_payload?.decision || {};
  const wallet = dashboard?.wallet_snapshot;
  const liveHistory = dashboard?.live_history || [];
  const marketScan = dashboard?.market_scan || [];
  const riskPlan = dashboard?.risk_plan || {};
  const sizePlan = dashboard?.size_plan || {};
  const cyclePlan = dashboard?.cycle_plan || {};
  const llmStatus = dashboard?.llm_status || {};
  const aiRuntimeTone = toneForAiRuntime(llmStatus);
  const aiRuntimeHeadline =
    llmStatus?.health_status === "error"
      ? "Primary trained model is unavailable."
      : llmStatus?.fallback_active
        ? "Fallback model is active instead of your trained model."
        : llmStatus?.enabled && llmStatus?.is_trained_model
          ? "Your trained model is active."
          : "Local AI runtime is disabled.";
  const aiRuntimeDetail =
    llmStatus?.health_error ||
    llmStatus?.fallback_reason ||
    llmStatus?.active_model_path ||
    llmStatus?.active_model ||
    "No local AI model is active.";
  const autopilotLogs = autopilotEvents?.events || autopilotState?.logs || [];
  const autopilotPreview = dashboard?.autopilot_preview || {};
  const latestTradeState = autopilotState?.latest_trade_result || {};
  const autopilotSignal = latestTradeState?.final_signal || decision.signal || "HOLD";
  const autopilotAction = latestTradeState?.final_action || (autopilotSignal === "BUY" ? "market_buy" : autopilotSignal === "SELL" ? "market_sell" : "hold");
  const autopilotTone = autopilotSignal === "BUY" ? "good" : autopilotSignal === "SELL" ? "bad" : "default";
  const autopilotExecutionPlan = latestTradeState?.execution_plan || autopilotPreview?.execution_plan || {};
  const autopilotGuardMode = latestTradeState?.signal_trade?.guard_mode || autopilotExecutionPlan?.guard_mode || autopilotState?.execution_mode || "normal";
  const autopilotWalletSummary = latestTradeState?.wallet_summary || autopilotState?.wallet_summary || autopilotPreview?.wallet_summary || {};
  const latestConversionResult = latestTradeState?.conversion_result || {};
  const latestConversionPlan = latestTradeState?.conversion_plan || autopilotState?.conversion_plan || autopilotPreview?.conversion_plan || {};
  const autopilotActiveSymbol = autopilotState?.symbol || config.live_symbol || "--";
  const autopilotRequiredQuoteAsset = latestTradeState?.required_quote_asset || autopilotState?.required_quote_asset || autopilotExecutionPlan?.required_quote_asset || "--";
  const autopilotRequiredBaseAsset = latestTradeState?.required_base_asset || autopilotState?.required_base_asset || autopilotExecutionPlan?.required_base_asset || "--";
  const autopilotSkipReason = latestTradeState?.signal_trade?.skip_reason || latestTradeState?.skip_reason || autopilotExecutionPlan?.skip_reason || autopilotState?.skip_reason || "--";
  const autopilotComputedSizeBase = autopilotExecutionPlan?.computed_order_size_base ?? autopilotState?.computed_order_size_base ?? 0;
  const autopilotComputedSizeQuote = autopilotExecutionPlan?.computed_order_size_quote ?? autopilotState?.computed_order_size_quote ?? 0;
  const fundingDiagnostics = latestTradeState?.funding_diagnostics || autopilotState?.funding_diagnostics || {};
  const eligibleFundingAssets = latestTradeState?.eligible_funding_assets || autopilotState?.eligible_funding_assets || [];
  const freeBnb = Number(latestTradeState?.free_bnb ?? autopilotState?.free_bnb ?? 0);
  const autopilotStartingValue = Number(autopilotState?.starting_value ?? 0);
  const autopilotCurrentValue = Number(autopilotState?.current_value ?? 0);
  const autopilotGoalValue = Number(autopilotState?.goal_value ?? 0);
  const autopilotProgressPct = Number(autopilotState?.progress_pct ?? 0);
  const autopilotContinueUntilGoal = Boolean(autopilotState?.continue_until_goal);
  const autopilotExtraCyclesUsed = Number(autopilotState?.extra_cycles_used ?? 0);
  const autopilotFailedCycles = Number(autopilotState?.failed_cycles_in_row ?? 0);
  const autopilotFinalStableAsset = autopilotState?.final_stable_target_asset || "--";
  const autopilotFinalizationStatus = autopilotState?.finalization_status || "--";
  const autopilotFinalStopReason = autopilotState?.final_stop_reason || "--";
  const opportunityRankings = latestTradeState?.opportunity_rankings || autopilotState?.opportunity_rankings || dashboard?.opportunity_panel?.ranked || [];
  const opportunityWinner = latestTradeState?.opportunity_winner || autopilotState?.opportunity_winner || dashboard?.opportunity_panel?.winner || {};
  const opportunityMeta = latestTradeState?.opportunity_meta || autopilotState?.opportunity_meta || dashboard?.opportunity_panel?.meta || {};
  const currentTradeSymbol = tradeResult?.symbol || tradePreview?.symbol || tradeForm.symbol;
  const currentTradeAction = tradeResult?.action || tradePreview?.action || tradeForm.action;
  const currentTradeTone = currentTradeAction === "market_buy" ? "good" : currentTradeAction === "market_sell" ? "bad" : "default";

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "live", label: "Live Terminal" },
    { id: "wallet", label: "Wallet" },
    { id: "account", label: "Account" },
    { id: "ai", label: "AI Desk" },
  ];

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Frontend in React, backend in Python</p>
          <h1>Trading Control Room</h1>
          <p className="hero-copy">
            The React app now owns the operator experience. The Python backend owns signals, autopilot, wallet access, live capture,
            trade execution, and AI support services.
          </p>
        </div>
        <div className="hero-actions">
          <button className="ghost-button" onClick={() => refreshDashboard()} disabled={loadingDashboard}>
            {loadingDashboard ? "Refreshing..." : "Refresh Dashboard"}
          </button>
          <button className="ghost-button" onClick={refreshHealth}>
            Refresh API
          </button>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard label="API Health" value={health.toUpperCase()} tone={health === "ok" ? "good" : "warn"} />
        <MetricCard label="AI Runtime" value={(llmStatus?.provider || "none").toUpperCase()} detail={llmStatus?.is_trained_model ? "trained model" : "fallback or disabled"} />
        <MetricCard label="Signal" value={decision.signal || "--"} tone={toneForSignal(decision.signal)} detail={decision.decision_engine || "ml"} />
        <MetricCard label="Probability Up" value={formatPercent(decision.probability_up)} detail={`ML ${formatPercent(decision.ml_probability_up)}`} />
        <MetricCard
          label="Autopilot"
          value={(autopilotState?.status || "idle").toUpperCase()}
          tone={autopilotState?.running ? "good" : "default"}
          detail={`${autopilotState?.current_cycle || 0}/${autopilotState?.target_cycles || 0} cycles`}
        />
        <MetricCard label="Wallet" value={formatCurrency(wallet?.estimated_total_usdt)} detail={`${wallet?.asset_count || 0} assets`} />
        <MetricCard label="Max Trade" value={formatCurrency(riskPlan.max_trade_size)} detail={`Reserve ${formatCurrency(riskPlan.reserve_cash)}`} />
      </section>

      <section className="intent-strip">
        <IntentCard
          title="Live Focus"
          symbol={config.live_symbol}
          action={formatActionLabel(decision.signal === "BUY" ? "market_buy" : decision.signal === "SELL" ? "market_sell" : "hold")}
          tone={toneForSignal(decision.signal)}
          detail={`Current model signal is ${decision.signal || "HOLD"} on ${config.live_symbol}.`}
          meta={[
            { label: "Probability", value: formatPercent(decision.probability_up) },
            { label: "Engine", value: decision.decision_engine || "ml" },
          ]}
        />
        <IntentCard
          title="Autopilot Intent"
          symbol={autopilotState?.symbol || config.live_symbol}
          action={formatActionLabel(autopilotAction)}
          tone={autopilotTone}
          detail={
            autopilotState?.running
              ? `Cycle ${autopilotState?.current_cycle || 0} of ${autopilotState?.target_cycles || 0}${latestTradeState?.override_reason ? `, override: ${latestTradeState.override_reason}` : ""}.`
              : "Autopilot is idle. Start a dry run or live run to stream actions here."
          }
          meta={[
            { label: "Signal", value: autopilotSignal || "--" },
            { label: "Size", value: formatNumber(latestTradeState?.signal_trade?.effective_quantity || latestTradeState?.size_plan?.base_size, 6) },
          ]}
        />
        <IntentCard
          title="Trade Ticket"
          symbol={currentTradeSymbol}
          action={formatActionLabel(currentTradeAction)}
          tone={currentTradeTone}
          detail={
            tradeResult?.message ||
            tradeResult?.guard_message ||
            tradePreview?.guard_message ||
            "Preview or run a ticket to inspect the exact symbol, size, and guard decision."
          }
          meta={[
            { label: "Requested", value: formatNumber(tradeForm.quantity, 6) },
            { label: "Effective", value: formatNumber(tradeResult?.effective_quantity || tradePreview?.effective_quantity, 6) },
          ]}
        />
      </section>

      {feedback.message ? (
        <section className="message-strip">
          <div className={`message ${feedback.kind === "error" ? "error" : "success"}`}>{feedback.message}</div>
        </section>
      ) : null}

      <section className={`ai-runtime-banner tone-${aiRuntimeTone}`}>
        <div className="ai-runtime-banner-head">
          <div>
            <span className="symbol-banner-label">AI Runtime Status</span>
            <h3>{aiRuntimeHeadline}</h3>
          </div>
          <StatusBadge tone={aiRuntimeTone}>{llmStatus?.provider || "disabled"}</StatusBadge>
        </div>
        <p className="warning-copy">{aiRuntimeDetail}</p>
        <div className="ai-runtime-meta">
          <span><small>Model</small><strong>{llmStatus?.active_model || "--"}</strong></span>
          <span><small>Path exists</small><strong>{String(Boolean(llmStatus?.path_exists))}</strong></span>
          <span><small>Trained</small><strong>{String(Boolean(llmStatus?.is_trained_model))}</strong></span>
          <span><small>Fallback</small><strong>{String(Boolean(llmStatus?.fallback_active))}</strong></span>
          <span><small>Health</small><strong>{llmStatus?.health_status || "--"}</strong></span>
          <span><small>Load status</small><strong>{llmStatus?.model_load_status || "--"}</strong></span>
        </div>
      </section>

      <section className="workspace-grid">
        <aside className="panel control-panel">
          <div className="panel-header">
            <h2>Autopilot Runtime</h2>
            <span className="panel-caption">Read-only backend state</span>
          </div>
          <p className="helper-copy">
            Manual runtime tuning has been removed from autopilot. The backend now resolves signal intent, wallet funding, exchange minimums, and guarded execution automatically.
          </p>
          <div className="data-list">
            <div><span>Active symbol</span><strong>{autopilotActiveSymbol}</strong></div>
            <div><span>AI runtime</span><strong>{llmStatus?.provider || "--"}</strong></div>
            <div><span>Active model</span><strong>{llmStatus?.active_model || llmStatus?.active_model_path || "--"}</strong></div>
            <div><span>Model path</span><strong>{llmStatus?.active_model_path || "--"}</strong></div>
            <div><span>Path exists</span><strong>{String(Boolean(llmStatus?.path_exists))}</strong></div>
            <div><span>Trained model</span><strong>{String(Boolean(llmStatus?.is_trained_model))}</strong></div>
            <div><span>Fallback active</span><strong>{String(Boolean(llmStatus?.fallback_active))}</strong></div>
            <div><span>Load status</span><strong>{llmStatus?.model_load_status || "--"}</strong></div>
            <div><span>Required quote asset</span><strong>{autopilotRequiredQuoteAsset}</strong></div>
            <div><span>Required base asset</span><strong>{autopilotRequiredBaseAsset}</strong></div>
            <div><span>Free quote</span><strong>{formatNumber(autopilotState?.free_quote ?? autopilotExecutionPlan?.free_quote, 4)}</strong></div>
            <div><span>Free base</span><strong>{formatNumber(autopilotState?.free_base ?? autopilotExecutionPlan?.free_base, 6)}</strong></div>
            <div><span>Direct trade possible</span><strong>{String(Boolean(latestTradeState?.direct_trade_possible ?? autopilotState?.direct_trade_possible ?? autopilotExecutionPlan?.direct_trade_possible))}</strong></div>
            <div><span>Funding path</span><strong>{latestTradeState?.funding_path || autopilotState?.funding_path || autopilotExecutionPlan?.funding_path || "none"}</strong></div>
            <div><span>Minimum valid buy</span><strong>{formatCurrency(autopilotState?.minimum_valid_quote ?? autopilotExecutionPlan?.minimum_valid_quote)}</strong></div>
            <div><span>Computed order size</span><strong>{formatNumber(autopilotComputedSizeBase, 6)} / {formatCurrency(autopilotComputedSizeQuote)}</strong></div>
            <div><span>Guard mode</span><strong>{autopilotGuardMode}</strong></div>
            <div><span>Skip reason</span><strong>{autopilotSkipReason}</strong></div>
            <div><span>Starting value</span><strong>{formatCurrency(autopilotStartingValue)}</strong></div>
            <div><span>Current value</span><strong>{formatCurrency(autopilotCurrentValue)}</strong></div>
            <div><span>Goal value</span><strong>{formatCurrency(autopilotGoalValue)}</strong></div>
            <div><span>Progress</span><strong>{formatPercent(autopilotProgressPct)}</strong></div>
            <div><span>Continue to goal</span><strong>{String(autopilotContinueUntilGoal)}</strong></div>
            <div><span>Extra cycles</span><strong>{autopilotExtraCyclesUsed}</strong></div>
            <div><span>Failed cycles</span><strong>{autopilotFailedCycles}</strong></div>
            <div><span>Stable exit asset</span><strong>{autopilotFinalStableAsset}</strong></div>
            <div><span>Finalization</span><strong>{autopilotFinalizationStatus}</strong></div>
            <div><span>Stop reason</span><strong>{autopilotFinalStopReason}</strong></div>
          </div>
          <div className="data-list">
            <div><span>Wallet est. total</span><strong>{formatCurrency(autopilotWalletSummary?.estimated_total_usdt)}</strong></div>
            <div><span>Top funding asset</span><strong>{autopilotWalletSummary?.largest_asset || "--"}</strong></div>
            <div><span>Conversion plan</span><strong>{latestConversionPlan?.conversion_symbol ? `${latestConversionPlan.action || "convert"} ${latestConversionPlan.conversion_symbol}` : "none"}</strong></div>
            <div><span>Conversion result</span><strong>{latestConversionResult?.status || "--"}</strong></div>
          </div>
        </aside>

        <section className="panel main-panel">
          <div className="tab-row">
            {tabs.map((tab) => (
              <button key={tab.id} className={tab.id === activeTab ? "tab-button active" : "tab-button"} onClick={() => setActiveTab(tab.id)}>
                {tab.label}
              </button>
            ))}
          </div>

          {activeTab === "overview" ? (
            <div className="tab-content">
              <div className="content-grid overview-grid">
                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Current Decision</h3>
                    <span className="panel-caption">{decision.decision_engine || "ml_primary"}</span>
                  </div>
                  <div className="data-list">
                    <div><span>Signal</span><strong>{decision.signal || "--"}</strong></div>
                    <div><span>Final probability</span><strong>{formatPercent(decision.probability_up)}</strong></div>
                    <div><span>ML probability</span><strong>{formatPercent(decision.ml_probability_up)}</strong></div>
                    <div><span>Buy threshold</span><strong>{formatNumber(decision.buy_threshold, 3)}</strong></div>
                    <div><span>Sell threshold</span><strong>{formatNumber(decision.sell_threshold, 3)}</strong></div>
                    <div><span>LLM overlay</span><strong>{decision.llm_overlay?.status || "disabled"}</strong></div>
                    <div><span>LLM model</span><strong>{llmStatus?.active_model || llmStatus?.active_model_path || "--"}</strong></div>
                  </div>
                </article>

                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Risk Plan</h3>
                    <span className="panel-caption">Backend computed</span>
                  </div>
                  <div className="data-list">
                    <div><span>Active capital</span><strong>{formatCurrency(riskPlan.active_capital)}</strong></div>
                    <div><span>Reserve cash</span><strong>{formatCurrency(riskPlan.reserve_cash)}</strong></div>
                    <div><span>Max trade size</span><strong>{formatCurrency(riskPlan.max_trade_size)}</strong></div>
                    <div><span>Daily loss limit</span><strong>{formatCurrency(riskPlan.daily_loss_limit)}</strong></div>
                    <div><span>Drawdown limit</span><strong>{formatCurrency(riskPlan.drawdown_limit)}</strong></div>
                    <div><span>Withdrawal target</span><strong>{formatCurrency(riskPlan.withdrawal_target)}</strong></div>
                  </div>
                </article>

                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>AI Planner</h3>
                    <span className="panel-caption">Cycle + size planning</span>
                  </div>
                  <div className="data-list">
                    <div><span>Size planner</span><strong>{sizePlan.status || "--"}</strong></div>
                    <div><span>Allocation</span><strong>{formatPercent(sizePlan.allocation_pct)}</strong></div>
                    <div><span>Quote size</span><strong>{formatCurrency(sizePlan.quote_size)}</strong></div>
                    <div><span>Base size</span><strong>{formatNumber(sizePlan.base_size, 6)}</strong></div>
                    <div><span>Cycle planner</span><strong>{cyclePlan.status || "--"}</strong></div>
                    <div><span>Recommended cycles</span><strong>{cyclePlan.recommended_cycles || 0}</strong></div>
                  </div>
                </article>
              </div>

              <LineChart
                title="Current Market Graph"
                rows={liveHistory.slice(-50)}
                lines={[
                  { key: "best_bid", label: "Best bid", color: "#55b7ff" },
                  { key: "best_ask", label: "Best ask", color: "#2fe38d" },
                  { key: "account_value", label: "Account value", color: "#ffcb57" },
                ]}
                yFormatter={formatCurrency}
              />

              <article className="subpanel-card">
                <div className="panel-header">
                  <h3>Market Scanner</h3>
                  <span className="panel-caption">{marketScan.length} symbols</span>
                </div>
                <DataTable
                  columns={[
                    { key: "symbol", label: "Symbol" },
                    { key: "last", label: "Last", render: (value) => formatNumber(value, 4) },
                    { key: "change_pct", label: "24h %", render: (value) => formatPercent(Number(value) / 100) },
                    { key: "spread_bps", label: "Spread", render: (value) => `${formatNumber(value, 1)} bps` },
                    { key: "ai_bias", label: "AI Bias" },
                    { key: "ai_score", label: "AI Score", render: (value) => formatNumber(value, 2) },
                  ]}
                  rows={marketScan.slice(0, 25)}
                  emptyMessage={dashboard?.market_scan_error || "Scanner is quiet right now."}
                />
              </article>
            </div>
          ) : null}

          {activeTab === "live" ? (
            <div className="tab-content live-terminal">
              <article className="subpanel-card live-market-strip">
                <div className="live-market-head">
                  <div>
                    <span className="symbol-banner-label">Active Market</span>
                    <h3>{autopilotActiveSymbol}</h3>
                  </div>
                  <div className="live-market-price">
                    <strong>{formatNumber(liveTicker?.last || chartData?.current_price, 4)}</strong>
                    <span>{loadingTicker ? "Updating..." : "Live"}</span>
                  </div>
                </div>
                <div className="ticket-summary">
                  <div><span className="ticket-label">Bid</span><strong>{formatNumber(liveTicker?.bid, 4)}</strong></div>
                  <div><span className="ticket-label">Ask</span><strong>{formatNumber(liveTicker?.ask, 4)}</strong></div>
                  <div><span className="ticket-label">Spread</span><strong>{formatNumber(liveTicker?.spread_bps, 1)} bps</strong></div>
                  <div><span className="ticket-label">Guard</span><strong>{autopilotGuardMode}</strong></div>
                  <div><span className="ticket-label">Last action</span><strong>{formatActionLabel(autopilotAction)}</strong></div>
                  <div><span className="ticket-label">Skip reason</span><strong>{autopilotSkipReason}</strong></div>
                </div>
              </article>

              <div className="live-terminal-layout">
                <section className="live-terminal-main">
                  <MarketMonitorChart
                    chart={chartData}
                    timeframe={chartInterval}
                    onTimeframeChange={setChartInterval}
                    overlays={chartOverlays}
                    onOverlayChange={updateChartOverlay}
                    loading={loadingChart}
                  />
                </section>

                <aside className="live-terminal-side">
                  <article className="subpanel-card">
                    <div className="panel-header">
                      <h3>Autopilot</h3>
                      <span className="panel-caption">{autopilotState?.status || "idle"}</span>
                    </div>
                    <div className="form-grid single-row">
                      <label>
                        <span>Goal value</span>
                        <input
                          type="number"
                          min="0"
                          step="0.01"
                          value={autopilotForm.goal_value}
                          onChange={(event) => updateAutopilotField("goal_value", Number(event.target.value))}
                        />
                      </label>
                    </div>
                    <div className="action-row">
                      <button className="primary-button" disabled={submittingAutopilot} onClick={() => handleStartAutopilot(false)}>Start Dry Run</button>
                      <button className="warning-button" disabled={submittingAutopilot} onClick={() => handleStartAutopilot(true)}>Start Live Auto Trade</button>
                      <button className="ghost-button" disabled={submittingAutopilot} onClick={handleStopAutopilot}>Cancel Auto Trade</button>
                    </div>
                    <div className="data-list">
                      <div><span>Cycle</span><strong>{autopilotState?.current_cycle || 0} / {autopilotState?.target_cycles || 0}</strong></div>
                      <div><span>Decision</span><strong>{autopilotSignal}</strong></div>
                      <div><span>Trade result</span><strong>{latestTradeState?.signal_trade?.status || latestTradeState?.status || "--"}</strong></div>
                      <div><span>Decision confidence</span><strong>{formatPercent(latestTradeState?.latest_decision?.decision_confidence || autopilotState?.latest_decision?.decision_confidence || chartData?.latest_decision?.decision_confidence || 0)}</strong></div>
                      <div><span>Current value</span><strong>{formatCurrency(autopilotCurrentValue)}</strong></div>
                      <div><span>Goal</span><strong>{formatCurrency(autopilotGoalValue)}</strong></div>
                      <div><span>Progress</span><strong>{formatPercent(autopilotProgressPct)}</strong></div>
                      <div><span>Extra cycles</span><strong>{autopilotExtraCyclesUsed}</strong></div>
                      <div><span>Failed cycles</span><strong>{autopilotFailedCycles}</strong></div>
                      <div><span>Stable exit</span><strong>{autopilotFinalStableAsset}</strong></div>
                      <div><span>Finalization</span><strong>{autopilotFinalizationStatus}</strong></div>
                      <div><span>Stop reason</span><strong>{autopilotFinalStopReason}</strong></div>
                      <div><span>Updated</span><strong>{formatDate(autopilotState?.updated_at)}</strong></div>
                    </div>
                  </article>

                  <article className="subpanel-card">
                    <div className="panel-header">
                      <h3>Funding Snapshot</h3>
                    </div>
                    <div className="data-list">
                      <div><span>Required quote</span><strong>{autopilotRequiredQuoteAsset}</strong></div>
                      <div><span>Free quote</span><strong>{formatNumber(autopilotState?.free_quote ?? autopilotExecutionPlan?.free_quote, 4)}</strong></div>
                      <div><span>Free BNB</span><strong>{formatNumber(freeBnb, 6)}</strong></div>
                      <div><span>Funding path</span><strong>{latestTradeState?.funding_path || autopilotState?.funding_path || autopilotExecutionPlan?.funding_path || "none"}</strong></div>
                      <div><span>Conversion pair</span><strong>{latestConversionPlan?.conversion_symbol || fundingDiagnostics?.selected_conversion_pair || "--"}</strong></div>
                      <div><span>Estimated quote after conversion</span><strong>{formatNumber(fundingDiagnostics?.estimated_quote_after_conversion, 4)}</strong></div>
                      <div><span>Minimum valid buy</span><strong>{formatCurrency(autopilotState?.minimum_valid_quote ?? autopilotExecutionPlan?.minimum_valid_quote)}</strong></div>
                      <div><span>Computed order size</span><strong>{formatNumber(autopilotComputedSizeBase, 6)} / {formatCurrency(autopilotComputedSizeQuote)}</strong></div>
                    </div>
                    <div className="funding-assets">
                      {eligibleFundingAssets.length === 0 ? <span className="panel-caption">No alternate funding assets detected.</span> : eligibleFundingAssets.slice(0, 5).map((asset) => (
                        <span key={`${asset.asset}-${asset.free}`}>
                          {asset.asset}: {formatNumber(asset.free, 4)}
                        </span>
                      ))}
                    </div>
                  </article>

                  <article className="subpanel-card">
                    <div className="panel-header">
                      <h3>Opportunity Winner</h3>
                    </div>
                    <div className="data-list">
                      <div><span>Winner symbol</span><strong>{opportunityWinner?.symbol || "--"}</strong></div>
                      <div><span>Reason</span><strong>{opportunityWinner?.selection_reason || "--"}</strong></div>
                      <div><span>Action</span><strong>{opportunityWinner?.final_action || "--"}</strong></div>
                      <div><span>Score</span><strong>{formatNumber(opportunityWinner?.score, 3)}</strong></div>
                      <div><span>Confidence</span><strong>{formatPercent(opportunityWinner?.confidence || 0)}</strong></div>
                      <div><span>Fundable</span><strong>{String(Boolean(opportunityWinner?.wallet_fundable))}</strong></div>
                    </div>
                    <span className="panel-caption">Universe: {opportunityMeta?.evaluated || 0} evaluated / {opportunityMeta?.universe_size || 0} scanned</span>
                  </article>
                </aside>
              </div>

              <article className="subpanel-card">
                <div className="panel-header">
                  <h3>Autopilot Event Tape</h3>
                </div>
                <DataTable
                  columns={[
                    { key: "cycle", label: "Cycle" },
                    { key: "timestamp", label: "Timestamp", render: (value) => formatDate(value) },
                    { key: "symbol", label: "Symbol" },
                    { key: "signal", label: "Signal" },
                    { key: "final_action", label: "Action", render: (value) => formatActionLabel(value) },
                    { key: "trade_status", label: "Result" },
                    { key: "funding_path", label: "Funding" },
                    { key: "skip_reason", label: "Skip Reason" },
                    { key: "conversion_pair", label: "Conversion", render: (_value, row) => row?.conversion_plan?.conversion_symbol || "--" },
                    { key: "probability_up", label: "Prob", render: (value) => formatPercent(value) },
                  ]}
                  rows={[...autopilotLogs].map((row) => ({ ...row, conversion_pair: row?.conversion_plan?.conversion_symbol || "--" })).slice(-24).reverse()}
                  emptyMessage="No autopilot events yet."
                />
              </article>

              <article className="subpanel-card">
                <div className="panel-header">
                  <h3>Opportunity Shortlist</h3>
                </div>
                <DataTable
                  columns={[
                    { key: "symbol", label: "Symbol" },
                    { key: "score", label: "Score", render: (value) => formatNumber(value, 3) },
                    { key: "final_action", label: "Action" },
                    { key: "confidence", label: "Confidence", render: (value) => formatPercent(value) },
                    { key: "expected_return", label: "Exp Return", render: (value) => formatPercent(value) },
                    { key: "wallet_fundable", label: "Fundable", render: (value) => String(Boolean(value)) },
                    { key: "funding_path", label: "Funding" },
                    { key: "rejection_reason", label: "Rejected Why" },
                  ]}
                  rows={opportunityRankings.slice(0, 12)}
                  emptyMessage="No ranked candidates yet."
                />
              </article>
            </div>
          ) : null}

          {activeTab === "wallet" ? (
            <div className="tab-content">
              <div className="content-grid wallet-grid">
                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Wallet Snapshot</h3>
                  </div>
                  <div className="data-list">
                    <div><span>Exchange</span><strong>{wallet?.exchange || "binance"}</strong></div>
                    <div><span>Mode</span><strong>{wallet?.testnet ? "testnet" : "live"}</strong></div>
                    <div><span>Asset count</span><strong>{wallet?.asset_count || 0}</strong></div>
                    <div><span>Estimated total</span><strong>{formatCurrency(wallet?.estimated_total_usdt)}</strong></div>
                  </div>
                  {dashboard?.wallet_error ? <p className="warning-copy">{dashboard.wallet_error}</p> : null}
                </article>
              </div>

              <DataTable
                columns={[
                  { key: "asset", label: "Asset" },
                  { key: "free", label: "Free", render: (value) => formatNumber(value, 6) },
                  { key: "used", label: "Used", render: (value) => formatNumber(value, 6) },
                  { key: "total", label: "Total", render: (value) => formatNumber(value, 6) },
                  { key: "est_usdt", label: "Est. USDT", render: (value) => formatCurrency(value) },
                ]}
                rows={wallet?.balances || []}
                emptyMessage={dashboard?.wallet_error || "No non-zero balances found."}
              />
            </div>
          ) : null}

          {activeTab === "account" ? (
            <div className="tab-content">
              <div className="content-grid account-grid">
                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Account Snapshot</h3>
                    <button className="ghost-button compact-button" onClick={handleRefreshAccount}>Refresh</button>
                  </div>
                  <div className="symbol-banner">
                    <div>
                      <span className="symbol-banner-label">Watching</span>
                      <strong>{tradeForm.symbol}</strong>
                    </div>
                    <StatusBadge tone={currentTradeTone}>{formatActionLabel(currentTradeAction)}</StatusBadge>
                  </div>
                  <div className="form-grid single-row">
                    <label>
                      <span>Symbol</span>
                      <input value={tradeForm.symbol} onChange={(event) => updateTradeField("symbol", event.target.value)} />
                    </label>
                  </div>
                  <div className="data-list">
                    <div><span>Best bid</span><strong>{formatNumber(accountSnapshot?.best_bid, 4)}</strong></div>
                    <div><span>Best ask</span><strong>{formatNumber(accountSnapshot?.best_ask, 4)}</strong></div>
                    <div><span>Base free</span><strong>{formatNumber(accountSnapshot?.base_free, 6)}</strong></div>
                    <div><span>Quote free</span><strong>{formatNumber(accountSnapshot?.quote_free, 4)}</strong></div>
                    <div><span>Account value</span><strong>{formatCurrency(accountSnapshot?.account_value_quote)}</strong></div>
                    <div><span>Spread</span><strong>{formatNumber(accountSnapshot?.spread_bps, 1)} bps</strong></div>
                    <div><span>Open orders</span><strong>{accountSnapshot?.open_orders_count || 0}</strong></div>
                  </div>
                </article>

                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Trade Ticket</h3>
                  </div>
                  <div className="ticket-summary">
                    <div>
                      <span className="ticket-label">Selected action</span>
                      <strong>{formatActionLabel(tradeForm.action)}</strong>
                    </div>
                    <div>
                      <span className="ticket-label">Symbol</span>
                      <strong>{tradeForm.symbol}</strong>
                    </div>
                    <div>
                      <span className="ticket-label">Requested size</span>
                      <strong>{formatNumber(tradeForm.quantity, 6)}</strong>
                    </div>
                    <div>
                      <span className="ticket-label">Mode</span>
                      <strong>{tradeForm.dry_run ? "Dry run" : "Live"}</strong>
                    </div>
                  </div>
                  <div className="form-grid">
                    <label>
                      <span>Action</span>
                      <select value={tradeForm.action} onChange={(event) => updateTradeField("action", event.target.value)}>
                        <option value="market_buy">Market Buy</option>
                        <option value="market_sell">Market Sell</option>
                        <option value="cancel_all_orders">Cancel All Orders</option>
                      </select>
                    </label>
                    <label>
                      <span>Quantity</span>
                      <input type="number" step="0.0001" value={tradeForm.quantity} onChange={(event) => updateTradeField("quantity", Number(event.target.value))} />
                    </label>
                    <label>
                      <span>Quote amount</span>
                      <input type="number" value={tradeForm.quote_amount} onChange={(event) => updateTradeField("quote_amount", Number(event.target.value))} />
                    </label>
                    <label>
                      <span>Max latency (ms)</span>
                      <input type="number" value={tradeForm.max_api_latency_ms} onChange={(event) => updateTradeField("max_api_latency_ms", Number(event.target.value))} />
                    </label>
                  </div>
                  <div className="toggle-row">
                    <label><input type="checkbox" checked={tradeForm.dry_run} onChange={(event) => updateTradeField("dry_run", event.target.checked)} /> Dry-run mode</label>
                  </div>
                  <div className="action-row">
                    <button className="ghost-button" disabled={submittingTrade} onClick={handlePreviewTrade}>Preview Ticket</button>
                    <button className="primary-button" disabled={submittingTrade} onClick={handleRunTrade}>
                      {tradeForm.dry_run ? "Run Dry Trade" : "Run Live Trade"}
                    </button>
                  </div>
                </article>
              </div>

              <div className="content-grid account-grid">
                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Guard Preview</h3>
                  </div>
                  <div className="data-list">
                    <div><span>Status</span><strong>{tradePreview?.status || "--"}</strong></div>
                    <div><span>Symbol</span><strong>{tradePreview?.symbol || tradeForm.symbol}</strong></div>
                    <div><span>Action</span><strong>{formatActionLabel(tradePreview?.action || tradeForm.action)}</strong></div>
                    <div><span>Guard</span><strong>{tradePreview?.guard_message || "--"}</strong></div>
                    <div><span>Minimums</span><strong>{tradePreview?.minimum_message || "--"}</strong></div>
                    <div><span>Effective size</span><strong>{formatNumber(tradePreview?.effective_quantity, 6)}</strong></div>
                    <div><span>Size cap</span><strong>{tradePreview?.size_cap_reason || "--"}</strong></div>
                    <div><span>Min qty</span><strong>{formatNumber(tradePreview?.min_qty, 6)}</strong></div>
                    <div><span>Min notional</span><strong>{formatNumber(tradePreview?.min_notional, 6)}</strong></div>
                    <div><span>Price</span><strong>{formatNumber(tradePreview?.market_price, 4)}</strong></div>
                  </div>
                </article>

                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Trade Result</h3>
                  </div>
                  <div className="data-list">
                    <div><span>Status</span><strong>{tradeResult?.status || "--"}</strong></div>
                    <div><span>Symbol</span><strong>{tradeResult?.symbol || tradeForm.symbol}</strong></div>
                    <div><span>Action</span><strong>{formatActionLabel(tradeResult?.action || tradeForm.action)}</strong></div>
                    <div><span>Message</span><strong>{tradeResult?.message || tradeResult?.guard_message || "--"}</strong></div>
                    <div><span>Effective size</span><strong>{formatNumber(tradeResult?.effective_quantity, 6)}</strong></div>
                    <div><span>Execution mode</span><strong>{tradeResult?.guard_mode || "--"}</strong></div>
                    <div><span>Spread</span><strong>{formatNumber(tradeResult?.spread_bps, 1)} bps</strong></div>
                    <div><span>Ticker age</span><strong>{formatNumber(tradeResult?.ticker_age_ms, 0)} ms</strong></div>
                  </div>
                </article>
              </div>
            </div>
          ) : null}

          {activeTab === "ai" ? (
            <div className="tab-content">
              <div className="content-grid ai-grid">
                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>AI Command Center</h3>
                  </div>
                  <p className="helper-copy">Example: <code>recommend</code></p>
                  <div className="action-input">
                    <input value={aiCommand} onChange={(event) => setAiCommand(event.target.value)} placeholder="Type a short AI command" />
                    <button className="primary-button" onClick={handleRunAiCommand}>Run Command</button>
                  </div>
                </article>

                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>LLM Support Chat</h3>
                    <span className="panel-caption">{decision.llm_overlay?.status || "disabled"}</span>
                  </div>
                  <div className="chat-shell">
                    {supportMessages.length === 0 ? <p className="empty-state">Ask about the current signal, merge status, or next safe action.</p> : null}
                    {supportMessages.map((message, index) => (
                      <div key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
                        <strong>{message.role === "user" ? "You" : "AI"}</strong>
                        <p>{message.content}</p>
                      </div>
                    ))}
                  </div>
                  <form className="chat-form" onSubmit={handleSupportChat}>
                    <input value={supportPrompt} onChange={(event) => setSupportPrompt(event.target.value)} placeholder="Ask support about the current trading state" />
                    <button className="primary-button" type="submit" disabled={submittingChat}>{submittingChat ? "Sending..." : "Send"}</button>
                  </form>
                </article>
              </div>
            </div>
          ) : null}
        </section>
      </section>
    </main>
  );
}

export default App;
