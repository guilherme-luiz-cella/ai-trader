import { useEffect, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8765").replace(/\/$/, "");

const initialConfig = {
  buy_threshold: 0.55,
  sell_threshold: 0.45,
  adaptive_threshold_enabled: true,
  deposit_amount: 1000,
  active_capital_pct: 0.7,
  reserve_pct: 0.3,
  max_trade_pct: 0.1,
  stop_loss_pct: 0.03,
  take_profit_pct: 0.05,
  max_daily_loss_pct: 0.05,
  max_drawdown_pct: 0.15,
  withdrawal_target_pct: 0.25,
  live_symbol: "BTC/USDT",
  market_scan_enabled: true,
  market_scan_max_symbols: 60,
  market_scan_quote_asset: "USDT",
};

const initialAutopilotForm = {
  interval_seconds: 15,
  cycles: 5,
  auto_cycles_enabled: true,
  auto_size_enabled: true,
  goal_value: 1250,
  current_value: 1000,
  fee_drag_pct: 0.003,
  order_size: 0.001,
  max_trade_size_quote: 100,
  allow_live: false,
  max_api_latency_ms: 1200,
  max_ticker_age_ms: 3000,
  max_spread_bps: 20,
  min_trade_cooldown_seconds: 5,
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

function buildDashboardQuery(config) {
  const params = new URLSearchParams();
  Object.entries(config).forEach(([key, value]) => {
    params.set(key, String(value));
  });
  return params.toString();
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

  async function refreshHealth() {
    try {
      const payload = await request("/health");
      setHealth(payload.status || "ok");
    } catch (error) {
      setHealth("offline");
      setFeedback({ kind: "error", message: error.message });
    }
  }

  async function refreshDashboard(nextConfig = config, options = {}) {
    const { silent = false } = options;
    if (!silent) {
      setLoadingDashboard(true);
      setFeedback({ kind: "", message: "" });
    }

    try {
      const query = buildDashboardQuery(nextConfig);
      const payload = await request(`/dashboard?${query}`);
      setDashboard(payload);
      setAutopilotState(payload.autopilot || null);
      setConfig(payload.config || nextConfig);
      setAccountSnapshot(payload.account_snapshot || null);

      const walletValue = payload.wallet_snapshot?.estimated_total_usdt || payload.risk_plan?.deposit_amount || nextConfig.deposit_amount;
      const maxTradeSize = payload.risk_plan?.max_trade_size || nextConfig.deposit_amount * nextConfig.max_trade_pct;

      setAutopilotForm((current) => ({
        ...current,
        current_value: walletValue,
        goal_value: current.goal_value || walletValue * 1.25,
        max_trade_size_quote: maxTradeSize,
      }));

      setTradeForm((current) => ({
        ...current,
        symbol: nextConfig.live_symbol,
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

  useEffect(() => {
    refreshHealth();
    refreshDashboard(initialConfig);
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      refreshHealth();
      refreshAutopilotStatus();
    }, 5000);
    return () => window.clearInterval(interval);
  }, []);

  function updateConfigField(name, value) {
    setConfig((current) => ({ ...current, [name]: value }));
  }

  function updateAutopilotField(name, value) {
    setAutopilotForm((current) => ({ ...current, [name]: value }));
  }

  function updateTradeField(name, value) {
    setTradeForm((current) => ({ ...current, [name]: value }));
  }

  async function handleCaptureLivePoint() {
    const decision = dashboard?.decision_payload?.decision || {};
    try {
      await request("/live/capture", {
        method: "POST",
        body: JSON.stringify({
          symbol: config.live_symbol,
          signal: decision.signal || "HOLD",
          probability_up: Number(decision.probability_up || 0.5),
        }),
      });
      setFeedback({ kind: "success", message: "Live point captured from the backend feed." });
      refreshDashboard(config, { silent: true });
    } catch (error) {
      setFeedback({ kind: "error", message: error.message });
    }
  }

  async function handleStartAutopilot(allowLive) {
    setSubmittingAutopilot(true);
    setFeedback({ kind: "", message: "" });
    try {
      const payload = await request("/autopilot/start", {
        method: "POST",
        body: JSON.stringify({
          ...autopilotForm,
          allow_live: allowLive,
          symbol: config.live_symbol,
          buy_threshold: config.buy_threshold,
          sell_threshold: config.sell_threshold,
          adaptive_threshold_enabled: config.adaptive_threshold_enabled,
          take_profit_pct: config.take_profit_pct,
          stop_loss_pct: config.stop_loss_pct,
        }),
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
      refreshDashboard(config, { silent: true });
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
      if (result.patch) {
        const nextConfig = { ...config, ...result.patch };
        setConfig(nextConfig);
        await refreshDashboard(nextConfig);
      }
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
  const autopilotLogs = autopilotState?.logs || [];

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
          <button className="ghost-button" onClick={() => refreshDashboard(config)} disabled={loadingDashboard}>
            {loadingDashboard ? "Refreshing..." : "Refresh Dashboard"}
          </button>
          <button className="ghost-button" onClick={refreshHealth}>
            Refresh API
          </button>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard label="API Health" value={health.toUpperCase()} tone={health === "ok" ? "good" : "warn"} />
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

      {feedback.message ? (
        <section className="message-strip">
          <div className={`message ${feedback.kind === "error" ? "error" : "success"}`}>{feedback.message}</div>
        </section>
      ) : null}

      <section className="workspace-grid">
        <aside className="panel control-panel">
          <div className="panel-header">
            <h2>Runtime Controls</h2>
            <span className="panel-caption">State sent to Python</span>
          </div>
          <div className="form-grid">
            <label>
              <span>Live symbol</span>
              <input value={config.live_symbol} onChange={(event) => updateConfigField("live_symbol", event.target.value)} />
            </label>
            <label>
              <span>Buy threshold</span>
              <input type="number" step="0.01" value={config.buy_threshold} onChange={(event) => updateConfigField("buy_threshold", Number(event.target.value))} />
            </label>
            <label>
              <span>Sell threshold</span>
              <input type="number" step="0.01" value={config.sell_threshold} onChange={(event) => updateConfigField("sell_threshold", Number(event.target.value))} />
            </label>
            <label>
              <span>Deposit amount</span>
              <input type="number" value={config.deposit_amount} onChange={(event) => updateConfigField("deposit_amount", Number(event.target.value))} />
            </label>
            <label>
              <span>Capital deployed</span>
              <input type="number" step="0.01" value={config.active_capital_pct} onChange={(event) => updateConfigField("active_capital_pct", Number(event.target.value))} />
            </label>
            <label>
              <span>Max trade %</span>
              <input type="number" step="0.01" value={config.max_trade_pct} onChange={(event) => updateConfigField("max_trade_pct", Number(event.target.value))} />
            </label>
            <label>
              <span>Market scan quote</span>
              <input value={config.market_scan_quote_asset} onChange={(event) => updateConfigField("market_scan_quote_asset", event.target.value)} />
            </label>
            <label>
              <span>Market scan max</span>
              <input type="number" value={config.market_scan_max_symbols} onChange={(event) => updateConfigField("market_scan_max_symbols", Number(event.target.value))} />
            </label>
          </div>
          <div className="toggle-row">
            <label><input type="checkbox" checked={config.adaptive_threshold_enabled} onChange={(event) => updateConfigField("adaptive_threshold_enabled", event.target.checked)} /> Adaptive thresholds</label>
            <label><input type="checkbox" checked={config.market_scan_enabled} onChange={(event) => updateConfigField("market_scan_enabled", event.target.checked)} /> Market scanner</label>
          </div>
          <div className="action-row">
            <button className="primary-button" onClick={() => refreshDashboard(config)} disabled={loadingDashboard}>Apply Controls</button>
            <button className="ghost-button" onClick={handleCaptureLivePoint}>Capture Live Point</button>
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
            <div className="tab-content">
              <article className="subpanel-card">
                <div className="panel-header">
                  <h3>Autopilot Controls</h3>
                  <span className="panel-caption">Button-only backend automation</span>
                </div>
                <div className="form-grid">
                  <label>
                    <span>Interval (s)</span>
                    <input type="number" value={autopilotForm.interval_seconds} onChange={(event) => updateAutopilotField("interval_seconds", Number(event.target.value))} />
                  </label>
                  <label>
                    <span>Cycles</span>
                    <input type="number" value={autopilotForm.cycles} onChange={(event) => updateAutopilotField("cycles", Number(event.target.value))} />
                  </label>
                  <label>
                    <span>Current value</span>
                    <input type="number" value={autopilotForm.current_value} onChange={(event) => updateAutopilotField("current_value", Number(event.target.value))} />
                  </label>
                  <label>
                    <span>Goal value</span>
                    <input type="number" value={autopilotForm.goal_value} onChange={(event) => updateAutopilotField("goal_value", Number(event.target.value))} />
                  </label>
                  <label>
                    <span>Manual order size</span>
                    <input type="number" step="0.0001" value={autopilotForm.order_size} onChange={(event) => updateAutopilotField("order_size", Number(event.target.value))} />
                  </label>
                  <label>
                    <span>Max trade quote</span>
                    <input type="number" value={autopilotForm.max_trade_size_quote} onChange={(event) => updateAutopilotField("max_trade_size_quote", Number(event.target.value))} />
                  </label>
                </div>
                <div className="toggle-row">
                  <label><input type="checkbox" checked={autopilotForm.auto_cycles_enabled} onChange={(event) => updateAutopilotField("auto_cycles_enabled", event.target.checked)} /> AI cycle planner</label>
                  <label><input type="checkbox" checked={autopilotForm.auto_size_enabled} onChange={(event) => updateAutopilotField("auto_size_enabled", event.target.checked)} /> AI size planner</label>
                </div>
                <div className="action-row">
                  <button className="primary-button" disabled={submittingAutopilot} onClick={() => handleStartAutopilot(false)}>Start Dry Run</button>
                  <button className="warning-button" disabled={submittingAutopilot} onClick={() => handleStartAutopilot(true)}>Start Live Auto Trade</button>
                  <button className="ghost-button" disabled={submittingAutopilot} onClick={handleStopAutopilot}>Cancel Auto Trade</button>
                </div>
              </article>

              <div className="content-grid live-grid">
                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>Autopilot Status</h3>
                  </div>
                  <div className="data-list">
                    <div><span>Status</span><strong>{autopilotState?.status || "idle"}</strong></div>
                    <div><span>Running</span><strong>{String(Boolean(autopilotState?.running))}</strong></div>
                    <div><span>Current cycle</span><strong>{autopilotState?.current_cycle || 0}</strong></div>
                    <div><span>Target cycles</span><strong>{autopilotState?.target_cycles || 0}</strong></div>
                    <div><span>Updated</span><strong>{formatDate(autopilotState?.updated_at)}</strong></div>
                    <div><span>Last error</span><strong>{autopilotState?.last_error || "--"}</strong></div>
                  </div>
                </article>

                <article className="subpanel-card">
                  <div className="panel-header">
                    <h3>What The AI Is Doing</h3>
                  </div>
                  <DataTable
                    columns={[
                      { key: "cycle", label: "Cycle" },
                      { key: "timestamp", label: "Timestamp", render: (value) => formatDate(value) },
                      { key: "signal", label: "Signal" },
                      { key: "probability_up", label: "Prob", render: (value) => formatPercent(value) },
                      { key: "trade_status", label: "Trade status" },
                      { key: "trade_message", label: "Result" },
                    ]}
                    rows={[...autopilotLogs].slice(-12).reverse()}
                    emptyMessage="No autopilot cycles have completed yet."
                  />
                </article>
              </div>

              <LineChart
                title="Autopilot Cycle Graph"
                rows={autopilotLogs.slice(-50)}
                lines={[
                  { key: "bid", label: "Bid", color: "#55b7ff" },
                  { key: "ask", label: "Ask", color: "#2fe38d" },
                  { key: "probability_up", label: "Probability", color: "#ff7d67" },
                  { key: "account_value", label: "Account value", color: "#ffcb57" },
                ]}
              />
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
                  <div className="form-grid single-row">
                    <label>
                      <span>Symbol</span>
                      <input value={tradeForm.symbol} onChange={(event) => updateTradeField("symbol", event.target.value)} />
                    </label>
                  </div>
                  <div className="data-list">
                    <div><span>Best bid</span><strong>{formatNumber(accountSnapshot?.best_bid, 4)}</strong></div>
                    <div><span>Best ask</span><strong>{formatNumber(accountSnapshot?.best_ask, 4)}</strong></div>
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
                    <div><span>Guard</span><strong>{tradePreview?.guard_message || "--"}</strong></div>
                    <div><span>Minimums</span><strong>{tradePreview?.minimum_message || "--"}</strong></div>
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
                    <div><span>Message</span><strong>{tradeResult?.message || tradeResult?.guard_message || "--"}</strong></div>
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
                  <p className="helper-copy">Examples: <code>set deposit 2000</code>, <code>set buy threshold 0.62</code>, <code>recommend</code></p>
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
