import { ActivityIndicator, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";

import { MetricTile } from "../components/MetricTile";
import { SectionCard } from "../components/SectionCard";
import { useTradingControlApp } from "../hooks/useTradingControlApp";
import { theme } from "../styles/theme";

function formatCurrency(value: number) {
  return `$${Number(value || 0).toFixed(2)}`;
}

function formatPercent(value: number) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatNumber(value: number, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function toneForSignal(signal: string): "good" | "warn" | "bad" {
  if (signal === "BUY") {
    return "good";
  }

  if (signal === "SELL") {
    return "bad";
  }

  return "warn";
}

function ActionChip({
  active,
  label,
  onPress,
}: {
  active: boolean;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={[styles.chip, active ? styles.chipActive : undefined]}>
      <Text style={[styles.chipText, active ? styles.chipTextActive : undefined]}>{label}</Text>
    </Pressable>
  );
}

export function TradingControlScreen() {
  const {
    backendUrl,
    setBackendUrl,
    email,
    setEmail,
    password,
    setPassword,
    session,
    overview,
    previewRequest,
    previewResult,
    tradeResult,
    loading,
    error,
    signIn,
    signOut,
    refresh,
    previewTrade,
    executeTrade,
    updatePreviewField,
  } = useTradingControlApp();

  const dashboard = overview?.dashboard;
  const autopilot = overview?.autopilot;
  const health = overview?.health;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>Personal iPhone Operator Client</Text>
          <Text style={styles.title}>AI Trader Mobile</Text>
          <Text style={styles.subtitle}>
            Operator status, trade preview, and live trade execution from your own device against the hosted API.
          </Text>
        </View>

        {!session ? (
          <SectionCard
            title="Sign In"
            subtitle="Use the backend app login. This repo is currently set up to use the hosted reverse-proxy path https://cella.website/api."
          >
            <View style={styles.form}>
              <View style={styles.field}>
                <Text style={styles.label}>Backend URL</Text>
                <TextInput
                  autoCapitalize="none"
                  autoCorrect={false}
                  value={backendUrl}
                  onChangeText={setBackendUrl}
                  placeholder="https://cella.website/api"
                  placeholderTextColor={theme.colors.textMuted}
                  style={styles.input}
                />
              </View>
              <View style={styles.field}>
                <Text style={styles.label}>Email</Text>
                <TextInput
                  autoCapitalize="none"
                  keyboardType="email-address"
                  autoCorrect={false}
                  value={email}
                  onChangeText={setEmail}
                  placeholder="operator@example.com"
                  placeholderTextColor={theme.colors.textMuted}
                  style={styles.input}
                />
              </View>
              <View style={styles.field}>
                <Text style={styles.label}>Password</Text>
                <TextInput
                  secureTextEntry
                  value={password}
                  onChangeText={setPassword}
                  placeholder="Your backend app password"
                  placeholderTextColor={theme.colors.textMuted}
                  style={styles.input}
                />
              </View>

              {error ? <Text style={styles.errorText}>{error}</Text> : null}

              <Pressable onPress={signIn} style={styles.primaryButton} disabled={loading}>
                {loading ? <ActivityIndicator color={theme.colors.buttonText} /> : <Text style={styles.primaryButtonText}>Sign In</Text>}
              </Pressable>
            </View>
          </SectionCard>
        ) : (
          <>
            <SectionCard
              title="Session"
              subtitle={`Signed in as ${session.email}`}
              action={
                <View style={styles.headerActions}>
                  <Pressable onPress={() => refresh()} style={styles.secondaryButton}>
                    <Text style={styles.secondaryButtonText}>Refresh</Text>
                  </Pressable>
                  <Pressable onPress={signOut} style={styles.ghostButton}>
                    <Text style={styles.ghostButtonText}>Sign Out</Text>
                  </Pressable>
                </View>
              }
            >
              <Text style={styles.helperText}>Session TTL from backend: {session.expiresInSeconds} seconds.</Text>
              {error ? <Text style={styles.errorText}>{error}</Text> : null}
            </SectionCard>

            <View style={styles.metricGrid}>
              <MetricTile
                label="API"
                value={(health?.status ?? "--").toUpperCase()}
                detail={health?.service ?? "signal-api"}
                tone={health?.status === "ok" ? "good" : "warn"}
              />
              <MetricTile
                label="Signal"
                value={dashboard?.decision.signal ?? "--"}
                detail={dashboard?.decision.decisionEngine ?? "ml"}
                tone={toneForSignal(dashboard?.decision.signal ?? "HOLD")}
              />
              <MetricTile
                label="Wallet"
                value={formatCurrency(dashboard?.estimatedWalletUsd ?? 0)}
                detail={`${dashboard?.walletAssetCount ?? 0} assets`}
                tone="default"
              />
              <MetricTile
                label="Autopilot"
                value={(autopilot?.status ?? "--").toUpperCase()}
                detail={`${autopilot?.currentCycle ?? 0}/${autopilot?.targetCycles ?? 0} cycles`}
                tone={autopilot?.running ? "good" : "warn"}
              />
            </View>

            <SectionCard title="Overview" subtitle="Backend-computed signal, risk, and readiness data">
              <View style={styles.dataList}>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Live symbol</Text>
                  <Text style={styles.dataValue}>{dashboard?.liveSymbol ?? "--"}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Probability up</Text>
                  <Text style={styles.dataValue}>{formatPercent(dashboard?.decision.probabilityUp ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Decision confidence</Text>
                  <Text style={styles.dataValue}>{formatPercent(dashboard?.decision.decisionConfidence ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Safe next action</Text>
                  <Text style={styles.dataValue}>{dashboard?.decision.safeNextAction ?? "--"}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Reserve cash</Text>
                  <Text style={styles.dataValue}>{formatCurrency(dashboard?.reserveCashUsd ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Max trade size</Text>
                  <Text style={styles.dataValue}>{formatCurrency(dashboard?.maxTradeSizeUsd ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Preview gate</Text>
                  <Text style={styles.dataValue}>{String(Boolean(dashboard?.previewGatePassed))}</Text>
                </View>
              </View>
              {dashboard?.readinessReason ? <Text style={styles.helperText}>{dashboard.readinessReason}</Text> : null}
            </SectionCard>

            <SectionCard title="Autopilot Snapshot" subtitle="Read-only runtime telemetry from the backend">
              <View style={styles.dataList}>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Symbol</Text>
                  <Text style={styles.dataValue}>{autopilot?.symbol ?? "--"}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Progress</Text>
                  <Text style={styles.dataValue}>{formatPercent(autopilot?.progressPct ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Current value</Text>
                  <Text style={styles.dataValue}>{formatCurrency(autopilot?.currentValue ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Goal value</Text>
                  <Text style={styles.dataValue}>{formatCurrency(autopilot?.goalValue ?? 0)}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Latest trade status</Text>
                  <Text style={styles.dataValue}>{autopilot?.latestTradeStatus ?? "--"}</Text>
                </View>
                <View style={styles.dataRow}>
                  <Text style={styles.dataLabel}>Safe next action</Text>
                  <Text style={styles.dataValue}>{autopilot?.safeNextAction ?? "--"}</Text>
                </View>
              </View>
              {autopilot?.updatedAt ? <Text style={styles.helperText}>Updated at {autopilot.updatedAt}</Text> : null}
              {autopilot?.lastError ? <Text style={styles.errorText}>{autopilot.lastError}</Text> : null}
            </SectionCard>

            <SectionCard title="Trade Desk" subtitle="Preview or send the selected request to the backend trade endpoint.">
              <View style={styles.form}>
                <View style={styles.field}>
                  <Text style={styles.label}>Action</Text>
                  <View style={styles.chipRow}>
                    <ActionChip
                      active={previewRequest.action === "market_buy"}
                      label="Market Buy"
                      onPress={() => updatePreviewField("action", "market_buy")}
                    />
                    <ActionChip
                      active={previewRequest.action === "market_sell"}
                      label="Market Sell"
                      onPress={() => updatePreviewField("action", "market_sell")}
                    />
                    <ActionChip
                      active={previewRequest.action === "cancel_all_orders"}
                      label="Cancel Orders"
                      onPress={() => updatePreviewField("action", "cancel_all_orders")}
                    />
                  </View>
                </View>

                <View style={styles.field}>
                  <Text style={styles.label}>Symbol</Text>
                  <TextInput
                    autoCapitalize="characters"
                    autoCorrect={false}
                    value={previewRequest.symbol}
                    onChangeText={(value) => updatePreviewField("symbol", value)}
                    style={styles.input}
                  />
                </View>

                <View style={styles.row}>
                  <View style={[styles.field, styles.halfField]}>
                    <Text style={styles.label}>Quantity</Text>
                    <TextInput
                      keyboardType="decimal-pad"
                      value={String(previewRequest.quantity)}
                      onChangeText={(value) => updatePreviewField("quantity", Number(value || 0))}
                      style={styles.input}
                    />
                  </View>
                  <View style={[styles.field, styles.halfField]}>
                    <Text style={styles.label}>Quote amount</Text>
                    <TextInput
                      keyboardType="decimal-pad"
                      value={String(previewRequest.quoteAmount)}
                      onChangeText={(value) => updatePreviewField("quoteAmount", Number(value || 0))}
                      style={styles.input}
                    />
                  </View>
                </View>

                <View style={styles.row}>
                  <View style={[styles.field, styles.halfField]}>
                    <Text style={styles.label}>Max latency ms</Text>
                    <TextInput
                      keyboardType="number-pad"
                      value={String(previewRequest.maxApiLatencyMs)}
                      onChangeText={(value) => updatePreviewField("maxApiLatencyMs", Number(value || 0))}
                      style={styles.input}
                    />
                  </View>
                  <View style={[styles.field, styles.halfField]}>
                    <Text style={styles.label}>Max spread bps</Text>
                    <TextInput
                      keyboardType="decimal-pad"
                      value={String(previewRequest.maxSpreadBps)}
                      onChangeText={(value) => updatePreviewField("maxSpreadBps", Number(value || 0))}
                      style={styles.input}
                    />
                  </View>
                </View>

                <View style={styles.field}>
                  <Text style={styles.label}>Mode</Text>
                  <View style={styles.chipRow}>
                    <ActionChip
                      active={!previewRequest.dryRun}
                      label="Live"
                      onPress={() => updatePreviewField("dryRun", false)}
                    />
                    <ActionChip
                      active={previewRequest.dryRun}
                      label="Dry Run"
                      onPress={() => updatePreviewField("dryRun", true)}
                    />
                  </View>
                </View>

                <View style={styles.actionRow}>
                  <Pressable onPress={previewTrade} style={styles.secondaryButtonWide} disabled={loading}>
                    {loading ? <ActivityIndicator color={theme.colors.text} /> : <Text style={styles.secondaryButtonText}>Preview</Text>}
                  </Pressable>
                  <Pressable
                    onPress={executeTrade}
                    style={[styles.primaryButton, styles.actionButton]}
                    disabled={loading}
                  >
                    {loading ? (
                      <ActivityIndicator color={theme.colors.buttonText} />
                    ) : (
                      <Text style={styles.primaryButtonText}>{previewRequest.dryRun ? "Run Dry Trade" : "Run Live Trade"}</Text>
                    )}
                  </Pressable>
                </View>
              </View>

              {previewResult ? (
                <View style={styles.dataList}>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Status</Text>
                    <Text style={styles.dataValue}>{previewResult.status}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Action</Text>
                    <Text style={styles.dataValue}>{previewResult.action}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Effective quantity</Text>
                    <Text style={styles.dataValue}>{formatNumber(previewResult.effectiveQuantity, 6)}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Market price</Text>
                    <Text style={styles.dataValue}>{formatNumber(previewResult.marketPrice, 4)}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Minimums</Text>
                    <Text style={styles.dataValue}>{previewResult.minimumMessage || "--"}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Size cap</Text>
                    <Text style={styles.dataValue}>{previewResult.sizeCapReason || "--"}</Text>
                  </View>
                  {previewResult.guardMessage ? <Text style={styles.helperText}>{previewResult.guardMessage}</Text> : null}
                </View>
              ) : null}

              {tradeResult ? (
                <View style={styles.dataList}>
                  <View style={styles.separator} />
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Executed status</Text>
                    <Text style={styles.dataValue}>{tradeResult.status}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Executed action</Text>
                    <Text style={styles.dataValue}>{tradeResult.action}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Guard mode</Text>
                    <Text style={styles.dataValue}>{tradeResult.guardMode || "--"}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Spread bps</Text>
                    <Text style={styles.dataValue}>{formatNumber(tradeResult.spreadBps || 0, 1)}</Text>
                  </View>
                  <View style={styles.dataRow}>
                    <Text style={styles.dataLabel}>Ticker age ms</Text>
                    <Text style={styles.dataValue}>{formatNumber(tradeResult.tickerAgeMs || 0, 0)}</Text>
                  </View>
                  {tradeResult.message ? <Text style={styles.helperText}>{tradeResult.message}</Text> : null}
                  {tradeResult.guardMessage ? <Text style={styles.helperText}>{tradeResult.guardMessage}</Text> : null}
                </View>
              ) : null}
            </SectionCard>

            <SectionCard title="Connection Notes" subtitle="What changes when you move from simulator to your own iPhone">
              <Text style={styles.helperText}>
                `127.0.0.1` works only in the iOS simulator running on your Mac. For a real iPhone on your local network, use your Mac LAN address such as `http://192.168.x.x:8765`.
              </Text>
              <Text style={styles.helperText}>
                For a Mac-independent setup, deploy the backend and point this app at the hosted URL `https://cella.website/api` in this repo's Oracle layout.
              </Text>
            </SectionCard>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  content: {
    padding: theme.spacing.lg,
    gap: theme.spacing.lg,
  },
  hero: {
    gap: theme.spacing.xs,
  },
  eyebrow: {
    color: theme.colors.accent,
    textTransform: "uppercase",
    letterSpacing: 1.4,
    fontSize: 12,
    fontWeight: "700",
  },
  title: {
    color: theme.colors.text,
    fontSize: 34,
    fontWeight: "800",
  },
  subtitle: {
    color: theme.colors.textMuted,
    fontSize: 15,
    lineHeight: 22,
  },
  form: {
    gap: theme.spacing.md,
  },
  field: {
    gap: 6,
  },
  row: {
    flexDirection: "row",
    gap: theme.spacing.md,
  },
  actionRow: {
    flexDirection: "row",
    gap: theme.spacing.md,
  },
  halfField: {
    flex: 1,
  },
  label: {
    color: theme.colors.textMuted,
    fontSize: 13,
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  input: {
    backgroundColor: theme.colors.inputBackground,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.colors.border,
    color: theme.colors.text,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    fontSize: 16,
  },
  primaryButton: {
    backgroundColor: theme.colors.button,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 48,
    paddingHorizontal: theme.spacing.lg,
  },
  actionButton: {
    flex: 1,
  },
  primaryButtonText: {
    color: theme.colors.buttonText,
    fontSize: 16,
    fontWeight: "700",
  },
  headerActions: {
    flexDirection: "row",
    gap: theme.spacing.sm,
  },
  secondaryButton: {
    backgroundColor: theme.colors.panelMuted,
    borderRadius: 999,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  secondaryButtonWide: {
    backgroundColor: theme.colors.panelMuted,
    borderRadius: 999,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: theme.colors.border,
    minHeight: 48,
    justifyContent: "center",
    alignItems: "center",
    flex: 1,
  },
  secondaryButtonText: {
    color: theme.colors.text,
    fontWeight: "600",
  },
  ghostButton: {
    borderRadius: 999,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: theme.colors.border,
  },
  ghostButtonText: {
    color: theme.colors.textMuted,
    fontWeight: "600",
  },
  errorText: {
    color: theme.colors.danger,
    fontSize: 14,
    lineHeight: 20,
  },
  helperText: {
    color: theme.colors.textMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  metricGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: theme.spacing.md,
  },
  dataList: {
    gap: theme.spacing.sm,
  },
  separator: {
    height: 1,
    backgroundColor: theme.colors.border,
    marginVertical: theme.spacing.sm,
  },
  dataRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: theme.spacing.md,
  },
  dataLabel: {
    color: theme.colors.textMuted,
    flex: 1,
    fontSize: 14,
  },
  dataValue: {
    color: theme.colors.text,
    flex: 1,
    fontSize: 14,
    fontWeight: "600",
    textAlign: "right",
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: theme.spacing.sm,
  },
  chip: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: theme.colors.border,
    backgroundColor: theme.colors.inputBackground,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: 10,
  },
  chipActive: {
    backgroundColor: "rgba(31, 127, 255, 0.22)",
    borderColor: theme.colors.accent,
  },
  chipText: {
    color: theme.colors.textMuted,
    fontWeight: "600",
  },
  chipTextActive: {
    color: theme.colors.text,
  },
});
