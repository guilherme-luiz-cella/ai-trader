import { StyleSheet, Text, View } from "react-native";

import { theme } from "../styles/theme";

type Props = {
  label: string;
  value: string;
  detail?: string;
  tone?: "default" | "good" | "warn" | "bad";
};

export function MetricTile({ label, value, detail, tone = "default" }: Props) {
  const toneColor =
    tone === "good"
      ? theme.colors.success
      : tone === "warn"
        ? theme.colors.warning
        : tone === "bad"
          ? theme.colors.danger
          : theme.colors.accent;

  return (
    <View style={styles.tile}>
      <Text style={styles.label}>{label}</Text>
      <Text style={[styles.value, { color: toneColor }]}>{value}</Text>
      {detail ? <Text style={styles.detail}>{detail}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  tile: {
    minWidth: "47%",
    flexGrow: 1,
    backgroundColor: theme.colors.panelMuted,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.colors.border,
    padding: theme.spacing.md,
    gap: 6,
  },
  label: {
    color: theme.colors.textMuted,
    fontSize: 12,
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  value: {
    fontSize: 22,
    fontWeight: "700",
  },
  detail: {
    color: theme.colors.textMuted,
    fontSize: 13,
    lineHeight: 18,
  },
});
