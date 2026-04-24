import { StyleSheet, Text, View } from "react-native";

import { theme } from "../styles/theme";

export function DryModeBanner() {
  return (
    <View style={styles.banner}>
      <Text style={styles.title}>Dry mode only</Text>
      <Text style={styles.copy}>
        This iPhone client can read status and request trade previews, but it does not call live execution endpoints.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "rgba(255, 203, 84, 0.12)",
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: "rgba(255, 203, 84, 0.35)",
    padding: theme.spacing.md,
    gap: 6,
  },
  title: {
    color: theme.colors.warning,
    fontSize: 15,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  copy: {
    color: theme.colors.text,
    fontSize: 14,
    lineHeight: 20,
  },
});
