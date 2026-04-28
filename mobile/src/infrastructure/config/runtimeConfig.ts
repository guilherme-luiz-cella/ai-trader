const DEFAULT_HOSTED_BACKEND_URL = "https://cella.website/api";

export function resolveInitialBackendUrl(envValue?: string): string {
  const configured = String(envValue ?? "").trim();
  return configured || DEFAULT_HOSTED_BACKEND_URL;
}

export const initialBackendUrl = resolveInitialBackendUrl(process.env.EXPO_PUBLIC_API_BASE_URL);
