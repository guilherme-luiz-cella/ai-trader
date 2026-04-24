export type BackendConfig = {
  baseUrl: string;
};

export function normalizeBackendBaseUrl(input: string): BackendConfig {
  const trimmed = input.trim();
  if (!trimmed) {
    throw new Error("Backend URL is required.");
  }

  const normalized = trimmed.replace(/\/+$/, "");

  if (!/^https?:\/\//i.test(normalized)) {
    throw new Error("Backend URL must start with http:// or https://.");
  }

  return {
    baseUrl: normalized,
  };
}
