import { describe, expect, it } from "vitest";

import { resolveInitialBackendUrl } from "../../src/infrastructure/config/runtimeConfig";

describe("runtimeConfig", () => {
  it("uses the configured hosted API URL when provided", () => {
    expect(resolveInitialBackendUrl("https://example.com/api")).toBe("https://example.com/api");
  });

  it("falls back to the hosted cella.website API when no env value is provided", () => {
    expect(resolveInitialBackendUrl("")).toBe("https://cella.website/api");
  });
});
