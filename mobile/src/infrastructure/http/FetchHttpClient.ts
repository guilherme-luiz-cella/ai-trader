import { HttpClient, RequestError, type HttpRequestOptions } from "./HttpClient";

export class FetchHttpClient implements HttpClient {
  async request<TResponse>(url: string, options: HttpRequestOptions = {}): Promise<TResponse> {
    const response = await fetch(url, {
      method: options.method ?? "GET",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers ?? {}),
      },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
    });

    const payload = await response.json().catch(() => ({}));

    if (response.status === 401 || payload?.status === "unauthorized") {
      throw new RequestError(payload?.message ?? "Authentication required.", 401);
    }

    if (!response.ok || payload?.status === "error") {
      throw new RequestError(payload?.message ?? `Request failed for ${url}`, response.status);
    }

    return payload as TResponse;
  }
}
