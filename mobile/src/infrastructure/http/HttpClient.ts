export type HttpRequestOptions = {
  method?: "GET" | "POST";
  headers?: Record<string, string>;
  body?: unknown;
};

export interface HttpClient {
  request<TResponse>(url: string, options?: HttpRequestOptions): Promise<TResponse>;
}

export class RequestError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "RequestError";
  }
}
