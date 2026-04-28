import { useState } from "react";

import type { AuthSession } from "../../domain/entities/AuthSession";
import type { OperatorOverview } from "../../domain/entities/OperatorOverview";
import type { TradePreview, TradePreviewRequest, TradeResult } from "../../domain/entities/TradePreview";
import { SignInUseCase } from "../../application/use-cases/SignInUseCase";
import { LoadOperatorOverviewUseCase } from "../../application/use-cases/LoadOperatorOverviewUseCase";
import { ExecuteTradeUseCase } from "../../application/use-cases/ExecuteTradeUseCase";
import { PreviewTradeUseCase } from "../../application/use-cases/PreviewTradeUseCase";
import { initialBackendUrl } from "../../infrastructure/config/runtimeConfig";
import { FetchHttpClient } from "../../infrastructure/http/FetchHttpClient";
import { HttpTradingControlRepository } from "../../infrastructure/repositories/HttpTradingControlRepository";

const repository = new HttpTradingControlRepository(new FetchHttpClient());
const signInUseCase = new SignInUseCase(repository);
const loadOperatorOverviewUseCase = new LoadOperatorOverviewUseCase(repository);
const previewTradeUseCase = new PreviewTradeUseCase(repository);
const executeTradeUseCase = new ExecuteTradeUseCase(repository);

const initialPreviewRequest: TradePreviewRequest = {
  action: "market_buy",
  symbol: "BTC/USDT",
  quantity: 0.001,
  quoteAmount: 0,
  dryRun: false,
  maxApiLatencyMs: 1200,
  maxTickerAgeMs: 3000,
  maxSpreadBps: 20,
  minTradeCooldownSeconds: 5,
};

export function useTradingControlApp() {
  const [backendUrl, setBackendUrl] = useState(initialBackendUrl);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [session, setSession] = useState<AuthSession | null>(null);
  const [overview, setOverview] = useState<OperatorOverview | null>(null);
  const [previewRequest, setPreviewRequest] = useState<TradePreviewRequest>(initialPreviewRequest);
  const [previewResult, setPreviewResult] = useState<TradePreview | null>(null);
  const [tradeResult, setTradeResult] = useState<TradeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refresh(activeSession = session) {
    if (!activeSession) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const nextOverview = await loadOperatorOverviewUseCase.execute({
        backendUrl,
        sessionToken: activeSession.token,
      });
      setOverview(nextOverview);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Unable to refresh the operator overview.");
    } finally {
      setLoading(false);
    }
  }

  async function signIn() {
    setLoading(true);
    setError("");

    try {
      const nextSession = await signInUseCase.execute({
        backendUrl,
        email,
        password,
      });
      setSession(nextSession);
      setPassword("");

      const nextOverview = await loadOperatorOverviewUseCase.execute({
        backendUrl,
        sessionToken: nextSession.token,
      });
      setOverview(nextOverview);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Unable to sign in.");
      setSession(null);
      setOverview(null);
    } finally {
      setLoading(false);
    }
  }

  async function previewTrade() {
    if (!session) {
      setError("Sign in first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const result = await previewTradeUseCase.execute({
        backendUrl,
        sessionToken: session.token,
        request: previewRequest,
      });
      setPreviewResult(result);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Unable to preview the trade.");
    } finally {
      setLoading(false);
    }
  }

  async function executeTrade() {
    if (!session) {
      setError("Sign in first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const result = await executeTradeUseCase.execute({
        backendUrl,
        sessionToken: session.token,
        request: previewRequest,
      });
      setTradeResult(result);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Unable to execute the trade.");
    } finally {
      setLoading(false);
    }
  }

  function signOut() {
    setSession(null);
    setOverview(null);
    setPreviewResult(null);
    setTradeResult(null);
    setPassword("");
    setError("");
  }

  function updatePreviewField<Key extends keyof TradePreviewRequest>(field: Key, value: TradePreviewRequest[Key]) {
    setPreviewRequest((current) => ({
      ...current,
      [field]: value,
    }));
  }

  return {
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
  };
}
