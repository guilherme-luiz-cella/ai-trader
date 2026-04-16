from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = BASE_DIR / "data_sets"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "prod_alpha"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_CACHE_LOCK = threading.Lock()
PIPELINE_CACHE: Dict[str, Dict[Any, Any]] = {
    "raw": {},
    "features": {},
    "splits": {},
}
_LAST_DEVICE_LOG: str | None = None
TABULAR_PROFILES = {"default", "fast-tabular", "very-light-debug"}


def _log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[prod_alpha {ts}] {message}", flush=True)


def _safe_import_psutil() -> Any | None:
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _cache_put(cache_name: str, key: Any, value: Any, max_entries: int) -> None:
    store = PIPELINE_CACHE[cache_name]
    store[key] = value
    while len(store) > max(1, int(max_entries)):
        oldest_key = next(iter(store.keys()))
        if oldest_key == key and len(store) == 1:
            break
        store.pop(oldest_key, None)


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024.0**3)


def _detect_system_resources() -> Dict[str, Any]:
    cpu_total = max(1, int(os.cpu_count() or 1))
    psutil = _safe_import_psutil()
    ram_total_gb = 0.0
    ram_available_gb = 0.0
    ram_pct = 0.0
    if psutil is not None:
        vm = psutil.virtual_memory()
        ram_total_gb = _bytes_to_gb(vm.total)
        ram_available_gb = _bytes_to_gb(vm.available)
        ram_pct = float(vm.percent)

    gpu: Dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
        "selected_gpu": "",
        "selected_index": None,
        "total_memory_gb": 0.0,
        "free_memory_gb": 0.0,
        "nvidia_smi_detected": False,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(idx)
        total_mem = float(props.total_memory)
        free_mem = float(total_mem)
        try:
            free_mem, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            pass
        gpu.update(
            {
                "selected_gpu": torch.cuda.get_device_name(idx),
                "selected_index": idx,
                "total_memory_gb": _bytes_to_gb(total_mem),
                "free_memory_gb": _bytes_to_gb(free_mem),
            }
        )

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            gpu["nvidia_smi_detected"] = True
            rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            parsed = []
            for line in rows:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    parsed.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": float(parts[2]),
                            "memory_free_mb": float(parts[3]),
                            "utilization_gpu_pct": float(parts[4]),
                        }
                    )
            gpu["nvidia_smi_gpus"] = parsed
    except Exception:
        gpu["nvidia_smi_detected"] = False

    return {
        "platform": platform.platform(),
        "cpu_total_cores": cpu_total,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "ram_percent_used": ram_pct,
        "gpu": gpu,
    }


def _runtime_utilization_snapshot() -> Dict[str, Any]:
    psutil = _safe_import_psutil()
    snap: Dict[str, Any] = {}
    if psutil is not None:
        vm = psutil.virtual_memory()
        snap.update(
            {
                "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
                "ram_percent": float(vm.percent),
                "ram_used_gb": _bytes_to_gb(vm.used),
            }
        )

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        idx = int(torch.cuda.current_device())
        total_mem = float(torch.cuda.get_device_properties(idx).total_memory)
        try:
            free_mem, _ = torch.cuda.mem_get_info(idx)
            used_mem = total_mem - float(free_mem)
            snap.update(
                {
                    "gpu_index": idx,
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "gpu_mem_used_gb": _bytes_to_gb(used_mem),
                    "gpu_mem_total_gb": _bytes_to_gb(total_mem),
                    "gpu_mem_used_pct": float((used_mem / max(1.0, total_mem)) * 100.0),
                }
            )
        except Exception:
            pass
    return snap


def _choose_gru_batch_size(base_batch_size: int, auto_batch_size: bool, device: torch.device) -> int:
    base = max(8, int(base_batch_size))
    if not auto_batch_size or device.type != "cuda":
        return base

    try:
        idx = int(torch.cuda.current_device())
        free_mem, _ = torch.cuda.mem_get_info(idx)
        free_gb = _bytes_to_gb(free_mem)
        if free_gb >= 10:
            return max(base, 256)
        if free_gb >= 7:
            return max(base, 192)
        if free_gb >= 5:
            return max(base, 128)
        if free_gb >= 3:
            return max(base, 96)
    except Exception:
        pass
    return max(base, 64)


def _configure_runtime_resources(config: "AlphaConfig", workers: int) -> Dict[str, Any]:
    resources = _detect_system_resources()
    cpu_total = int(resources.get("cpu_total_cores", max(1, int(os.cpu_count() or 1))))
    requested_threads = int(config.cpu_thread_count) if int(config.cpu_thread_count) > 0 else cpu_total
    per_worker_threads = max(1, requested_threads // max(1, int(workers)))
    if workers <= 1:
        per_worker_threads = requested_threads

    os.environ["OMP_NUM_THREADS"] = str(per_worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(per_worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(per_worker_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(per_worker_threads)

    try:
        torch.set_num_threads(max(1, per_worker_threads))
    except Exception:
        pass

    effective = {
        "requested_cpu_threads": requested_threads,
        "effective_cpu_threads_per_worker": per_worker_threads,
        "parallel_workers": int(workers),
        "oversubscription_guard": "enabled" if workers > 1 else "not_needed",
    }
    return {"detected": resources, "effective": effective}


def _bound_status_from_runtime(stage_seconds: Dict[str, float], util: Dict[str, Any]) -> str:
    if not stage_seconds:
        return "waiting_or_unknown"
    dominant_stage = max(stage_seconds.items(), key=lambda x: float(x[1]))[0]
    cpu_pct = float(util.get("cpu_percent", 0.0) or 0.0)
    ram_pct = float(util.get("ram_percent", 0.0) or 0.0)
    gpu_mem_pct = float(util.get("gpu_mem_used_pct", 0.0) or 0.0)

    if "data" in dominant_stage or "artifact" in dominant_stage:
        return "io_bound"
    if ram_pct >= 88.0:
        return "ram_bound"
    if gpu_mem_pct >= 70.0 and cpu_pct < 65.0:
        return "gpu_bound"
    if cpu_pct >= 70.0:
        return "cpu_bound"
    return "waiting_or_mixed"


@contextmanager
def _stage_timer(stage: str, timings: Dict[str, float], enabled: bool = True):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[stage] = timings.get(stage, 0.0) + elapsed
        if enabled:
            _log(f"stage={stage} elapsed={elapsed:.3f}s")


def _file_signature(path: Path) -> Tuple[str, bool, int, int]:
    exists = path.exists()
    if not exists:
        return str(path), False, 0, 0
    stat = path.stat()
    return str(path), True, int(stat.st_mtime_ns), int(stat.st_size)


def _data_cache_key(config: "AlphaConfig") -> Tuple[Any, ...]:
    asset_sig = tuple(sorted((sym, _file_signature(path)) for sym, path in config.asset_paths.items()))
    context_paths = [
        DATA_DIR / "fear_greed_features.csv",
        DATA_DIR / "fred_macro_features.csv",
        DATA_DIR / "gdelt_event_features.csv",
        DATA_DIR / "market_news.csv",
    ]
    context_sig = tuple(_file_signature(path) for path in context_paths)
    return (
        config.benchmark_asset,
        asset_sig,
        context_sig,
    )


def _feature_cache_key(config: "AlphaConfig", data_key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return (
        data_key,
        config.horizon,
        config.buy_threshold,
        config.sell_threshold,
        config.feature_corr_threshold,
    )


def _split_cache_key(config: "AlphaConfig", feature_key: Tuple[Any, ...], feat: pd.DataFrame) -> Tuple[Any, ...]:
    if feat.empty:
        ts_bounds: Tuple[str, str, int] = ("", "", 0)
    else:
        ts_bounds = (str(feat["timestamp"].min()), str(feat["timestamp"].max()), int(len(feat)))
    return (feature_key, config.walk_forward_splits, ts_bounds)


def _resolve_torch_device(log_device: bool = True) -> torch.device:
    global _LAST_DEVICE_LOG
    if torch.cuda.is_available():
        device = torch.device("cuda")
        desc = f"cuda:{torch.cuda.get_device_name(0)}"
    else:
        device = torch.device("cpu")
        desc = "cpu"

    if log_device and _LAST_DEVICE_LOG != desc:
        _LAST_DEVICE_LOG = desc
        _log(f"GRU selected_device={desc}")
    return device


def get_prepared_pipeline_inputs(
    config: "AlphaConfig",
    *,
    enable_cache: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    cache_hits = {"raw": False, "features": False, "splits": False}

    data_key = _data_cache_key(config)
    feature_key = _feature_cache_key(config, data_key)

    raw: pd.DataFrame
    feat: pd.DataFrame
    feature_cols: List[str]
    dropped_features: List[str]
    splits: List[Tuple[np.ndarray, np.ndarray]]

    with _stage_timer("data_loading", timings, enabled=config.enable_timing_logs):
        with PIPELINE_CACHE_LOCK:
            has_raw = enable_cache and (not force_refresh) and (data_key in PIPELINE_CACHE["raw"])
            raw = PIPELINE_CACHE["raw"].get(data_key) if has_raw else pd.DataFrame()
        if has_raw:
            cache_hits["raw"] = True
            if config.enable_timing_logs:
                _log("cache=data_loading hit")
        else:
            if config.enable_timing_logs:
                _log("cache=data_loading miss")
            raw = load_multi_source_data(config)
            if enable_cache:
                with PIPELINE_CACHE_LOCK:
                    _cache_put("raw", data_key, raw, max_entries=config.max_cache_entries)

    with _stage_timer("feature_engineering", timings, enabled=config.enable_timing_logs):
        with PIPELINE_CACHE_LOCK:
            has_feat = enable_cache and (not force_refresh) and (feature_key in PIPELINE_CACHE["features"])
            feat_bundle = PIPELINE_CACHE["features"].get(feature_key) if has_feat else None
        if has_feat and feat_bundle is not None:
            cache_hits["features"] = True
            if config.enable_timing_logs:
                _log("cache=feature_engineering hit")
            feat, feature_cols, dropped_features = feat_bundle
        else:
            if config.enable_timing_logs:
                _log("cache=feature_engineering miss")
            feat = engineer_features(raw, config)
            feature_cols = _feature_columns(feat)
            feature_cols, dropped_features = prune_feature_set(feat, feature_cols, corr_threshold=config.feature_corr_threshold)
            if config.very_light_debug_mode:
                tail_size = max(120, min(320, len(feat) // 4))
                feat = feat.groupby("symbol", group_keys=False).tail(tail_size).reset_index(drop=True)
            elif config.debug_mode:
                tail_size = max(400, min(1200, len(feat) // 2))
                feat = feat.groupby("symbol", group_keys=False).tail(tail_size).reset_index(drop=True)
            if enable_cache:
                with PIPELINE_CACHE_LOCK:
                    _cache_put("features", feature_key, (feat, feature_cols, dropped_features), max_entries=config.max_cache_entries)

    with _stage_timer("walk_forward_split_build", timings, enabled=config.enable_timing_logs):
        split_key = _split_cache_key(config, feature_key, feat)
        with PIPELINE_CACHE_LOCK:
            has_splits = enable_cache and (not force_refresh) and (split_key in PIPELINE_CACHE["splits"])
            splits = PIPELINE_CACHE["splits"].get(split_key, []) if has_splits else []
        if has_splits:
            cache_hits["splits"] = True
            if config.enable_timing_logs:
                _log("cache=walk_forward_split_build hit")
        else:
            if config.enable_timing_logs:
                _log("cache=walk_forward_split_build miss")
            n_splits = config.walk_forward_splits
            if config.very_light_debug_mode:
                n_splits = 1
            elif config.debug_mode:
                n_splits = min(3, max(2, n_splits))
            splits = time_series_walk_forward_splits(feat["timestamp"], n_splits=n_splits)
            if enable_cache:
                with PIPELINE_CACHE_LOCK:
                    _cache_put("splits", split_key, splits, max_entries=config.max_cache_entries)

    return {
        "raw": raw,
        "feat": feat,
        "feature_cols": feature_cols,
        "dropped_features": dropped_features,
        "splits": splits,
        "timings": timings,
        "cache_hits": cache_hits,
    }


@dataclass
class AlphaConfig:
    asset_paths: Dict[str, Path]
    benchmark_asset: str = "BTC"
    horizon: int = 3
    buy_threshold: float = 0.008
    sell_threshold: float = -0.008
    min_stable_reserve: float = 0.25
    fee_bps: float = 8.0
    slippage_bps: float = 5.0
    latency_steps: int = 1
    walk_forward_splits: int = 5
    sequence_length: int = 20
    gru_epochs: int = 6
    min_confidence: float = 0.52
    min_edge: float = 0.05
    signal_cooldown_steps: int = 2
    max_daily_turnover: float = 0.35
    price_impact_coef_bps: float = 4.0
    max_position_weight: float = 0.35
    vol_scale_target: float = 0.03
    vol_scale_floor: float = 0.20
    vol_scale_ceiling: float = 1.00
    feature_corr_threshold: float = 0.985
    blend_lookback: int = 20
    tabular_profile: str = "default"
    sklearn_n_jobs: int = -1
    cpu_thread_count: int = 0
    gru_batch_size: int = 64
    gru_auto_batch_size: bool = True
    max_cache_entries: int = 8
    resource_monitoring: bool = True
    enable_cache: bool = True
    enable_timing_logs: bool = True
    debug_mode: bool = False
    very_light_debug_mode: bool = False
    random_seed: int = 42


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / period, adjust=False).mean()
    loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _max_drawdown(returns: pd.Series) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return dd


def _ulcer_index(returns: pd.Series, window: int = 20) -> pd.Series:
    dd = _max_drawdown(returns)
    return (dd.pow(2).rolling(window, min_periods=3).mean()).pow(0.5)


def _winsorize(frame: pd.DataFrame, cols: Iterable[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        if col not in out.columns:
            continue
        low = out[col].quantile(lower_q)
        high = out[col].quantile(upper_q)
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def _normalize_tabular_profile(profile: str | None) -> str:
    normalized = str(profile or "default").strip().lower()
    return normalized if normalized in TABULAR_PROFILES else "default"


def _active_tabular_profile(config: AlphaConfig) -> str:
    if config.very_light_debug_mode:
        return "very-light-debug"
    return _normalize_tabular_profile(config.tabular_profile)


def load_multi_source_data(config: AlphaConfig) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol, path in config.asset_paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ts_col = "datetime" if "datetime" in df.columns else "timestamp"
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {path}")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        if symbol != config.benchmark_asset:
            df = (
                df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
                .resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )
        else:
            # benchmark asset can be higher frequency; keep daily alignment for portfolio engine
            df = (
                df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
                .resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )

        df["symbol"] = symbol
        frames.append(df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise FileNotFoundError("No valid asset price files found for production pipeline.")

    price = pd.concat(frames, ignore_index=True)
    price["date"] = price["timestamp"].dt.date

    def _load_context_csv(path: Path, date_col: str = "date") -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["date"])
        frame = pd.read_csv(path)
        if date_col in frame.columns:
            frame["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.date
            if date_col != "date":
                frame = frame.drop(columns=[date_col], errors="ignore")
        elif "timestamp" in frame.columns:
            frame["date"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.date
        else:
            return pd.DataFrame(columns=["date"])
        frame = frame.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
        return frame

    context_files = [
        DATA_DIR / "fear_greed_features.csv",
        DATA_DIR / "fred_macro_features.csv",
        DATA_DIR / "gdelt_event_features.csv",
    ]

    context = pd.DataFrame({"date": sorted(price["date"].unique())})
    for file in context_files:
        c = _load_context_csv(file)
        if not c.empty:
            context = context.merge(c, on="date", how="left")

    # optional sentiment from market news (daily score proxy)
    market_news_path = DATA_DIR / "market_news.csv"
    if market_news_path.exists():
        news = pd.read_csv(market_news_path)
        if "datetime" in news.columns:
            news["date"] = pd.to_datetime(news["datetime"], errors="coerce").dt.date
            news["headline"] = news.get("headline", "").fillna("").astype(str)
            pos_words = ["bull", "surge", "gain", "upgrade", "strong", "rally"]
            neg_words = ["bear", "drop", "loss", "downgrade", "weak", "selloff"]

            def score(text: str) -> int:
                t = text.lower()
                return sum(w in t for w in pos_words) - sum(w in t for w in neg_words)

            daily = (
                news.dropna(subset=["date"])
                .assign(news_score=lambda x: x["headline"].map(score))
                .groupby("date", as_index=False)
                .agg(news_score=("news_score", "mean"), news_count=("headline", "count"))
            )
            context = context.merge(daily, on="date", how="left")

    context_cols = [c for c in context.columns if c != "date"]
    context[context_cols] = context[context_cols].ffill().bfill().fillna(0.0)

    merged = price.merge(context, on="date", how="left")
    return merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def engineer_features(raw: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    frame = raw.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    benchmark_close = (
        frame[frame["symbol"] == config.benchmark_asset][["timestamp", "close"]]
        .rename(columns={"close": "benchmark_close"})
        .drop_duplicates(subset=["timestamp"], keep="last")
    )

    feature_groups: List[pd.DataFrame] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        g = group.sort_values("timestamp").copy()
        g = g.merge(benchmark_close, on="timestamp", how="left")
        g["benchmark_close"] = g["benchmark_close"].ffill().bfill()

        g["ret_1"] = g["close"].pct_change(1)
        g["ret_3"] = g["close"].pct_change(3)
        g["ret_7"] = g["close"].pct_change(7)
        g["ret_14"] = g["close"].pct_change(14)
        g["log_ret"] = np.log(g["close"]).diff()

        g["volatility_10"] = g["ret_1"].rolling(10, min_periods=5).std()
        g["volatility_20"] = g["ret_1"].rolling(20, min_periods=10).std()
        g["vol_ewma"] = g["ret_1"].ewm(span=20, adjust=False).std()
        g["vol_cluster"] = g["ret_1"].pow(2).rolling(20, min_periods=8).mean()

        g["momentum_5"] = g["close"].pct_change(5)
        g["momentum_20"] = g["close"].pct_change(20)
        g["sma_10"] = g["close"].rolling(10, min_periods=5).mean()
        g["sma_30"] = g["close"].rolling(30, min_periods=10).mean()
        g["trend_factor"] = g["sma_10"] / g["sma_30"] - 1

        g["rsi_14"] = _rsi(g["close"], 14)
        macd, macd_signal, macd_hist = _macd(g["close"])
        g["macd"] = macd
        g["macd_signal"] = macd_signal
        g["macd_hist"] = macd_hist

        g["range_pct"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
        g["body_pct"] = (g["close"] - g["open"]) / g["open"].replace(0, np.nan)
        g["volume_z"] = (
            (g["volume"] - g["volume"].rolling(20, min_periods=5).mean())
            / g["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
        )

        # order book proxies when depth snapshots are unavailable
        g["spread_proxy"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
        g["order_imbalance_proxy"] = g["body_pct"] / g["range_pct"].replace(0, np.nan)
        g["liquidity_proxy"] = g["volume"] / g["range_pct"].replace(0, np.nan)

        g["rolling_corr_benchmark_20"] = g["ret_1"].rolling(20, min_periods=8).corr(g["benchmark_close"].pct_change())
        dd = _max_drawdown(g["ret_1"])
        g["drawdown"] = dd
        g["max_drawdown_30"] = dd.rolling(30, min_periods=10).min()
        g["ulcer_20"] = _ulcer_index(g["ret_1"], 20)

        # regime indicators
        vol_rank = g["volatility_20"].rolling(120, min_periods=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        g["regime_high_vol"] = (vol_rank > 0.75).astype(float)
        g["regime_low_vol"] = (vol_rank < 0.25).astype(float)
        g["regime_trend_up"] = (g["trend_factor"] > 0).astype(float)
        g["regime_trend_down"] = (g["trend_factor"] < 0).astype(float)

        g["future_return"] = g["close"].shift(-config.horizon) / g["close"] - 1
        g["target_class"] = 0
        g.loc[g["future_return"] >= config.buy_threshold, "target_class"] = 2
        g.loc[g["future_return"] <= config.sell_threshold, "target_class"] = 1

        feature_groups.append(g)

    features = pd.concat(feature_groups, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # missing/noisy handling
    numeric_cols = [c for c in features.columns if c not in {"timestamp", "date", "symbol"} and pd.api.types.is_numeric_dtype(features[c])]
    features = _winsorize(features, numeric_cols)
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
    features[numeric_cols] = features.groupby("symbol", sort=False)[numeric_cols].transform(lambda x: x.ffill().bfill())
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    return features.dropna(subset=["future_return"]).reset_index(drop=True)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"timestamp", "date", "symbol", "future_return", "target_class"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def time_series_walk_forward_splits(timestamps: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_times = np.array(sorted(pd.Series(timestamps).dropna().unique()))
    if len(unique_times) < n_splits + 3:
        return []

    fold_size = max(10, len(unique_times) // (n_splits + 1))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, n_splits + 1):
        train_end_idx = fold * fold_size
        val_end_idx = min(len(unique_times), (fold + 1) * fold_size)
        if val_end_idx - train_end_idx < 5 or train_end_idx < 30:
            continue
        train_times = set(unique_times[:train_end_idx])
        val_times = set(unique_times[train_end_idx:val_end_idx])
        train_idx = np.where(pd.Series(timestamps).isin(train_times))[0]
        val_idx = np.where(pd.Series(timestamps).isin(val_times))[0]
        if len(train_idx) > 30 and len(val_idx) > 10:
            splits.append((train_idx, val_idx))
    return splits


def _risk_metrics(returns: pd.Series, annualization: int = 252) -> Dict[str, float]:
    r = returns.fillna(0.0).astype(float)
    if len(r) == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "win_loss_ratio": 0.0, "cagr": 0.0}

    mean_r = float(r.mean())
    std_r = float(r.std())
    downside_std = float(r[r < 0].std()) if (r < 0).any() else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(annualization) if std_r > 1e-12 else 0.0
    sortino = (mean_r / downside_std) * math.sqrt(annualization) if downside_std > 1e-12 else 0.0

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    max_dd = float(drawdown.min())

    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    win_loss_ratio = float(wins / losses) if losses > 0 else float(wins)

    years = max(1e-6, len(r) / annualization)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "win_loss_ratio": float(win_loss_ratio),
        "cagr": float(cagr),
    }


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str], idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = df.iloc[idx][feature_cols].values.astype(np.float32)
    y = df.iloc[idx]["target_class"].values.astype(int)
    return x, y


def _strategy_utility(metrics: Dict[str, Any]) -> float:
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    sortino = float(metrics.get("sortino", 0.0) or 0.0)
    cagr = float(metrics.get("cagr", 0.0) or 0.0)
    max_dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    avg_turnover = float(metrics.get("avg_turnover", 0.0) or 0.0)
    trades = float(metrics.get("num_trades", 0.0) or 0.0)
    return float(sharpe + (0.35 * sortino) + (0.45 * cagr) - (0.55 * max_dd) - (0.22 * avg_turnover) - (0.0015 * trades))


def prune_feature_set(df: pd.DataFrame, feature_cols: List[str], corr_threshold: float = 0.985) -> Tuple[List[str], List[str]]:
    if not feature_cols:
        return [], []
    x = df[feature_cols].copy()
    stds = x.std(numeric_only=True)
    low_var = set(stds[stds <= 1e-8].index.tolist())
    kept = [c for c in feature_cols if c not in low_var]
    if len(kept) <= 1:
        return kept, sorted(list(low_var))

    corr = x[kept].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > corr_threshold]
        if len(high_corr) > 0:
            to_drop.add(col)

    selected = [c for c in kept if c not in to_drop]
    dropped = sorted(list(low_var.union(to_drop)))
    return selected, dropped


def _temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-9
    t = max(0.5, float(temperature))
    logits = np.log(np.clip(probs, eps, 1.0)) / t
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), eps, None)


def calibrate_temperature(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    if len(probs) == 0 or len(y_true) == 0:
        return 1.0, probs
    temps = np.linspace(0.7, 2.0, 14)
    best_t = 1.0
    best_loss = float("inf")
    for t in temps:
        scaled = _temperature_scale_probs(probs, float(t))
        try:
            loss = float(log_loss(y_true, scaled, labels=[0, 1, 2]))
        except ValueError:
            continue
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t, _temperature_scale_probs(probs, best_t)


def apply_regime_overlay(df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    if len(df) == 0 or len(probs) == 0:
        return probs
    adjusted = probs.copy()
    high_vol = (df.get("regime_high_vol", 0.0).values > 0.5)[: len(adjusted)]
    trend_up = (df.get("regime_trend_up", 0.0).values > 0.5)[: len(adjusted)]
    trend_down = (df.get("regime_trend_down", 0.0).values > 0.5)[: len(adjusted)]

    adjusted[high_vol, 0] += 0.08
    adjusted[high_vol, 2] *= 0.92
    adjusted[high_vol, 1] *= 0.96
    adjusted[trend_up, 2] += 0.04
    adjusted[trend_down, 1] += 0.04

    adjusted = np.clip(adjusted, 1e-9, None)
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    return adjusted


def build_execution_filtered_signal(df: pd.DataFrame, probs: np.ndarray, config: AlphaConfig) -> Tuple[np.ndarray, Dict[str, int]]:
    if len(df) == 0 or len(probs) == 0:
        return np.zeros((0,), dtype=int), {"blocked_low_conf": 0, "blocked_low_edge": 0, "blocked_cooldown": 0, "flip_count": 0}

    raw_pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    filtered = np.zeros(len(raw_pred), dtype=int)

    prev_signal: Dict[str, int] = {}
    cooldown: Dict[str, int] = {}
    blocked_low_conf = 0
    blocked_low_edge = 0
    blocked_cooldown = 0
    flip_count = 0

    for idx, row in enumerate(df.itertuples(index=False)):
        sym = str(row.symbol)
        sig = int(raw_pred[idx])
        p0, p1, p2 = float(probs[idx, 0]), float(probs[idx, 1]), float(probs[idx, 2])

        edge = 0.0
        if sig == 2:
            edge = p2 - max(p0, p1)
        elif sig == 1:
            edge = p1 - max(p0, p2)

        min_conf = config.min_confidence + (0.05 if float(getattr(row, "regime_high_vol", 0.0) or 0.0) > 0.5 else 0.0)

        if confidence[idx] < min_conf:
            sig = 0
            blocked_low_conf += 1
        elif sig in (1, 2) and edge < config.min_edge:
            sig = 0
            blocked_low_edge += 1

        cd = int(cooldown.get(sym, 0))
        prev = int(prev_signal.get(sym, 0))
        if cd > 0 and sig in (1, 2) and prev in (1, 2) and sig != prev:
            sig = 0
            blocked_cooldown += 1

        if sig in (1, 2):
            if prev in (1, 2) and sig != prev:
                flip_count += 1
            prev_signal[sym] = sig
            cooldown[sym] = config.signal_cooldown_steps
        else:
            cooldown[sym] = max(0, cd - 1)

        filtered[idx] = sig

    return filtered, {
        "blocked_low_conf": int(blocked_low_conf),
        "blocked_low_edge": int(blocked_low_edge),
        "blocked_cooldown": int(blocked_cooldown),
        "flip_count": int(flip_count),
    }


def _sequence_data(df: pd.DataFrame, feature_cols: List[str], indices: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    id_set = set(indices.tolist())
    x_seq: List[np.ndarray] = []
    y_seq: List[int] = []
    for _, group in df.groupby("symbol", sort=False):
        g = group.reset_index()
        for i in range(seq_len, len(g)):
            raw_idx = int(g.loc[i, "index"])
            if raw_idx not in id_set:
                continue
            window = g.loc[i - seq_len:i - 1, feature_cols].values.astype(np.float32)
            target = int(g.loc[i, "target_class"])
            x_seq.append(window)
            y_seq.append(target)
    if not x_seq:
        return np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.stack(x_seq), np.array(y_seq, dtype=int)


def train_gru_classifier(
    x_train_seq: np.ndarray,
    y_train: np.ndarray,
    x_val_seq: np.ndarray,
    input_dim: int,
    epochs: int,
    seed: int,
    gru_batch_size: int = 64,
    gru_auto_batch_size: bool = True,
) -> Tuple[GRUClassifier | None, np.ndarray]:
    if len(x_train_seq) == 0 or len(x_val_seq) == 0:
        return None, np.zeros((len(x_val_seq), 3), dtype=np.float32)

    set_seeds(seed)
    device = _resolve_torch_device(log_device=True)
    use_cuda = device.type == "cuda"
    model = GRUClassifier(input_dim=input_dim).to(device)
    batch_size = _choose_gru_batch_size(
        base_batch_size=gru_batch_size,
        auto_batch_size=gru_auto_batch_size,
        device=device,
    )
    _log(f"GRU runtime device={device} batch_size={batch_size} auto_batch={gru_auto_batch_size}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.tensor(x_train_seq, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)

    first_param = next(model.parameters())
    if first_param.device.type != device.type:
        raise RuntimeError(f"GRU model/device mismatch: model={first_param.device} expected={device}")

    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if xb.device.type != device.type or yb.device.type != device.type:
                raise RuntimeError(f"GRU tensor/device mismatch: xb={xb.device} yb={yb.device} expected={device}")
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_val_tensor = torch.tensor(x_val_seq, dtype=torch.float32, device=device)
        logits = model(x_val_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

    return model, probs


def _baseline_signal(df: pd.DataFrame) -> pd.Series:
    momentum = df["ret_3"].fillna(0.0)
    signal = pd.Series(0, index=df.index)
    signal[momentum > 0.003] = 2
    signal[momentum < -0.003] = 1
    return signal


def simulate_portfolio(
    test_df: pd.DataFrame,
    signal_col: str,
    config: AlphaConfig,
    allocation_scores: Dict[pd.Timestamp, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    df = test_df.copy().sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    timestamps = sorted(df["timestamp"].unique())
    symbols = sorted(df["symbol"].unique())

    cash_weight = 1.0
    weights = {sym: 0.0 for sym in symbols}
    entry_price = {sym: np.nan for sym in symbols}

    portfolio_returns: List[float] = []
    return_timestamps: List[str] = []
    daily_turnover: List[float] = []
    trades: List[Dict[str, Any]] = []

    fee = config.fee_bps / 10000
    slippage = config.slippage_bps / 10000

    entry_ts: Dict[str, pd.Timestamp | None] = {sym: None for sym in symbols}

    for i, ts in enumerate(timestamps):
        day = df[df["timestamp"] == ts]
        signal_map = {r.symbol: int(getattr(r, signal_col)) for r in day.itertuples(index=False)}
        close_map = {r.symbol: float(r.close) for r in day.itertuples(index=False)}
        vol_map = {r.symbol: float(getattr(r, "volatility_20", 0.0) or 0.0) for r in day.itertuples(index=False)}
        liq_map = {r.symbol: float(getattr(r, "liquidity_proxy", 0.0) or 0.0) for r in day.itertuples(index=False)}
        conf_map = {r.symbol: float(getattr(r, "signal_confidence", 0.0) or 0.0) for r in day.itertuples(index=False)}
        edge_map = {r.symbol: float(getattr(r, "signal_edge", 0.0) or 0.0) for r in day.itertuples(index=False)}

        day_cost = 0.0
        day_turnover_used = 0.0
        prev_weights = weights.copy()

        if allocation_scores and ts in allocation_scores:
            alloc = allocation_scores[ts]
            cash_weight = float(alloc.get("CASH", cash_weight))
            for sym in symbols:
                weights[sym] = float(alloc.get(sym, weights[sym]))

        # dynamic de-risk in high volatility regime
        high_vol_count = sum(1 for sym in symbols if vol_map.get(sym, 0.0) > 0.05)
        if high_vol_count >= max(1, len(symbols) // 2):
            de_risk = min(0.35, max(0.0, 1.0 - config.min_stable_reserve))
            cash_weight = min(1.0, cash_weight + de_risk)
            scale = max(1e-6, sum(weights.values()))
            for sym in symbols:
                weights[sym] = max(0.0, weights[sym] * (1 - de_risk) / scale)

        # trade decisions
        for sym in symbols:
            signal = signal_map.get(sym, 0)
            price = close_map.get(sym)
            if price is None or np.isnan(price):
                continue

            vol = float(vol_map.get(sym, 0.02) or 0.02)
            liq = max(0.0, float(liq_map.get(sym, 0.0) or 0.0))
            illiq = min(1.5, 1.0 / max(1.0, math.log1p(liq + 1.0)))
            dyn_slippage = slippage * (1 + 3.0 * vol + illiq)

            if signal == 2 and cash_weight > config.min_stable_reserve:
                kelly_p = 0.56
                tp = max(0.012, float(vol_map.get(sym, 0.02) * 1.2))
                sl = max(0.008, float(vol_map.get(sym, 0.02) * 0.8))
                edge = (kelly_p * (tp / sl)) - (1 - kelly_p)
                kelly_fraction = max(0.0, min(0.25, edge / (tp / sl + 1e-6)))
                vol_scale = max(
                    config.vol_scale_floor,
                    min(
                        config.vol_scale_ceiling,
                        config.vol_scale_target / (vol_map.get(sym, config.vol_scale_target) + 1e-6),
                    ),
                )
                add_weight = min(cash_weight - config.min_stable_reserve, kelly_fraction * vol_scale)
                add_weight = min(add_weight, config.max_position_weight - weights[sym])
                if day_turnover_used + add_weight > config.max_daily_turnover:
                    add_weight = max(0.0, config.max_daily_turnover - day_turnover_used)
                if add_weight > 0.0:
                    weights[sym] += add_weight
                    cash_weight -= add_weight
                    entry_price[sym] = price
                    entry_ts[sym] = ts
                    impact = (config.price_impact_coef_bps / 10000.0) * (add_weight / max(config.max_position_weight, 1e-6))
                    trade_cost_rate = fee + dyn_slippage + impact
                    day_cost += add_weight * trade_cost_rate
                    day_turnover_used += add_weight
                    trades.append(
                        {
                            "timestamp": str(ts),
                            "symbol": sym,
                            "action": "BUY",
                            "weight": add_weight,
                            "price": price,
                            "confidence": conf_map.get(sym, 0.0),
                            "edge": edge_map.get(sym, 0.0),
                            "cost_rate": trade_cost_rate,
                        }
                    )

            elif signal == 1 and weights[sym] > 0:
                # sell or reduce on bearish prediction
                reduce_weight = min(weights[sym], max(0.15, weights[sym] * 0.5))
                if day_turnover_used + reduce_weight > config.max_daily_turnover:
                    reduce_weight = max(0.0, config.max_daily_turnover - day_turnover_used)
                weights[sym] -= reduce_weight
                cash_weight += reduce_weight
                impact = (config.price_impact_coef_bps / 10000.0) * (reduce_weight / max(config.max_position_weight, 1e-6))
                trade_cost_rate = fee + dyn_slippage + impact
                day_cost += reduce_weight * trade_cost_rate
                day_turnover_used += reduce_weight
                hold_days = None
                if entry_ts.get(sym) is not None:
                    hold_days = max(0, int((pd.Timestamp(ts) - pd.Timestamp(entry_ts[sym])).days))
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "action": "SELL",
                        "weight": reduce_weight,
                        "price": price,
                        "confidence": conf_map.get(sym, 0.0),
                        "edge": edge_map.get(sym, 0.0),
                        "hold_days": hold_days,
                        "cost_rate": trade_cost_rate,
                    }
                )

            # target-based exits + dynamic stop/take
            if weights[sym] > 0 and not np.isnan(entry_price[sym]):
                ret_from_entry = price / entry_price[sym] - 1
                dynamic_tp = max(0.02, float(vol_map.get(sym, 0.02) * 1.5))
                dynamic_sl = -max(0.015, float(vol_map.get(sym, 0.02) * 1.0))
                if ret_from_entry >= dynamic_tp or ret_from_entry <= dynamic_sl:
                    exit_weight = weights[sym]
                    if day_turnover_used + exit_weight > config.max_daily_turnover:
                        exit_weight = max(0.0, config.max_daily_turnover - day_turnover_used)
                    if exit_weight > 0:
                        weights[sym] -= exit_weight
                        cash_weight += exit_weight
                        action = "TAKE_PROFIT" if ret_from_entry >= dynamic_tp else "STOP_LOSS"
                        impact = (config.price_impact_coef_bps / 10000.0) * (exit_weight / max(config.max_position_weight, 1e-6))
                        trade_cost_rate = fee + dyn_slippage + impact
                        day_cost += exit_weight * trade_cost_rate
                        day_turnover_used += exit_weight
                        hold_days = None
                        if entry_ts.get(sym) is not None:
                            hold_days = max(0, int((pd.Timestamp(ts) - pd.Timestamp(entry_ts[sym])).days))
                        trades.append(
                            {
                                "timestamp": str(ts),
                                "symbol": sym,
                                "action": action,
                                "weight": exit_weight,
                                "price": price,
                                "ret_from_entry": float(ret_from_entry),
                                "hold_days": hold_days,
                                "cost_rate": trade_cost_rate,
                            }
                        )
                        entry_ts[sym] = None

        # normalize weights
        total_risk = cash_weight + sum(weights.values())
        if total_risk > 0:
            cash_weight /= total_risk
            for sym in symbols:
                weights[sym] /= total_risk

        realized_turnover = 0.5 * sum(abs(weights[sym] - prev_weights[sym]) for sym in symbols)
        daily_turnover.append(float(max(realized_turnover, day_turnover_used)))

        # latency-adjusted returns use next timestamp move
        if i + config.latency_steps < len(timestamps):
            future_ts = timestamps[i + config.latency_steps]
            nxt = df[df["timestamp"] == future_ts]
            next_close = {r.symbol: float(r.close) for r in nxt.itertuples(index=False)}
            day_ret = 0.0
            for sym in symbols:
                c0 = close_map.get(sym)
                c1 = next_close.get(sym)
                if c0 and c1 and c0 > 0:
                    gross = c1 / c0 - 1
                    day_ret += weights[sym] * gross
            day_ret -= day_cost
            portfolio_returns.append(day_ret)
            return_timestamps.append(str(ts))

    returns_series = pd.Series(portfolio_returns, dtype=float)
    metrics = _risk_metrics(returns_series)
    metrics.update(
        {
            "total_return": float((1 + returns_series).prod() - 1) if len(returns_series) else 0.0,
            "avg_daily_return": float(returns_series.mean()) if len(returns_series) else 0.0,
            "num_trades": len(trades),
            "avg_turnover": float(np.mean(daily_turnover)) if daily_turnover else 0.0,
            "max_turnover": float(np.max(daily_turnover)) if daily_turnover else 0.0,
        }
    )
    action_counts = pd.Series([t.get("action", "") for t in trades]).value_counts().to_dict() if trades else {}
    hold_days = [float(t["hold_days"]) for t in trades if t.get("hold_days") is not None]
    avg_cost_rate = float(np.mean([float(t.get("cost_rate", 0.0)) for t in trades])) if trades else 0.0
    diagnostics = {
        "action_counts": action_counts,
        "avg_hold_days": float(np.mean(hold_days)) if hold_days else 0.0,
        "avg_trade_cost_rate": avg_cost_rate,
        "cost_drag_proxy": float(avg_cost_rate * len(trades)),
    }
    return {
        "metrics": metrics,
        "returns": returns_series,
        "return_timestamps": return_timestamps,
        "daily_turnover": daily_turnover,
        "trades": trades,
        "diagnostics": diagnostics,
    }


class SimpleRLAllocator:
    def __init__(self, assets: List[str], stable_asset: str = "CASH", seed: int = 42, fee_bps: float = 8.0, slippage_bps: float = 5.0):
        self.assets = assets
        self.stable_asset = stable_asset
        self.rng = np.random.default_rng(seed)
        self.fee = fee_bps / 10000.0
        self.slippage = slippage_bps / 10000.0
        self.q: Dict[Tuple[int, int], np.ndarray] = {}
        self.actions = [
            np.array([0.20, 0.20, 0.20]),
            np.array([0.30, 0.25, 0.20]),
            np.array([0.35, 0.30, 0.20]),
            np.array([0.15, 0.15, 0.10]),
            np.array([0.10, 0.10, 0.08]),
        ]

    def _state(self, row: pd.Series) -> Tuple[int, int]:
        vol_state = int(min(2, max(0, round(float(row.get("volatility_20", 0.02) * 100)) // 2)))
        trend_state = 1 if float(row.get("trend_factor", 0.0)) > 0 else 0
        return vol_state, trend_state

    def fit(self, df: pd.DataFrame, signal_col: str = "upgraded_signal", epochs: int = 4) -> None:
        grouped = df.sort_values("timestamp").groupby("timestamp")
        dates = list(grouped.groups.keys())
        if len(dates) < 50:
            return

        alpha = 0.2
        gamma = 0.95
        epsilon = 0.25

        for _ in range(epochs):
            prev_alloc_use: np.ndarray | None = None
            for i in range(len(dates) - 2):
                cur = grouped.get_group(dates[i])
                nxt = grouped.get_group(dates[i + 1])
                fwd = grouped.get_group(dates[i + 2])
                row_ref = cur.iloc[0]
                state = self._state(row_ref)
                if state not in self.q:
                    self.q[state] = np.zeros(len(self.actions), dtype=float)

                if self.rng.random() < epsilon:
                    action_idx = self.rng.integers(0, len(self.actions))
                else:
                    action_idx = int(np.argmax(self.q[state]))

                alloc = self.actions[action_idx]
                cur_close = dict(zip(cur["symbol"], cur["close"]))
                fwd_close = dict(zip(fwd["symbol"], fwd["close"]))
                symbols = [s for s in self.assets if s in cur_close and s in fwd_close]
                if not symbols:
                    continue
                alloc_use = alloc[: len(symbols)]
                alloc_use = alloc_use / max(1e-9, alloc_use.sum()) * 0.75

                port_ret = 0.0
                step_rets: List[float] = []
                for k, sym in enumerate(symbols):
                    r = fwd_close[sym] / cur_close[sym] - 1
                    step_rets.append(float(r))
                    port_ret += alloc_use[k] * r

                downside_penalty = 0.5 * max(0.0, -port_ret)
                realized_vol = float(np.std(step_rets)) if step_rets else 0.0
                vol_penalty = 0.35 * max(0.0, realized_vol - 0.03)

                turnover = 0.0
                if prev_alloc_use is not None and len(prev_alloc_use) == len(alloc_use):
                    turnover = 0.5 * float(np.sum(np.abs(alloc_use - prev_alloc_use)))
                cost_penalty = turnover * (self.fee + self.slippage) * 6.0

                signal_vec = cur[signal_col].values.astype(int) if signal_col in cur.columns else np.zeros((len(cur),), dtype=int)
                long_frac = float(np.mean(signal_vec == 2)) if len(signal_vec) else 0.0
                short_frac = float(np.mean(signal_vec == 1)) if len(signal_vec) else 0.0

                reward = (port_ret * (1.0 + 0.4 * long_frac)) - downside_penalty - vol_penalty - cost_penalty - (0.05 * short_frac)
                prev_alloc_use = alloc_use.copy()

                next_state = self._state(nxt.iloc[0])
                if next_state not in self.q:
                    self.q[next_state] = np.zeros(len(self.actions), dtype=float)

                td_target = reward + gamma * np.max(self.q[next_state])
                td_error = td_target - self.q[state][action_idx]
                self.q[state][action_idx] += alpha * td_error

            epsilon = max(0.05, epsilon * 0.8)

    def allocation_schedule(self, df: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
        schedule: Dict[pd.Timestamp, Dict[str, float]] = {}
        for ts, group in df.groupby("timestamp"):
            state = self._state(group.iloc[0])
            if state in self.q:
                action_idx = int(np.argmax(self.q[state]))
            else:
                action_idx = 0
            alloc = self.actions[action_idx]
            active_symbols = [s for s in self.assets if s in set(group["symbol"])]
            if not active_symbols:
                continue
            alloc_use = alloc[: len(active_symbols)]
            alloc_use = alloc_use / max(1e-9, alloc_use.sum()) * 0.75
            high_vol = float(group["regime_high_vol"].mean()) if "regime_high_vol" in group.columns else 0.0
            if high_vol > 0.5:
                alloc_use = alloc_use * 0.7
            mapping = {sym: float(alloc_use[i]) for i, sym in enumerate(active_symbols)}
            mapping["CASH"] = max(0.25, 1.0 - sum(mapping.values()))
            schedule[ts] = mapping
        return schedule


def _compute_blended_backtest(upgraded_backtest: Dict[str, Any], baseline_backtest: Dict[str, Any], lookback: int) -> Dict[str, Any]:
    up_ret = upgraded_backtest["returns"].reset_index(drop=True)
    base_ret = baseline_backtest["returns"].reset_index(drop=True)
    n = min(len(up_ret), len(base_ret))
    if n == 0:
        return {
            "metrics": {
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "win_loss_ratio": 0.0,
                "cagr": 0.0,
                "total_return": 0.0,
                "avg_daily_return": 0.0,
                "num_trades": 0,
                "avg_turnover": 0.0,
                "max_turnover": 0.0,
            },
            "returns": pd.Series(dtype=float),
            "trades": [],
        }

    up_use = up_ret.iloc[-n:]
    base_use = base_ret.iloc[-n:]
    blend_rets: List[float] = []
    for i in range(n):
        start = max(0, i - lookback + 1)
        up_win = up_use.iloc[start : i + 1]
        base_win = base_use.iloc[start : i + 1]
        up_sharpe = float(_risk_metrics(up_win).get("sharpe", 0.0))
        base_sharpe = float(_risk_metrics(base_win).get("sharpe", 0.0))
        blend_rets.append(float(up_use.iloc[i] if up_sharpe >= base_sharpe else base_use.iloc[i]))

    blend_series = pd.Series(blend_rets, dtype=float)
    metrics = _risk_metrics(blend_series)
    metrics.update(
        {
            "total_return": float((1 + blend_series).prod() - 1) if len(blend_series) else 0.0,
            "avg_daily_return": float(blend_series.mean()) if len(blend_series) else 0.0,
            "num_trades": int(min(upgraded_backtest["metrics"].get("num_trades", 0), baseline_backtest["metrics"].get("num_trades", 0))),
            "avg_turnover": float(min(upgraded_backtest["metrics"].get("avg_turnover", 0.0), baseline_backtest["metrics"].get("avg_turnover", 0.0))),
            "max_turnover": float(min(upgraded_backtest["metrics"].get("max_turnover", 0.0), baseline_backtest["metrics"].get("max_turnover", 0.0))),
        }
    )
    return {"metrics": metrics, "returns": blend_series, "trades": []}


def _build_tabular_models(config: AlphaConfig, seed: int, phase: str = "cv") -> List[Tuple[str, Any]]:
    debug = config.debug_mode or config.very_light_debug_mode
    profile = _active_tabular_profile(config)
    gb = GradientBoostingClassifier(random_state=seed)

    if profile == "fast-tabular":
        if phase == "cv":
            rf_estimators = 70 if debug else 110
            rf_depth = 5 if debug else 6
            lr_iter = 90 if debug else 140
        else:
            rf_estimators = 100 if debug else 140
            rf_depth = 6 if debug else 7
            lr_iter = 120 if debug else 160
        rf = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=config.sklearn_n_jobs,
        )
        lr = LogisticRegression(
            max_iter=lr_iter,
            random_state=seed,
            solver="lbfgs",
            multi_class="auto",
        )
        return [("gb", gb), ("rf", rf), ("lr", lr)]

    if phase == "cv":
        if profile == "very-light-debug":
            rf_estimators, rf_depth, mlp_layers, mlp_iter = 60, 5, (32, 16), 70
        else:
            rf_estimators = 120 if debug else 300
            rf_depth = 6 if debug else 8
            mlp_layers = (64, 32) if debug else (96, 48)
            mlp_iter = 120 if debug else 250
    else:
        if profile == "very-light-debug":
            rf_estimators, rf_depth, mlp_layers, mlp_iter = 90, 6, (48, 24), 90
        else:
            rf_estimators = 180 if debug else 400
            rf_depth = 7 if debug else 9
            mlp_layers = (96, 48) if debug else (128, 64)
            mlp_iter = 180 if debug else 300

    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        max_depth=rf_depth,
        random_state=seed,
        n_jobs=config.sklearn_n_jobs,
    )
    mlp = MLPClassifier(hidden_layer_sizes=mlp_layers, random_state=seed, max_iter=mlp_iter)
    return [("gb", gb), ("rf", rf), ("mlp", mlp)]


def _evaluate_single_split(
    feat: pd.DataFrame,
    feature_cols: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: AlphaConfig,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    train_df = feat.iloc[train_idx].copy()
    val_df = feat.iloc[val_idx].copy()
    if train_df.empty or val_df.empty:
        return {
            "upgraded": {"sharpe": 0.0, "max_drawdown": 0.0, "avg_turnover": 0.0},
            "baseline": {"sharpe": 0.0, "max_drawdown": 0.0, "avg_turnover": 0.0},
            "blended": {"sharpe": 0.0, "max_drawdown": 0.0, "avg_turnover": 0.0},
            "timings": timings,
        }

    with _stage_timer("split_preprocess", timings, enabled=config.enable_timing_logs):
        imp = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        x_train = imp.fit_transform(train_df[feature_cols])
        x_val = imp.transform(val_df[feature_cols])
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        y_train = train_df["target_class"].values.astype(int)
        y_val = val_df["target_class"].values.astype(int)

    tabular_models = _build_tabular_models(config, seed=config.random_seed, phase="cv")

    model_probs: Dict[str, np.ndarray] = {}
    model_utility_scores: Dict[str, float] = {}
    with _stage_timer("split_tabular_train_backtest", timings, enabled=config.enable_timing_logs):
        for name, model in tabular_models:
            model_fit_start = time.perf_counter()
            model.fit(x_train, y_train)
            timings[f"split_tabular_model_fit_{name}"] = float(
                timings.get(f"split_tabular_model_fit_{name}", 0.0) + (time.perf_counter() - model_fit_start)
            )
            probs = model.predict_proba(x_val)
            probs = apply_regime_overlay(val_df, probs)
            pred, _ = build_execution_filtered_signal(val_df, probs, config)
            local_df = val_df.copy()
            local_df["signal"] = pred
            local_df["signal_confidence"] = np.max(probs, axis=1)
            local_df["signal_edge"] = np.maximum(probs[:, 1], probs[:, 2]) - probs[:, 0]
            back = simulate_portfolio(local_df, signal_col="signal", config=config)
            model_probs[name] = probs
            model_utility_scores[name] = float(_strategy_utility(back["metrics"]))

    model_names = [name for name, _ in tabular_models]
    score_vec = np.array([model_utility_scores.get(name, 0.0) for name in model_names], dtype=float)
    score_vec = score_vec - np.max(score_vec)
    weights = np.exp(score_vec)
    weights = np.clip(weights, 1e-4, None)
    weights = weights / weights.sum()

    ensemble_probs = np.zeros_like(model_probs[model_names[0]])
    for idx, name in enumerate(model_names):
        ensemble_probs += weights[idx] * model_probs[name]
    ensemble_probs = apply_regime_overlay(val_df, ensemble_probs)

    with _stage_timer("split_gru_train", timings, enabled=config.enable_timing_logs):
        x_train_seq, y_train_seq = _sequence_data(feat, feature_cols, train_idx, config.sequence_length)
        x_val_seq, _ = _sequence_data(feat, feature_cols, val_idx, config.sequence_length)
        split_gru_epochs = 1 if config.very_light_debug_mode else max(2, config.gru_epochs)
        _, gru_probs = train_gru_classifier(
            x_train_seq=x_train_seq,
            y_train=y_train_seq,
            x_val_seq=x_val_seq,
            input_dim=len(feature_cols),
            epochs=split_gru_epochs,
            seed=config.random_seed,
            gru_batch_size=config.gru_batch_size,
            gru_auto_batch_size=config.gru_auto_batch_size,
        )

    eval_df = val_df.copy()
    eval_y = y_val.copy()
    if len(gru_probs) > 0:
        trim = min(len(ensemble_probs), len(gru_probs))
        eval_df = eval_df.iloc[-trim:].copy()
        eval_y = eval_y[-trim:]
        gru_adj = apply_regime_overlay(eval_df, gru_probs[-trim:])
        ensemble_probs = ensemble_probs[-trim:] * 0.80 + gru_adj * 0.20

    _, ensemble_probs = calibrate_temperature(ensemble_probs, eval_y)
    upgraded_pred, _ = build_execution_filtered_signal(eval_df, ensemble_probs, config)
    confidence = np.max(ensemble_probs, axis=1)
    edge = np.maximum(ensemble_probs[:, 1], ensemble_probs[:, 2]) - ensemble_probs[:, 0]
    baseline_pred = _baseline_signal(eval_df)

    upgraded_df = eval_df.copy()
    upgraded_df["upgraded_signal"] = upgraded_pred
    upgraded_df["signal_confidence"] = confidence
    upgraded_df["signal_edge"] = edge

    baseline_df = eval_df.copy()
    baseline_df["baseline_signal"] = baseline_pred.values

    with _stage_timer("split_backtest", timings, enabled=config.enable_timing_logs):
        allocator = SimpleRLAllocator(
            assets=sorted(upgraded_df["symbol"].unique()),
            seed=config.random_seed,
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
        )
        allocator.fit(upgraded_df, signal_col="upgraded_signal", epochs=4)
        alloc_schedule = allocator.allocation_schedule(upgraded_df)

        upgraded_backtest = simulate_portfolio(
            upgraded_df,
            signal_col="upgraded_signal",
            config=config,
            allocation_scores=alloc_schedule,
        )
        baseline_backtest = simulate_portfolio(
            baseline_df,
            signal_col="baseline_signal",
            config=config,
            allocation_scores=None,
        )
        blended_backtest = _compute_blended_backtest(upgraded_backtest, baseline_backtest, lookback=config.blend_lookback)

    return {
        "upgraded": upgraded_backtest["metrics"],
        "baseline": baseline_backtest["metrics"],
        "blended": blended_backtest["metrics"],
        "timings": timings,
    }


def evaluate_walk_forward_robustness(config: AlphaConfig) -> Dict[str, Any]:
    set_seeds(config.random_seed)
    util_start = _runtime_utilization_snapshot() if config.resource_monitoring else {}
    prepared = get_prepared_pipeline_inputs(config, enable_cache=config.enable_cache, force_refresh=False)
    feat = prepared["feat"]
    feature_cols = prepared["feature_cols"]
    dropped_features = prepared["dropped_features"]
    splits = prepared["splits"]
    if not splits:
        raise RuntimeError("Not enough data for walk-forward evaluation.")

    fold_rows: List[Dict[str, Any]] = []
    split_timing_rows: List[Dict[str, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        split_res = _evaluate_single_split(feat, feature_cols, train_idx, val_idx, config)
        split_timing_rows.append(split_res.get("timings", {}))
        row = {
            "fold": int(fold_idx),
            "upgraded_sharpe": float(split_res["upgraded"].get("sharpe", 0.0)),
            "upgraded_max_drawdown": float(split_res["upgraded"].get("max_drawdown", 0.0)),
            "upgraded_avg_turnover": float(split_res["upgraded"].get("avg_turnover", 0.0)),
            "baseline_sharpe": float(split_res["baseline"].get("sharpe", 0.0)),
            "baseline_max_drawdown": float(split_res["baseline"].get("max_drawdown", 0.0)),
            "baseline_avg_turnover": float(split_res["baseline"].get("avg_turnover", 0.0)),
            "blended_sharpe": float(split_res["blended"].get("sharpe", 0.0)),
            "blended_max_drawdown": float(split_res["blended"].get("max_drawdown", 0.0)),
            "blended_avg_turnover": float(split_res["blended"].get("avg_turnover", 0.0)),
        }
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    beat_baseline_rate = float((fold_df["upgraded_sharpe"] > fold_df["baseline_sharpe"]).mean())
    beat_blend_rate = float((fold_df["upgraded_sharpe"] > fold_df["blended_sharpe"]).mean())

    mean_up_sh = float(fold_df["upgraded_sharpe"].mean())
    std_up_sh = float(fold_df["upgraded_sharpe"].std(ddof=0))
    mean_up_dd = float(fold_df["upgraded_max_drawdown"].mean())
    std_up_dd = float(fold_df["upgraded_max_drawdown"].std(ddof=0))
    mean_up_to = float(fold_df["upgraded_avg_turnover"].mean())
    std_up_to = float(fold_df["upgraded_avg_turnover"].std(ddof=0))

    objective = (
        mean_up_sh
        - 0.60 * abs(mean_up_dd)
        - 0.30 * mean_up_to
        - 0.55 * std_up_sh
        - 0.20 * std_up_dd
        - 0.10 * std_up_to
        + 0.35 * beat_baseline_rate
    )

    summary = {
        "num_folds": int(len(fold_df)),
        "upgraded": {
            "sharpe_mean": mean_up_sh,
            "sharpe_std": std_up_sh,
            "max_drawdown_mean": mean_up_dd,
            "max_drawdown_std": std_up_dd,
            "avg_turnover_mean": mean_up_to,
            "avg_turnover_std": std_up_to,
        },
        "baseline": {
            "sharpe_mean": float(fold_df["baseline_sharpe"].mean()),
            "max_drawdown_mean": float(fold_df["baseline_max_drawdown"].mean()),
            "avg_turnover_mean": float(fold_df["baseline_avg_turnover"].mean()),
        },
        "blended": {
            "sharpe_mean": float(fold_df["blended_sharpe"].mean()),
            "max_drawdown_mean": float(fold_df["blended_max_drawdown"].mean()),
            "avg_turnover_mean": float(fold_df["blended_avg_turnover"].mean()),
        },
        "beat_baseline_rate": beat_baseline_rate,
        "beat_blend_rate": beat_blend_rate,
        "objective": float(objective),
        "feature_count": int(len(feature_cols)),
        "dropped_features": dropped_features,
        "cache_hits": prepared.get("cache_hits", {}),
        "prep_timings_seconds": prepared.get("timings", {}),
    }

    if split_timing_rows:
        timing_df = pd.DataFrame(split_timing_rows).fillna(0.0)
        summary["fold_stage_mean_seconds"] = {col: float(timing_df[col].mean()) for col in timing_df.columns}
        summary["fold_stage_total_seconds"] = {col: float(timing_df[col].sum()) for col in timing_df.columns}

    util_end = _runtime_utilization_snapshot() if config.resource_monitoring else {}
    summary["utilization_start"] = util_start
    summary["utilization_end"] = util_end
    summary["bound_status"] = _bound_status_from_runtime(summary.get("fold_stage_total_seconds", {}), util_end)

    return {"summary": summary, "fold_metrics": fold_df.to_dict(orient="records")}


def _sample_from_bounds(rng: np.random.Generator, bounds: Dict[str, Tuple[float, float]], int_keys: set[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, (low, high) in bounds.items():
        val = float(rng.uniform(low, high))
        if key in int_keys:
            out[key] = int(round(val))
        else:
            out[key] = val
    return out


def _adaptive_parameter_candidates(
    rng: np.random.Generator,
    bounds: Dict[str, Tuple[float, float]],
    int_keys: set[str],
    trials: List[Dict[str, Any]],
    n_trials: int,
) -> List[Dict[str, float]]:
    n_init = min(n_trials, max(2, n_trials // 3))
    candidates: List[Dict[str, float]] = []

    for _ in range(n_init):
        candidates.append(_sample_from_bounds(rng, bounds, int_keys))

    remaining = max(0, n_trials - n_init)
    if remaining <= 0:
        return candidates

    if not trials:
        for _ in range(remaining):
            candidates.append(_sample_from_bounds(rng, bounds, int_keys))
        return candidates

    ranked = sorted(trials, key=lambda item: float(item.get("objective", -1e9)), reverse=True)
    elites = ranked[: max(2, len(ranked) // 4)]
    for _ in range(remaining):
        if rng.random() < 0.20:
            candidates.append(_sample_from_bounds(rng, bounds, int_keys))
            continue

        elite = elites[int(rng.integers(0, len(elites)))]
        point = {}
        for key, (low, high) in bounds.items():
            center = float(elite[key])
            spread = (high - low) * 0.18
            val = float(rng.normal(center, spread))
            val = max(low, min(high, val))
            if key in int_keys:
                point[key] = int(round(val))
            else:
                point[key] = val
        candidates.append(point)
    return candidates


def _sensitivity_report(trials_df: pd.DataFrame, baseline_sharpe_reference: float) -> Dict[str, Any]:
    param_cols = [
        "min_confidence",
        "max_daily_turnover",
        "signal_cooldown_steps",
        "blend_lookback",
        "max_position_weight",
        "vol_scale_target",
        "vol_scale_floor",
        "vol_scale_ceiling",
    ]
    corr: Dict[str, float] = {}
    for col in param_cols:
        if col not in trials_df.columns:
            continue
        corr[col] = float(trials_df[col].corr(trials_df["objective"], method="spearman")) if len(trials_df) > 1 else 0.0

    robust_pool = trials_df[
        (trials_df["upgraded_sharpe_mean"] > baseline_sharpe_reference)
        & (trials_df["upgraded_sharpe_std"] <= trials_df["upgraded_sharpe_std"].quantile(0.65))
        & (trials_df["objective"] >= trials_df["objective"].quantile(0.65))
    ].copy()
    if robust_pool.empty:
        robust_pool = trials_df.nlargest(max(3, min(10, len(trials_df))), "objective").copy()

    robust_ranges: Dict[str, Dict[str, float]] = {}
    for col in param_cols:
        if col not in robust_pool.columns:
            continue
        robust_ranges[col] = {
            "min": float(robust_pool[col].min()),
            "p10": float(robust_pool[col].quantile(0.10)),
            "median": float(robust_pool[col].median()),
            "p90": float(robust_pool[col].quantile(0.90)),
            "max": float(robust_pool[col].max()),
        }

    return {
        "spearman_vs_objective": corr,
        "robust_pool_size": int(len(robust_pool)),
        "robust_ranges": robust_ranges,
    }


def optimize_upgraded_strategy_hyperparameters(
    base_config: AlphaConfig,
    n_trials: int = 24,
    random_seed: int = 42,
    parallel_workers: int | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    active_profile = _active_tabular_profile(base_config)
    workers = max(1, int(parallel_workers or 1))
    workers = min(workers, max(1, n_trials))
    parallel_mode = "parallel" if workers > 1 else "sequential"
    resource_cfg = _configure_runtime_resources(base_config, workers=workers)
    cpu_total = int(resource_cfg["detected"].get("cpu_total_cores", max(1, int(os.cpu_count() or 1))))

    if os.name == "nt" and workers > 1 and n_trials <= max(4, workers * 2):
        _log(
            f"windows_parallel_overhead_warning workers={workers} trials={n_trials}; "
            "for small trial counts, run sequentially (--parallel-workers 1)"
        )

    bounds: Dict[str, Tuple[float, float]] = {
        "min_confidence": (0.45, 0.72),
        "max_daily_turnover": (0.12, 0.45),
        "signal_cooldown_steps": (1, 6),
        "blend_lookback": (8, 64),
        "max_position_weight": (0.15, 0.45),
        "vol_scale_target": (0.015, 0.060),
        "vol_scale_floor": (0.10, 0.40),
        "vol_scale_ceiling": (0.70, 1.40),
    }
    int_keys = {"signal_cooldown_steps", "blend_lookback"}

    t0 = time.perf_counter()
    baseline_eval = evaluate_walk_forward_robustness(base_config)
    baseline_sharpe_ref = float(baseline_eval["summary"]["baseline"]["sharpe_mean"])
    _log(f"hyperopt baseline evaluated in {time.perf_counter() - t0:.2f}s")
    _log(
        "startup_banner "
        f"tabular_profile={active_profile} "
        f"debug_mode={base_config.debug_mode} "
        f"very_light_debug_mode={base_config.very_light_debug_mode} "
        f"cache_enabled={base_config.enable_cache} "
        f"gpu_available={torch.cuda.is_available()} "
        f"parallel_workers={workers} "
        f"parallel_mode={parallel_mode} "
        f"cpu_threads_per_worker={resource_cfg['effective']['effective_cpu_threads_per_worker']} "
        f"ram_total_gb={resource_cfg['detected'].get('ram_total_gb', 0.0):.2f}"
    )
    _log(f"resource_detected {json.dumps(resource_cfg.get('detected', {}), default=str)}")
    _log(f"resource_effective {json.dumps(resource_cfg.get('effective', {}), default=str)}")

    trials: List[Dict[str, Any]] = []
    candidates = _adaptive_parameter_candidates(rng, bounds, int_keys, trials=[], n_trials=n_trials)

    def _build_cfg(params: Dict[str, float], trial_idx: int) -> AlphaConfig:
        thread_budget = int(base_config.cpu_thread_count) if int(base_config.cpu_thread_count) > 0 else cpu_total
        if base_config.sklearn_n_jobs == -1:
            n_jobs = max(1, thread_budget // max(1, workers))
        else:
            n_jobs = max(1, min(int(base_config.sklearn_n_jobs), thread_budget))
        return AlphaConfig(
            asset_paths=base_config.asset_paths,
            benchmark_asset=base_config.benchmark_asset,
            horizon=base_config.horizon,
            buy_threshold=base_config.buy_threshold,
            sell_threshold=base_config.sell_threshold,
            min_stable_reserve=base_config.min_stable_reserve,
            fee_bps=base_config.fee_bps,
            slippage_bps=base_config.slippage_bps,
            latency_steps=base_config.latency_steps,
            walk_forward_splits=base_config.walk_forward_splits,
            sequence_length=base_config.sequence_length,
            gru_epochs=base_config.gru_epochs,
            min_confidence=float(params["min_confidence"]),
            min_edge=base_config.min_edge,
            signal_cooldown_steps=int(params["signal_cooldown_steps"]),
            max_daily_turnover=float(params["max_daily_turnover"]),
            price_impact_coef_bps=base_config.price_impact_coef_bps,
            max_position_weight=float(params["max_position_weight"]),
            vol_scale_target=float(params["vol_scale_target"]),
            vol_scale_floor=float(params["vol_scale_floor"]),
            vol_scale_ceiling=float(params["vol_scale_ceiling"]),
            feature_corr_threshold=base_config.feature_corr_threshold,
            blend_lookback=int(params["blend_lookback"]),
            tabular_profile=base_config.tabular_profile,
            sklearn_n_jobs=n_jobs,
            cpu_thread_count=base_config.cpu_thread_count,
            gru_batch_size=base_config.gru_batch_size,
            gru_auto_batch_size=base_config.gru_auto_batch_size,
            max_cache_entries=base_config.max_cache_entries,
            resource_monitoring=base_config.resource_monitoring,
            enable_cache=base_config.enable_cache,
            enable_timing_logs=base_config.enable_timing_logs,
            debug_mode=base_config.debug_mode,
            very_light_debug_mode=base_config.very_light_debug_mode,
            random_seed=base_config.random_seed + trial_idx,
        )

    def _run_one(trial_idx: int, params: Dict[str, float]) -> Dict[str, Any]:
        trial_start = time.perf_counter()
        worker_name = threading.current_thread().name
        util_start = _runtime_utilization_snapshot() if base_config.resource_monitoring else {}
        _log(f"trial_start trial={trial_idx}/{n_trials} worker={worker_name} params={json.dumps(params, default=str)}")
        cfg = _build_cfg(params, trial_idx=trial_idx)
        wf = evaluate_walk_forward_robustness(cfg)
        summary = wf["summary"]

        prep = summary.get("prep_timings_seconds", {})
        fold_total = summary.get("fold_stage_total_seconds", {})
        model_training_seconds = float(fold_total.get("split_preprocess", 0.0) + fold_total.get("split_tabular_train_backtest", 0.0) + fold_total.get("split_gru_train", 0.0))
        backtest_seconds = float(fold_total.get("split_backtest", 0.0))
        _log(
            "trial_stage "
            f"trial={trial_idx} "
            f"load_cache={float(prep.get('data_loading', 0.0)):.2f}s "
            f"feature_engineering={float(prep.get('feature_engineering', 0.0)):.2f}s "
            f"split_generation={float(prep.get('walk_forward_split_build', 0.0)):.2f}s "
            f"model_training={model_training_seconds:.2f}s "
            f"backtest={backtest_seconds:.2f}s "
            "artifact_write=0.00s(skipped)"
        )

        trial_elapsed = time.perf_counter() - trial_start
        util_end = _runtime_utilization_snapshot() if base_config.resource_monitoring else {}
        if util_end:
            _log(f"trial_utilization trial={trial_idx} start={json.dumps(util_start, default=str)} end={json.dumps(util_end, default=str)}")
        _log(f"trial_finish trial={trial_idx}/{n_trials} worker={worker_name} elapsed={trial_elapsed:.2f}s objective={float(summary['objective']):.6f}")
        return {
            "trial": int(trial_idx),
            **params,
            "objective": float(summary["objective"]),
            "upgraded_sharpe_mean": float(summary["upgraded"]["sharpe_mean"]),
            "upgraded_sharpe_std": float(summary["upgraded"]["sharpe_std"]),
            "upgraded_max_drawdown_mean": float(summary["upgraded"]["max_drawdown_mean"]),
            "upgraded_avg_turnover_mean": float(summary["upgraded"]["avg_turnover_mean"]),
            "baseline_sharpe_mean": float(summary["baseline"]["sharpe_mean"]),
            "blended_sharpe_mean": float(summary["blended"]["sharpe_mean"]),
            "beat_baseline_rate": float(summary["beat_baseline_rate"]),
            "beat_blend_rate": float(summary["beat_blend_rate"]),
            "fold_metrics": wf["fold_metrics"],
            "trial_elapsed_seconds": float(trial_elapsed),
            "tabular_profile": _active_tabular_profile(cfg),
            "stage_timings": summary.get("fold_stage_mean_seconds", {}),
            "prep_timings": summary.get("prep_timings_seconds", {}),
            "cache_hits": summary.get("cache_hits", {}),
            "utilization_start": util_start,
            "utilization_end": util_end,
        }

    total_start = time.perf_counter()
    first_eta_seconds: float | None = None

    param_rows: List[Tuple[int, Dict[str, float]]] = []
    for idx, point in enumerate(candidates, start=1):
        params = dict(point)
        if params["vol_scale_floor"] >= params["vol_scale_ceiling"]:
            mid = 0.5 * (params["vol_scale_floor"] + params["vol_scale_ceiling"])
            params["vol_scale_floor"] = max(bounds["vol_scale_floor"][0], min(mid - 0.05, bounds["vol_scale_floor"][1]))
            params["vol_scale_ceiling"] = min(bounds["vol_scale_ceiling"][1], max(mid + 0.05, bounds["vol_scale_ceiling"][0]))
        param_rows.append((idx, params))

    if workers <= 1:
        _log("parallel_workers_confirmed workers=1 mode=sequential")
        for idx, params in param_rows:
            trial = _run_one(idx, params)
            trials.append(trial)
            done = len(trials)
            elapsed = time.perf_counter() - total_start
            avg_trial = elapsed / max(1, done)
            if done == 1:
                first_eta_seconds = avg_trial * max(0, n_trials - 1)
            remaining = max(0.0, avg_trial * (n_trials - done))
            _log(f"hyperopt trial={done}/{n_trials} elapsed={trial['trial_elapsed_seconds']:.2f}s eta_remaining={remaining:.1f}s")
    else:
        _log(f"parallel_workers_confirmed workers={workers} mode=threadpool")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_one, idx, params) for idx, params in param_rows]
            for fut in as_completed(futures):
                trial = fut.result()
                trials.append(trial)
                done = len(trials)
                elapsed = time.perf_counter() - total_start
                avg_trial = elapsed / max(1, done)
                if done == 1:
                    first_eta_seconds = max(0.0, (n_trials - 1) * avg_trial / workers)
                remaining = max(0.0, avg_trial * (n_trials - done) / workers)
                _log(f"hyperopt trial={done}/{n_trials} elapsed={trial['trial_elapsed_seconds']:.2f}s eta_remaining={remaining:.1f}s workers={workers}")

    trials_df = pd.DataFrame([{k: v for k, v in row.items() if k != "fold_metrics"} for row in trials])
    if trials_df.empty:
        raise RuntimeError("Hyperparameter optimization produced no valid trial.")

    best_idx = int(trials_df["objective"].idxmax())
    best_row = trials_df.iloc[best_idx].to_dict()
    best_trial_num = int(best_row["trial"])
    best_fold = next(item["fold_metrics"] for item in trials if int(item["trial"]) == best_trial_num)

    sens = _sensitivity_report(trials_df, baseline_sharpe_reference=baseline_sharpe_ref)

    stage_df = pd.DataFrame([row.get("stage_timings", {}) for row in trials]).fillna(0.0)
    prep_df = pd.DataFrame([row.get("prep_timings", {}) for row in trials]).fillna(0.0)
    avg_stage = {col: float(stage_df[col].mean()) for col in stage_df.columns} if not stage_df.empty else {}
    avg_prep = {col: float(prep_df[col].mean()) for col in prep_df.columns} if not prep_df.empty else {}

    avg_trial_seconds = float(np.mean([float(row.get("trial_elapsed_seconds", 0.0)) for row in trials])) if trials else 0.0
    gpu_stage_seconds = float(avg_stage.get("split_gru_train", 0.0))
    cpu_stage_seconds = float(
        avg_stage.get("split_preprocess", 0.0)
        + avg_stage.get("split_tabular_train_backtest", 0.0)
        + avg_stage.get("split_backtest", 0.0)
    )
    io_stage_seconds = float(avg_prep.get("data_loading", 0.0))
    total_stage_seconds = max(1e-9, gpu_stage_seconds + cpu_stage_seconds + io_stage_seconds)

    assumed_gpu_gain = 2.0
    gpu_speedup = 1.0 / ((1.0 - (gpu_stage_seconds / total_stage_seconds)) + ((gpu_stage_seconds / total_stage_seconds) / assumed_gpu_gain))
    cache_saved = max(0.0, float(avg_prep.get("data_loading", 0.0) + avg_prep.get("feature_engineering", 0.0) + avg_prep.get("walk_forward_split_build", 0.0)))
    cache_speedup = 1.0 + min(3.0, cache_saved / max(1e-6, avg_trial_seconds))
    parallel_speedup = float(min(workers, 0.85 * workers + 0.15))

    return {
        "baseline_reference": baseline_eval["summary"],
        "search": {
            "n_trials": int(len(trials)),
            "method": "adaptive_bayesian_like",
            "objective": "maximize upgraded_sharpe_mean with penalties for drawdown/turnover/instability",
            "parallel_workers": workers,
            "parallel_mode": parallel_mode,
            "tabular_profile": active_profile,
            "parallel_fallback_recommendation": (
                "Use --parallel-workers 1 on Windows for small trial counts or if logs stall between trials."
                if os.name == "nt"
                else "Not required"
            ),
            "avg_trial_seconds": avg_trial_seconds,
            "eta_after_first_trial_seconds": float(first_eta_seconds or 0.0),
            "resource_config": resource_cfg,
        },
        "best_parameters": {
            "min_confidence": float(best_row["min_confidence"]),
            "max_daily_turnover": float(best_row["max_daily_turnover"]),
            "cooldown_period": int(best_row["signal_cooldown_steps"]),
            "blend_lookback_window": int(best_row["blend_lookback"]),
            "position_cap": float(best_row["max_position_weight"]),
            "volatility_scaling": {
                "vol_scale_target": float(best_row["vol_scale_target"]),
                "vol_scale_floor": float(best_row["vol_scale_floor"]),
                "vol_scale_ceiling": float(best_row["vol_scale_ceiling"]),
            },
        },
        "best_summary": {
            "objective": float(best_row["objective"]),
            "upgraded_sharpe_mean": float(best_row["upgraded_sharpe_mean"]),
            "upgraded_sharpe_std": float(best_row["upgraded_sharpe_std"]),
            "upgraded_max_drawdown_mean": float(best_row["upgraded_max_drawdown_mean"]),
            "upgraded_avg_turnover_mean": float(best_row["upgraded_avg_turnover_mean"]),
            "baseline_sharpe_mean": float(best_row["baseline_sharpe_mean"]),
            "blended_sharpe_mean": float(best_row["blended_sharpe_mean"]),
            "beat_baseline_rate": float(best_row["beat_baseline_rate"]),
            "beat_blend_rate": float(best_row["beat_blend_rate"]),
        },
        "best_fold_metrics": best_fold,
        "sensitivity": sens,
        "runtime_profile": {
            "gpu_bound": ["GRU sequence training/inference"],
            "cpu_bound": ["sklearn model training", "RL allocation", "portfolio simulation/backtesting"],
            "io_bound": ["CSV loading", "artifact JSON/CSV writing"],
            "windows_parallel_note": (
                "Uses ThreadPool for parallel trials to avoid ProcessPool spawn/pickle overhead on Windows; "
                "for small trial counts prefer sequential (--parallel-workers 1)."
            ),
            "windows_high_priority_recommendation": (
                "Start-Process -FilePath c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe "
                "-ArgumentList 'run_prod_alpha_hyperopt.py --trials 20 --run-production' -Priority High"
                if os.name == "nt"
                else "not_applicable"
            ),
            "wsl2_vs_native_note": (
                "WSL2/Linux container often provides lower process/thread scheduling overhead for heavy Python/sklearn workloads; "
                "native Windows remains fine for single-worker runs with CUDA-enabled PyTorch."
            ),
            "average_stage_seconds": {
                "prep": avg_prep,
                "split": avg_stage,
            },
            "estimated_speedup": {
                "gpu": float(gpu_speedup),
                "caching": float(cache_speedup),
                "parallelization": float(parallel_speedup),
            },
            "resource_detected": resource_cfg["detected"],
            "resource_effective": resource_cfg["effective"],
            "tabular_profile": active_profile,
            "bound_status": _bound_status_from_runtime(avg_stage, _runtime_utilization_snapshot()),
        },
        "trials": trials_df.sort_values("objective", ascending=False).to_dict(orient="records"),
    }


def run_production_pipeline(config: AlphaConfig) -> Dict[str, Any]:
    set_seeds(config.random_seed)
    pipeline_timing: Dict[str, float] = {}
    resource_cfg = _configure_runtime_resources(config, workers=1)
    util_start = _runtime_utilization_snapshot() if config.resource_monitoring else {}

    _log(
        "startup_banner "
        f"tabular_profile={_active_tabular_profile(config)} "
        f"debug_mode={config.debug_mode} "
        f"very_light_debug_mode={config.very_light_debug_mode} "
        f"cache_enabled={config.enable_cache} "
        f"gpu_available={torch.cuda.is_available()} "
        f"cpu_threads={resource_cfg['effective']['effective_cpu_threads_per_worker']} "
        f"gru_batch_size={config.gru_batch_size} "
        f"gru_auto_batch_size={config.gru_auto_batch_size} "
        f"max_cache_entries={config.max_cache_entries} "
        "parallel_workers=1 "
        "parallel_mode=sequential"
    )
    _log(f"resource_detected {json.dumps(resource_cfg.get('detected', {}), default=str)}")
    _log(f"resource_effective {json.dumps(resource_cfg.get('effective', {}), default=str)}")

    with _stage_timer("prepare_inputs", pipeline_timing, enabled=config.enable_timing_logs):
        prepared = get_prepared_pipeline_inputs(config, enable_cache=config.enable_cache, force_refresh=False)
        feat = prepared["feat"]
        feature_cols = prepared["feature_cols"]
        dropped_features = prepared["dropped_features"]
        splits = prepared["splits"]

    if not splits:
        raise RuntimeError("Not enough data for walk-forward validation.")

    model_scores: Dict[str, List[float]] = {"gru": []}

    with _stage_timer("walk_forward_model_selection", pipeline_timing, enabled=config.enable_timing_logs):
        for train_idx, val_idx in splits:
            train_df = feat.iloc[train_idx]
            val_df = feat.iloc[val_idx]

            imp = SimpleImputer(strategy="median")
            scaler = RobustScaler()
            x_train = imp.fit_transform(train_df[feature_cols])
            x_val = imp.transform(val_df[feature_cols])
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)
            y_train = train_df["target_class"].values.astype(int)
            y_val = val_df["target_class"].values.astype(int)

            tabular_models = _build_tabular_models(config, seed=config.random_seed, phase="cv")
            for name, _ in tabular_models:
                model_scores.setdefault(name, [])

            for name, model in tabular_models:
                model.fit(x_train, y_train)
                probs = model.predict_proba(x_val)
                probs = apply_regime_overlay(val_df, probs)
                pred, _ = build_execution_filtered_signal(val_df, probs, config)
                val_df_local = val_df.copy()
                val_df_local["signal"] = pred
                val_df_local["signal_confidence"] = np.max(probs, axis=1)
                val_df_local["signal_edge"] = np.maximum(probs[:, 1], probs[:, 2]) - probs[:, 0]
                back = simulate_portfolio(val_df_local, signal_col="signal", config=config)
                model_scores[name].append(float(_strategy_utility(back["metrics"])))

            x_train_seq, y_train_seq = _sequence_data(feat, feature_cols, train_idx, config.sequence_length)
            x_val_seq, _ = _sequence_data(feat, feature_cols, val_idx, config.sequence_length)
            _, gru_probs = train_gru_classifier(
                x_train_seq=x_train_seq,
                y_train=y_train_seq,
                x_val_seq=x_val_seq,
                input_dim=len(feature_cols),
                epochs=config.gru_epochs,
                seed=config.random_seed,
                gru_batch_size=config.gru_batch_size,
                gru_auto_batch_size=config.gru_auto_batch_size,
            )
            if len(gru_probs) > 0:
                seq_df = val_df.iloc[-len(gru_probs):].copy()
                gru_probs = apply_regime_overlay(seq_df, gru_probs)
                gru_pred, _ = build_execution_filtered_signal(seq_df, gru_probs, config)
                val_seq_df = val_df.iloc[-len(gru_pred):].copy()
                val_seq_df["signal"] = gru_pred
                val_seq_df["signal_confidence"] = np.max(gru_probs, axis=1)
                val_seq_df["signal_edge"] = np.maximum(gru_probs[:, 1], gru_probs[:, 2]) - gru_probs[:, 0]
                back_seq = simulate_portfolio(val_seq_df, signal_col="signal", config=config)
                model_scores["gru"].append(float(_strategy_utility(back_seq["metrics"])))

    avg_scores = {k: (float(np.mean(v)) if v else 0.0) for k, v in model_scores.items()}

    # final train/test holdout
    timestamps = sorted(feat["timestamp"].unique())
    cutoff = timestamps[int(len(timestamps) * 0.8)]
    train_df = feat[feat["timestamp"] < cutoff].copy()
    test_df = feat[feat["timestamp"] >= cutoff].copy()

    train_times = sorted(train_df["timestamp"].unique())
    calib_cut = train_times[int(len(train_times) * 0.85)] if len(train_times) > 20 else train_times[-1]
    model_train_df = train_df[train_df["timestamp"] < calib_cut].copy()
    calib_df = train_df[train_df["timestamp"] >= calib_cut].copy()
    if len(model_train_df) < 50 or len(calib_df) < 25:
        model_train_df = train_df.copy()
        calib_df = test_df.copy()

    with _stage_timer("holdout_training", pipeline_timing, enabled=config.enable_timing_logs):
        imp = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        x_train = scaler.fit_transform(imp.fit_transform(model_train_df[feature_cols]))
        x_calib = scaler.transform(imp.transform(calib_df[feature_cols]))
        x_test = scaler.transform(imp.transform(test_df[feature_cols]))
        y_train = model_train_df["target_class"].values.astype(int)
        y_calib = calib_df["target_class"].values.astype(int)
        y_test = test_df["target_class"].values.astype(int)

        fitted_models: Dict[str, Any] = {}
        tabular_models_final = _build_tabular_models(config, seed=config.random_seed, phase="final")
        for name, model in tabular_models_final:
            fitted_models[name] = model.fit(x_train, y_train)

    model_names = list(fitted_models.keys())
    p_calib = {name: fitted_models[name].predict_proba(x_calib) for name in model_names}
    p_test = {name: fitted_models[name].predict_proba(x_test) for name in model_names}

    def _model_utility(calib_probs: np.ndarray) -> float:
        adj_probs = apply_regime_overlay(calib_df, calib_probs)
        sig, _ = build_execution_filtered_signal(calib_df, adj_probs, config)
        local_df = calib_df.copy()
        local_df["signal"] = sig
        local_df["signal_confidence"] = np.max(adj_probs, axis=1)
        local_df["signal_edge"] = np.maximum(adj_probs[:, 1], adj_probs[:, 2]) - adj_probs[:, 0]
        bt = simulate_portfolio(local_df, signal_col="signal", config=config)
        return _strategy_utility(bt["metrics"])

    calib_scores = np.array([_model_utility(p_calib[name]) for name in model_names], dtype=float)
    centered = calib_scores - np.max(calib_scores)
    weights = np.exp(centered)
    weights = np.clip(weights, 1e-4, None)
    weights = weights / weights.sum()

    ensemble_probs = np.zeros_like(p_test[model_names[0]])
    for idx, name in enumerate(model_names):
        ensemble_probs += weights[idx] * p_test[name]
    ensemble_probs = apply_regime_overlay(test_df, ensemble_probs)

    x_train_seq, y_train_seq = _sequence_data(feat, feature_cols, model_train_df.index.values, config.sequence_length)
    x_test_seq, _ = _sequence_data(feat, feature_cols, test_df.index.values, config.sequence_length)
    holdout_gru_epochs = 1 if config.very_light_debug_mode else config.gru_epochs
    _, gru_probs = train_gru_classifier(
        x_train_seq=x_train_seq,
        y_train=y_train_seq,
        x_val_seq=x_test_seq,
        input_dim=len(feature_cols),
        epochs=holdout_gru_epochs,
        seed=config.random_seed,
        gru_batch_size=config.gru_batch_size,
        gru_auto_batch_size=config.gru_auto_batch_size,
    )

    if len(gru_probs) > 0:
        trim = min(len(ensemble_probs), len(gru_probs))
        test_df = test_df.iloc[-trim:].copy()
        y_test = y_test[-trim:]
        gru_probs = apply_regime_overlay(test_df, gru_probs[-trim:])
        ensemble_probs = ensemble_probs[-trim:] * 0.80 + gru_probs * 0.20

    temp, ensemble_probs = calibrate_temperature(ensemble_probs, y_test)

    raw_pred = np.argmax(ensemble_probs, axis=1)
    confidence = np.max(ensemble_probs, axis=1)
    upgraded_pred, signal_diag = build_execution_filtered_signal(test_df, ensemble_probs, config)
    edge = np.maximum(ensemble_probs[:, 1], ensemble_probs[:, 2]) - ensemble_probs[:, 0]
    baseline_pred = _baseline_signal(test_df)

    upgraded_df = test_df.copy()
    upgraded_df["upgraded_signal"] = upgraded_pred
    upgraded_df["signal_confidence"] = confidence
    upgraded_df["signal_edge"] = edge
    baseline_df = test_df.copy()
    baseline_df["baseline_signal"] = baseline_pred.values

    with _stage_timer("backtesting", pipeline_timing, enabled=config.enable_timing_logs):
        allocator = SimpleRLAllocator(
            assets=sorted(upgraded_df["symbol"].unique()),
            seed=config.random_seed,
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
        )
        allocator.fit(upgraded_df, signal_col="upgraded_signal", epochs=5)
        alloc_schedule = allocator.allocation_schedule(upgraded_df)

        upgraded_backtest = simulate_portfolio(
            upgraded_df,
            signal_col="upgraded_signal",
            config=config,
            allocation_scores=alloc_schedule,
        )
        baseline_backtest = simulate_portfolio(
            baseline_df,
            signal_col="baseline_signal",
            config=config,
            allocation_scores=None,
        )

        blended_backtest = _compute_blended_backtest(upgraded_backtest, baseline_backtest, lookback=config.blend_lookback)

    deployed_strategy = "upgraded"
    deployed_backtest = upgraded_backtest
    if _strategy_utility(blended_backtest["metrics"]) > _strategy_utility(deployed_backtest["metrics"]):
        deployed_strategy = "meta_blend"
        deployed_backtest = blended_backtest
    if _strategy_utility(baseline_backtest["metrics"]) > _strategy_utility(deployed_backtest["metrics"]):
        deployed_strategy = "baseline_fallback"
        deployed_backtest = baseline_backtest

    prediction_metrics = {
        "upgraded_accuracy": float(accuracy_score(y_test, upgraded_pred)),
        "upgraded_macro_f1": float(f1_score(y_test, upgraded_pred, average="macro")),
        "baseline_accuracy": float(accuracy_score(y_test, baseline_pred.values[: len(y_test)])),
        "baseline_macro_f1": float(f1_score(y_test, baseline_pred.values[: len(y_test)], average="macro")),
    }

    monitor_report = {
        "feature_drift_flags": detect_feature_drift(train_df[feature_cols], test_df[feature_cols]),
        "performance_decay": detect_performance_decay(
            baseline_sharpe=float(baseline_backtest["metrics"]["sharpe"]),
            upgraded_sharpe=float(upgraded_backtest["metrics"]["sharpe"]),
        ),
        "signal_filtering": signal_diag,
    }

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pred_log = ARTIFACT_DIR / f"predictions_{now}.csv"
    trade_log = ARTIFACT_DIR / f"trades_{now}.json"
    metrics_file = ARTIFACT_DIR / f"comparison_{now}.json"

    with _stage_timer("artifact_writing", pipeline_timing, enabled=config.enable_timing_logs):
        if config.very_light_debug_mode:
            _log("artifact_write=light mode (skipping prediction/trade logs)")
            pred_log = ARTIFACT_DIR / "skipped_in_very_light_debug.csv"
            trade_log = ARTIFACT_DIR / "skipped_in_very_light_debug.json"
        else:
            pd.DataFrame(
                {
                    "timestamp": upgraded_df["timestamp"].astype(str).values,
                    "symbol": upgraded_df["symbol"].values,
                    "target_class": y_test,
                    "baseline_signal": baseline_pred.values[: len(y_test)],
                    "upgraded_signal": upgraded_pred,
                }
            ).to_csv(pred_log, index=False)

            with open(trade_log, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "baseline_trades": baseline_backtest["trades"],
                        "upgraded_trades": upgraded_backtest["trades"],
                        "blended_trades": blended_backtest["trades"],
                    },
                    f,
                    indent=2,
                )

    result = {
        "config": {
            "assets": list(config.asset_paths.keys()),
            "horizon": config.horizon,
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "latency_steps": config.latency_steps,
            "min_confidence": config.min_confidence,
            "max_daily_turnover": config.max_daily_turnover,
            "signal_cooldown_steps": config.signal_cooldown_steps,
            "blend_lookback": config.blend_lookback,
            "max_position_weight": config.max_position_weight,
            "vol_scale_target": config.vol_scale_target,
            "vol_scale_floor": config.vol_scale_floor,
            "vol_scale_ceiling": config.vol_scale_ceiling,
            "sklearn_n_jobs": config.sklearn_n_jobs,
            "cpu_thread_count": config.cpu_thread_count,
            "gru_batch_size": config.gru_batch_size,
            "gru_auto_batch_size": config.gru_auto_batch_size,
            "max_cache_entries": config.max_cache_entries,
            "resource_monitoring": config.resource_monitoring,
            "tabular_profile": _active_tabular_profile(config),
            "debug_mode": config.debug_mode,
            "very_light_debug_mode": config.very_light_debug_mode,
            "torch_device": str(_resolve_torch_device(log_device=True)),
        },
        "walk_forward_model_selection": avg_scores,
        "ensemble_weights": {name: float(weights[idx]) for idx, name in enumerate(model_names)},
        "calibration": {"temperature": float(temp)},
        "feature_pruning": {"kept": len(feature_cols), "dropped": dropped_features[:40]},
        "prediction_metrics": prediction_metrics,
        "baseline_backtest": baseline_backtest["metrics"],
        "upgraded_backtest": upgraded_backtest["metrics"],
        "blended_backtest": blended_backtest["metrics"],
        "deployed_strategy": deployed_strategy,
        "deployed_backtest": deployed_backtest["metrics"],
        "diagnostics": {
            "baseline": baseline_backtest.get("diagnostics", {}),
            "upgraded": upgraded_backtest.get("diagnostics", {}),
        },
        "monitoring": monitor_report,
        "runtime": {
            "pipeline_stage_seconds": pipeline_timing,
            "prep_cache_hits": prepared.get("cache_hits", {}),
            "prep_stage_seconds": prepared.get("timings", {}),
            "resource_detected": resource_cfg.get("detected", {}),
            "resource_effective": resource_cfg.get("effective", {}),
            "tabular_profile": _active_tabular_profile(config),
            "utilization_start": util_start,
            "utilization_end": (_runtime_utilization_snapshot() if config.resource_monitoring else {}),
            "bound_status": _bound_status_from_runtime(pipeline_timing, (_runtime_utilization_snapshot() if config.resource_monitoring else {})),
            "bound_classification": {
                "gpu_bound": ["GRU sequence training/inference"],
                "cpu_bound": ["sklearn model training", "RL allocation", "portfolio simulation"],
                "io_bound": ["CSV/context loading", "artifact writing"],
            },
            "windows_high_priority_recommendation": (
                "Start-Process -FilePath c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe "
                "-ArgumentList 'run_prod_alpha_hyperopt.py --trials 20 --run-production' -Priority High"
                if os.name == "nt"
                else "not_applicable"
            ),
            "wsl2_vs_native_note": (
                "WSL2/Linux container may improve scheduler and file I/O behavior for long sklearn-heavy runs; "
                "native Windows is typically best kept to sequential hyperopt workers unless trial count is large."
            ),
        },
        "artifacts": {
            "predictions": str(pred_log),
            "trades": str(trade_log),
        },
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["artifacts"]["comparison"] = str(metrics_file)
    return result


def detect_feature_drift(train_x: pd.DataFrame, test_x: pd.DataFrame) -> Dict[str, Any]:
    flags: Dict[str, Any] = {"drifted_columns": [], "total_columns": int(train_x.shape[1])}
    if train_x.empty or test_x.empty:
        return flags
    for col in train_x.columns:
        train_med = float(np.nanmedian(train_x[col].values))
        test_med = float(np.nanmedian(test_x[col].values))
        train_std = float(np.nanstd(train_x[col].values))
        if train_std <= 1e-12:
            continue
        z_shift = abs(test_med - train_med) / train_std
        if z_shift > 1.5:
            flags["drifted_columns"].append(col)
    flags["drift_ratio"] = float(len(flags["drifted_columns"]) / max(1, train_x.shape[1]))
    return flags


def detect_performance_decay(baseline_sharpe: float, upgraded_sharpe: float) -> Dict[str, Any]:
    delta = upgraded_sharpe - baseline_sharpe
    return {
        "baseline_sharpe": float(baseline_sharpe),
        "upgraded_sharpe": float(upgraded_sharpe),
        "delta_sharpe": float(delta),
        "status": "improved" if delta >= 0 else "degraded",
    }


def periodic_retrain(config: AlphaConfig) -> Dict[str, Any]:
    return run_production_pipeline(config)


__all__ = [
    "AlphaConfig",
    "run_production_pipeline",
    "optimize_upgraded_strategy_hyperparameters",
    "periodic_retrain",
    "detect_feature_drift",
    "detect_performance_decay",
]
