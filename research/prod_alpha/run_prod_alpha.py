from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from pipeline import AlphaConfig, run_production_pipeline


def _phase_banner(title: str, elapsed_seconds: float | None = None) -> None:
    banner = "=" * 18
    if elapsed_seconds is None:
        print(f"\n{banner} {title} {banner}", flush=True)
        return
    print(f"\n{banner} {title} | elapsed={elapsed_seconds:.2f}s {banner}", flush=True)


def _load_best_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {}) if isinstance(data, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def default_config() -> AlphaConfig:
    root = Path(__file__).resolve().parents[2]
    return AlphaConfig(
        asset_paths={
            "BTC": root / "backtest" / "data" / "BTC-6h-1000wks-data.csv",
            "ETH": root / "backtest" / "data" / "ETH-1d-1000wks-data.csv",
            "SOL": root / "backtest" / "data" / "SOL-1d-1000wks-data.csv",
        },
        benchmark_asset="BTC",
        horizon=3,
        buy_threshold=0.008,
        sell_threshold=-0.008,
        min_stable_reserve=0.25,
        fee_bps=8.0,
        slippage_bps=5.0,
        latency_steps=1,
        walk_forward_splits=5,
        sequence_length=20,
        gru_epochs=5,
        random_seed=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prod_alpha production pipeline.")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "fast-tabular", "very-light-debug"],
        help="Runtime profile for tabular stack and debug depth.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility.")
    parser.add_argument("--config-json", type=str, default="", help="Path to hyperopt best config JSON (best_config_latest.json).")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU threads to use (0=auto).")
    parser.add_argument("--sklearn-n-jobs", type=int, default=-1, help="n_jobs for sklearn models (-1=all).")
    parser.add_argument("--gru-batch-size", type=int, default=64, help="Base GRU batch size.")
    parser.add_argument("--no-gru-auto-batch", action="store_true", help="Disable GRU auto batch sizing by available GPU memory.")
    parser.add_argument("--max-cache-entries", type=int, default=8, help="Maximum in-memory cache entries per cache bucket.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable in-memory data/feature/split cache.")
    parser.add_argument("--disable-resource-monitoring", action="store_true", help="Disable CPU/RAM/GPU utilization snapshots.")
    parser.add_argument("--debug", action="store_true", help="Fast lightweight mode for quick iteration.")
    parser.add_argument("--very-light-debug", action="store_true", help="Ultra-light debug mode (1 split, 1 GRU epoch, minimal writes).")
    args = parser.parse_args()

    cfg = default_config()
    cfg.random_seed = int(args.seed)
    cfg.tabular_profile = str(args.profile)
    cfg.cpu_thread_count = max(0, int(args.cpu_threads))
    cfg.sklearn_n_jobs = int(args.sklearn_n_jobs)
    cfg.gru_batch_size = max(8, int(args.gru_batch_size))
    cfg.gru_auto_batch_size = not bool(args.no_gru_auto_batch)
    cfg.max_cache_entries = max(1, int(args.max_cache_entries))
    cfg.enable_cache = not bool(args.disable_cache)
    cfg.resource_monitoring = not bool(args.disable_resource_monitoring)

    if args.config_json:
        config_path = Path(args.config_json).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Best config file not found: {config_path}")
        loaded = _load_best_config(config_path)
        if loaded:
            cfg.min_confidence = float(loaded.get("min_confidence", cfg.min_confidence))
            cfg.max_daily_turnover = float(loaded.get("max_daily_turnover", cfg.max_daily_turnover))
            cfg.signal_cooldown_steps = int(loaded.get("signal_cooldown_steps", cfg.signal_cooldown_steps))
            cfg.blend_lookback = int(loaded.get("blend_lookback", cfg.blend_lookback))
            cfg.max_position_weight = float(loaded.get("max_position_weight", cfg.max_position_weight))
            cfg.vol_scale_target = float(loaded.get("vol_scale_target", cfg.vol_scale_target))
            cfg.vol_scale_floor = float(loaded.get("vol_scale_floor", cfg.vol_scale_floor))
            cfg.vol_scale_ceiling = float(loaded.get("vol_scale_ceiling", cfg.vol_scale_ceiling))
            cfg.tabular_profile = str(loaded.get("tabular_profile", cfg.tabular_profile))
            cfg.random_seed = int(loaded.get("random_seed", cfg.random_seed))
            cfg.cpu_thread_count = int(loaded.get("cpu_thread_count", cfg.cpu_thread_count))
            cfg.sklearn_n_jobs = int(loaded.get("sklearn_n_jobs", cfg.sklearn_n_jobs))
            cfg.gru_batch_size = int(loaded.get("gru_batch_size", cfg.gru_batch_size))
            cfg.gru_auto_batch_size = bool(loaded.get("gru_auto_batch_size", cfg.gru_auto_batch_size))
            cfg.max_cache_entries = int(loaded.get("max_cache_entries", cfg.max_cache_entries))
            cfg.enable_cache = bool(loaded.get("enable_cache", cfg.enable_cache))
            cfg.resource_monitoring = bool(loaded.get("resource_monitoring", cfg.resource_monitoring))

    if args.profile == "very-light-debug" or args.very_light_debug:
        cfg.debug_mode = True
        cfg.very_light_debug_mode = True
        cfg.tabular_profile = "very-light-debug"
        cfg.walk_forward_splits = 1
        cfg.gru_epochs = 1
        cfg.sequence_length = min(8, cfg.sequence_length)
    elif args.debug:
        cfg.debug_mode = True
        cfg.walk_forward_splits = min(3, cfg.walk_forward_splits)
        cfg.gru_epochs = min(2, cfg.gru_epochs)
        cfg.sequence_length = min(12, cfg.sequence_length)

    if cfg.debug_mode:
        cfg.enable_timing_logs = True

    print(
        json.dumps(
            {
                "startup_banner": {
                    "profile": str(cfg.tabular_profile),
                    "debug_mode": bool(cfg.debug_mode),
                    "very_light_debug_mode": bool(cfg.very_light_debug_mode),
                    "config_json": str(args.config_json) if args.config_json else "",
                    "seed": int(cfg.random_seed),
                    "cpu_threads": int(cfg.cpu_thread_count),
                    "sklearn_n_jobs": int(cfg.sklearn_n_jobs),
                    "gru_batch_size": int(cfg.gru_batch_size),
                    "gru_auto_batch_size": bool(cfg.gru_auto_batch_size),
                    "cache_enabled": bool(cfg.enable_cache),
                    "max_cache_entries": int(cfg.max_cache_entries),
                    "resource_monitoring": bool(cfg.resource_monitoring),
                }
            },
            indent=2,
        ),
        flush=True,
    )

    _phase_banner("PHASE 1/1: PRODUCTION START")
    prod_start = time.perf_counter()
    result = run_production_pipeline(cfg)
    prod_elapsed = float(time.perf_counter() - prod_start)
    _phase_banner("PHASE 1/1: PRODUCTION COMPLETE", elapsed_seconds=prod_elapsed)
    print(
        json.dumps(
            {
                "deployed_strategy": result.get("deployed_strategy", "unknown"),
                "artifacts": result.get("artifacts", {}),
                "comparison": result.get("artifacts", {}).get("comparison", ""),
                "phase_timings_seconds": {"production": prod_elapsed},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
