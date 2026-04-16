from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline import AlphaConfig, optimize_upgraded_strategy_hyperparameters, run_production_pipeline


def _phase_banner(title: str, elapsed_seconds: float | None = None) -> None:
    banner = "=" * 18
    if elapsed_seconds is None:
        print(f"\n{banner} {title} {banner}", flush=True)
        return
    print(f"\n{banner} {title} | elapsed={elapsed_seconds:.2f}s {banner}", flush=True)


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


def _latest_comparison_file(folder: Path) -> Path | None:
    files = sorted(folder.glob("comparison_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_config_with_best(base: AlphaConfig, best: Dict[str, Any]) -> AlphaConfig:
    vol = best.get("volatility_scaling", {})
    return AlphaConfig(
        asset_paths=base.asset_paths,
        benchmark_asset=base.benchmark_asset,
        horizon=base.horizon,
        buy_threshold=base.buy_threshold,
        sell_threshold=base.sell_threshold,
        min_stable_reserve=base.min_stable_reserve,
        fee_bps=base.fee_bps,
        slippage_bps=base.slippage_bps,
        latency_steps=base.latency_steps,
        walk_forward_splits=base.walk_forward_splits,
        sequence_length=base.sequence_length,
        gru_epochs=base.gru_epochs,
        min_confidence=float(best.get("min_confidence", base.min_confidence)),
        min_edge=base.min_edge,
        signal_cooldown_steps=int(best.get("cooldown_period", base.signal_cooldown_steps)),
        max_daily_turnover=float(best.get("max_daily_turnover", base.max_daily_turnover)),
        price_impact_coef_bps=base.price_impact_coef_bps,
        max_position_weight=float(best.get("position_cap", base.max_position_weight)),
        vol_scale_target=float(vol.get("vol_scale_target", base.vol_scale_target)),
        vol_scale_floor=float(vol.get("vol_scale_floor", base.vol_scale_floor)),
        vol_scale_ceiling=float(vol.get("vol_scale_ceiling", base.vol_scale_ceiling)),
        feature_corr_threshold=base.feature_corr_threshold,
        blend_lookback=int(best.get("blend_lookback_window", base.blend_lookback)),
        tabular_profile=base.tabular_profile,
        sklearn_n_jobs=base.sklearn_n_jobs,
        cpu_thread_count=base.cpu_thread_count,
        gru_batch_size=base.gru_batch_size,
        gru_auto_batch_size=base.gru_auto_batch_size,
        max_cache_entries=base.max_cache_entries,
        resource_monitoring=base.resource_monitoring,
        enable_cache=base.enable_cache,
        enable_timing_logs=base.enable_timing_logs,
        debug_mode=base.debug_mode,
        very_light_debug_mode=base.very_light_debug_mode,
        random_seed=base.random_seed,
    )


def _metric_delta(before: Dict[str, Any], after: Dict[str, Any], key: str) -> Dict[str, float]:
    b = float(before.get(key, 0.0) or 0.0)
    a = float(after.get(key, 0.0) or 0.0)
    return {"before": b, "after": a, "delta": a - b}


def _recommendation(opt_summary: Dict[str, Any], prod_result: Dict[str, Any]) -> str:
    upgraded_mean = float(opt_summary.get("upgraded_sharpe_mean", 0.0) or 0.0)
    baseline_mean = float(opt_summary.get("baseline_sharpe_mean", 0.0) or 0.0)
    blended_mean = float(opt_summary.get("blended_sharpe_mean", 0.0) or 0.0)
    beat_baseline = float(opt_summary.get("beat_baseline_rate", 0.0) or 0.0)
    beat_blend = float(opt_summary.get("beat_blend_rate", 0.0) or 0.0)
    deployed = str(prod_result.get("deployed_strategy", ""))

    if upgraded_mean > baseline_mean and beat_baseline >= 0.70 and upgraded_mean >= blended_mean and beat_blend >= 0.60:
        return "Move to standalone upgraded strategy (robustly beats baseline and blend)."
    if deployed == "meta_blend" or blended_mean > upgraded_mean:
        return "Keep meta_blend (upgraded standalone still less robust than blend)."
    return "Keep baseline fallback safety while iterating upgraded execution filters."


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust walk-forward hyperparameter optimization for prod_alpha upgraded strategy.")
    parser.add_argument("--trials", type=int, default=18, help="Number of adaptive search trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for optimizer sampling.")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel hyperopt trial workers.")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "fast-tabular", "very-light-debug"],
        help="Runtime profile for tabular stack and debug depth.",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU threads to use (0=auto).")
    parser.add_argument("--sklearn-n-jobs", type=int, default=-1, help="n_jobs for sklearn models (-1=all).")
    parser.add_argument("--gru-batch-size", type=int, default=64, help="Base GRU batch size.")
    parser.add_argument("--no-gru-auto-batch", action="store_true", help="Disable GRU auto batch sizing by available GPU memory.")
    parser.add_argument("--max-cache-entries", type=int, default=8, help="Maximum in-memory cache entries per cache bucket.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable in-memory data/feature/split cache.")
    parser.add_argument("--disable-resource-monitoring", action="store_true", help="Disable CPU/RAM/GPU utilization snapshots.")
    parser.add_argument("--debug", action="store_true", help="Fast lightweight mode for quick iteration.")
    parser.add_argument("--very-light-debug", action="store_true", help="Ultra-light debug mode (1 split, 1 GRU epoch, minimal writes).")
    parser.add_argument("--run-production", action="store_true", help="Run full production pipeline with best parameters after optimization.")
    args = parser.parse_args()

    cfg = default_config()
    cfg.cpu_thread_count = max(0, int(args.cpu_threads))
    cfg.sklearn_n_jobs = int(args.sklearn_n_jobs)
    cfg.gru_batch_size = max(8, int(args.gru_batch_size))
    cfg.gru_auto_batch_size = not bool(args.no_gru_auto_batch)
    cfg.max_cache_entries = max(1, int(args.max_cache_entries))
    cfg.enable_cache = not bool(args.disable_cache)
    cfg.resource_monitoring = not bool(args.disable_resource_monitoring)
    cfg.tabular_profile = str(args.profile)

    if args.profile == "very-light-debug" or args.very_light_debug:
        cfg.debug_mode = True
        cfg.tabular_profile = "very-light-debug"
        cfg.very_light_debug_mode = True
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
                    "debug_mode": bool(cfg.debug_mode),
                    "very_light_debug_mode": bool(cfg.very_light_debug_mode),
                    "tabular_profile": str(cfg.tabular_profile),
                    "cache_enabled": bool(cfg.enable_cache),
                    "gpu_available": bool(__import__("torch").cuda.is_available()),
                    "parallel_workers": max(1, int(args.parallel_workers)),
                    "cpu_threads": int(cfg.cpu_thread_count),
                    "sklearn_n_jobs": int(cfg.sklearn_n_jobs),
                    "gru_batch_size": int(cfg.gru_batch_size),
                    "gru_auto_batch_size": bool(cfg.gru_auto_batch_size),
                    "max_cache_entries": int(cfg.max_cache_entries),
                    "resource_monitoring": bool(cfg.resource_monitoring),
                    "run_production": bool(args.run_production),
                }
            },
            indent=2,
        ),
        flush=True,
    )

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "prod_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)

    before_file = _latest_comparison_file(out_dir)
    before_comp = _load_json(before_file)

    _phase_banner("PHASE 1/2: HYPEROPT START")
    hyperopt_start = time.perf_counter()
    opt = optimize_upgraded_strategy_hyperparameters(
        cfg,
        n_trials=max(1, int(args.trials)),
        random_seed=args.seed,
        parallel_workers=max(1, int(args.parallel_workers)),
    )
    hyperopt_elapsed = float(time.perf_counter() - hyperopt_start)
    _phase_banner("PHASE 1/2: HYPEROPT COMPLETE", elapsed_seconds=hyperopt_elapsed)

    best_cfg = _build_config_with_best(cfg, opt["best_parameters"])
    best_cfg.debug_mode = cfg.debug_mode
    best_cfg.enable_timing_logs = cfg.enable_timing_logs

    after_prod: Dict[str, Any] = {}
    production_elapsed = 0.0
    if args.run_production:
        _phase_banner("TRANSITION: HYPEROPT -> PRODUCTION")
        _phase_banner("PHASE 2/2: PRODUCTION START")
        production_start = time.perf_counter()
        after_prod = run_production_pipeline(best_cfg)
        production_elapsed = float(time.perf_counter() - production_start)
        _phase_banner("PHASE 2/2: PRODUCTION COMPLETE", elapsed_seconds=production_elapsed)
    else:
        print(
            "\nSkipping production phase by default. Recommended workflow: "
            "1) run hyperopt, 2) save best config/artifacts, 3) run production separately.",
            flush=True,
        )

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trials_df = pd.DataFrame(opt.get("trials", []))
    trials_csv = out_dir / f"hyperopt_trials_{now}.csv"
    if not trials_df.empty:
        trials_df.to_csv(trials_csv, index=False)

    best_config_path = out_dir / f"best_config_{now}.json"
    best_config_latest_path = out_dir / "best_config_latest.json"
    best_config_payload = {
        "source": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(cfg.random_seed),
            "profile": str(cfg.tabular_profile),
            "trials": int(max(1, int(args.trials))),
            "parallel_workers": int(max(1, int(args.parallel_workers))),
        },
        "config": {
            "min_confidence": float(best_cfg.min_confidence),
            "max_daily_turnover": float(best_cfg.max_daily_turnover),
            "signal_cooldown_steps": int(best_cfg.signal_cooldown_steps),
            "blend_lookback": int(best_cfg.blend_lookback),
            "max_position_weight": float(best_cfg.max_position_weight),
            "vol_scale_target": float(best_cfg.vol_scale_target),
            "vol_scale_floor": float(best_cfg.vol_scale_floor),
            "vol_scale_ceiling": float(best_cfg.vol_scale_ceiling),
            "tabular_profile": str(best_cfg.tabular_profile),
            "random_seed": int(best_cfg.random_seed),
            "cpu_thread_count": int(best_cfg.cpu_thread_count),
            "sklearn_n_jobs": int(best_cfg.sklearn_n_jobs),
            "gru_batch_size": int(best_cfg.gru_batch_size),
            "gru_auto_batch_size": bool(best_cfg.gru_auto_batch_size),
            "max_cache_entries": int(best_cfg.max_cache_entries),
            "enable_cache": bool(best_cfg.enable_cache),
            "resource_monitoring": bool(best_cfg.resource_monitoring),
        },
    }
    best_config_text = json.dumps(best_config_payload, indent=2)
    best_config_path.write_text(best_config_text, encoding="utf-8")
    best_config_latest_path.write_text(best_config_text, encoding="utf-8")

    before_upgraded = (before_comp.get("upgraded_backtest") or {}) if before_comp else {}
    after_upgraded = (after_prod.get("upgraded_backtest") or {}) if after_prod else {}

    before_deployed = (before_comp.get("deployed_backtest") or {}) if before_comp else {}
    after_deployed = (after_prod.get("deployed_backtest") or {}) if after_prod else {}

    comparison = {
        "upgraded": {
            "sharpe": _metric_delta(before_upgraded, after_upgraded, "sharpe") if after_prod else {},
            "max_drawdown": _metric_delta(before_upgraded, after_upgraded, "max_drawdown") if after_prod else {},
            "avg_turnover": _metric_delta(before_upgraded, after_upgraded, "avg_turnover") if after_prod else {},
        },
        "deployed": {
            "sharpe": _metric_delta(before_deployed, after_deployed, "sharpe") if after_prod else {},
            "max_drawdown": _metric_delta(before_deployed, after_deployed, "max_drawdown") if after_prod else {},
            "avg_turnover": _metric_delta(before_deployed, after_deployed, "avg_turnover") if after_prod else {},
        },
    }

    report = {
        "search": opt.get("search", {}),
        "runtime_profile": opt.get("runtime_profile", {}),
        "best_parameters": opt.get("best_parameters", {}),
        "best_summary": opt.get("best_summary", {}),
        "robust_ranges": (opt.get("sensitivity") or {}).get("robust_ranges", {}),
        "sensitivity": opt.get("sensitivity", {}),
        "baseline_reference": opt.get("baseline_reference", {}),
        "before_comparison_file": str(before_file) if before_file else "",
        "after_production": {
            "enabled": bool(args.run_production),
            "result": after_prod,
        },
        "phase_timings_seconds": {
            "hyperopt": hyperopt_elapsed,
            "production": production_elapsed,
            "total": float(hyperopt_elapsed + production_elapsed),
        },
        "before_vs_after": comparison,
        "recommendation": _recommendation(opt.get("best_summary", {}), after_prod),
        "artifacts": {
            "trials_csv": str(trials_csv),
            "best_config": str(best_config_path),
            "best_config_latest": str(best_config_latest_path),
        },
    }

    report_path = out_dir / f"hyperopt_report_{now}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    recommended_production_cmd = (
        "c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe "
        "run_prod_alpha.py --config-json ../artifacts/prod_alpha/best_config_latest.json"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "trials_csv": str(trials_csv),
                "best_config": str(best_config_path),
                "best_config_latest": str(best_config_latest_path),
                "best_parameters": report.get("best_parameters", {}),
                "best_summary": report.get("best_summary", {}),
                "recommendation": report.get("recommendation", ""),
                "phase_timings_seconds": report.get("phase_timings_seconds", {}),
                "next_step": {
                    "workflow": [
                        "Run hyperopt",
                        "Save best config/artifacts",
                        "Run production separately",
                    ],
                    "recommended_production_command": recommended_production_cmd,
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
