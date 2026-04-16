from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _latest_report(folder: Path) -> Path | None:
    files = sorted(folder.glob("hyperopt_report_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _run_profile(
    script_dir: Path,
    profile: str,
    trials: int,
    seed: int,
    cpu_threads: int,
    sklearn_n_jobs: int,
    parallel_workers: int,
    debug: bool,
) -> Dict[str, Any]:
    out_dir = script_dir.parent / "artifacts" / "prod_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _latest_report(out_dir)

    cmd = [
        sys.executable,
        "run_prod_alpha_hyperopt.py",
        "--profile",
        profile,
        "--trials",
        str(max(1, trials)),
        "--seed",
        str(seed),
        "--parallel-workers",
        str(max(1, parallel_workers)),
        "--cpu-threads",
        str(max(0, cpu_threads)),
        "--sklearn-n-jobs",
        str(sklearn_n_jobs),
    ]
    if debug and profile != "very-light-debug":
        cmd.append("--debug")

    run = subprocess.run(cmd, cwd=script_dir, check=False, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"Profile {profile} run failed (code={run.returncode}).\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}")

    after = _latest_report(out_dir)
    if after is None or (before is not None and after == before):
        raise RuntimeError(f"Could not locate new report for profile {profile}.")

    report = json.loads(after.read_text(encoding="utf-8"))
    return {"profile": profile, "report_path": str(after), "report": report}


def _profile_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    profile = str(result["profile"])
    report = result["report"]
    search = report.get("search", {})
    best = report.get("best_summary", {})
    runtime_profile = report.get("runtime_profile", {})
    run_profile = str(search.get("tabular_profile", profile))
    stage_data = (runtime_profile.get("average_stage_seconds") or {}).get("split", {})

    fit_keys = [k for k in stage_data.keys() if k.startswith("split_tabular_model_fit_")]
    model_bottleneck = "unknown"
    model_bottleneck_seconds = 0.0
    if fit_keys:
        model_bottleneck = max(fit_keys, key=lambda k: float(stage_data.get(k, 0.0)))
        model_bottleneck_seconds = float(stage_data.get(model_bottleneck, 0.0))

    return {
        "profile": run_profile,
        "report_path": str(result["report_path"]),
        "avg_trial_seconds": float((report.get("search") or {}).get("avg_trial_seconds", 0.0)),
        "tabular_stage_seconds": float(stage_data.get("split_tabular_train_backtest", 0.0)),
        "objective": float(best.get("objective", 0.0)),
        "upgraded_sharpe_mean": float(best.get("upgraded_sharpe_mean", 0.0)),
        "baseline_sharpe_mean": float(best.get("baseline_sharpe_mean", 0.0)),
        "beat_baseline_rate": float(best.get("beat_baseline_rate", 0.0)),
        "upgraded_avg_turnover_mean": float(best.get("upgraded_avg_turnover_mean", 0.0)),
        "upgraded_max_drawdown_mean": float(best.get("upgraded_max_drawdown_mean", 0.0)),
        "tabular_model_bottleneck": model_bottleneck,
        "tabular_model_bottleneck_seconds": model_bottleneck_seconds,
        "split_stage": stage_data,
    }


def _decision(default_row: Dict[str, Any], fast_row: Dict[str, Any]) -> Dict[str, Any]:
    default_runtime = max(1e-9, float(default_row["avg_trial_seconds"]))
    fast_runtime = float(fast_row["avg_trial_seconds"])
    runtime_reduction = (default_runtime - fast_runtime) / default_runtime

    objective_drop = float(default_row["objective"]) - float(fast_row["objective"])
    sharpe_drop = float(default_row["upgraded_sharpe_mean"]) - float(fast_row["upgraded_sharpe_mean"])
    beat_baseline_drop = float(default_row["beat_baseline_rate"]) - float(fast_row["beat_baseline_rate"])

    major_runtime_reduction = runtime_reduction >= 0.30
    small_quality_degradation = objective_drop <= 0.12 and sharpe_drop <= 0.08 and beat_baseline_drop <= 0.10
    should_default = bool(major_runtime_reduction and small_quality_degradation)

    return {
        "runtime_reduction_ratio": float(runtime_reduction),
        "objective_drop": float(objective_drop),
        "sharpe_drop": float(sharpe_drop),
        "beat_baseline_drop": float(beat_baseline_drop),
        "recommendation": (
            "Promote fast-tabular as default tuning profile" if should_default else "Keep fast-tabular as optional speed profile"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark for tabular profiles in prod_alpha hyperopt.")
    parser.add_argument("--trials", type=int, default=4, help="Trials per profile.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for both profiles.")
    parser.add_argument("--cpu-threads", type=int, default=12, help="CPU threads per run.")
    parser.add_argument("--sklearn-n-jobs", type=int, default=-1, help="sklearn n_jobs.")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel workers per run.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode during benchmark runs.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    debug = not bool(args.no_debug)

    default_run = _run_profile(
        script_dir=script_dir,
        profile="default",
        trials=args.trials,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
        sklearn_n_jobs=args.sklearn_n_jobs,
        parallel_workers=args.parallel_workers,
        debug=debug,
    )
    fast_run = _run_profile(
        script_dir=script_dir,
        profile="fast-tabular",
        trials=args.trials,
        seed=args.seed,
        cpu_threads=args.cpu_threads,
        sklearn_n_jobs=args.sklearn_n_jobs,
        parallel_workers=args.parallel_workers,
        debug=debug,
    )

    default_row = _profile_summary(default_run)
    fast_row = _profile_summary(fast_run)
    verdict = _decision(default_row, fast_row)

    output = {
        "benchmark": {
            "trials_per_profile": int(max(1, args.trials)),
            "seed": int(args.seed),
            "debug_mode": bool(debug),
            "profiles": [default_row, fast_row],
            "decision": verdict,
        }
    }

    out_dir = script_dir.parent / "artifacts" / "prod_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tabular_profile_benchmark_latest.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2), flush=True)
    print(f"Saved benchmark summary: {out_path}", flush=True)


if __name__ == "__main__":
    main()
