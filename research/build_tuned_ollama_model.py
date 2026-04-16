from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LORA_RUNS_DIR = ARTIFACTS_DIR / "lora_runs"
MERGED_MODELS_DIR = ARTIFACTS_DIR / "merged_models"
MERGED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def latest_lora_adapter_dir() -> Path:
    if not LORA_RUNS_DIR.exists():
        raise FileNotFoundError(f"LoRA runs directory not found: {LORA_RUNS_DIR}")
    candidates = [path for path in LORA_RUNS_DIR.iterdir() if path.is_dir() and (path / "adapter_model.safetensors").exists()]
    if not candidates:
        raise FileNotFoundError("No LoRA adapter run found under research/artifacts/lora_runs")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def infer_base_model(adapter_dir: Path, explicit_base_model: str | None) -> str:
    if explicit_base_model:
        return explicit_base_model
    metadata_path = adapter_dir / "run_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        base_model = str(payload.get("base_model", "")).strip()
        if base_model:
            return base_model
    raise ValueError("Could not infer base model. Pass --base-model explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter and create a tuned Ollama model.")
    parser.add_argument("--adapter-dir", default="", help="Path to LoRA adapter run directory. Defaults to latest run.")
    parser.add_argument("--base-model", default="", help="HF base model ID/path used for LoRA training.")
    parser.add_argument("--ollama-model", default="", help="Target Ollama model name (default: ai-trader-tuned:<timestamp>).")
    parser.add_argument("--skip-create", action="store_true", help="Only merge model; skip 'ollama create'.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()

    adapter_dir = Path(args.adapter_dir).resolve() if args.adapter_dir else latest_lora_adapter_dir().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    base_model = infer_base_model(adapter_dir, args.base_model.strip() or None)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    merged_dir = (MERGED_MODELS_DIR / f"merged_{stamp}").resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Adapter dir: {adapter_dir}")
    print(f"Base model: {base_model}")
    print(f"Merged output: {merged_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    modelfile_path = merged_dir / "Modelfile"
    modelfile_path.write_text(
        "\n".join(
            [
                f"FROM {merged_dir.as_posix()}",
                "PARAMETER temperature 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ollama_model = args.ollama_model.strip() or f"ai-trader-tuned:{stamp}"

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "merged_dir": str(merged_dir),
        "modelfile": str(modelfile_path),
        "ollama_model": ollama_model,
        "ollama_create_skipped": bool(args.skip_create),
    }
    (merged_dir / "ollama_build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.skip_create:
        print("Skipped Ollama create step (--skip-create).")
        print(f"To create manually: ollama create {ollama_model} -f \"{modelfile_path}\"")
        return

    ollama_bin = "ollama"
    local_ollama = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if local_ollama.exists():
        ollama_bin = str(local_ollama)

    create_cmd = [ollama_bin, "create", ollama_model, "-f", str(modelfile_path)]
    print("Running:", " ".join(create_cmd))
    subprocess.run(create_cmd, check=True)

    print("Done")
    print(f"Created Ollama model: {ollama_model}")
    print("Set this in .env and restart backend:")
    print(f"LLM_MODEL={ollama_model}")


if __name__ == "__main__":
    run()
