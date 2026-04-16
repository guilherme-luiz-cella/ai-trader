# Local LoRA Fine-Tuning

This workflow trains a local LoRA adapter from `research/artifacts/aapl_llm_finetune.jsonl`.

## 1) Install dependencies

```powershell
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe -m pip install -r requirements-llm.txt
```

## 2) Dry-run validation

```powershell
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe research/train_lora_local.py --dry-run
```

## 3) First short training pass

```powershell
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe research/train_lora_local.py --base-model Qwen/Qwen2.5-0.5B-Instruct --sample-size 512 --max-steps 10
```

## Notes

- Outputs are saved under `research/artifacts/lora_runs/lora_YYYYMMDD_HHMMSS/`.
- Use `--use-4bit` only if `bitsandbytes` works in your environment.
- This produces a LoRA adapter, not a fully merged model.

## 4) Build a tuned Ollama model from adapter

```powershell
c:/Users/lastf/Desktop/ai-trader/.venv/Scripts/python.exe research/build_tuned_ollama_model.py
```

This will:

- merge LoRA adapter + base model into `research/artifacts/merged_models/merged_*/`
- run `ollama create ai-trader-tuned:<timestamp> -f Modelfile`

Then set `LLM_MODEL` in `.env` to the created model name and restart Docker:

```powershell
docker compose up --build -d
```
