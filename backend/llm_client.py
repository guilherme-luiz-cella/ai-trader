from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
ARTIFACTS_DIR = RESEARCH_DIR / "artifacts"
load_dotenv(PROJECT_ROOT / ".env", override=False)

_MODEL_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()
_STARTUP_LOG_LOCK = threading.Lock()
_SESSION: requests.Session | None = None
_LOCAL_MODEL_CACHE: dict[str, Any] = {
    "path": "",
    "tokenizer": None,
    "model": None,
}
_STARTUP_LOGGED = False
_STARTUP_VALIDATION: dict[str, Any] = {
    "attempted": False,
    "path_exists": False,
    "model_load_status": "not_attempted",
    "model_load_error": "",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _latest_merged_model_dir() -> Path | None:
    folder = ARTIFACTS_DIR / "merged_models"
    if not folder.exists():
        return None
    candidates = sorted((path for path in folder.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _normalize_model_path(raw_path: str) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        return Path("")
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.exists():
        return path
    normalized = expanded.replace("\\", "/")
    if normalized.startswith("/app/"):
        relative = normalized[len("/app/") :]
        candidate = PROJECT_ROOT / Path(relative)
        return candidate
    return path


def get_llm_runtime_config() -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "local_transformers").strip().lower() or "local_transformers"
    primary_model = os.getenv("PRIMARY_MODEL", os.getenv("LLM_MODEL", "")).strip()
    primary_model_path = os.getenv("PRIMARY_MODEL_PATH", os.getenv("LLM_LOCAL_MODEL_PATH", "")).strip()
    allow_fallback = _env_bool("ALLOW_MODEL_FALLBACK", False)
    fallback_model = os.getenv("FALLBACK_MODEL", "").strip()
    fallback_model_path = os.getenv("FALLBACK_MODEL_PATH", "").strip()
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    bypass_proxy = _env_bool("LLM_BYPASS_ENV_PROXY", True)
    llm_enabled = _env_bool("LLM_ENABLED", False)
    explicit_trained = _env_bool("PRIMARY_MODEL_IS_TRAINED", True)
    return {
        "enabled": llm_enabled,
        "provider": provider,
        "primary_model": primary_model,
        "primary_model_path": primary_model_path,
        "allow_fallback": allow_fallback,
        "fallback_model": fallback_model,
        "fallback_model_path": fallback_model_path,
        "timeout_seconds": timeout_seconds,
        "base_url": base_url,
        "api_key": api_key,
        "bypass_proxy": bypass_proxy,
        "is_trained_model": explicit_trained,
    }


def _resolved_model_target(config: dict[str, Any]) -> dict[str, Any]:
    provider = str(config.get("provider") or "")
    primary_model = str(config.get("primary_model") or "")
    primary_model_path = str(config.get("primary_model_path") or "")
    fallback_model = str(config.get("fallback_model") or "")
    fallback_model_path = str(config.get("fallback_model_path") or "")
    allow_fallback = bool(config.get("allow_fallback", False))

    if provider == "local_transformers":
        if primary_model_path:
            candidate = _normalize_model_path(primary_model_path)
            if candidate.exists():
                return {
                    "model": primary_model or candidate.name,
                    "model_path": str(candidate),
                    "is_trained_model": bool(config.get("is_trained_model", True)),
                    "fallback_active": False,
                    "fallback_reason": "",
                    "model_source": "primary_model_path",
                }
            if not allow_fallback:
                raise FileNotFoundError(f"PRIMARY_MODEL_PATH does not exist: {candidate}")
        else:
            if not allow_fallback:
                raise FileNotFoundError("PRIMARY_MODEL_PATH must be set for local_transformers runtime.")

        if allow_fallback:
            if fallback_model_path:
                candidate = _normalize_model_path(fallback_model_path)
                if candidate.exists():
                    return {
                        "model": fallback_model or candidate.name,
                        "model_path": str(candidate),
                        "is_trained_model": False,
                        "fallback_active": True,
                        "fallback_reason": "configured_fallback_model_path",
                        "model_source": "fallback_model_path",
                    }
            latest_merged = _latest_merged_model_dir()
            if latest_merged is not None:
                return {
                    "model": fallback_model or latest_merged.name,
                    "model_path": str(latest_merged),
                    "is_trained_model": False,
                    "fallback_active": True,
                    "fallback_reason": "latest_merged_model_dir",
                    "model_source": "latest_merged_model_dir",
                }
        raise FileNotFoundError("No local model available. Set PRIMARY_MODEL_PATH or configure explicit fallback.")

    if primary_model:
        return {
            "model": primary_model,
            "model_path": primary_model_path,
            "is_trained_model": bool(config.get("is_trained_model", bool(primary_model_path))),
            "fallback_active": False,
            "fallback_reason": "",
            "model_source": "primary_model",
        }

    if allow_fallback and fallback_model:
        return {
            "model": fallback_model,
            "model_path": fallback_model_path,
            "is_trained_model": False,
            "fallback_active": True,
            "fallback_reason": "configured_fallback_model",
            "model_source": "fallback_model",
        }

    raise ValueError("PRIMARY_MODEL must be set for server-based LLM runtimes.")


def _log_startup_status_once() -> None:
    global _STARTUP_LOGGED
    with _STARTUP_LOG_LOCK:
        if _STARTUP_LOGGED:
            return
        validate_llm_startup()
        status = get_llm_status()
        print(
            "[llm] "
            f"provider={status.get('provider')} "
            f"model={status.get('active_model')} "
            f"model_path={status.get('active_model_path')} "
            f"path_exists={status.get('path_exists')} "
            f"is_trained_model={status.get('is_trained_model')} "
            f"fallback_active={status.get('fallback_active')} "
            f"health_status={status.get('health_status')} "
            f"model_load_status={status.get('model_load_status')}"
        )
        _STARTUP_LOGGED = True


def _get_session(config: dict[str, Any]) -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        session = requests.Session()
        if bool(config.get("bypass_proxy", True)):
            session.trust_env = False
        _SESSION = session
        return _SESSION


def _load_local_transformers_model(model_path: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing local transformers dependencies. Install transformers/torch/sentencepiece/safetensors."
        ) from exc

    with _MODEL_LOCK:
        if _LOCAL_MODEL_CACHE.get("path") != model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
            _LOCAL_MODEL_CACHE["path"] = model_path
            _LOCAL_MODEL_CACHE["tokenizer"] = tokenizer
            _LOCAL_MODEL_CACHE["model"] = model
        return _LOCAL_MODEL_CACHE["tokenizer"], _LOCAL_MODEL_CACHE["model"]


def validate_llm_startup() -> dict[str, Any]:
    global _STARTUP_VALIDATION
    config = get_llm_runtime_config()
    validation = {
        "attempted": True,
        "path_exists": False,
        "model_load_status": "not_attempted",
        "model_load_error": "",
    }
    if not bool(config.get("enabled", False)):
        validation["model_load_status"] = "disabled"
        _STARTUP_VALIDATION = validation
        return validation
    try:
        target = _resolved_model_target(config)
        model_path = str(target.get("model_path") or "")
        if model_path:
            validation["path_exists"] = Path(model_path).exists()
        if str(config.get("provider") or "") == "local_transformers":
            if not validation["path_exists"]:
                raise FileNotFoundError(f"Resolved model path does not exist: {model_path}")
            _load_local_transformers_model(model_path)
            validation["model_load_status"] = "loaded"
        else:
            validation["model_load_status"] = "configured"
    except Exception as exc:
        validation["model_load_status"] = "error"
        validation["model_load_error"] = str(exc)
    _STARTUP_VALIDATION = validation
    return validation


def _chat_local_transformers(system_prompt: str, user_content: str, max_new_tokens: int) -> str:
    model_target = _resolved_model_target(get_llm_runtime_config())
    tokenizer, model = _load_local_transformers_model(str(model_target["model_path"]))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"SYSTEM: {system_prompt}\nUSER: {user_content}\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    import torch

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def llm_chat(system_prompt: str, user_content: str, max_new_tokens: int = 300, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
    config = get_llm_runtime_config()
    _log_startup_status_once()
    if not bool(config.get("enabled", False)):
        return {"status": "disabled", "content": "", "error": "LLM is disabled.", **get_llm_status()}

    try:
        model_target = _resolved_model_target(config)
    except Exception as exc:
        return {"status": "error", "content": "", "error": str(exc), **get_llm_status()}

    provider = str(config.get("provider") or "")
    if provider == "local_transformers":
        try:
            content = _chat_local_transformers(system_prompt=system_prompt, user_content=user_content, max_new_tokens=max_new_tokens)
            result = {
                "status": "ok",
                "content": content,
                "endpoint": "local://transformers",
                "handled_by_model": model_target["model"],
                **get_llm_status(),
            }
            print(f"[llm] inference provider={result.get('provider')} model={result.get('handled_by_model')} fallback_active={result.get('fallback_active')}")
            return result
        except Exception as exc:
            result = {"status": "error", "content": "", "error": str(exc), **get_llm_status()}
            print(f"[llm] inference_error provider={result.get('provider')} model={result.get('active_model') or result.get('active_model_path')} error={result.get('error')}")
            return result

    headers = {"Content-Type": "application/json"}
    if provider in {"openai_compatible", "deepseek", "huggingface_inference"} and str(config.get("api_key") or ""):
        headers["Authorization"] = f"Bearer {config['api_key']}"

    if provider == "ollama":
        endpoint = f"{str(config.get('base_url') or '').rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": model_target["model"],
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if response_format:
            payload["format"] = "json"
    elif provider in {"openai_compatible", "deepseek"}:
        endpoint = f"{str(config.get('base_url') or '').rstrip('/')}/chat/completions"
        payload = {
            "model": model_target["model"],
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if response_format:
            payload["response_format"] = response_format
    elif provider == "huggingface_inference":
        endpoint = f"{str(config.get('base_url') or '').rstrip('/')}/models/{model_target['model']}"
        payload = {
            "inputs": f"{system_prompt}\n{user_content}",
            "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.2, "return_full_text": False},
        }
    else:
        return {"status": "error", "content": "", "error": f"Unsupported LLM provider: {provider}", **get_llm_status()}

    try:
        session = _get_session(config)
        response = session.post(endpoint, headers=headers, json=payload, timeout=int(config.get("timeout_seconds") or 30))
        response.raise_for_status()
        data = response.json()
        if provider == "ollama":
            content = str(data.get("message", {}).get("content", ""))
        elif provider == "huggingface_inference":
            if isinstance(data, list) and data:
                content = str(data[0].get("generated_text", ""))
            elif isinstance(data, dict):
                content = str(data.get("generated_text", ""))
            else:
                content = str(data)
        else:
            content = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        result = {
            "status": "ok",
            "content": content,
            "endpoint": endpoint,
            "handled_by_model": model_target["model"],
            **get_llm_status(),
        }
        print(f"[llm] inference provider={result.get('provider')} model={result.get('handled_by_model')} fallback_active={result.get('fallback_active')}")
        return result
    except Exception as exc:
        result = {"status": "error", "content": "", "error": str(exc), "endpoint": endpoint, **get_llm_status()}
        print(f"[llm] inference_error provider={result.get('provider')} model={result.get('active_model') or result.get('active_model_path')} error={result.get('error')}")
        return result


def get_llm_status() -> dict[str, Any]:
    config = get_llm_runtime_config()
    validation = dict(_STARTUP_VALIDATION)
    try:
        target = _resolved_model_target(config) if bool(config.get("enabled", False)) else {
            "model": str(config.get("primary_model") or ""),
            "model_path": str(config.get("primary_model_path") or ""),
            "is_trained_model": bool(config.get("is_trained_model", True)),
            "fallback_active": False,
            "fallback_reason": "",
            "model_source": "disabled",
        }
        health_status = "ok" if bool(config.get("enabled", False)) else "disabled"
        health_error = ""
    except Exception as exc:
        target = {
            "model": str(config.get("primary_model") or ""),
            "model_path": str(config.get("primary_model_path") or ""),
            "is_trained_model": bool(config.get("is_trained_model", True)),
            "fallback_active": False,
            "fallback_reason": "",
            "model_source": "error",
        }
        health_status = "error"
        health_error = str(exc)

    model_path = str(target.get("model_path") or "")
    if model_path and not validation.get("attempted", False):
        validation["path_exists"] = Path(model_path).exists()
        validation["model_load_status"] = "not_attempted"
        validation["model_load_error"] = ""

    return {
        "provider": str(config.get("provider") or ""),
        "active_model": str(target.get("model") or ""),
        "active_model_path": model_path,
        "is_trained_model": bool(target.get("is_trained_model", False)),
        "fallback_active": bool(target.get("fallback_active", False)),
        "fallback_reason": str(target.get("fallback_reason") or ""),
        "model_source": str(target.get("model_source") or ""),
        "health_status": health_status,
        "health_error": health_error,
        "path_exists": bool(validation.get("path_exists", False)),
        "model_load_status": str(validation.get("model_load_status") or "not_attempted"),
        "model_load_error": str(validation.get("model_load_error") or ""),
        "base_url": str(config.get("base_url") or ""),
        "enabled": bool(config.get("enabled", False)),
    }


def ensure_llm_startup_logged() -> None:
    _log_startup_status_once()
