"""Persistent operator configuration for mergedog."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mergedog.paths import CONFIG_FILE

LLM_PROVIDERS = ("claude", "codex", "metacode")
DEFAULT_LLM_PROVIDER = "codex"
DEFAULT_CLAUDE_MODEL = "opus"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = DEFAULT_LLM_PROVIDER
    model: str | None = None

    @property
    def effective_model(self) -> str | None:
        if self.model:
            return self.model
        if self.provider == "claude":
            return DEFAULT_CLAUDE_MODEL
        return None


def _read_config_file(path: Path = CONFIG_FILE) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"failed to read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_config_file(data: dict[str, Any], path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def get_llm_config(path: Path = CONFIG_FILE) -> LLMConfig:
    data = _read_config_file(path)
    raw_llm = data.get("llm", {})
    if raw_llm is None:
        raw_llm = {}
    if not isinstance(raw_llm, dict):
        raise ValueError(f"{path}: llm must be an object")
    provider = raw_llm.get("provider", DEFAULT_LLM_PROVIDER)
    if not isinstance(provider, str) or provider not in LLM_PROVIDERS:
        choices = ", ".join(LLM_PROVIDERS)
        raise ValueError(f"{path}: llm.provider must be one of: {choices}")
    model = raw_llm.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(f"{path}: llm.model must be a non-empty string")
    return LLMConfig(provider=provider, model=model)


def set_llm_config(
    provider: str,
    *,
    model: str | None = None,
    clear_model: bool = False,
    path: Path = CONFIG_FILE,
) -> LLMConfig:
    if provider not in LLM_PROVIDERS:
        choices = ", ".join(LLM_PROVIDERS)
        raise ValueError(f"provider must be one of: {choices}")
    if model is not None and not model.strip():
        raise ValueError("model must be a non-empty string")
    data = _read_config_file(path)
    llm = data.get("llm", {})
    if not isinstance(llm, dict):
        llm = {}
    llm["provider"] = provider
    if clear_model:
        llm.pop("model", None)
    elif model is not None:
        llm["model"] = model
    data["llm"] = llm
    _write_config_file(data, path)
    return get_llm_config(path)
