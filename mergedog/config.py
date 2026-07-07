"""Persistent operator configuration for mergedog."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mergedog.paths import CONFIG_FILE, atomic_write_text

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


@dataclass(frozen=True)
class CiSevConfig:
    ignored_numbers: tuple[int, ...] = ()


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
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def parse_ci_sev_number(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("ci: sev issue number must be a positive integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.startswith("#"):
            raw = raw[1:]
        if not raw.isdigit():
            raise ValueError("ci: sev issue number must be a positive integer")
        number = int(raw)
    else:
        raise ValueError("ci: sev issue number must be a positive integer")
    if number <= 0:
        raise ValueError("ci: sev issue number must be a positive integer")
    return number


def get_ci_sev_config(path: Path = CONFIG_FILE) -> CiSevConfig:
    data = _read_config_file(path)
    raw_ci_sev = data.get("ci_sev", {})
    if raw_ci_sev is None:
        raw_ci_sev = {}
    if not isinstance(raw_ci_sev, dict):
        raise ValueError(f"{path}: ci_sev must be an object")
    raw_ignored = raw_ci_sev.get("ignored", [])
    if raw_ignored is None:
        raw_ignored = []
    if not isinstance(raw_ignored, list):
        raise ValueError(f"{path}: ci_sev.ignored must be a list")
    try:
        ignored = tuple(sorted({parse_ci_sev_number(v) for v in raw_ignored}))
    except ValueError as e:
        raise ValueError(
            f"{path}: ci_sev.ignored contains invalid issue: {e}"
        ) from e
    return CiSevConfig(ignored_numbers=ignored)


def get_ignored_ci_sev_numbers(path: Path = CONFIG_FILE) -> set[int]:
    return set(get_ci_sev_config(path).ignored_numbers)


def set_ignored_ci_sev_numbers(
    numbers: list[int] | tuple[int, ...] | set[int],
    *,
    path: Path = CONFIG_FILE,
) -> CiSevConfig:
    ignored = sorted({parse_ci_sev_number(n) for n in numbers})
    data = _read_config_file(path)
    ci_sev = data.get("ci_sev", {})
    if not isinstance(ci_sev, dict):
        ci_sev = {}
    ci_sev["ignored"] = ignored
    data["ci_sev"] = ci_sev
    _write_config_file(data, path)
    return get_ci_sev_config(path)


def add_ignored_ci_sev(number: int, *, path: Path = CONFIG_FILE) -> CiSevConfig:
    ignored = get_ignored_ci_sev_numbers(path)
    ignored.add(parse_ci_sev_number(number))
    return set_ignored_ci_sev_numbers(ignored, path=path)


def remove_ignored_ci_sev(
    number: int, *, path: Path = CONFIG_FILE
) -> CiSevConfig:
    ignored = get_ignored_ci_sev_numbers(path)
    ignored.discard(parse_ci_sev_number(number))
    return set_ignored_ci_sev_numbers(ignored, path=path)


def clear_ignored_ci_sevs(*, path: Path = CONFIG_FILE) -> CiSevConfig:
    return set_ignored_ci_sev_numbers(set(), path=path)


def format_ci_sev_ignored_numbers(
    numbers: tuple[int, ...] | list[int] | set[int],
) -> str:
    ordered = sorted(numbers)
    if not ordered:
        return "none"
    return ", ".join(f"#{n}" for n in ordered)


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
