"""Classify CI log lines using dr.ci's ruleset.toml.

This is a Python reimplementation of the core matching logic from
pytorch/test-infra's Rust log classifier.  The ruleset.toml is vendored
from ``aws/lambda/log-classifier/ruleset.toml`` in that repo.

Algorithm (matching the Rust implementation):
  - Rules are ordered by priority (first in file = highest).
  - Each rule is evaluated against the log in *reverse line order*
    (last matching line wins for a given rule).
  - The highest-priority rule that matches anywhere wins overall.
"""
from __future__ import annotations

import re
import signal
import threading
from dataclasses import dataclass
from pathlib import Path

try:
    import re2
except ImportError:
    re2 = None

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


_RULESET_PATH = Path(__file__).parent / "ruleset.toml"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: object
    priority: int


@dataclass(frozen=True)
class Match:
    rule_name: str
    line: str
    line_num: int  # 0-indexed
    captures: tuple[str, ...]
    priority: int


def _load_rules(path: Path = _RULESET_PATH) -> list[Rule]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    rules: list[Rule] = []
    for i, entry in enumerate(data.get("rule", [])):
        if re2 is not None:
            try:
                pat = re2.compile(entry["pattern"])
            except re2.error:
                try:
                    pat = re.compile(entry["pattern"])
                except re.error:
                    continue
        else:
            try:
                pat = re.compile(entry["pattern"])
            except re.error:
                continue
        rules.append(Rule(name=entry["name"], pattern=pat, priority=i))
    return rules


_RULES: list[Rule] | None = None


def _get_rules() -> list[Rule]:
    global _RULES
    if _RULES is None:
        _RULES = _load_rules()
    return _RULES


_CLASSIFY_TIMEOUT_SEC = 5


class _RegexTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: object) -> None:
    raise _RegexTimeout


def _match_rule(rule: Rule, lines: list[str]) -> Match | None:
    """Scan lines in reverse for the last line matching ``rule``."""
    for line_num in range(len(lines) - 1, -1, -1):
        m = rule.pattern.search(lines[line_num])
        if m:
            captures = m.groups() if m.groups() else (m.group(0),)
            return Match(
                rule_name=rule.name,
                line=lines[line_num],
                line_num=line_num,
                captures=captures,
                priority=rule.priority,
            )
    return None


def classify(lines: list[str]) -> Match | None:
    """Find the highest-priority rule match, scanning lines in reverse."""
    rules = _get_rules()
    best: Match | None = None
    use_alarm = threading.current_thread() is threading.main_thread()
    if not use_alarm:
        return _classify_without_alarm(rules, lines)
    try:
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    except ValueError:
        return _classify_without_alarm(rules, lines)
    try:
        signal.alarm(_CLASSIFY_TIMEOUT_SEC)
        for rule in rules:
            if best is not None and rule.priority >= best.priority:
                continue
            try:
                candidate = _match_rule(rule, lines)
                if candidate is not None:
                    best = candidate
            except _RegexTimeout:
                signal.alarm(_CLASSIFY_TIMEOUT_SEC)
                continue
        signal.alarm(0)
    except _RegexTimeout:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)
    return best


def _classify_without_alarm(rules: list[Rule], lines: list[str]) -> Match | None:
    best: Match | None = None
    for rule in rules:
        if best is not None and rule.priority >= best.priority:
            continue
        candidate = _match_rule(rule, lines)
        if candidate is not None:
            best = candidate
    return best
