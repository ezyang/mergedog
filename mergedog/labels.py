"""PR autolabeling via Claude.

Fetches available labels from GitHub (with a file-based cache), asks Claude
to classify the PR, and applies the suggested labels. Runs at most once per
PR: if the PR already carries a ``release notes:`` or ``topic:`` label, the
labeling step is skipped entirely.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time

from mergedog import github
from mergedog.log import log
from mergedog.paths import REPO_SLUG, ROOT
from mergedog.taint import taint, untaint

_LABEL_CACHE_TTL_SEC = 24 * 60 * 60
_LABEL_CACHE_PATH = ROOT / "label-cache.json"

_RELEASE_NOTES_PREFIX = "release notes:"
_TOPIC_PREFIX = "topic:"
_TOPIC_NOT_USER_FACING = "topic: not user facing"


def _get_cached_labels() -> list[dict] | None:
    if not _LABEL_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(_LABEL_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - data.get("fetched_at", 0) > _LABEL_CACHE_TTL_SEC:
        return None
    return data.get("labels")


def _fetch_and_cache_labels() -> list[dict]:
    labels = github.get_repo_labels()
    _LABEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LABEL_CACHE_PATH.write_text(json.dumps({
        "fetched_at": time.time(),
        "labels": labels,
    }))
    return labels


def _get_relevant_labels() -> tuple[list[dict], list[dict], list[dict]]:
    """Return ``(ciflow, release_notes, topic)`` label lists."""
    labels = _get_cached_labels()
    if labels is None:
        labels = _fetch_and_cache_labels()
    ciflow = [l for l in labels if l["name"].startswith("ciflow/")]
    release_notes = [l for l in labels if l["name"].startswith(_RELEASE_NOTES_PREFIX)]
    topic = [l for l in labels if l["name"].startswith(_TOPIC_PREFIX)]
    return ciflow, release_notes, topic


def _pr_needs_autolabel(pr_data: dict) -> bool:
    existing = {l.get("name", "") for l in pr_data.get("labels", [])}
    if _TOPIC_NOT_USER_FACING in existing:
        return False
    has_release_notes = any(n.startswith(_RELEASE_NOTES_PREFIX) for n in existing)
    has_topic = any(n.startswith(_TOPIC_PREFIX) for n in existing)
    return not (has_release_notes or has_topic)


_LABEL_PROMPT = """\
You are a PR classifier for pytorch/pytorch. Given a PR's title, body, and \
changed files, decide which labels to apply.

There are two independent label groups to consider:

1. **Release notes / topic labels** (exactly one required):
   These classify the PR for release note generation.
   - If the PR is NOT user-facing (internal refactoring, CI fixes, test-only \
     changes, build system changes, documentation-only, etc.), apply ONLY \
     ``topic: not user facing``. Do NOT also add a ``release notes:`` label.
   - If the PR IS user-facing, apply exactly one ``release notes: <category>`` \
     label AND exactly one ``topic: <category>`` label.

2. **ciflow labels** (zero or more):
   These trigger additional CI testing beyond the default. Only add ciflow \
   labels when the PR's changes clearly affect a specific platform or \
   subsystem that warrants targeted testing. Most PRs need zero ciflow labels. \
   Be conservative -- unnecessary ciflow labels waste CI resources.

Available labels:

{labels_section}

PR information:

  Title: {title}
  URL: {url}

  Body:
{body}

  Changed files:
{changed_files}

Respond with ONLY a JSON array of label name strings to apply. No explanation, \
no markdown fencing, just the JSON array. Example: ["topic: not user facing"]
"""


def _format_labels_section(
    ciflow: list[dict], release_notes: list[dict], topic: list[dict]
) -> str:
    sections: list[str] = []
    for heading, group in [
        ("Release notes labels", release_notes),
        ("Topic labels", topic),
        ("ciflow labels", ciflow),
    ]:
        lines = [f"  {heading}:"]
        for l in sorted(group, key=lambda x: x["name"]):
            desc = l.get("description") or ""
            if desc:
                lines.append(f"    - {l['name']}: {desc}")
            else:
                lines.append(f"    - {l['name']}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _get_changed_files(pr: int) -> list[str]:
    try:
        proc = github._gh(
            ["pr", "diff", str(pr), "--repo", REPO_SLUG, "--name-only"],
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return [taint(f, "pr_diff") for f in proc.stdout.strip().splitlines()]
    except Exception:
        pass
    return []


def _invoke_claude_for_labels(prompt: str) -> list[str]:
    cmd = [
        "claude",
        "-p", prompt,
        "--model", "haiku",
        "--output-format", "text",
    ]
    redacted = [c if c is not prompt else "<prompt>" for c in cmd]
    log("autolabel: $ " + " ".join(shlex.quote(c) for c in redacted))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        log("autolabel: claude timed out")
        return []
    if proc.returncode != 0:
        log(f"autolabel: claude exited {proc.returncode}")
        return []
    text = proc.stdout.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log(f"autolabel: could not parse claude response: {text[:200]}")
        return []
    try:
        labels = json.loads(match.group(0))
    except json.JSONDecodeError:
        log(f"autolabel: invalid JSON in claude response: {text[:200]}")
        return []
    if not isinstance(labels, list):
        return []
    return [l for l in labels if isinstance(l, str)]


def _validate_labels(
    suggested: list[str],
    ciflow: list[dict],
    release_notes: list[dict],
    topic: list[dict],
) -> list[str]:
    """Filter suggested labels to only those that actually exist."""
    valid_names = {l["name"] for l in ciflow + release_notes + topic}
    result = [l for l in suggested if l in valid_names]
    has_not_user_facing = _TOPIC_NOT_USER_FACING in result
    if has_not_user_facing:
        result = [
            l for l in result if not l.startswith(_RELEASE_NOTES_PREFIX)
        ]
    return result


def autolabel_if_needed(pr: int, pr_data: dict) -> None:
    if not _pr_needs_autolabel(pr_data):
        log("autolabel: PR already has release notes / topic labels; skipping")
        return

    ciflow, release_notes, topic = _get_relevant_labels()
    changed_files = _get_changed_files(pr)
    body = pr_data.get("body", "") or ""
    if len(body) > 3000:
        body = body[:3000] + "\n... (truncated)"
    files_str = "\n".join(f"    {f}" for f in changed_files[:200])
    if len(changed_files) > 200:
        files_str += f"\n    ... and {len(changed_files) - 200} more files"

    # Declassify: output is constrained to validated label names, limiting
    # blast radius of any injection in title/body/filenames.
    prompt = _LABEL_PROMPT.format(
        labels_section=_format_labels_section(ciflow, release_notes, topic),
        title=untaint(pr_data.get("title", "")),
        url=pr_data.get("url", ""),
        body=untaint(body),
        changed_files=untaint(files_str) if files_str else "    (unavailable)",
    )

    suggested = _invoke_claude_for_labels(prompt)
    validated = _validate_labels(suggested, ciflow, release_notes, topic)

    if not validated:
        log("autolabel: no valid labels suggested")
        return

    existing = {l.get("name", "") for l in pr_data.get("labels", [])}
    for label in validated:
        if label not in existing:
            log(f"autolabel: adding {label!r}")
            try:
                github.add_label(pr, label)
            except Exception as e:
                log(f"autolabel: failed to add {label!r}: {e}")
