"""Thin wrappers over the ``gh`` CLI.

We deliberately keep this layer small: every call shells out to ``gh`` and
parses JSON. No third-party HTTP client.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

from mergedog.paths import REPO_SLUG
from mergedog.process import run

REPO = REPO_SLUG


def _gh(
    args: list[str], *, check: bool = True, loud: bool = False
) -> subprocess.CompletedProcess[str]:
    return run(["gh", *args], check=check, loud=loud)


def _gh_json(args: list[str]) -> Any:
    return json.loads(_gh(args).stdout)


def _gh_json_lenient(args: list[str], allowed_exit_codes: tuple[int, ...]) -> Any:
    """Like ``_gh_json`` but tolerates specific non-zero exit codes.

    ``gh pr checks`` exits 8 when checks are still pending; that's not an
    error from our perspective, the JSON is still on stdout.
    """
    proc = _gh(args, check=False)
    if proc.returncode != 0 and proc.returncode not in allowed_exit_codes:
        proc.check_returncode()
    return json.loads(proc.stdout)


def _gh_pr_checks_json(args: list[str]) -> list[dict]:
    """Run ``gh pr checks --json ...`` and tolerate the no-checks case.

    Right after a push, GitHub may not yet have registered any check runs
    for the new commit; ``gh pr checks`` then exits 1 with a stderr message
    like "no checks reported" and empty stdout. We treat that as "no checks
    yet -> pending". Exit 8 means "checks pending", which is also fine.
    """
    proc = _gh(args, check=False)
    if proc.returncode == 0 or proc.returncode == 8:
        return json.loads(proc.stdout)
    if proc.returncode == 1 and (
        not proc.stdout.strip() or "no checks" in (proc.stderr or "").lower()
    ):
        return []
    proc.check_returncode()
    return []  # unreachable


def get_pr(pr: int) -> dict:
    return _gh_json(
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            REPO,
            "--json",
            ",".join(
                [
                    "number",
                    "title",
                    "body",
                    "state",
                    "isDraft",
                    "headRefName",
                    "headRefOid",
                    "headRepository",
                    "headRepositoryOwner",
                    "labels",
                    "maintainerCanModify",
                    "url",
                ]
            ),
        ]
    )


def get_pr_head_sha(pr: int) -> str:
    data = _gh_json(
        ["pr", "view", str(pr), "--repo", REPO, "--json", "headRefOid"]
    )
    return data["headRefOid"]


def list_workflow_runs_for_sha(sha: str) -> list[dict]:
    data = _gh_json(
        [
            "api",
            f"repos/{REPO}/actions/runs?head_sha={sha}&per_page=100",
        ]
    )
    return data.get("workflow_runs", [])


def approve_workflow_run(run_id: int | str) -> tuple[bool, str]:
    """Approve a workflow run that is waiting for first-time-contributor approval.

    Returns ``(ok, message)``. ``ok`` is True on HTTP 2xx; ``message`` is
    a short stderr-derived explanation when ``ok`` is False (so we can log
    what GitHub said no to).
    """
    proc = _gh(
        [
            "api",
            "-X",
            "POST",
            f"repos/{REPO}/actions/runs/{run_id}/approve",
        ],
        check=False,
        loud=True,
    )
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    msg = err[-1] if err else f"exit {proc.returncode}"
    return False, msg


def get_pr_checks(pr: int) -> list[dict]:
    """Return the latest-commit checks for a PR.

    ``gh pr checks --json`` already filters to the head commit.
    """
    return _gh_pr_checks_json(
        [
            "pr",
            "checks",
            str(pr),
            "--repo",
            REPO,
            "--json",
            "name,state,workflow,link,bucket",
            "--required",  # avoid drowning in optional informational checks
        ]
    )


def get_pr_checks_all(pr: int) -> list[dict]:
    return _gh_pr_checks_json(
        [
            "pr",
            "checks",
            str(pr),
            "--repo",
            REPO,
            "--json",
            "name,state,workflow,link,bucket",
        ]
    )


def add_label(pr: int, label: str) -> None:
    _gh(
        ["pr", "edit", str(pr), "--repo", REPO, "--add-label", label],
        loud=True,
    )


def has_label(pr_data: dict, label: str) -> bool:
    return any(l.get("name") == label for l in pr_data.get("labels", []))


def get_commit_subject(sha: str) -> str:
    data = _gh_json(["api", f"repos/{REPO}/commits/{sha}"])
    msg = data.get("commit", {}).get("message", "")
    return msg.splitlines()[0] if msg else ""


def get_failed_job_logs(pr: int, max_jobs: int = 8, max_chars: int = 30000) -> list[tuple[str, str]]:
    """Return ``(name, log_excerpt)`` pairs for failing jobs on the PR head."""
    checks = get_pr_checks_all(pr)
    failed = [
        c
        for c in checks
        if c.get("bucket") in ("fail", "cancel")
        or c.get("state") in ("FAILURE", "STARTUP_FAILURE", "TIMED_OUT")
    ]
    out: list[tuple[str, str]] = []
    for c in failed[:max_jobs]:
        link = c.get("link") or ""
        run_id, job_id = _parse_run_link(link)
        if run_id is None:
            continue
        args = ["run", "view", str(run_id), "--repo", REPO, "--log-failed"]
        if job_id is not None:
            args += ["--job", str(job_id)]
        proc = _gh(args, check=False)
        text = proc.stdout or proc.stderr or "<no log available>"
        if len(text) > max_chars:
            text = text[-max_chars:]
        out.append((c.get("name", "<unknown>"), text))
    return out


def _parse_run_link(link: str) -> tuple[int | None, int | None]:
    """Parse ``.../actions/runs/<run_id>[/job/<job_id>]`` out of a check link."""
    parts = link.rstrip("/").split("/")
    run_id: int | None = None
    job_id: int | None = None
    try:
        if "runs" in parts:
            i = parts.index("runs")
            run_id = int(parts[i + 1])
        if "job" in parts:
            j = parts.index("job")
            job_id = int(parts[j + 1])
    except (ValueError, IndexError):
        pass
    return run_id, job_id


def evaluate_checks(checks: list[dict]) -> str:
    """Reduce a list of checks to a single status: pending, failed, passed.

    We wait for every check to complete before declaring failure, so that
    when we hand the failed-log bundle to claude it has the full picture.
    """
    if not checks:
        return "pending"
    pending_buckets = {"pending", None}
    fail_buckets = {"fail", "cancel"}
    is_pending = any(c.get("bucket") in pending_buckets for c in checks)
    if is_pending:
        return "pending"
    is_failed = any(c.get("bucket") in fail_buckets for c in checks)
    if is_failed:
        return "failed"
    return "passed"
