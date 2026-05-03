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


def get_pr_comments(pr: int) -> list[dict]:
    """Return issue-tab comments on the PR, oldest first.

    Each dict has ``author`` (login string), ``body``, ``created_at``.
    Review comments (per-line on the diff) are intentionally not included
    -- they're tied to specific commits and don't fit the sidecar format.
    """
    data = _gh_json(
        ["pr", "view", str(pr), "--repo", REPO, "--json", "comments"]
    )
    out: list[dict] = []
    for c in data.get("comments", []) or []:
        author = (c.get("author") or {}).get("login") or "?"
        out.append(
            {
                "author": author,
                "body": c.get("body", "") or "",
                "created_at": c.get("createdAt", "") or "",
            }
        )
    return out


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


def get_pr_checks_all(pr: int) -> list[dict]:
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
        ]
    )


def add_label(pr: int, label: str) -> None:
    _gh(
        ["pr", "edit", str(pr), "--repo", REPO, "--add-label", label],
        loud=True,
    )


def post_pr_comment(pr: int, body: str) -> None:
    """Post a comment on the PR. Body is passed via stdin to dodge argv limits."""
    subprocess.run(
        ["gh", "pr", "comment", str(pr), "--repo", REPO, "--body-file", "-"],
        input=body,
        text=True,
        check=True,
    )


def has_label(pr_data: dict, label: str) -> bool:
    return any(l.get("name") == label for l in pr_data.get("labels", []))


CI_SEV_LABEL = "ci: sev"


def list_active_ci_sevs() -> list[dict]:
    """Return open GitHub issues tagged ``ci: sev``.

    pytorch/pytorch's dev-infra team puts this label on an issue when
    trunk CI is broken in a way that affects everyone. mergedog parks
    itself on the presence of any such issue so we don't pile new
    pushes onto already-broken CI.
    """
    return _gh_json(
        [
            "issue",
            "list",
            "--repo",
            REPO,
            "--state",
            "open",
            "--label",
            CI_SEV_LABEL,
            "--json",
            "number,title,url",
            "--limit",
            "10",
        ]
    )


def has_mergedog_handoff_comment(pr: int) -> bool:
    """True if any existing PR comment carries the mergedog handoff marker.

    Lets the shepherd be restart-safe: after a ctrl-c & rerun, we won't
    re-post the same handoff. The marker is the ``<!-- mergedog:handoff ...``
    HTML comment embedded by ``_format_handoff_comment``.
    """
    return latest_mergedog_handoff_iso(pr) is not None


def latest_mergedog_handoff_iso(pr: int) -> str | None:
    """``created_at`` of the most recent mergedog handoff comment, or None.

    Used so that a restart picks up where the previous shepherd left off:
    the post-handoff watch loop scopes "what counts as new pytorchmergebot
    activity" to comments newer than this timestamp, instead of newer than
    ``now`` (which would miss a merge-failed reply that happened before
    we restarted).
    """
    matches = [
        c.get("created_at") or ""
        for c in get_pr_comments(pr)
        if "<!-- mergedog:handoff" in (c.get("body") or "")
    ]
    return max(matches) if matches else None


def get_commit_subject(sha: str) -> str:
    data = _gh_json(["api", f"repos/{REPO}/commits/{sha}"])
    msg = data.get("commit", {}).get("message", "")
    return msg.splitlines()[0] if msg else ""


# Author associations that we treat as "real maintainer" for review-approval
# trust. Anyone outside this set can leave an APPROVED review on GitHub but
# their approval does not seed our trust DB. See:
# https://docs.github.com/en/graphql/reference/enums#commentauthorassociation
_TRUSTED_ASSOCIATIONS = {"MEMBER", "COLLABORATOR", "OWNER"}


def get_pr_review_audit(pr: int) -> dict:
    """Return enough review state to decide who, if anyone, approved this PR.

    Uses GraphQL because:
    - ``reviewDecision`` is the authoritative aggregate (APPROVED /
      CHANGES_REQUESTED / REVIEW_REQUIRED) and already accounts for
      dismissals and per-user supersession.
    - ``latestOpinionatedReviews`` collapses each reviewer's history to
      their single most recent non-comment review, which is the only one
      that "counts" for that user.

    Returns ``{"decision": str | None, "reviews": list[dict]}``. Each review
    dict has: ``login``, ``association``, ``state``, ``commit_id``,
    ``submitted_at``.
    """
    owner, name = REPO.split("/", 1)
    query = (
        "query($owner:String!, $name:String!, $pr:Int!) {"
        "  repository(owner:$owner, name:$name) {"
        "    pullRequest(number:$pr) {"
        "      reviewDecision"
        "      latestOpinionatedReviews(first:50) {"
        "        nodes {"
        "          author { login }"
        "          authorAssociation"
        "          state"
        "          commit { oid }"
        "          submittedAt"
        "        }"
        "      }"
        "    }"
        "  }"
        "}"
    )
    proc = _gh(
        [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"pr={pr}",
        ]
    )
    data = json.loads(proc.stdout)
    pr_node = (
        data.get("data", {}).get("repository", {}).get("pullRequest", {}) or {}
    )
    nodes = (pr_node.get("latestOpinionatedReviews") or {}).get("nodes") or []
    reviews = []
    for n in nodes:
        author = (n.get("author") or {}).get("login")
        if not author:
            continue  # ghost / deleted user; skip
        reviews.append(
            {
                "login": author,
                "association": n.get("authorAssociation"),
                "state": n.get("state"),
                "commit_id": ((n.get("commit") or {}).get("oid")),
                "submitted_at": n.get("submittedAt"),
            }
        )
    return {"decision": pr_node.get("reviewDecision"), "reviews": reviews}


def is_trusted_association(association: str | None) -> bool:
    return (association or "").upper() in _TRUSTED_ASSOCIATIONS


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
