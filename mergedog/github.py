"""Thin wrappers over the ``gh`` CLI.

We deliberately keep this layer small: every call shells out to ``gh`` and
parses JSON. No third-party HTTP client.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any

from mergedog.log import log
from mergedog.paths import CI_LOGS_DIR, REPO_SLUG
from mergedog.process import run
from mergedog.project import get_project_policy
from mergedog.sanitize import sanitize_untrusted_text
from mergedog.taint import taint, taint_dict

REPO = REPO_SLUG
PROJECT = get_project_policy()

_GH_TRANSIENT_CODES = ("502", "504", "503")
_GH_TRANSIENT_MESSAGES = (
    "error connecting to api.github.com",
    "connection reset",
    "connection refused",
    "connection timed out",
    "i/o timeout",
    "tls handshake timeout",
    "temporary failure",
    "unexpected eof",
)
_GH_STARTUP_CRASH_MESSAGES = (
    "fatal error: lfstack.push",
    "runtime: lfstack.push invalid packing",
)
_GH_FALLBACK_PATHS = ("/usr/local/bin/gh",)
_GH_MAX_RETRIES = 3
_GH_RETRY_DELAY = 5  # seconds
_GH_COMMAND = ["gh"]


def _is_gh_startup_crash(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    stderr_lower = (proc.stderr or "").lower()
    return any(msg in stderr_lower for msg in _GH_STARTUP_CRASH_MESSAGES)


def _is_transient_gh_failure(proc: subprocess.CompletedProcess[str]) -> bool:
    """True if the gh CLI failed due to a transient HTTP error."""
    if proc.returncode == 0:
        return False
    if _is_gh_startup_crash(proc):
        return True
    stderr = proc.stderr or ""
    stderr_lower = stderr.lower()
    return any(code in stderr for code in _GH_TRANSIENT_CODES) or any(
        msg in stderr_lower for msg in _GH_TRANSIENT_MESSAGES
    )


def _candidate_gh_paths(current: str) -> list[str]:
    current_path = shutil.which(current) or current
    current_real = os.path.realpath(current_path)
    seen: set[str] = set()
    paths: list[str] = []

    def add(path: str) -> None:
        real = os.path.realpath(path)
        if real == current_real or path in seen or real in seen:
            return
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            return
        seen.add(path)
        seen.add(real)
        paths.append(path)

    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory:
            add(os.path.join(directory, "gh"))
    for path in _GH_FALLBACK_PATHS:
        add(path)
    return paths


def _find_working_gh_executable(current: str) -> str | None:
    for candidate in _candidate_gh_paths(current):
        try:
            proc = subprocess.run(
                [candidate, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            return candidate
    return None


def _raise_gh_failure(
    proc: subprocess.CompletedProcess[str], args: list[str]
) -> None:
    err = (proc.stderr or "").rstrip()
    if err:
        log(f"  ! gh {' '.join(args[:3])}")
        for line in err.splitlines():
            log(f"    stderr: {line}")
    proc.check_returncode()


def _gh(
    args: list[str],
    *,
    check: bool = True,
    loud: bool = False,
    log_context: str | None = None,
) -> subprocess.CompletedProcess[str]:
    global _GH_COMMAND
    for attempt in range(_GH_MAX_RETRIES):
        proc = run([*_GH_COMMAND, *args], check=False, loud=(loud and attempt == 0))
        if _is_gh_startup_crash(proc):
            replacement = _find_working_gh_executable(_GH_COMMAND[0])
            if replacement is not None:
                current = shutil.which(_GH_COMMAND[0]) or _GH_COMMAND[0]
                log(
                    f"  ! gh startup crash from {current}; "
                    f"retrying with {replacement}"
                )
                _GH_COMMAND = [replacement]
                proc = run([*_GH_COMMAND, *args], check=False, loud=False)
        if proc.returncode == 0 or not _is_transient_gh_failure(proc):
            if attempt > 0 and proc.returncode == 0:
                suffix = f" while {log_context}" if log_context else ""
                log(f"  gh recovered after transient failure{suffix}")
            if check and proc.returncode != 0:
                _raise_gh_failure(proc, args)
            return proc
        if attempt + 1 == _GH_MAX_RETRIES:
            log(f"  ! gh transient failure after {_GH_MAX_RETRIES} attempts")
            if check:
                _raise_gh_failure(proc, args)
            return proc
        log(
            f"  ! gh transient failure (attempt {attempt + 1}/{_GH_MAX_RETRIES}), "
            f"retrying in {_GH_RETRY_DELAY}s"
        )
        time.sleep(_GH_RETRY_DELAY)
    raise AssertionError("unreachable")


def _gh_json(args: list[str], *, log_context: str | None = None) -> Any:
    return json.loads(_gh(args, log_context=log_context).stdout)


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


def get_pr(pr: int, *, log_context: str | None = None) -> dict:
    data = _gh_json(
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
                    "author",
                    "headRefName",
                    "headRefOid",
                    "headRepository",
                    "headRepositoryOwner",
                    "labels",
                    "maintainerCanModify",
                    "mergeStateStatus",
                    "reviewDecision",
                    "url",
                ]
            ),
        ],
        log_context=log_context,
    )
    return taint_dict(data, "pr_metadata", ["title", "body", "headRefName"])


def get_pr_merge_commit_sha(pr: int) -> str | None:
    data = _gh_json(
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            REPO,
            "--json",
            "state,mergeCommit",
        ],
    )
    if data.get("state") != "MERGED":
        return None
    merge_commit = data.get("mergeCommit") or {}
    oid = merge_commit.get("oid")
    return oid if isinstance(oid, str) and oid else None


def viewer_login() -> str:
    """The GitHub login of the user the local ``gh`` CLI is authenticated as."""
    return _gh(["api", "user", "--jq", ".login"]).stdout.strip()


def is_self_pr(pr_data: dict, viewer: str) -> bool:
    """True if ``viewer`` authored ``pr_data``.

    Author lookup tolerates both ``{"login": "..."}`` (REST/gh shape) and
    raw-string authors; we lowercase to match GitHub's case-insensitive
    login semantics.
    """
    author = pr_data.get("author") or {}
    if isinstance(author, dict):
        login = author.get("login") or ""
    else:
        login = str(author)
    return bool(login) and login.lower() == viewer.lower()


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
                "author": taint(author, "pr_comment"),
                "body": taint(c.get("body", "") or "", "pr_comment"),
                "created_at": c.get("createdAt", "") or "",
            }
        )
    return out


# GitHub login of the dr. ci bot. A GitHub App, so the login may surface as
# either ``pytorch-bot`` or ``pytorch-bot[bot]`` depending on the API path.
# dr. ci re-summarizes the head commit's checks as a structured comment with
# the salient one-line error from each failing job; it is bot-generated from
# CI metadata (not user-controlled) and so is treated as trusted input by
# the agent prompt.
_DRCI_LOGINS = PROJECT.drci_logins
_DRCI_TRAILER = PROJECT.drci_trailer
# dr. ci's body opens with "As of commit <40-hex-sha> with merge base ...". We
# parse this to confirm the summary describes the current head, since dr. ci
# only refreshes every ~15 minutes -- a stale summary on a freshly-pushed head
# can claim "no failures" while CI is actually red.
_DRCI_COMMIT_RE = re.compile(r"As of commit ([0-9a-f]{7,40})")


def latest_drci_summary(
    comments: list[dict], head_sha: str | None = None
) -> str | None:
    """Return the most recent dr. ci comment body, or None.

    Filters for both author login (``pytorch-bot``) and the dr. ci trailer
    so we don't accidentally treat a different pytorch-bot comment as a
    failure summary. When ``head_sha`` is provided, also requires the body
    to reference that SHA -- otherwise the summary is dropped (treated as
    stale) so it can't override the live check list.

    The author + trailer filter is the trust boundary — the returned
    string is declassified (untainted) because it is bot-generated from
    CI metadata, not user-authored.
    """
    from mergedog.taint import untaint

    matches = [
        c
        for c in comments
        if _DRCI_TRAILER is not None
        and (c.get("author") or "") in _DRCI_LOGINS
        and _DRCI_TRAILER in (c.get("body") or "")
    ]
    if not matches:
        return None
    body = matches[-1].get("body") or None
    if body is None or head_sha is None:
        if body is None:
            return None
        return sanitize_untrusted_text(untaint(body))
    m = _DRCI_COMMIT_RE.search(body)
    if m is None:
        return None
    return (
        sanitize_untrusted_text(untaint(body))
        if head_sha.startswith(m.group(1))
        else None
    )


def get_pr_head_sha(pr: int) -> str:
    data = _gh_json(
        ["pr", "view", str(pr), "--repo", REPO, "--json", "headRefOid"]
    )
    return data["headRefOid"]


def get_pr_labels(pr: int) -> list[str]:
    """Return the names of labels currently applied to the PR."""
    data = _gh_json(
        ["pr", "view", str(pr), "--repo", REPO, "--json", "labels"]
    )
    return [l.get("name", "") for l in data.get("labels", []) or []]


def get_pr_status_fields(pr: int) -> tuple[list[str], str | None]:
    """Return ``(labels, reviewDecision)`` in a single ``gh pr view`` call.

    Used by the shepherd's per-iteration log-prefix refresh, where we
    need both pieces (labels for [MERGING], reviewDecision for [APPROVED])
    and don't want to pay for two separate round trips.
    """
    data = _gh_json(
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            REPO,
            "--json",
            "labels,reviewDecision",
        ]
    )
    labels = [l.get("name", "") for l in data.get("labels", []) or []]
    return labels, data.get("reviewDecision") or None


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


def rerun_failed_jobs(run_id: int | str) -> tuple[bool, str]:
    """Re-run only the failed jobs in a workflow run via ``gh run rerun --failed``.

    Used by the intervention path when a transient upstream failure
    (e.g. GitHub GraphQL 5xx) calls for a retry rather than a fix.
    Returns ``(ok, message)``; on failure ``message`` is the last line
    of stderr/stdout for the log.
    """
    proc = _gh(
        [
            "run",
            "rerun",
            str(run_id),
            "--repo",
            REPO,
            "--failed",
        ],
        check=False,
        loud=True,
    )
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    msg = err[-1] if err else f"exit {proc.returncode}"
    return False, msg


def run_id_for_check(check: dict) -> int | None:
    """Extract the workflow run id from a check dict's ``link`` field.

    Returns None if the link is missing or doesn't carry a run id (e.g.
    status-only checks like Dr. CI).
    """
    run_id, _ = _parse_run_link(check.get("link") or "")
    return run_id


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
            "name,state,workflow,link,bucket,completedAt",
        ]
    )


def add_label(pr: int, label: str, *, loud: bool = True) -> None:
    _gh(
        ["pr", "edit", str(pr), "--repo", REPO, "--add-label", label],
        loud=loud,
    )


def remove_label(pr: int, label: str) -> None:
    _gh(
        ["pr", "edit", str(pr), "--repo", REPO, "--remove-label", label],
        loud=True,
    )


def post_pr_comment(pr: int, body: str) -> None:
    """Post a comment on the PR. Body is passed via stdin to dodge argv limits."""
    body = sanitize_untrusted_text(body)
    subprocess.run(
        ["gh", "pr", "comment", str(pr), "--repo", REPO, "--body-file", "-"],
        input=body,
        text=True,
        check=True,
    )


def has_label(pr_data: dict, label: str | None) -> bool:
    if label is None:
        return False
    return any(l.get("name") == label for l in pr_data.get("labels", []))


CI_SEV_LABEL = PROJECT.ci_sev_label
# pytorchmergebot stamps this label while a PR is actively being merged.
MERGING_LABEL = PROJECT.merging_label


def list_active_ci_sevs() -> list[dict]:
    """Return open GitHub issues tagged ``ci: sev``.

    pytorch/pytorch's dev-infra team puts this label on an issue when
    trunk CI is broken in a way that affects everyone. mergedog parks
    itself on the presence of any such issue so we don't pile new
    pushes onto already-broken CI.
    """
    if CI_SEV_LABEL is None:
        return []
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


_HANDOFF_MARKER_RE = re.compile(
    r"<!--\s*mergedog:handoff(?:\s+head=([^\s>]+))?\s*-->"
)


def _mergedog_handoff_marker(body: str) -> tuple[bool, str | None]:
    match = _HANDOFF_MARKER_RE.search(body)
    if not match:
        return False, None
    return True, match.group(1)


def has_mergedog_handoff_comment(
    pr: int, *, head_sha: str | None = None
) -> bool:
    """True if an existing PR comment carries the mergedog handoff marker.

    Lets the shepherd be restart-safe: after a ctrl-c & rerun, we won't
    re-post the same handoff. When ``head_sha`` is supplied, require the
    marker to describe that exact PR head; an older handoff before a rebase
    is stale and should not suppress a new comment.
    """
    for c in get_pr_comments(pr):
        body = c.get("body") or ""
        has_marker, marker_head = _mergedog_handoff_marker(body)
        if not has_marker:
            continue
        if head_sha is None or marker_head == head_sha:
            return True
    return False


def latest_mergedog_handoff_iso(pr: int) -> str | None:
    """``created_at`` of the most recent mergedog handoff comment, or None.

    Used so that a restart picks up where the previous shepherd left off:
    the post-handoff watch loop scopes "what counts as new pytorchmergebot
    activity" to comments newer than this timestamp, instead of newer than
    ``now`` (which would miss a merge-failed reply that happened before
    we restarted).
    """
    return latest_mergedog_handoff_iso_from_comments(get_pr_comments(pr))


def latest_mergedog_handoff_iso_from_comments(comments: list[dict]) -> str | None:
    """Return the latest mergedog handoff timestamp from an existing comment list."""
    matches = [
        c.get("created_at") or ""
        for c in comments
        if _mergedog_handoff_marker(c.get("body") or "")[0]
    ]
    return max(matches) if matches else None


def get_repo_labels(per_page: int = 100) -> list[dict]:
    """Return all labels defined on the repo, with name and description."""
    labels: list[dict] = []
    page = 1
    while True:
        data = _gh_json(
            [
                "api",
                f"repos/{REPO}/labels?per_page={per_page}&page={page}",
            ]
        )
        if not data:
            break
        for l in data:
            labels.append(
                {"name": l.get("name", ""), "description": l.get("description") or ""}
            )
        if len(data) < per_page:
            break
        page += 1
    return labels


def get_commit_subject(sha: str) -> str:
    data = _gh_json(["api", f"repos/{REPO}/commits/{sha}"])
    msg = data.get("commit", {}).get("message", "")
    subject = msg.splitlines()[0] if msg else ""
    return taint(subject, "commit_message")


def get_commit_actor_logins(sha: str) -> tuple[str | None, str | None]:
    """Return ``(author_login, committer_login)`` for a commit."""
    data = _gh_json(["api", f"repos/{REPO}/commits/{sha}"])
    author = data.get("author") or {}
    committer = data.get("committer") or {}
    return author.get("login"), committer.get("login")


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


# Tokens that mark "the actual failure is somewhere around here" in CI logs.
# Order doesn't matter -- we pick the *last* occurrence across all of them.
# Add new ones here as they show up in HALTed runs; tweaks are cheap.
_LOG_FAILURE_MARKERS = (
    "error:",                   # compiler/linker, plus most lower-level tools
    "FAILED:",                  # ninja
    "ninja: build stopped",     # ninja's terminal line
    "##[error]",                # GitHub Actions error annotation
    "fatal error",              # clang/gcc fatal
    "Traceback (most recent",   # Python tracebacks
    "AssertionError",
    "RuntimeError:",
    "Segmentation fault",
    "FATAL:",
    "Test Failed",
    "FAILED test",              # pytest summary lines
    "short test summary info",  # pytest section header before FAILED lines
    " failed,",                 # pytest "= N failed, M passed =" final line
    " failed =",                # pytest "= N failed =" final line (no passed)
    "= FAILURES =",             # pytest FAILURES section header
)

_PYTEST_FAILURES_RE = re.compile(r"(?m)^=+\s+FAILURES\s+=+\s*$")


def _strip_gh_log_prefix(line: str) -> str:
    """Strip the ``<job>\\tSTEP\\t<timestamp> `` prefix from a ``gh run view --log`` line.

    ``gh`` prefixes every line with the same job name, step name (often
    ``UNKNOWN STEP``), and an RFC3339 timestamp. The prefix is the same on
    every line and eats ~70 chars; stripping it ~doubles the useful content
    that fits in our character budget.
    """
    sp = line.find(" ")
    if sp > 0:
        head = line[:sp]
        # ``gh api repos/.../actions/jobs/<job>/logs`` returns raw lines
        # prefixed by only an RFC3339 timestamp.
        if "T" in head and (head.endswith("Z") or "+" in head):
            return line[sp + 1:]
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return line
    rest = parts[2]
    sp = rest.find(" ")
    if sp <= 0:
        return rest
    head = rest[:sp]
    # RFC3339-ish timestamp: contains 'T' and ends with 'Z' (or has a '+' offset).
    if "T" in head and (head.endswith("Z") or "+" in head):
        return rest[sp + 1:]
    return rest


def _extract_pytest_failures(cleaned: str, max_chars: int) -> str | None:
    """Try to extract the pytest FAILURES section + short summary.

    Pytest prints failures as::

        ============================= FAILURES =============================
        ___ test_name ___
        <traceback>
        ...
        = short test summary info =
        FAILED test/foo.py::test_name - AssertionError: ...
        = N failed, M passed =

    If both delimiters are present, extract from ``= FAILURES =`` through
    end-of-log. This is far more useful than a generic window around an
    arbitrary marker, because it contains the actual tracebacks.
    """
    match = _PYTEST_FAILURES_RE.search(cleaned)
    if match is None:
        return None
    fail_start = match.start()
    section = cleaned[fail_start:]
    if len(section) <= max_chars:
        return "... [head truncated] ...\n" + section if fail_start > 0 else section
    # Section itself is over budget — trim from the front of the section,
    # keeping the tail (short test summary + final summary are at the end).
    return (
        "... [head truncated] ...\n"
        + section[: max_chars // 4]
        + "\n... [middle truncated] ...\n"
        + section[-(max_chars * 3 // 4):]
    )


def _find_anchor(cleaned: str, lines: list[str]) -> int:
    """Find the best character offset to anchor the trimming window on.

    Tries, in order:
      1. dr.ci's ruleset regexes (highest-priority match, last occurrence)
      2. Pytest ``= FAILURES =`` section header
      3. Generic ``_LOG_FAILURE_MARKERS`` (last occurrence of any)

    Returns -1 if nothing matches.
    """
    from mergedog.log_classifier import classify

    match = classify(lines)
    if match is not None:
        # Convert line number to character offset in the joined string.
        offset = 0
        for i, line in enumerate(lines):
            if i == match.line_num:
                return offset
            offset += len(line) + 1  # +1 for \n
        return offset

    best = -1
    for m in _LOG_FAILURE_MARKERS:
        i = cleaned.rfind(m)
        if i > best:
            best = i
    return best


def _trim_log_for_prompt(text: str, max_chars: int) -> str:
    """Trim a CI log to ~``max_chars``, biased toward the actual failure line.

    Strategy:
      1. Strip per-line ``gh`` prefixes (saves ~40-50% of bytes).
      2. If still over budget, try to extract the pytest ``= FAILURES =``
         section which contains actual tracebacks.
      3. Otherwise use dr.ci's ruleset regexes to find the failure line,
         falling back to generic ``_LOG_FAILURE_MARKERS``. Keep a window
         around the anchor (most chars before, fewer after) plus a small
         tail.
      4. If no anchor is found, fall back to head+tail so the agent
         at least sees both ends rather than just post-job cleanup.

    Why bias *before* the anchor: compiler errors typically print the error
    line, then the source location, then a caret pointing at the column.
    Walking back from the anchor reaches the entire diagnostic; walking
    forward mostly reaches "ninja: build stopped" and shutdown noise.
    """
    text = sanitize_untrusted_text(text)
    lines = [_strip_gh_log_prefix(l) for l in text.splitlines()]
    cleaned = "\n".join(lines)
    if len(cleaned) <= max_chars:
        return cleaned
    # Pytest logs: extract the FAILURES section directly.
    pytest_extract = _extract_pytest_failures(cleaned, max_chars)
    if pytest_extract is not None:
        return pytest_extract
    best = _find_anchor(cleaned, lines)
    if best < 0:
        # No anchor; fall back to head+tail so the agent sees both ends.
        half = max_chars // 2
        return (
            cleaned[: max_chars - half]
            + "\n... [middle truncated] ...\n"
            + cleaned[-half:]
        )
    head_marker = "... [head truncated] ...\n"
    tail_marker = "\n... [tail truncated] ...\n"
    tail_budget = min(2000, max_chars // 8)
    body_budget = max_chars - tail_budget - len(head_marker) - len(tail_marker)
    before = int(body_budget * 0.8)
    after = body_budget - before
    start = max(0, best - before)
    end = min(len(cleaned), best + after)
    pieces: list[str] = []
    if start > 0:
        pieces.append(head_marker)
    pieces.append(cleaned[start:end])
    if end < len(cleaned) - tail_budget:
        pieces.append(tail_marker)
        pieces.append(cleaned[-tail_budget:])
    elif end < len(cleaned):
        pieces.append(cleaned[end:])
    return "".join(pieces)


def _fetch_job_log(run_id: int, job_id: int | None) -> str:
    """Fetch log text for a failed job, with fallback for in-progress runs.

    Prefer the per-job logs API when a job id is known. ``gh run view
    --log-failed --job`` can return only a prefix of the failed step log,
    while the job logs API includes the actual pytest failure body.
    """
    if job_id is not None:
        api_proc = _gh(
            ["api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
            check=False,
        )
        if api_proc.returncode == 0 and (api_proc.stdout or "").strip():
            return api_proc.stdout

    args = ["run", "view", str(run_id), "--repo", REPO, "--log-failed"]
    if job_id is not None:
        args += ["--job", str(job_id)]
    proc = _gh(args, check=False)
    if proc.returncode == 0 and (proc.stdout or "").strip():
        return proc.stdout
    return "<no log available>"


def _cache_failed_job_log(
    pr: int, name: str, run_id: int, job_id: int | None, text: str
) -> None:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")[:120]
    if not safe_name:
        safe_name = "unknown"
    job_part = str(job_id) if job_id is not None else "run"
    path = CI_LOGS_DIR / str(pr) / f"{run_id}-{job_part}-{safe_name}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"PR #{pr}: could not cache full log for {name!r}: {e}")


def _fetch_raw_job_log(
    spec: tuple[str, int, int | None], pr: int | None = None
) -> tuple[str, str]:
    name, run_id, job_id = spec
    name = sanitize_untrusted_text(name)
    text = sanitize_untrusted_text(_fetch_job_log(run_id, job_id))
    if pr is not None and text != "<no log available>":
        _cache_failed_job_log(pr, name, run_id, job_id, text)
    return (name, text)


def _trim_failed_job_log(raw: tuple[str, str], max_chars: int) -> tuple[str, str]:
    name, text = raw
    text = _trim_log_for_prompt(text, max_chars)
    return (taint(name, "ci_log"), taint(text, "ci_log"))


def _fetch_failed_job_logs_parallel(
    specs: list[tuple[str, int, int | None]], max_chars: int, pr: int | None = None
) -> list[tuple[str, str]]:
    if len(specs) <= 1:
        return [
            _trim_failed_job_log(_fetch_raw_job_log(spec, pr), max_chars)
            for spec in specs
        ]
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures = [
            ex.submit(_fetch_raw_job_log, spec, pr)
            for spec in specs
        ]
        raw_logs = [future.result() for future in futures]
    return [_trim_failed_job_log(raw, max_chars) for raw in raw_logs]


def get_failed_job_logs(
    pr: int, max_jobs: int = 8, max_chars: int = 30000
) -> list[tuple[str, str]]:
    """Return ``(name, log_excerpt)`` pairs for failing jobs on the PR head."""
    checks = get_pr_checks_all(pr)
    failed = [
        c
        for c in checks
        if c.get("state") in ("FAILURE", "STARTUP_FAILURE", "TIMED_OUT", "ERROR")
    ]
    if failed:
        capped = min(len(failed), max_jobs)
        suffix = f" (capped at {max_jobs})" if len(failed) > max_jobs else ""
        log(
            f"PR #{pr}: fetching failed-job logs for {capped}/"
            f"{len(failed)} failed check(s){suffix}"
        )
    specs: list[tuple[str, int, int | None]] = []
    for c in failed[:max_jobs]:
        name = c.get("name", "<unknown>")
        link = c.get("link") or ""
        run_id, job_id = _parse_run_link(link)
        if run_id is None:
            log(f"PR #{pr}: skipping log fetch for {name!r}; no run id")
            continue
        job_part = f"/job/{job_id}" if job_id is not None else ""
        log(f"PR #{pr}: fetching log for {name!r} from run {run_id}{job_part}")
        specs.append((name, run_id, job_id))
    return _fetch_failed_job_logs_parallel(specs, max_chars, pr)


def get_failed_job_logs_for_runs(
    run_ids: list[int], max_jobs: int = 8, max_chars: int = 30000
) -> list[tuple[str, str]]:
    """Return ``(name, log_excerpt)`` for failing jobs in specific workflow runs.

    Used when the workflow-run API reports conclusion=failure but
    ``gh pr checks`` disagrees (e.g. a re-run made the check-run API
    show the latest passing attempt).
    """
    specs: list[tuple[str, int, int | None]] = []
    for run_id in run_ids:
        jobs_proc = _gh(
            ["api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100"],
            check=False,
        )
        if jobs_proc.returncode != 0:
            continue
        jobs_data = json.loads(jobs_proc.stdout)
        failed_jobs = [
            j
            for j in (jobs_data.get("jobs") or [])
            if j.get("conclusion") == "failure"
        ]
        for j in failed_jobs[:max_jobs - len(specs)]:
            name = j.get("name", f"<run {run_id}>")
            job_id = j.get("id")
            specs.append((name, run_id, job_id))
            if len(specs) >= max_jobs:
                break
        if len(specs) >= max_jobs:
            break
    return _fetch_failed_job_logs_parallel(specs, max_chars)


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

    Declare failure as soon as any check fails, even with others still
    pending. The point is to push a fix immediately so fresh CI starts
    sooner -- waiting for slow unrelated workflows before invoking claude
    can add tens of minutes before any new signal returns. Any failures
    that crystallize after the fix is pushed are caught by the next loop.
    """
    if not checks:
        return "pending"
    fail_buckets = {"fail", "cancel"}
    pending_buckets = {"pending", None}
    is_failed = any(c.get("bucket") in fail_buckets for c in checks)
    if is_failed:
        return "failed"
    is_pending = any(c.get("bucket") in pending_buckets for c in checks)
    if is_pending:
        return "pending"
    return "passed"
