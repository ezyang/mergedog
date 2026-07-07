"""Persistent trust database for a single PR shepherding session.

The trust DB tracks the set of SHAs we consider safe to land:

- The initial PR head SHA at the moment mergedog was started (this is what
  the human implicitly approved by running mergedog).
- Every SHA we ourselves produced and pushed, identifiable by the
  ``[MERGEDOG]`` commit-title prefix.
- Repository automation rebases whose patch-id matches an already trusted
  commit.

If GitHub ever reports a PR head SHA that is not in this set, we halt:
the PR has been touched by someone other than mergedog after approval.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mergedog.paths import atomic_write_text, state_file

SCHEMA_VERSION = 1


@dataclass
class TrustDB:
    pr: int
    trusted_shas: list[str] = field(default_factory=list)
    head_branch: str = ""
    head_repo_clone_url: str = ""
    # ISO 8601 timestamp of the most recent pytorchmergebot "Merge failed"
    # comment we already halted on. Acts as a monotonic floor for the
    # post-handoff watcher so a restart doesn't re-react to a failure we
    # already saw and exited on.
    last_observed_failure_iso: str = ""
    # Body of the most recent pytorchmergebot failure comment, so that a
    # restart can classify it (e.g. merge conflict) and act accordingly.
    last_observed_failure_body: str = ""
    # Check names claude already judged spurious (unrelated to this PR).
    # Persisted so that a restart doesn't re-invoke claude for the same
    # failures. Cleared whenever a fix commit is pushed (fresh CI
    # invalidates prior judgments).
    spurious_check_names: list[str] = field(default_factory=list)
    # How many LLM fix commits this PR has consumed against the
    # --max-fix-commits cap, and the human-approval SHA that budget is
    # scoped to. Persisted so a restart can't silently grant a fresh
    # budget; a new approval baseline (or --reassess) resets it.
    fix_commits_pushed: int = 0
    fix_budget_ack_sha: str = ""
    # Auto-retry comments posted for retryable (infra-flake) merge
    # failures. Persisted so restarts can't grant a fresh budget and
    # spam the merge command during an outage. Reset by --reassess.
    merge_auto_retries: int = 0
    # The /orig commit a ghstack submit is about to publish. Between
    # ghstack's push and trusting the resulting /head SHA there is a
    # window where a kill leaves the PR head untrusted (and the next run
    # halting); a restart re-establishes trust when origin's /orig
    # patch-id matches this record. Cleared once the publish is trusted.
    pending_publish_orig_sha: str = ""
    # Fields written by a newer mergedog than this one. Carried through
    # save() so that running old code against new state files doesn't
    # silently strip what the newer code persisted.
    extra_fields: dict = field(default_factory=dict)
    path: Path | None = None

    _KNOWN_FIELDS = frozenset(
        {
            "schema_version",
            "pr",
            "trusted_shas",
            "head_branch",
            "head_repo_clone_url",
            "last_observed_failure_iso",
            "last_observed_failure_body",
            "spurious_check_names",
            "fix_commits_pushed",
            "fix_budget_ack_sha",
            "merge_auto_retries",
            "pending_publish_orig_sha",
        }
    )

    @classmethod
    def load_or_create(cls, pr: int) -> TrustDB:
        path = state_file(pr)
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                pr=data["pr"],
                trusted_shas=list(data.get("trusted_shas", [])),
                head_branch=data.get("head_branch", ""),
                head_repo_clone_url=data.get("head_repo_clone_url", ""),
                last_observed_failure_iso=data.get(
                    "last_observed_failure_iso", ""
                ),
                last_observed_failure_body=data.get(
                    "last_observed_failure_body", ""
                ),
                spurious_check_names=list(
                    data.get("spurious_check_names", [])
                ),
                fix_commits_pushed=int(data.get("fix_commits_pushed", 0)),
                fix_budget_ack_sha=data.get("fix_budget_ack_sha", ""),
                merge_auto_retries=int(data.get("merge_auto_retries", 0)),
                pending_publish_orig_sha=data.get(
                    "pending_publish_orig_sha", ""
                ),
                extra_fields={
                    k: v
                    for k, v in data.items()
                    if k not in cls._KNOWN_FIELDS
                },
                path=path,
            )
        return cls(pr=pr, path=path)

    def save(self) -> None:
        assert self.path is not None
        # Atomic write: the trust DB is the only piece of state mergedog
        # itself authors that can't be regenerated on restart, so we go
        # through a tempfile + rename instead of a bare ``write_text``.
        data = json.dumps(
            {
                **self.extra_fields,
                "schema_version": SCHEMA_VERSION,
                "pr": self.pr,
                "trusted_shas": self.trusted_shas,
                "head_branch": self.head_branch,
                "head_repo_clone_url": self.head_repo_clone_url,
                "last_observed_failure_iso": self.last_observed_failure_iso,
                "last_observed_failure_body": self.last_observed_failure_body,
                "spurious_check_names": self.spurious_check_names,
                "fix_commits_pushed": self.fix_commits_pushed,
                "fix_budget_ack_sha": self.fix_budget_ack_sha,
                "merge_auto_retries": self.merge_auto_retries,
                "pending_publish_orig_sha": self.pending_publish_orig_sha,
            },
            indent=2,
        )
        atomic_write_text(self.path, data)

    def trust(self, sha: str) -> None:
        if sha and sha not in self.trusted_shas:
            self.trusted_shas.append(sha)
            self.save()

    def is_trusted(self, sha: str) -> bool:
        return sha in self.trusted_shas
