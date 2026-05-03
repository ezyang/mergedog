"""Persistent trust database for a single PR shepherding session.

The trust DB tracks the set of SHAs we consider safe to land:

- The initial PR head SHA at the moment mergedog was started (this is what
  the human implicitly approved by running mergedog).
- Every SHA we ourselves produced and pushed, identifiable by the
  ``[MERGEDOG]`` commit-title prefix.

If GitHub ever reports a PR head SHA that is not in this set, we halt:
the PR has been touched by someone other than mergedog after approval.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from mergedog.paths import state_file


@dataclass
class TrustDB:
    pr: int
    trusted_shas: list[str] = field(default_factory=list)
    head_branch: str = ""
    head_repo_clone_url: str = ""
    path: Path | None = None

    @classmethod
    def load_or_create(cls, pr: int) -> "TrustDB":
        path = state_file(pr)
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                pr=data["pr"],
                trusted_shas=list(data.get("trusted_shas", [])),
                head_branch=data.get("head_branch", ""),
                head_repo_clone_url=data.get("head_repo_clone_url", ""),
                path=path,
            )
        return cls(pr=pr, path=path)

    def save(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: ``write_text`` is open->write->close, which can leave
        # a half-written or empty file if we're SIGKILL'd mid-write. The
        # trust DB is the only piece of state mergedog itself authors that
        # can't be regenerated on restart, so we go through a tempfile +
        # rename instead.
        data = json.dumps(
            {
                "pr": self.pr,
                "trusted_shas": self.trusted_shas,
                "head_branch": self.head_branch,
                "head_repo_clone_url": self.head_repo_clone_url,
            },
            indent=2,
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(data)
        os.replace(tmp, self.path)

    def trust(self, sha: str) -> None:
        if sha and sha not in self.trusted_shas:
            self.trusted_shas.append(sha)
            self.save()

    def is_trusted(self, sha: str) -> bool:
        return sha in self.trusted_shas
