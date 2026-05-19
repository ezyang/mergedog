"""Trust helpers for PR head movements after approval."""
from __future__ import annotations

from collections.abc import Callable

from mergedog import github, repo
from mergedog.log import log
from mergedog.project import get_project_policy
from mergedog.state import TrustDB


PROJECT = get_project_policy()


def trust_mergebot_rebase_if_equivalent(
    trust: TrustDB,
    current_sha: str,
    *,
    ensure_current_available: Callable[[], bool] | None = None,
) -> bool:
    """Trust a mergebot-authored rebase when it preserves reviewed changes.

    The mergebot is allowed to rewrite a PR branch during handoff, but we do
    not trust the bot identity by itself: that would let an untrusted author
    use repo automation as a confused deputy. The new commit must also have
    the same stable patch-id as a SHA already in this PR's trust DB.
    """
    if PROJECT.mergebot_login is None:
        return False

    _, committer_login = github.get_commit_actor_logins(current_sha)
    if (committer_login or "").lower() != PROJECT.mergebot_login.lower():
        return False

    if ensure_current_available is not None and not ensure_current_available():
        return False

    if not repo.patch_id_matches_any(current_sha, trust.trusted_shas):
        return False

    trust.trust(current_sha)
    log(
        f"trusted {PROJECT.mergebot_login} rebase {current_sha[:12]} "
        "because its patch-id matches an already trusted commit"
    )
    return True
