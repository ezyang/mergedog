"""Seed the trust DB from the GitHub review state at startup.

Run once per shepherd invocation, before the inner loop begins. Determines
which SHA on the PR represents the "human approved this" anchor, and
either trusts the current PR head (if it matches or has already been
trusted in a prior run) or halts.
"""
from __future__ import annotations

from mergedog import github
from mergedog.log import die, log
from mergedog.state import TrustDB


def seed_trust_from_reviews(
    trust: TrustDB,
    pr: int,
    pr_data: dict,
    accept_divergence: bool,
    self_pr: bool = False,
) -> None:
    """Establish the initial trusted SHA from GitHub PR reviews.

    Queries the PR's reviews via GraphQL and finds the most recent
    APPROVED review by an author with a *trusted* association
    (MEMBER / COLLABORATOR / OWNER -- i.e. someone with push access,
    not a drive-by approval from any GitHub user). The ``commit_id`` of
    that approval is the trust seed.

    If the PR's current head differs from the approval SHA and the
    head isn't already in the trust DB (e.g. from a previous mergedog
    run that pushed [MERGEDOG] commits), we halt -- the contributor has
    pushed unblessed work after the approval. ``--accept-divergence``
    overrides this for cases where the operator has personally
    re-reviewed the new commits.

    When ``self_pr`` is set, the operator is the PR author. We skip the
    maintainer-approval gate entirely (you're allowed to iterate on your
    own CI before review) and trust the current head directly; the main
    loop continues to roll the trust forward as the author pushes.
    """
    if self_pr:
        head_sha = pr_data["headRefOid"]
        log(f"self-authored PR; trusting current head {head_sha[:12]} directly")
        trust.trust(head_sha)
        return

    audit = github.get_pr_review_audit(pr)
    decision = audit.get("decision")
    log(f"PR review decision: {decision}")

    untrusted_approvals = [
        r for r in audit["reviews"]
        if r.get("state") == "APPROVED"
        and not github.is_trusted_association(r.get("association"))
    ]
    for r in untrusted_approvals:
        log(
            f"  ignoring APPROVED review by {r.get('login')!r} "
            f"(association={r.get('association')!r}, not a maintainer)"
        )

    trusted_approvals = [
        r for r in audit["reviews"]
        if r.get("state") == "APPROVED"
        and github.is_trusted_association(r.get("association"))
        and r.get("commit_id")
    ]
    if not trusted_approvals:
        die(
            "no APPROVED review from a maintainer "
            "(MEMBER/COLLABORATOR/OWNER) was found on this PR; "
            "halting. Get a real approval before running mergedog."
        )

    trusted_approvals.sort(key=lambda r: r.get("submitted_at") or "")
    latest = trusted_approvals[-1]
    approval_sha: str = latest["commit_id"]
    log(
        f"latest maintainer approval: {latest['login']} "
        f"({latest['association']}) at {approval_sha[:12]} "
        f"on {latest.get('submitted_at')}"
    )

    if decision != "APPROVED":
        die(
            f"PR review decision is {decision!r}, not APPROVED "
            f"(another reviewer may have requested changes after the "
            f"latest approval). Halting."
        )

    trust.trust(approval_sha)

    head_sha = pr_data["headRefOid"]
    if head_sha == approval_sha:
        log(f"current PR head matches the approval SHA")
    elif trust.is_trusted(head_sha):
        log(
            f"current PR head {head_sha[:12]} is already in our trust DB "
            f"(presumably a [MERGEDOG] commit from a previous run); "
            f"approval seed at {approval_sha[:12]}"
        )
    elif accept_divergence:
        log(
            f"WARNING: current PR head {head_sha[:12]} differs from the "
            f"latest maintainer approval at {approval_sha[:12]}. "
            f"--accept-divergence given; trusting head as well."
        )
        trust.trust(head_sha)
    else:
        die(
            f"current PR head {head_sha[:12]} differs from the latest "
            f"maintainer approval at {approval_sha[:12]} (by "
            f"{latest['login']}). The contributor pushed unreviewed "
            f"commits after approval. Rerun with --accept-divergence "
            f"after re-reviewing, or get a fresh approval."
        )
