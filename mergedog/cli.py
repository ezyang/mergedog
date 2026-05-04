"""``mergedog`` command-line entry point."""
from __future__ import annotations

import argparse
import sys

from mergedog import shepherd


def _parse_pr(value: str) -> int:
    """Accept either a bare PR number or a full PR URL."""
    value = value.strip()
    if value.isdigit():
        return int(value)
    if "/pull/" in value:
        tail = value.rsplit("/pull/", 1)[-1]
        num = tail.split("/", 1)[0].split("#", 1)[0]
        if num.isdigit():
            return int(num)
    raise argparse.ArgumentTypeError(
        f"expected a PR number or pytorch/pytorch PR URL, got {value!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mergedog",
        description=(
            "Autonomously shepherd an approved pytorch/pytorch PR through "
            "CI to the point a human can comment `@pytorchbot merge`."
        ),
    )
    parser.add_argument(
        "pr",
        type=_parse_pr,
        help="PR number (or full PR URL) on pytorch/pytorch",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help=(
            "Before polling CI, refresh the PR's view of origin/main: for "
            "regular fork PRs this is a merge of origin/main into the PR "
            "branch (and a push); for ghstack PRs it's a ``git rebase`` of "
            "/orig followed by ``ghstack submit``. Default behavior is to "
            "never auto-refresh based on merge-base age -- mergedog only "
            "refreshes against main as a piggyback on a fix commit it was "
            "going to push anyway. Use --rebase when you want a one-shot "
            "upfront refresh on this run."
        ),
    )
    parser.add_argument(
        "--accept-divergence",
        action="store_true",
        help=(
            "Proceed even if the PR head differs from the latest maintainer "
            "approval's commit. Use this only after re-reviewing the "
            "additional commits yourself."
        ),
    )
    parser.add_argument(
        "--ignore-sev",
        action="store_true",
        help=(
            "Don't park on open ``ci: sev`` issues. By default mergedog "
            "waits before any action that would trigger fresh CI (claude "
            "fix invocations, pushes, ciflow/trunk label) so we don't "
            "stampede already-broken trunk."
        ),
    )
    args = parser.parse_args(argv)

    try:
        shepherd.shepherd(
            args.pr,
            rebase=args.rebase,
            accept_divergence=args.accept_divergence,
            ignore_sev=args.ignore_sev,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; partial state left in ~/.mergedog/", file=sys.stderr)
        return 130
    return 0
