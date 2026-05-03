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
        "--max-base-age",
        type=int,
        default=shepherd.DEFAULT_MAX_BASE_AGE_DAYS,
        metavar="DAYS",
        help=(
            "If the PR's merge-base with origin/main is older than this many "
            "days, merge origin/main into the PR branch before starting "
            f"(default: {shepherd.DEFAULT_MAX_BASE_AGE_DAYS})."
        ),
    )
    args = parser.parse_args(argv)

    try:
        shepherd.shepherd(args.pr, max_base_age_days=args.max_base_age)
    except KeyboardInterrupt:
        print("\ninterrupted; partial state left in ~/.mergedog/", file=sys.stderr)
        return 130
    return 0
