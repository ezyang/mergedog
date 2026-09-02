"""Identify ghstack stacks and the PRs they contain.

ghstack writes a markdown table at the top of each PR body listing every
PR in the stack, with the current PR marked by ``__->__``::

    Stack from [ghstack](https://github.com/ezyang/ghstack) (oldest at bottom):
    * #150
    * #149
    * __->__ #148
    * #147
    * #146

We rely on ezyang's upstream fix that preserves this list across
mid-stack ``ghstack submit`` runs -- without it, the listing can drift
when only the bottom PR is re-submitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mergedog.log import die

_STACK_HEADER_RE = re.compile(r"Stack\s+from\s+\[?ghstack\]?", re.IGNORECASE)
# Each line is one of:
#   * #123
#   * __->__ #123
#   -> Direct ghstack title #123
#   Direct ghstack title #123
# We accept any leading whitespace and ignore anything after the number in
# markdown-list format. Direct format puts the PR number at the end of the line.
_STACK_LINE_RE = re.compile(
    r"^\s*(?:\*\s+(?:__->__\s+)?#(\d+)\b|(?:->\s+)?\S.*\s#(\d+)\s*$)"
)


def parse_stack_from_body(body: str) -> list[int]:
    """Return PR numbers in stack order (bottom -> top), or ``[]`` if absent.

    ghstack writes the list with "oldest at bottom", i.e. the first
    list entry is the topmost PR. We reverse so the returned ordering
    is bottom-first, matching the natural processing order (parents
    before children).
    """
    if not body or not _STACK_HEADER_RE.search(body):
        return []
    prs_top_first: list[int] = []
    in_block = False
    for line in body.splitlines():
        if not in_block:
            if _STACK_HEADER_RE.search(line):
                in_block = True
            continue
        m = _STACK_LINE_RE.match(line)
        if m:
            prs_top_first.append(int(m.group(1) or m.group(2)))
            continue
        if not line.strip():
            # Tolerate a blank line between header and first list entry.
            if prs_top_first:
                break
            continue
        # Any other non-list, non-blank line ends the block.
        if prs_top_first:
            break
    return list(reversed(prs_top_first))


def parse_stack_from_discussion(body: str, comments: list[dict]) -> list[int]:
    """Return stack PR numbers from the PR body or latest ghstack comment."""
    nums = parse_stack_from_body(body)
    if nums:
        return nums
    for comment in reversed(comments):
        nums = parse_stack_from_body(comment.get("body", "") or "")
        if nums:
            return nums
    return []


@dataclass
class StackMember:
    """One PR in a ghstack stack, with the refs we need to push to.

    ``head_ref`` is the synthetic ``gh/<user>/<id>/head`` branch the
    GitHub PR is built on; ``orig_ref`` is the ``/orig`` companion
    where the contributor's actual single-commit change lives.
    """

    pr: int
    head_ref: str
    orig_ref: str

    @classmethod
    def from_pr_data(cls, pr_data: dict) -> StackMember:
        head = pr_data.get("headRefName") or ""
        if not (head.startswith("gh/") and head.endswith("/head")):
            die(
                f"PR #{pr_data.get('number')} is not a ghstack PR "
                f"(headRefName={head!r}); use the regular shepherd"
            )
        return cls(
            pr=pr_data["number"],
            head_ref=head,
            orig_ref=head[: -len("/head")] + "/orig",
        )
