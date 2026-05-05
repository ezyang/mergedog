import time
import unittest
from unittest import mock

from mergedog import stack_shepherd
from mergedog.shepherd import CI_STABILITY_WINDOW_SEC
from mergedog.stack import StackMember


def _mk_ctx(*, orig_sha: str, status: str | None, stable_for: float = 0.0):
    """Build a stub _MemberCtx with just the fields _propagation_needed reads.

    ``status`` is the verdict ("passed"/"failed"/"pending") to expose via
    stable_observation; None means "not yet observed". ``stable_for`` sets
    how long the observation has been stable (seconds before now).
    """
    member = StackMember(
        pr=0, head_ref="gh/u/0/head", orig_ref="gh/u/0/orig"
    )
    ctx = stack_shepherd._MemberCtx(
        member=member,
        pr_data={},
        trust=mock.MagicMock(),
        self_pr=False,
        head_sha="x",
        orig_sha=orig_sha,
    )
    if status is not None:
        ctx.stable_observation = (status, 1)
        ctx.stable_since = time.time() - stable_for
    return ctx


class TestPropagationNeeded(unittest.TestCase):
    def test_single_member_never_propagates(self):
        ctx = _mk_ctx(orig_sha="A", status="passed", stable_for=999)
        self.assertFalse(stack_shepherd._propagation_needed([ctx], time.time()))

    def test_no_stale_pair_returns_false(self):
        # Child's /orig parent IS the parent's /orig -- not stale.
        parent = _mk_ctx(orig_sha="A", status="passed", stable_for=999)
        child = _mk_ctx(orig_sha="B", status="passed", stable_for=999)
        with mock.patch.object(stack_shepherd.repo, "parent_sha", return_value="A"):
            self.assertFalse(
                stack_shepherd._propagation_needed([parent, child], time.time())
            )

    def test_stale_with_green_stable_parent_returns_true(self):
        parent = _mk_ctx(orig_sha="A_NEW", status="passed", stable_for=CI_STABILITY_WINDOW_SEC + 1)
        child = _mk_ctx(orig_sha="B", status="pending")
        # child's /orig parent is "A_OLD", parent's current is "A_NEW" -> stale
        with mock.patch.object(stack_shepherd.repo, "parent_sha", return_value="A_OLD"):
            self.assertTrue(
                stack_shepherd._propagation_needed([parent, child], time.time())
            )

    def test_stale_with_failing_parent_blocks_propagation(self):
        parent = _mk_ctx(orig_sha="A_NEW", status="failed", stable_for=999)
        child = _mk_ctx(orig_sha="B", status="pending")
        with mock.patch.object(stack_shepherd.repo, "parent_sha", return_value="A_OLD"):
            self.assertFalse(
                stack_shepherd._propagation_needed([parent, child], time.time())
            )

    def test_stale_with_unstable_parent_blocks_propagation(self):
        # Just-flipped to passed -- needs to hold for the window.
        parent = _mk_ctx(orig_sha="A_NEW", status="passed", stable_for=1)
        child = _mk_ctx(orig_sha="B", status="pending")
        with mock.patch.object(stack_shepherd.repo, "parent_sha", return_value="A_OLD"):
            self.assertFalse(
                stack_shepherd._propagation_needed([parent, child], time.time())
            )

    def test_stale_pair_below_failing_pair_blocks_propagation(self):
        # parent_0 green-stable, parent_1 failing, child_2 stale w.r.t. parent_1.
        # Even though parent_0 is fine, propagation would touch the failing
        # parent_1's content -- block until parent_1 is fixed first.
        p0 = _mk_ctx(orig_sha="A", status="passed", stable_for=999)
        p1 = _mk_ctx(orig_sha="B_NEW", status="failed", stable_for=999)
        p2 = _mk_ctx(orig_sha="C", status="pending")

        # parent_sha("B_NEW") -> "A"  (p1 not stale w.r.t. p0)
        # parent_sha("C") -> "B_OLD"  (p2 stale w.r.t. p1)
        def fake_parent(sha):
            return {"B_NEW": "A", "C": "B_OLD"}[sha]

        with mock.patch.object(stack_shepherd.repo, "parent_sha", side_effect=fake_parent):
            self.assertFalse(
                stack_shepherd._propagation_needed([p0, p1, p2], time.time())
            )


if __name__ == "__main__":
    unittest.main()
