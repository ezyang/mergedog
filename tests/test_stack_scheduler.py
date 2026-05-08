import time
import unittest
from unittest import mock

from mergedog import stack_shepherd
from mergedog.shepherd import CI_STABILITY_WINDOW_SEC
from mergedog.stack import StackMember


def _mk_ctx(
    *,
    orig_sha: str,
    status: str | None,
    stable_for: float = 0.0,
    trunk_applied: bool = False,
):
    """Build a stub _MemberCtx with just the fields the predicates read.

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
    ctx.trunk_applied = trunk_applied
    return ctx


class TestStackMergedogLabelManagement(unittest.TestCase):
    def _run_until_after_optional_label(self, **kwargs):
        member = StackMember(
            pr=101,
            head_ref="gh/user/101/head",
            orig_ref="gh/user/101/orig",
        )
        pr_data_by_pr = {101: {"number": 101}}
        with mock.patch.object(
            stack_shepherd.repo, "ensure_clone"
        ), mock.patch.object(
            stack_shepherd.repo, "fetch_main"
        ), mock.patch.object(
            stack_shepherd, "resolve_stack", return_value=([member], pr_data_by_pr)
        ), mock.patch.object(
            stack_shepherd, "configure_log_file"
        ), mock.patch.object(
            stack_shepherd.signal, "signal"
        ), mock.patch.object(
            stack_shepherd.faulthandler, "enable"
        ), mock.patch.object(
            stack_shepherd.faulthandler, "register"
        ), mock.patch.object(
            stack_shepherd, "_add_mergedog_labels_parallel", return_value=[101]
        ) as add_labels, mock.patch.object(
            stack_shepherd.repo, "fetch_stack_refs", side_effect=SystemExit
        ), mock.patch.object(
            stack_shepherd.github, "remove_label"
        ) as remove_label:
            with self.assertRaises(SystemExit):
                stack_shepherd.run_stack(101, **kwargs)
        return add_labels, remove_label

    def test_default_does_not_touch_mergedog_labels(self):
        add_labels, remove_label = self._run_until_after_optional_label()
        add_labels.assert_not_called()
        remove_label.assert_not_called()

    def test_explicit_flag_adds_and_removes_mergedog_labels(self):
        add_labels, remove_label = self._run_until_after_optional_label(
            manage_mergedog_label=True
        )
        add_labels.assert_called_once()
        remove_label.assert_called_once_with(101, stack_shepherd.MERGEDOG_LABEL)


class TestInspectMember(unittest.TestCase):
    def _ctx(self, *, spurious=()):
        ctx = _mk_ctx(orig_sha="A", status=None)
        ctx.spurious_check_names = set(spurious)
        return ctx

    def test_caches_failing_check_names_post_overrides(self):
        ctx = self._ctx(spurious={"flaky_dynamo"})
        checks = [
            {"name": "pull / linux", "bucket": "fail"},
            {"name": "lint", "bucket": "cancel"},
            {"name": "flaky_dynamo", "bucket": "fail"},
            {"name": "ok_check", "bucket": "pass"},
        ]
        with mock.patch.object(
            stack_shepherd.github, "get_pr_checks_all", return_value=checks
        ), mock.patch.object(
            stack_shepherd.github, "evaluate_checks", return_value="failed"
        ):
            stack_shepherd._inspect_member(ctx)
        # The spurious-overridden ``flaky_dynamo`` is treated as
        # ``skipping`` and so is not in the cached failing set.
        self.assertEqual(ctx.failing_check_names, ["lint", "pull / linux"])

    def test_caches_empty_when_no_failures(self):
        ctx = self._ctx()
        checks = [
            {"name": "pull / linux", "bucket": "pass"},
            {"name": "lint", "bucket": "pass"},
        ]
        with mock.patch.object(
            stack_shepherd.github, "get_pr_checks_all", return_value=checks
        ), mock.patch.object(
            stack_shepherd.github, "evaluate_checks", return_value="passed"
        ):
            stack_shepherd._inspect_member(ctx)
        self.assertEqual(ctx.failing_check_names, [])


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


class TestTrunkPromotionTarget(unittest.TestCase):
    def test_bottom_eligible_when_green_stable(self):
        bottom = _mk_ctx(orig_sha="A", status="passed", stable_for=CI_STABILITY_WINDOW_SEC + 1)
        target = stack_shepherd._trunk_promotion_target([bottom], time.time())
        self.assertIs(target, bottom)

    def test_bottom_not_eligible_when_unstable(self):
        bottom = _mk_ctx(orig_sha="A", status="passed", stable_for=1)
        self.assertIsNone(
            stack_shepherd._trunk_promotion_target([bottom], time.time())
        )

    def test_child_blocked_until_parent_trunk_applied_and_stable(self):
        # Parent has trunk applied and is green-stable; child is green-stable.
        parent = _mk_ctx(
            orig_sha="A",
            status="passed",
            stable_for=CI_STABILITY_WINDOW_SEC + 1,
            trunk_applied=True,
        )
        child = _mk_ctx(orig_sha="B", status="passed", stable_for=CI_STABILITY_WINDOW_SEC + 1)
        target = stack_shepherd._trunk_promotion_target(
            [parent, child], time.time()
        )
        self.assertIs(target, child)

    def test_child_blocked_when_parent_not_trunk_applied(self):
        # Parent green-stable but never had trunk applied (the bottom should
        # have been promoted first, but the predicate must not skip).
        parent = _mk_ctx(orig_sha="A", status="passed", stable_for=CI_STABILITY_WINDOW_SEC + 1)
        child = _mk_ctx(orig_sha="B", status="passed", stable_for=CI_STABILITY_WINDOW_SEC + 1)
        # The lowest eligible is parent itself, not child.
        target = stack_shepherd._trunk_promotion_target(
            [parent, child], time.time()
        )
        self.assertIs(target, parent)

    def test_child_blocked_when_parent_trunk_applied_but_unstable(self):
        # Parent trunk-applied but trunk-CI hasn't settled yet (unstable).
        parent = _mk_ctx(
            orig_sha="A",
            status="passed",
            stable_for=1,  # too fresh
            trunk_applied=True,
        )
        child = _mk_ctx(orig_sha="B", status="passed", stable_for=CI_STABILITY_WINDOW_SEC + 1)
        self.assertIsNone(
            stack_shepherd._trunk_promotion_target(
                [parent, child], time.time()
            )
        )


class TestAllTrunkGreenStable(unittest.TestCase):
    def test_all_satisfied(self):
        a = _mk_ctx(
            orig_sha="A",
            status="passed",
            stable_for=CI_STABILITY_WINDOW_SEC + 1,
            trunk_applied=True,
        )
        b = _mk_ctx(
            orig_sha="B",
            status="passed",
            stable_for=CI_STABILITY_WINDOW_SEC + 1,
            trunk_applied=True,
        )
        self.assertTrue(
            stack_shepherd._all_trunk_green_stable([a, b], time.time())
        )

    def test_one_member_not_trunk_applied_blocks(self):
        a = _mk_ctx(
            orig_sha="A",
            status="passed",
            stable_for=CI_STABILITY_WINDOW_SEC + 1,
            trunk_applied=True,
        )
        b = _mk_ctx(
            orig_sha="B",
            status="passed",
            stable_for=CI_STABILITY_WINDOW_SEC + 1,
        )
        self.assertFalse(
            stack_shepherd._all_trunk_green_stable([a, b], time.time())
        )

    def test_one_member_unstable_blocks(self):
        a = _mk_ctx(
            orig_sha="A",
            status="passed",
            stable_for=CI_STABILITY_WINDOW_SEC + 1,
            trunk_applied=True,
        )
        b = _mk_ctx(
            orig_sha="B",
            status="passed",
            stable_for=1,  # too fresh
            trunk_applied=True,
        )
        self.assertFalse(
            stack_shepherd._all_trunk_green_stable([a, b], time.time())
        )


class TestRebaseStackPrefix(unittest.TestCase):
    def test_rebases_and_submits_full_stack(self):
        bottom = _mk_ctx(orig_sha="A", status="passed", stable_for=999)
        bottom.member = StackMember(
            pr=101, head_ref="gh/u/101/head", orig_ref="gh/u/101/orig"
        )
        top = _mk_ctx(orig_sha="B", status="passed", stable_for=999)
        top.member = StackMember(
            pr=102, head_ref="gh/u/102/head", orig_ref="gh/u/102/orig"
        )
        contexts = [bottom, top]
        refs = {
            "gh/u/101/head": "A_HEAD2",
            "gh/u/101/orig": "A_ORIG2",
            "gh/u/102/head": "B_HEAD2",
            "gh/u/102/orig": "B_ORIG2",
        }
        with mock.patch.object(
            stack_shepherd, "_wait_for_no_active_sev", return_value=False
        ), mock.patch.object(
            stack_shepherd.repo,
            "select_rebase_target",
            return_value=("origin/main", "origin/main"),
        ), mock.patch.object(
            stack_shepherd, "_reconstruct_stack_up_to"
        ) as reconstruct, mock.patch.object(
            stack_shepherd.repo,
            "attempt_rebase_main",
            return_value=("ok", "B_REBASED"),
        ), mock.patch.object(
            stack_shepherd.repo, "ghstack_submit"
        ) as submit, mock.patch.object(
            stack_shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            stack_shepherd, "_record_pushed_commit"
        ) as record, mock.patch.object(
            stack_shepherd, "_wait_for_pr_head"
        ) as wait_head:
            stack_shepherd._rebase_stack_prefix_onto_main(
                contexts,
                1,
                mock.Mock(),
                [],
                ignore_sev=False,
                force_ghstack=True,
            )

        reconstruct.assert_called_once_with(contexts, 1, mock.ANY)
        submit.assert_called_once_with(
            mock.ANY,
            "Rebase stack onto origin/main",
            no_stack=False,
            force=True,
        )
        self.assertEqual(bottom.head_sha, "A_HEAD2")
        self.assertEqual(top.head_sha, "B_HEAD2")
        self.assertEqual(record.call_count, 2)
        wait_head.assert_has_calls(
            [mock.call(101, "A_HEAD2"), mock.call(102, "B_HEAD2")]
        )

    def test_conflict_resolver_allows_multiple_rebased_commits(self):
        bottom = _mk_ctx(orig_sha="A", status="passed", stable_for=999)
        bottom.member = StackMember(
            pr=101, head_ref="gh/u/101/head", orig_ref="gh/u/101/orig"
        )
        bottom.pr_data = {"url": "https://github.com/pytorch/pytorch/pull/101"}
        top = _mk_ctx(orig_sha="B", status="passed", stable_for=999)
        top.member = StackMember(
            pr=102, head_ref="gh/u/102/head", orig_ref="gh/u/102/orig"
        )
        top.pr_data = {"url": "https://github.com/pytorch/pytorch/pull/102"}
        contexts = [bottom, top]
        refs = {
            "gh/u/101/head": "A_HEAD2",
            "gh/u/101/orig": "A_ORIG2",
            "gh/u/102/head": "B_HEAD2",
            "gh/u/102/orig": "B_ORIG2",
        }
        with mock.patch.object(
            stack_shepherd, "_wait_for_no_active_sev", return_value=False
        ), mock.patch.object(
            stack_shepherd.repo,
            "select_rebase_target",
            return_value=("origin/main", "origin/main"),
        ), mock.patch.object(
            stack_shepherd, "_reconstruct_stack_up_to"
        ), mock.patch.object(
            stack_shepherd.repo,
            "attempt_rebase_main",
            return_value=("conflict", None),
        ), mock.patch.object(
            stack_shepherd, "_refresh_context_for", return_value=(mock.Mock(), [])
        ), mock.patch.object(
            stack_shepherd.repo, "head_sha", return_value="B_OLD"
        ), mock.patch.object(
            stack_shepherd.claude_mod,
            "invoke_rebase_resolver",
            return_value=(True, "B_REBASED", []),
        ) as invoke, mock.patch.object(
            stack_shepherd.repo, "ghstack_submit"
        ), mock.patch.object(
            stack_shepherd.repo, "fetch_stack_refs", return_value=refs
        ), mock.patch.object(
            stack_shepherd, "_record_pushed_commit"
        ), mock.patch.object(
            stack_shepherd, "_wait_for_pr_head"
        ):
            stack_shepherd._rebase_stack_prefix_onto_main(
                contexts,
                1,
                mock.Mock(),
                [],
                ignore_sev=False,
                force_ghstack=False,
            )

        invoke.assert_called_once()
        self.assertTrue(invoke.call_args.kwargs["allow_multiple_commits"])


class TestLatestUnhandledStackFailure(unittest.TestCase):
    def test_finds_newest_post_handoff_failure(self):
        bottom_sha = "a" * 40
        top_sha = "b" * 40
        bottom = _mk_ctx(orig_sha=bottom_sha, status=None)
        bottom.member = StackMember(
            pr=101, head_ref="gh/u/101/head", orig_ref="gh/u/101/orig"
        )
        bottom.trust.last_observed_failure_iso = ""
        top = _mk_ctx(orig_sha=top_sha, status=None)
        top.member = StackMember(
            pr=102, head_ref="gh/u/102/head", orig_ref="gh/u/102/orig"
        )
        top.trust.last_observed_failure_iso = "2026-05-08T13:00:00Z"

        def mergebot_failure_event(pr, since_iso):
            expected_since = (
                "2026-05-08T13:00:00Z"
                if pr == 102
                else ""
            )
            self.assertEqual(since_iso, expected_since)
            if pr == 102:
                return (
                    "2026-05-08T14:00:00Z",
                    "## Merge failed\n"
                    f"Command `git cherry-pick -x {bottom_sha}` failed\n"
                    "CONFLICT (content): Merge conflict",
                )
            return None

        with mock.patch.object(
            stack_shepherd,
            "latest_mergebot_failure_event",
            side_effect=mergebot_failure_event,
        ):
            result = stack_shepherd._latest_unhandled_stack_failure([bottom, top])

        self.assertIsNotNone(result)
        assert result is not None
        ctx, event_iso, body = result
        self.assertIs(ctx, top)
        self.assertEqual(event_iso, "2026-05-08T14:00:00Z")
        self.assertIn("Merge failed", body)

    def test_ignores_failure_for_old_stack_sha(self):
        bottom = _mk_ctx(orig_sha="a" * 40, status=None)
        bottom.member = StackMember(
            pr=101, head_ref="gh/u/101/head", orig_ref="gh/u/101/orig"
        )
        bottom.trust.last_observed_failure_iso = ""
        top = _mk_ctx(orig_sha="b" * 40, status=None)
        top.member = StackMember(
            pr=102, head_ref="gh/u/102/head", orig_ref="gh/u/102/orig"
        )
        top.trust.last_observed_failure_iso = ""

        def mergebot_failure_event(pr, since_iso):
            if pr == 102:
                return (
                    "2026-05-08T14:00:00Z",
                    "## Merge failed\n"
                    f"Command `git cherry-pick -x {'c' * 40}` failed\n"
                    "CONFLICT (content): Merge conflict",
                )
            return None

        with mock.patch.object(
            stack_shepherd,
            "latest_mergebot_failure_event",
            side_effect=mergebot_failure_event,
        ):
            result = stack_shepherd._latest_unhandled_stack_failure([bottom, top])

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
