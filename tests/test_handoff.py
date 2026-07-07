import unittest
from unittest import mock

from mergedog import github
from mergedog import handoff
from mergedog.taint import taint


class TestHandoffComments(unittest.TestCase):
    def test_handoff_comment_records_current_head(self):
        body = handoff._format_handoff_comment(
            {
                "number": 101,
                "headRefOid": "a" * 40,
            },
            [],
        )

        self.assertIn(f"<!-- mergedog:handoff head={'a' * 40} -->", body)
        self.assertIn(f"Current PR head: `{'a' * 40}`.", body)

    def test_handoff_comment_leads_with_pushed_changes(self):
        body = handoff._format_handoff_comment(
            {
                "number": 101,
                "headRefOid": "b" * 40,
            },
            [
                handoff.ClaudeSession(
                    mode="fix-CI",
                    started_at="2026-05-08T13:00:00+00:00",
                    sha_before="a" * 40,
                    sha_after="c" * 40,
                    verdict="pushed fix commit cccccccccccc",
                    transcript=["fixed it"],
                )
            ],
            pushed_changes=[
                handoff.PushedChange(
                    sha="d" * 40,
                    summary="merged main into the PR branch",
                    subject="[MERGEDOG] Merge main into PR branch",
                )
            ],
        )

        pushed_idx = body.index("### Autonomous changes pushed")
        sessions_idx = body.index("### Session 1")
        self.assertLess(pushed_idx, sessions_idx)
        self.assertIn(
            "- [`dddddddddddd`](https://github.com/pytorch/pytorch/commit/"
            f"{'d' * 40}) — merged main into the PR branch: "
            "[MERGEDOG] Merge main into PR branch",
            body,
        )
        self.assertIn(
            "- [`cccccccccccc`](https://github.com/pytorch/pytorch/commit/"
            f"{'c' * 40}) — pushed fix commit cccccccccccc "
            "(fix-CI (2026-05-08T13:00:00+00:00))",
            body,
        )

    def test_handoff_comment_surfaces_suppressed_failures(self):
        body = handoff._format_handoff_comment(
            {
                "number": 101,
                "headRefOid": "b" * 40,
            },
            [],
            suppressed_failures=[
                "lintrunner-noclang-all / lint",
                "dtensor-test / test-osdc",
            ],
            drci_summary=(
                "Dr. CI detected failures:\n"
                "- lintrunner-noclang-all / lint failed"
            ),
        )

        self.assertIn("### CI notes at handoff", body)
        self.assertIn(
            "still-failing checks as unrelated/spurious", body
        )
        self.assertIn("`lintrunner-noclang-all / lint`", body)
        self.assertIn("Latest Dr. CI summary", body)
        self.assertIn("lintrunner-noclang-all / lint failed", body)

    def test_handoff_comment_strips_drci_machine_markers(self):
        body = handoff._format_handoff_comment(
            {
                "number": 101,
                "headRefOid": "b" * 40,
            },
            [],
            drci_summary=(
                "<!-- drci-comment-start -->\n"
                "## :x: 1 New Failure\n"
                "As of commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:\n"
                "<!-- drci-comment-end -->"
            ),
        )

        self.assertNotIn("drci-comment-start", body)
        self.assertNotIn("drci-comment-end", body)
        self.assertIn("## :x: 1 New Failure", body)
        self.assertIn(
            "As of commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:",
            body,
        )

    def test_handoff_comment_warns_when_suppressions_differ_from_drci(self):
        drci_summary = (
            "<b>NEW FAILURES</b> - The following jobs have failed:<p>\n"
            "* [Lint / lintrunner-noclang-all / lint](https://hud)\n"
            "</p>\n"
            "<b>BROKEN TRUNK</b> - The following job failed on trunk:<p>\n"
            "* [inductor / unit-test / inductor-test / test-osdc](https://hud)\n"
            "</p>\n"
        )

        comparison = handoff.compare_suppressed_failures_with_drci(
            [
                "lintrunner-noclang-all / lint",
                "unit-test / inductor-test / test-osdc",
                "win-vs2022-cpu-py3 / build",
            ],
            drci_summary,
        )

        self.assertEqual(comparison.not_listed, ("win-vs2022-cpu-py3 / build",))
        self.assertEqual(
            comparison.not_marked_unrelated,
            ("lintrunner-noclang-all / lint",),
        )
        self.assertIn("not listed", comparison.status_warning() or "")
        self.assertIn("not marked unrelated", comparison.status_warning() or "")

        body = handoff._format_handoff_comment(
            {"number": 101, "headRefOid": "b" * 40},
            [],
            suppressed_failures=[
                "lintrunner-noclang-all / lint",
                "unit-test / inductor-test / test-osdc",
                "win-vs2022-cpu-py3 / build",
            ],
            drci_summary=drci_summary,
        )

        self.assertIn("suppressed check list differs", body)
        self.assertIn("not listed by Dr. CI", body)
        self.assertIn("not marked unrelated by Dr. CI", body)

    def test_handoff_comment_idempotency_is_scoped_to_head(self):
        comments = [
            {
                "author": "mergedog",
                "created_at": "2026-05-08T13:00:00Z",
                "body": f"<!-- mergedog:handoff head={'a' * 40} -->",
            },
            {
                "author": "mergedog",
                "created_at": "2026-05-08T14:00:00Z",
                "body": "<!-- mergedog:handoff -->",
            },
        ]

        with mock.patch.object(github, "get_pr_comments", return_value=comments):
            self.assertTrue(github.has_mergedog_handoff_comment(101))
            self.assertTrue(
                github.has_mergedog_handoff_comment(
                    101, head_sha="a" * 40
                )
            )
            self.assertFalse(
                github.has_mergedog_handoff_comment(
                    101, head_sha="b" * 40
                )
            )
            self.assertEqual(
                github.latest_mergedog_handoff_iso(101),
                "2026-05-08T14:00:00Z",
            )

    def test_handoff_marker_from_other_author_is_ignored(self):
        # A spoofed marker posted by someone other than mergedog's account
        # must not suppress the handoff or shift the watch-loop anchor.
        comments = [
            {
                "author": "attacker",
                "created_at": "2026-05-08T13:00:00Z",
                "body": f"<!-- mergedog:handoff head={'a' * 40} -->",
            },
        ]
        with mock.patch.object(github, "get_pr_comments", return_value=comments):
            self.assertFalse(
                github.has_mergedog_handoff_comment(101, author="mergedog")
            )
            self.assertFalse(
                github.has_mergedog_handoff_comment(
                    101, head_sha="a" * 40, author="mergedog"
                )
            )
            self.assertIsNone(
                github.latest_mergedog_handoff_iso(101, author="mergedog")
            )

    def test_handoff_marker_author_match_is_case_insensitive(self):
        comments = [
            {
                "author": "MergeDog",
                "created_at": "2026-05-08T13:00:00Z",
                "body": "<!-- mergedog:handoff -->",
            },
        ]
        with mock.patch.object(github, "get_pr_comments", return_value=comments):
            self.assertTrue(
                github.has_mergedog_handoff_comment(101, author="mergedog")
            )

    def test_latest_handoff_iso_filters_spoofed_author(self):
        comments = [
            {
                "author": "mergedog",
                "created_at": "2026-05-08T13:00:00Z",
                "body": "<!-- mergedog:handoff head=aaa -->",
            },
            {
                "author": "attacker",
                "created_at": "2026-05-08T23:00:00Z",
                "body": "<!-- mergedog:handoff head=bbb -->",
            },
        ]
        # Without the filter the attacker's later timestamp would win.
        self.assertEqual(
            github.latest_mergedog_handoff_iso_from_comments(
                comments, author="mergedog"
            ),
            "2026-05-08T13:00:00Z",
        )

    def test_latest_handoff_iso_from_existing_comments(self):
        comments = [
            {
                "created_at": "2026-05-08T13:00:00Z",
                "body": "<!-- mergedog:handoff head=aaa -->",
            },
            {"created_at": "2026-05-08T14:00:00Z", "body": "plain"},
            {
                "created_at": "2026-05-08T15:00:00Z",
                "body": "<!-- mergedog:handoff head=bbb -->",
            },
        ]

        self.assertEqual(
            github.latest_mergedog_handoff_iso_from_comments(comments),
            "2026-05-08T15:00:00Z",
        )

    def test_detects_easycla_merge_failure(self):
        body = (
            "## Merge failed\n"
            "**Reason**: 1 mandatory check(s) are pending/not yet run. "
            "The first few are:\n"
            "- EasyCLA\n"
        )

        self.assertTrue(handoff.is_cla_merge_failure(body))

    def test_handoff_status_waits_for_approval_when_not_approved(self):
        with mock.patch.object(handoff, "write_status") as write_status:
            handoff._write_handoff_status(
                101,
                approved=False,
                merging=False,
                intervention_count=2,
                human_ack_sha="a" * 40,
            )

        # Intervention count lives in the mux Int column, not the text.
        write_status.assert_called_once_with(
            101,
            phase="ready",
            category="ready",
            waiting_on="approval",
            user_action="approve the PR after reviewing mergedog interventions",
            message="waiting for maintainer approval",
            intervention_count=2,
            human_ack_sha="a" * 40,
            approved=False,
            merging=False,
            ci_suppressed=0,
        )

    def test_handoff_status_marks_approval_external_when_not_actionable(self):
        with mock.patch.object(handoff, "write_status") as write_status:
            handoff._write_handoff_status(
                101,
                approved=False,
                merging=False,
                intervention_count=0,
                approval_actionable=False,
            )

        kwargs = write_status.call_args.kwargs
        self.assertEqual(kwargs["phase"], "ready")
        self.assertEqual(kwargs["category"], "waiting")
        self.assertEqual(kwargs["waiting_on"], "approval")
        self.assertIsNone(kwargs["user_action"])
        self.assertEqual(kwargs["message"], "waiting for maintainer approval")


class TestMergebotIgnoredChecks(unittest.TestCase):
    def test_extracts_current_failed_checks_from_trusted_mergebot_comment(self):
        comments = [
            {
                "author": "pytorchmergebot",
                "created_at": "2026-05-08T14:00:00Z",
                "body": taint(
                    "Your change will be merged while ignoring the following "
                    "3 checks: pull / linux-jammy-py3.14-clang18 / test "
                    "(default, 4, 5, linux.4xlarge), B200 Smoke Tests / "
                    "linux-jammy-cuda13.0-py3.10-gcc11-sm100 / test "
                    "(smoke_b200, 1, 1, linux.dgx.b200), Limited CI on H100 / "
                    "linux-jammy-cuda13.0-py3.10-gcc11-sm90 / test "
                    "(smoke, 1, 1, linux.aws.h100)",
                    "pr_comment",
                ),
            }
        ]
        checks = [
            {
                "name": "pull / linux-jammy-py3.14-clang18 / test "
                "(default, 4, 5, linux.4xlarge)",
                "bucket": "fail",
            },
            {
                "name": "linux-jammy-cuda13.0-py3.10-gcc11-sm100 / test "
                "(smoke_b200, 1, 1, linux.dgx.b200)",
                "bucket": "fail",
            },
            {"name": "Lint / quick-check", "bucket": "fail"},
        ]

        self.assertEqual(
            handoff.mergebot_ignored_check_names(
                comments, checks, since_iso="2026-05-08T13:00:00Z"
            ),
            {
                "pull / linux-jammy-py3.14-clang18 / test "
                "(default, 4, 5, linux.4xlarge)",
                "linux-jammy-cuda13.0-py3.10-gcc11-sm100 / test "
                "(smoke_b200, 1, 1, linux.dgx.b200)",
            },
        )

    def test_ignores_old_or_untrusted_ignore_text(self):
        checks = [{"name": "pull / linux", "bucket": "fail"}]
        comments = [
            {
                "author": "pytorchmergebot",
                "created_at": "2026-05-08T12:00:00Z",
                "body": "Your change will be merged while ignoring the "
                "following 1 checks: pull / linux",
            },
            {
                "author": "random-user",
                "created_at": "2026-05-08T14:00:00Z",
                "body": "Your change will be merged while ignoring the "
                "following 1 checks: pull / linux",
            },
        ]

        self.assertEqual(
            handoff.mergebot_ignored_check_names(
                comments, checks, since_iso="2026-05-08T13:00:00Z"
            ),
            set(),
        )

    def test_trusted_merge_i_command_ignores_current_failed_checks(self):
        checks = [
            {"name": "pull / linux", "bucket": "fail"},
            {"name": "trunk / rocm", "bucket": "cancel"},
            {"name": "docs", "bucket": "pass"},
        ]
        comments = [
            {
                "author": "ezyang",
                "author_association": "MEMBER",
                "created_at": "2026-05-08T14:00:00Z",
                "body": taint("@pytorchbot merge -i", "pr_comment"),
            }
        ]

        self.assertEqual(
            handoff.mergebot_ignored_check_names(
                comments, checks, since_iso="2026-05-08T13:00:00Z"
            ),
            {"pull / linux", "trunk / rocm"},
        )

    def test_untrusted_merge_i_command_is_ignored(self):
        checks = [{"name": "pull / linux", "bucket": "fail"}]
        comments = [
            {
                "author": "external",
                "author_association": "CONTRIBUTOR",
                "created_at": "2026-05-08T14:00:00Z",
                "body": taint("@pytorchbot merge -i", "pr_comment"),
            }
        ]

        self.assertEqual(
            handoff.mergebot_ignored_check_names(
                comments, checks, since_iso="2026-05-08T13:00:00Z"
            ),
            set(),
        )


class TestWatchPostHandoff(unittest.TestCase):
    def test_post_handoff_ci_status_ignores_suppressed_failures(self):
        with mock.patch.object(
            handoff.github,
            "get_pr_checks_all",
            return_value=[
                {"name": "lint", "bucket": "fail"},
                {"name": "test", "bucket": "pass"},
            ],
        ):
            self.assertEqual(
                handoff._post_handoff_ci_status(
                    101, suppressed_check_names={"lint"}
                ),
                ("passed", 2, 2, 0, 1),
            )

    def test_post_handoff_ci_status_cache_reuses_unchanged_pending_result(self):
        cache = handoff._PostHandoffCiCache()
        with (
            mock.patch.object(
                handoff.github,
                "list_workflow_runs_for_sha",
                return_value=[{"id": 1, "status": "in_progress", "conclusion": None}],
            ),
            mock.patch.object(
                handoff.github,
                "get_pr_checks_all",
                return_value=[{"name": "lint", "bucket": "pending"}],
            ) as checks,
            mock.patch.object(handoff.time, "time", side_effect=[100.0, 120.0]),
        ):
            first = handoff._post_handoff_ci_status_cached(
                101, head_sha="abc", cache=cache
            )
            second = handoff._post_handoff_ci_status_cached(
                101, head_sha="abc", cache=cache
            )

        self.assertEqual(first, ("pending", 0, 1, 0, 0))
        self.assertEqual(second, first)
        checks.assert_called_once_with(101, head_sha="abc")

    def test_post_handoff_ci_status_cache_refetches_on_suppression_change(self):
        # The merging path polls without suppressions while the post-handoff
        # path polls with them; a result cached by one must not be reused by
        # the other, or suppressed failures leak into "CI regressed".
        cache = handoff._PostHandoffCiCache()
        runs = [{"id": 1, "status": "completed", "conclusion": "failure"}]
        with (
            mock.patch.object(
                handoff.github,
                "list_workflow_runs_for_sha",
                return_value=runs,
            ),
            mock.patch.object(
                handoff.github,
                "get_pr_checks_all",
                return_value=[{"name": "lint", "bucket": "fail"}],
            ) as checks,
            mock.patch.object(handoff.time, "time", side_effect=[100.0, 120.0]),
        ):
            unsuppressed = handoff._post_handoff_ci_status_cached(
                101, head_sha="abc", cache=cache
            )
            suppressed = handoff._post_handoff_ci_status_cached(
                101,
                head_sha="abc",
                cache=cache,
                suppressed_check_names={"lint"},
            )

        self.assertEqual(unsuppressed, ("failed", 1, 1, 1, 0))
        self.assertEqual(suppressed, ("passed", 1, 1, 0, 1))
        self.assertEqual(checks.call_count, 2)

    def test_post_handoff_ci_status_cache_refetches_on_workflow_change(self):
        cache = handoff._PostHandoffCiCache()
        with (
            mock.patch.object(
                handoff.github,
                "list_workflow_runs_for_sha",
                side_effect=[
                    [{"id": 1, "status": "in_progress", "conclusion": None}],
                    [{"id": 1, "status": "completed", "conclusion": "failure"}],
                ],
            ),
            mock.patch.object(
                handoff.github,
                "get_pr_checks_all",
                side_effect=[
                    [{"name": "lint", "bucket": "pending"}],
                    [{"name": "lint", "bucket": "fail"}],
                ],
            ) as checks,
            mock.patch.object(handoff.time, "time", side_effect=[100.0, 120.0]),
        ):
            handoff._post_handoff_ci_status_cached(
                101, head_sha="abc", cache=cache
            )
            second = handoff._post_handoff_ci_status_cached(
                101, head_sha="abc", cache=cache
            )

        self.assertEqual(second, ("failed", 1, 1, 1, 0))
        self.assertEqual(checks.call_count, 2)

    def test_watch_post_handoff_returns_conflict_from_github_merge_state(self):
        with mock.patch.object(
            handoff.github,
            "get_pr",
            return_value={
                "number": 101,
                "state": "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
                "mergeStateStatus": "DIRTY",
            },
        ), mock.patch.object(handoff.github, "get_pr_comments", return_value=[]):
            result = handoff.watch_post_handoff(
                101, "2026-05-08T13:00:00Z"
            )

        self.assertEqual(result, ("conflict", None, None))

    def test_watch_post_handoff_does_not_interrupt_active_merge_for_dirty_state(self):
        pr_states = [
            {
                "number": 101,
                "state": "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [{"name": github.MERGING_LABEL}],
                "mergeStateStatus": "DIRTY",
            },
            {
                "number": 101,
                "state": "CLOSED",
                "reviewDecision": "APPROVED",
                "labels": [],
                "mergeStateStatus": "DIRTY",
            },
        ]

        with mock.patch.object(
            handoff.github, "get_pr", side_effect=pr_states
        ), mock.patch.object(
            handoff.github, "get_pr_comments", return_value=[]
        ), mock.patch.object(
            handoff.github, "get_pr_checks_all", return_value=[]
        ), mock.patch.object(
            handoff.time, "sleep"
        ):
            result = handoff.watch_post_handoff(
                101, "2026-05-08T13:00:00Z"
            )

        self.assertEqual(result, ("closed", None, None))

    def test_watch_post_handoff_returns_ci_failed_when_checks_regress(self):
        with (
            mock.patch.object(
                handoff.github,
                "get_pr",
                return_value={
                    "number": 101,
                    "state": "OPEN",
                    "reviewDecision": "APPROVED",
                    "labels": [],
                    "mergeStateStatus": "CLEAN",
                },
            ),
            mock.patch.object(handoff.github, "get_pr_comments", return_value=[]),
            mock.patch.object(
                handoff.github,
                "get_pr_checks_all",
                return_value=[
                    {"name": "lint", "bucket": "fail"},
                    {"name": "test", "bucket": "pass"},
                ],
            ),
            mock.patch.object(handoff, "write_status") as write_status,
        ):
            result = handoff.watch_post_handoff(
                101,
                "2026-05-08T13:00:00Z",
                intervention_count=2,
                human_ack_sha="a" * 40,
            )

        self.assertEqual(result, ("ci_failed", None, None))
        write_status.assert_called_once()
        self.assertEqual(write_status.call_args.kwargs["phase"], "polling_ci")
        self.assertEqual(write_status.call_args.kwargs["category"], "action")
        self.assertEqual(write_status.call_args.kwargs["action"], "inspecting_ci")
        self.assertEqual(write_status.call_args.kwargs["ci_failed"], 1)
        self.assertIn("CI regressed after handoff", write_status.call_args.kwargs["message"])

    def test_handoff_status_surfaces_suppressed_failures(self):
        with mock.patch.object(handoff, "write_status") as write_status:
            handoff._write_handoff_status(
                101,
                approved=True,
                merging=False,
                intervention_count=0,
                suppressed_failure_count=2,
            )

        # Suppressed count is surfaced via the mux Sup column (ci_suppressed),
        # not repeated in the Status text.
        kwargs = write_status.call_args.kwargs
        self.assertEqual(kwargs["phase"], "ready")
        self.assertEqual(kwargs["message"], "ready for human merge")
        self.assertEqual(kwargs["ci_suppressed"], 2)

    def test_handoff_status_surfaces_comment_failure_and_drci_warning(self):
        with mock.patch.object(handoff, "write_status") as write_status:
            handoff._write_handoff_status(
                101,
                approved=True,
                merging=False,
                intervention_count=0,
                suppressed_failure_count=2,
                handoff_comment_ok=False,
                suppression_warning="suppression list differs from Dr. CI",
            )

        kwargs = write_status.call_args.kwargs
        self.assertIn("suppression list differs from Dr. CI", kwargs["message"])
        self.assertIn("handoff comment failed", kwargs["message"])
        self.assertEqual(
            kwargs["user_action"],
            "review local mergedog log and merge when satisfied",
        )
        self.assertFalse(kwargs["handoff_comment_ok"])
        self.assertEqual(
            kwargs["suppression_warning"],
            "suppression list differs from Dr. CI",
        )

    def test_handoff_status_marks_cla_as_contributor_wait(self):
        with mock.patch.object(handoff, "write_status") as write_status:
            handoff._write_handoff_status(
                101,
                approved=True,
                merging=False,
                intervention_count=0,
                cla_blocked=True,
            )

        kwargs = write_status.call_args.kwargs
        self.assertEqual(kwargs["category"], "waiting")
        self.assertEqual(kwargs["waiting_on"], "contributor")
        self.assertIsNone(kwargs["user_action"])
        self.assertIn("waiting for contributor CLA", kwargs["message"])

    def test_watch_post_handoff_reports_cla_wait_instead_of_ready(self):
        pr_states = [
            {
                "number": 101,
                "state": "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
                "mergeStateStatus": "CLEAN",
            },
            {
                "number": 101,
                "state": "CLOSED",
                "reviewDecision": "APPROVED",
                "labels": [],
                "mergeStateStatus": "CLEAN",
            },
        ]

        with (
            mock.patch.object(handoff.github, "get_pr", side_effect=pr_states),
            mock.patch.object(handoff.github, "get_pr_comments", return_value=[]),
            mock.patch.object(
                handoff.github,
                "get_pr_checks_all",
                return_value=[{"name": "test", "bucket": "pass"}],
            ),
            mock.patch.object(handoff, "write_status") as write_status,
            mock.patch.object(handoff.time, "sleep"),
        ):
            result = handoff.watch_post_handoff(
                101, "2026-05-08T13:00:00Z", cla_blocked=True
            )

        self.assertEqual(result, ("closed", None, None))
        first = write_status.call_args_list[0].kwargs
        self.assertEqual(first["phase"], "ready")
        self.assertEqual(first["category"], "waiting")
        self.assertEqual(first["waiting_on"], "contributor")
        self.assertIsNone(first["user_action"])
        self.assertIn("waiting for contributor CLA", first["message"])

    def test_watch_post_handoff_reports_pending_ci_instead_of_ready(self):
        pr_states = [
            {
                "number": 101,
                "state": "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
                "mergeStateStatus": "CLEAN",
            },
            {
                "number": 101,
                "state": "CLOSED",
                "reviewDecision": "APPROVED",
                "labels": [],
                "mergeStateStatus": "CLEAN",
            },
        ]

        with (
            mock.patch.object(handoff.github, "get_pr", side_effect=pr_states),
            mock.patch.object(handoff.github, "get_pr_comments", return_value=[]),
            mock.patch.object(
                handoff.github,
                "get_pr_checks_all",
                return_value=[{"name": "lint", "bucket": "pending"}],
            ),
            mock.patch.object(handoff, "write_status") as write_status,
            mock.patch.object(handoff.time, "sleep"),
        ):
            result = handoff.watch_post_handoff(
                101, "2026-05-08T13:00:00Z"
            )

        self.assertEqual(result, ("closed", None, None))
        first = write_status.call_args_list[0].kwargs
        self.assertEqual(first["phase"], "polling_ci")
        self.assertEqual(first["category"], "waiting")
        self.assertEqual(first["waiting_on"], "ci")
        self.assertIn("waiting for CI after handoff", first["message"])


if __name__ == "__main__":
    unittest.main()
