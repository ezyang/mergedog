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
            "- `dddddddddddd` — merged main into the PR branch: "
            "[MERGEDOG] Merge main into PR branch",
            body,
        )
        self.assertIn(
            "- `cccccccccccc` — pushed fix commit cccccccccccc "
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

    def test_handoff_status_waits_for_approval_when_not_approved(self):
        with mock.patch.object(handoff, "write_status") as write_status:
            handoff._write_handoff_status(
                101,
                approved=False,
                merging=False,
                intervention_count=2,
                human_ack_sha="a" * 40,
            )

        write_status.assert_called_once_with(
            101,
            phase="ready",
            category="ready",
            waiting_on="approval",
            user_action="approve the PR after reviewing mergedog interventions",
            message=(
                "waiting for maintainer approval; "
                "2 mergedog interventions since last approval"
            ),
            intervention_count=2,
            human_ack_sha="a" * 40,
            approved=False,
            merging=False,
        )


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


class TestWatchStackPostHandoff(unittest.TestCase):
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

        self.assertEqual(write_status.call_args.kwargs["phase"], "ready")
        self.assertIn(
            "ready for human merge; 2 suppressed failures",
            write_status.call_args.kwargs["message"],
        )

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

    def test_returns_failed_member(self):
        comments = {
            101: [],
            102: [
                {
                    "author": "pytorchmergebot",
                    "created_at": "2026-05-08T14:00:00Z",
                    "body": "## Merge failed\nCONFLICT (content): Merge conflict",
                }
            ],
        }

        def get_pr(pr, **kwargs):
            return {
                "number": pr,
                "state": "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
            }

        with mock.patch.object(
            handoff.github, "get_pr", side_effect=get_pr
        ), mock.patch.object(
            handoff.github,
            "get_pr_comments",
            side_effect=lambda pr: comments[pr],
        ):
            result = handoff.watch_stack_post_handoff(
                {
                    101: "2026-05-08T13:00:00Z",
                    102: "2026-05-08T13:00:00Z",
                }
            )

        self.assertEqual(result[0], "failed")
        self.assertEqual(result[1], 102)
        self.assertEqual(result[2], "2026-05-08T14:00:00Z")
        self.assertIn("Merge failed", result[3])

    def test_returns_closed_member(self):
        def get_pr(pr, **kwargs):
            return {
                "number": pr,
                "state": "CLOSED" if pr == 101 else "OPEN",
                "reviewDecision": "APPROVED",
                "labels": [],
            }

        with mock.patch.object(handoff.github, "get_pr", side_effect=get_pr):
            result = handoff.watch_stack_post_handoff(
                {
                    101: "2026-05-08T13:00:00Z",
                    102: "2026-05-08T13:00:00Z",
                }
            )

        self.assertEqual(result, ("closed", 101, None, None))


if __name__ == "__main__":
    unittest.main()
