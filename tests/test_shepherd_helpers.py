import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from mergedog.claude import LLMResult
from mergedog import shepherd
from mergedog.shepherd import (
    MIN_USEFUL_LOG_CHARS,
    _actionable_lint_failure_names,
    _failed_logs_are_content_free,
    _filter_spurious_failed_jobs,
    _inconclusive_refresh_target,
    _is_ghstack_mergeability_failure,
    _llm_halt_message,
    _llm_requested_rebase,
    _llm_signalled_inconclusive,
    _latest_completed_at,
    _count_mergedog_interventions_since_ack,
    _spurious_check_names_from_checks,
    describe_log_state,
)


class TestMergedogLabelManagement(unittest.TestCase):
    def _run_shepherd_wrapper(self, **kwargs):
        pr_data = {"number": 123, "labels": [], "isDraft": False, "state": "OPEN"}
        with mock.patch.object(shepherd.repo, "ensure_clone"), mock.patch.object(
            shepherd.repo, "fetch_origin"
        ), mock.patch.object(
            shepherd.github, "get_pr", return_value=pr_data
        ), mock.patch.object(
            shepherd, "_validate_pr"
        ), mock.patch.object(
            shepherd, "_shepherd_body"
        ), mock.patch.object(
            shepherd.signal, "signal"
        ), mock.patch.object(
            shepherd.faulthandler, "enable"
        ), mock.patch.object(
            shepherd.faulthandler, "register"
        ), mock.patch.object(
            shepherd.github, "add_label"
        ) as add_label, mock.patch.object(
            shepherd.github, "remove_label"
        ) as remove_label:
            shepherd.shepherd(123, **kwargs)
        return add_label, remove_label

    def test_default_does_not_touch_mergedog_label(self):
        add_label, remove_label = self._run_shepherd_wrapper()
        add_label.assert_not_called()
        remove_label.assert_not_called()

    def test_explicit_flag_adds_and_removes_mergedog_label(self):
        add_label, remove_label = self._run_shepherd_wrapper(
            manage_mergedog_label=True
        )
        add_label.assert_called_once_with(123, shepherd.MERGEDOG_LABEL)
        remove_label.assert_called_once_with(123, shepherd.MERGEDOG_LABEL)

    def test_max_fix_commits_passes_through_to_body(self):
        pr_data = {"number": 123, "labels": [], "isDraft": False, "state": "OPEN"}
        with mock.patch.object(shepherd.repo, "ensure_clone"), mock.patch.object(
            shepherd.repo, "fetch_origin"
        ), mock.patch.object(
            shepherd.github, "get_pr", return_value=pr_data
        ), mock.patch.object(
            shepherd, "_validate_pr"
        ), mock.patch.object(
            shepherd, "_shepherd_body"
        ) as body, mock.patch.object(
            shepherd.signal, "signal"
        ), mock.patch.object(
            shepherd.faulthandler, "enable"
        ), mock.patch.object(
            shepherd.faulthandler, "register"
        ):
            shepherd.shepherd(123, max_fix_commits=0)

        body.assert_called_once()
        self.assertEqual(body.call_args.args[6], 0)


class TestFailedLogsAreContentFree(unittest.TestCase):
    def test_empty_list_is_content_free(self):
        self.assertTrue(_failed_logs_are_content_free([]))

    def test_no_log_available_placeholder_is_content_free(self):
        self.assertTrue(
            _failed_logs_are_content_free(
                [
                    ("a", "<no log available>"),
                    ("b", "<no log available>"),
                ]
            )
        )

    def test_long_log_is_substantial(self):
        self.assertFalse(
            _failed_logs_are_content_free(
                [("a", "x" * (MIN_USEFUL_LOG_CHARS + 1))]
            )
        )

    def test_short_stub_below_threshold(self):
        # A few bytes of whitespace-padded noise still counts as content-free.
        self.assertTrue(
            _failed_logs_are_content_free([("a", "   error\n   ")])
        )

    def test_mixed_one_real_one_empty_is_substantial(self):
        # If even one job has real logs, claude has something to act on.
        self.assertFalse(
            _failed_logs_are_content_free(
                [
                    ("a", "<no log available>"),
                    ("b", "y" * (MIN_USEFUL_LOG_CHARS + 1)),
                ]
            )
        )


class TestLLMHaltMessage(unittest.TestCase):
    def test_uses_specific_halt_reason(self):
        result = LLMResult(
            ran_cleanly=False,
            new_sha=None,
            transcript=[],
            halt_reason="signalled INCONCLUSIVE; halting for human review",
        )

        with mock.patch("mergedog.shepherd._llm_label", return_value="claude"):
            self.assertEqual(
                _llm_halt_message(result, "claude exited abnormally"),
                "claude signalled INCONCLUSIVE; halting for human review",
            )

    def test_falls_back_without_specific_reason(self):
        result = LLMResult(
            ran_cleanly=False,
            new_sha=None,
            transcript=[],
        )

        self.assertEqual(
            _llm_halt_message(result, "claude exited abnormally"),
            "claude exited abnormally",
        )


class TestInconclusiveRefresh(unittest.TestCase):
    def test_detects_inconclusive_halt_reason(self):
        self.assertTrue(
            _llm_signalled_inconclusive(
                LLMResult(
                    ran_cleanly=False,
                    new_sha=None,
                    transcript=[],
                    halt_reason="signalled INCONCLUSIVE; halting for human review",
                )
            )
        )
        self.assertFalse(
            _llm_signalled_inconclusive(
                LLMResult(
                    ran_cleanly=False,
                    new_sha=None,
                    transcript=[],
                    halt_reason=(
                        "reported a real PR-related failure that is too hard "
                        "to fix safely"
                    ),
                )
            )
        )

    def test_refresh_target_reports_advancing_known_good_ref(self):
        with mock.patch.object(
            shepherd.repo,
            "select_rebase_target",
            return_value=("origin/viable/strict", "viable/strict"),
        ), mock.patch.object(
            shepherd.repo,
            "rebase_target_advances",
            return_value=True,
        ) as advances:
            can_refresh, reason = _inconclusive_refresh_target(Path("/tmp/wt"))

        self.assertTrue(can_refresh)
        self.assertEqual(reason, "viable/strict")
        advances.assert_called_once_with(Path("/tmp/wt"), "origin/viable/strict")


class TestRebaseRequest(unittest.TestCase):
    def test_detects_rebase_request_halt_reason(self):
        self.assertTrue(
            _llm_requested_rebase(
                LLMResult(
                    ran_cleanly=False,
                    new_sha=None,
                    transcript=[],
                    halt_reason="requested REBASE; refreshing stale base",
                )
            )
        )
        self.assertFalse(
            _llm_requested_rebase(
                LLMResult(
                    ran_cleanly=False,
                    new_sha=None,
                    transcript=[],
                    halt_reason="signalled INCONCLUSIVE; halting for human review",
                )
            )
        )


class TestGhstackMergeabilityFailure(unittest.TestCase):
    def test_detects_mergeability_check_name(self):
        self.assertTrue(
            _is_ghstack_mergeability_failure(["ghstack-mergeability-check"])
        )
        self.assertTrue(
            _is_ghstack_mergeability_failure(
                ["Check mergeability of ghstack PR"]
            )
        )
        self.assertFalse(_is_ghstack_mergeability_failure(["pull / linux"]))


class TestInterventionCount(unittest.TestCase):
    def _git(self, worktree: Path, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    def _commit(self, worktree: Path, subject: str) -> str:
        path = worktree / "file.txt"
        path.write_text(path.read_text() + subject + "\n")
        self._git(worktree, "add", "file.txt")
        self._git(worktree, "commit", "-m", subject)
        return self._git(worktree, "rev-parse", "HEAD")

    def test_counts_mergedog_commits_since_human_ack(self):
        with tempfile.TemporaryDirectory() as d:
            worktree = Path(d)
            self._git(worktree, "init")
            self._git(worktree, "config", "user.name", "Test User")
            self._git(worktree, "config", "user.email", "test@example.com")
            (worktree / "file.txt").write_text("")
            self._git(worktree, "add", "file.txt")
            self._git(worktree, "commit", "-m", "Contributor change")
            ack_sha = self._git(worktree, "rev-parse", "HEAD")

            self._commit(worktree, "[MERGEDOG] Fix CI")
            self._commit(worktree, "Contributor follow-up")
            self._commit(worktree, "[MERGEDOG] Merge main into PR branch")

            self.assertEqual(
                _count_mergedog_interventions_since_ack(worktree, ack_sha),
                2,
            )
            self.assertEqual(
                _count_mergedog_interventions_since_ack(
                    worktree, self._git(worktree, "rev-parse", "HEAD")
                ),
                0,
            )


class TestInlineHunkComments(unittest.TestCase):
    def test_posts_missing_inline_hunk_marker(self):
        sha = "a" * 40
        head_sha = "b" * 40
        target = shepherd.repo.DiffHunkCommentTarget("foo.py", "RIGHT", 10)

        with (
            mock.patch.object(
                shepherd.repo,
                "diff_hunk_comment_targets",
                return_value=[target],
            ),
            mock.patch.object(
                shepherd.github,
                "get_pr_review_comments",
                return_value=[],
            ),
            mock.patch.object(
                shepherd.github, "post_pr_review_comment"
            ) as post_comment,
            mock.patch.object(shepherd, "log"),
        ):
            shepherd._post_llm_hunk_comments(
                123, Path("/tmp/wt"), sha, commit_id=head_sha
            )

        post_comment.assert_called_once()
        self.assertEqual(post_comment.call_args.kwargs["commit_id"], head_sha)
        self.assertEqual(post_comment.call_args.kwargs["path"], "foo.py")
        self.assertEqual(post_comment.call_args.kwargs["line"], 10)
        self.assertEqual(post_comment.call_args.kwargs["side"], "RIGHT")
        self.assertIn(
            f"https://github.com/{shepherd.REPO_SLUG}/commit/{sha}",
            post_comment.call_args.kwargs["body"],
        )

    def test_skips_existing_inline_hunk_marker(self):
        sha = "a" * 40
        target = shepherd.repo.DiffHunkCommentTarget("foo.py", "RIGHT", 10)
        key = shepherd._inline_hunk_key(sha, target)

        with (
            mock.patch.object(
                shepherd.repo,
                "diff_hunk_comment_targets",
                return_value=[target],
            ),
            mock.patch.object(
                shepherd.github,
                "get_pr_review_comments",
                return_value=[
                    {"body": shepherd._inline_hunk_comment_body(sha, key)}
                ],
            ),
            mock.patch.object(
                shepherd.github, "post_pr_review_comment"
            ) as post_comment,
        ):
            shepherd._post_llm_hunk_comments(123, Path("/tmp/wt"), sha)

        post_comment.assert_not_called()


class TestDescribeLogState(unittest.TestCase):
    def test_empty_failed_list_calls_out_status_only_checks(self):
        self.assertIn(
            "0 of 3 failing checks have Actions logs",
            describe_log_state([], failing_check_count=3),
        )

    def test_non_empty_reports_run_count_and_chars(self):
        self.assertEqual(
            describe_log_state(
                [("a", "abcde"), ("b", "fghij")], failing_check_count=2
            ),
            "2 run(s), 10 chars",
        )

    def test_no_log_available_placeholder_doesnt_count_chars(self):
        self.assertEqual(
            describe_log_state(
                [("a", "<no log available>"), ("b", "real")],
                failing_check_count=2,
            ),
            "2 run(s), 4 chars",
        )


class TestLatestCompletedAt(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_latest_completed_at([]))

    def test_picks_max_timestamp(self):
        result = _latest_completed_at(
            [
                {"completedAt": "2026-05-06T10:00:00Z"},
                {"completedAt": "2026-05-06T11:30:00Z"},
                {"completedAt": "2026-05-06T09:15:00Z"},
            ]
        )
        expected = datetime(2026, 5, 6, 11, 30, tzinfo=timezone.utc).timestamp()
        self.assertEqual(result, expected)

    def test_missing_completed_at_returns_none(self):
        self.assertIsNone(
            _latest_completed_at(
                [
                    {"completedAt": "2026-05-06T10:00:00Z"},
                    {"completedAt": ""},
                ]
            )
        )

    def test_zero_placeholder_returns_none(self):
        self.assertIsNone(
            _latest_completed_at(
                [{"completedAt": "0001-01-01T00:00:00Z"}]
            )
        )

    def test_unparseable_returns_none(self):
        self.assertIsNone(
            _latest_completed_at([{"completedAt": "not-a-date"}])
        )


class TestSpuriousCheckNames(unittest.TestCase):
    def test_collects_only_named_failed_checks(self):
        self.assertEqual(
            _spurious_check_names_from_checks(
                [
                    {"name": "pull / linux", "bucket": "fail"},
                    {"name": "lint", "bucket": "cancel"},
                    {"name": "docs", "bucket": "pass"},
                    {"name": "", "bucket": "fail"},
                    {"bucket": "fail"},
                ]
            ),
            {"pull / linux", "lint"},
        )

    def test_workflow_only_failure_has_no_check_to_mark(self):
        self.assertEqual(_spurious_check_names_from_checks([]), set())


class TestFilterSpuriousFailedJobs(unittest.TestCase):
    def test_filters_logs_for_marked_spurious_checks(self):
        failed = [
            ("pull / linux", "real"),
            ("trunk / xpu", "unrelated"),
            ("trunk / rocm", "unrelated"),
        ]

        self.assertEqual(
            _filter_spurious_failed_jobs(
                failed, {"trunk / xpu", "trunk / rocm"}
            ),
            [("pull / linux", "real")],
        )

    def test_no_spurious_names_preserves_original_list(self):
        failed = [("pull / linux", "real")]

        self.assertIs(_filter_spurious_failed_jobs(failed, set()), failed)


class TestActionableLintFailureNames(unittest.TestCase):
    def test_detects_lintrunner_diagnostic(self):
        log = (
            "\x1b[1m>>>\x1b[0m Lint for \x1b[4mc10/core/TensorOptions.h\x1b[0m:\n"
            "  Error (CLANGTIDY) [modernize-use-constraints]\n"
            "Lint failed!\n"
        )

        self.assertEqual(
            _actionable_lint_failure_names(
                [("lintrunner-clang-partial / lint", log)]
            ),
            ["lintrunner-clang-partial / lint"],
        )

    def test_ignores_infra_lint_failure_without_diagnostic(self):
        log = "failed to download linter\nLint failed!\n"

        self.assertEqual(
            _actionable_lint_failure_names(
                [("lintrunner-clang-partial / lint", log)]
            ),
            [],
        )


class TestRecoverPendingGhstackPublish(unittest.TestCase):
    def _trust(self, **kwargs):
        from mergedog.state import TrustDB

        trust = TrustDB(pr=1, **kwargs)
        trust.save = mock.Mock()
        return trust

    def test_noop_without_pending_record(self):
        trust = self._trust()
        with mock.patch.object(shepherd.repo, "fetch_ghstack_orig") as fetch:
            shepherd._recover_pending_ghstack_publish(
                trust, {"headRefOid": "a" * 40}, "gh/u/1/head"
            )
        fetch.assert_not_called()
        trust.save.assert_not_called()

    def test_trusts_head_when_orig_patch_id_matches(self):
        trust = self._trust(pending_publish_orig_sha="b" * 40)
        with mock.patch.object(
            shepherd.repo, "fetch_ghstack_orig", return_value="c" * 40
        ), mock.patch.object(
            shepherd.repo, "patch_id_matches_any", return_value=True
        ) as match:
            shepherd._recover_pending_ghstack_publish(
                trust, {"headRefOid": "a" * 40}, "gh/u/1/head"
            )
        match.assert_called_once_with("c" * 40, ["b" * 40])
        self.assertIn("a" * 40, trust.trusted_shas)
        self.assertEqual(trust.pending_publish_orig_sha, "")

    def test_leaves_head_untrusted_on_mismatch(self):
        trust = self._trust(pending_publish_orig_sha="b" * 40)
        with mock.patch.object(
            shepherd.repo, "fetch_ghstack_orig", return_value="c" * 40
        ), mock.patch.object(
            shepherd.repo, "patch_id_matches_any", return_value=False
        ):
            shepherd._recover_pending_ghstack_publish(
                trust, {"headRefOid": "a" * 40}, "gh/u/1/head"
            )
        self.assertNotIn("a" * 40, trust.trusted_shas)
        # Record is cleared either way: it described a publish that no
        # longer matches reality.
        self.assertEqual(trust.pending_publish_orig_sha, "")

    def test_clears_record_when_head_already_trusted(self):
        trust = self._trust(
            pending_publish_orig_sha="b" * 40,
            trusted_shas=["a" * 40],
        )
        with mock.patch.object(shepherd.repo, "fetch_ghstack_orig") as fetch:
            shepherd._recover_pending_ghstack_publish(
                trust, {"headRefOid": "a" * 40}, "gh/u/1/head"
            )
        fetch.assert_not_called()
        self.assertEqual(trust.pending_publish_orig_sha, "")

    def test_keeps_record_when_orig_fetch_fails(self):
        trust = self._trust(pending_publish_orig_sha="b" * 40)
        with mock.patch.object(
            shepherd.repo,
            "fetch_ghstack_orig",
            side_effect=RuntimeError("network down"),
        ):
            shepherd._recover_pending_ghstack_publish(
                trust, {"headRefOid": "a" * 40}, "gh/u/1/head"
            )
        self.assertEqual(trust.pending_publish_orig_sha, "b" * 40)
        trust.save.assert_not_called()


class TestGhstackSubmitTrusted(unittest.TestCase):
    def test_records_orig_before_submit_and_clears_after(self):
        from mergedog.state import TrustDB

        trust = TrustDB(pr=1)
        trust.save = mock.Mock()
        recorded_at_submit: list[str] = []

        def fake_submit(worktree, message):
            recorded_at_submit.append(trust.pending_publish_orig_sha)

        with mock.patch.object(
            shepherd.repo, "head_sha", return_value="b" * 40
        ), mock.patch.object(
            shepherd.repo, "ghstack_submit", side_effect=fake_submit
        ), mock.patch.object(
            shepherd.repo, "fetch_ghstack_head", return_value="a" * 40
        ):
            new_head = shepherd._ghstack_submit_trusted(
                Path("/tmp/wt"), "gh/u/1/head", trust, "msg"
            )

        self.assertEqual(new_head, "a" * 40)
        # The pending record must be on disk before ghstack pushes.
        self.assertEqual(recorded_at_submit, ["b" * 40])
        self.assertEqual(trust.pending_publish_orig_sha, "")
        self.assertIn("a" * 40, trust.trusted_shas)


class TestPublishGhstackFix(unittest.TestCase):
    def test_pushes_audit_commit_before_fixup_and_returns_head(self):
        from mergedog.state import TrustDB

        trust = TrustDB(pr=1)
        worktree = Path("/tmp/wt")
        fix_sha = "b" * 40
        head_sha = "a" * 40
        events: list[str] = []

        with (
            mock.patch.object(
                shepherd.repo, "commit_message", return_value="[MERGEDOG] fix"
            ) as commit_message,
            mock.patch.object(shepherd, "_wait_for_no_active_sev") as wait_sev,
            mock.patch.object(
                shepherd.repo,
                "push_ref",
                side_effect=lambda *args: events.append("push"),
            ) as push_ref,
            mock.patch.object(
                shepherd.repo,
                "fixup_into_parent",
                side_effect=lambda *args: events.append("fixup"),
            ) as fixup,
            mock.patch.object(
                shepherd, "_ghstack_submit_trusted", return_value=head_sha
            ) as submit,
            mock.patch.object(shepherd, "_wait_for_pr_head") as wait_head,
            mock.patch.object(shepherd, "log"),
        ):
            result = shepherd._publish_ghstack_fix(
                123,
                worktree,
                "gh/u/1/head",
                fix_sha,
                trust,
                ignore_sev=False,
            )

        self.assertEqual(result, head_sha)
        commit_message.assert_called_once_with(worktree, fix_sha)
        wait_sev.assert_has_calls(
            [
                mock.call(
                    "pushing ghstack LLM audit commit", ignore_sev=False
                ),
                mock.call("re-publishing via ghstack submit", ignore_sev=False),
            ]
        )
        push_ref.assert_called_once_with(
            worktree,
            "origin",
            fix_sha,
            f"refs/heads/mergedog/123/{fix_sha}",
        )
        fixup.assert_called_once_with(worktree)
        self.assertEqual(events, ["push", "fixup"])
        submit.assert_called_once_with(
            worktree, "gh/u/1/head", trust, "[MERGEDOG] fix"
        )
        wait_head.assert_called_once_with(123, head_sha)


class TestLogRestoredState(unittest.TestCase):
    def _logged_lines(self, trust):
        with mock.patch.object(shepherd, "log") as logged:
            shepherd._log_restored_state(trust)
        return [c.args[0] for c in logged.call_args_list]

    def test_silent_on_fresh_state(self):
        from mergedog.state import TrustDB

        self.assertEqual(self._logged_lines(TrustDB(pr=1)), [])

    def test_reports_restored_fields(self):
        from mergedog.state import TrustDB

        trust = TrustDB(
            pr=1,
            trusted_shas=["a" * 40, "b" * 40],
            spurious_check_names=["lint / foo"],
            last_observed_failure_iso="2026-06-01T00:00:00Z",
            last_observed_failure_body="CONFLICT ... Merge conflict",
        )
        lines = self._logged_lines(trust)
        joined = "\n".join(lines)
        self.assertIn("restored state from previous run:", lines[0])
        self.assertIn("trusted SHAs: 2", joined)
        self.assertIn("lint / foo", joined)
        self.assertIn("2026-06-01T00:00:00Z (merge conflict)", joined)


class TestClassifyFailureBody(unittest.TestCase):
    def test_classifications(self):
        self.assertEqual(shepherd._classify_failure_body(""), "none")
        self.assertEqual(
            shepherd._classify_failure_body("CONFLICT x Merge conflict"),
            "merge conflict",
        )
        self.assertEqual(
            shepherd._classify_failure_body("HTTP Error 504"),
            "retryable infra flake",
        )
        self.assertEqual(
            shepherd._classify_failure_body("something else"),
            "unclassified",
        )


if __name__ == "__main__":
    unittest.main()
