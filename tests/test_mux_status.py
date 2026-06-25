import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import mux


class _FakeProc:
    def __init__(self, rc):
        self._rc = rc
        self.pid = 12345

    def poll(self):
        return self._rc


class _FakeTable:
    def __init__(self):
        self.rows = []

    def clear(self):
        self.rows.clear()

    def add_row(self, *cells):
        self.rows.append(cells)


class _FakeHint:
    def __init__(self):
        self.content = ""
        self.display = False

    def update(self, content):
        self.content = content


def _query_one_for(table, hint=None):
    hint = hint or _FakeHint()

    def query_one(selector, *args):
        if selector is mux.DataTable:
            return table
        if selector == "#cleanup-hint":
            return hint
        raise AssertionError(f"unexpected query_one selector: {selector!r}")

    return query_one


class TestMuxInput(unittest.TestCase):
    def test_command_suggester_completes_cleanup_prefix(self):
        suggester = mux.SuggestFromList(mux.COMMAND_SUGGESTIONS)

        suggestion = asyncio.run(suggester.get_suggestion("cle"))

        self.assertEqual(suggestion, "cleanup")

    def test_compose_wires_command_suggester_to_input(self):
        app = mux.MuxApp([], lock_fd=-1)
        widgets = list(app.compose())
        inputs = [w for w in widgets if isinstance(w, mux.HistoryInput)]

        self.assertEqual(len(inputs), 1)
        self.assertIsNotNone(inputs[0].suggester)


class TestMuxStructuredStatus(unittest.TestCase):
    def test_read_pr_title_prefers_context_sidecar_title(self):
        with tempfile.TemporaryDirectory() as d:
            context_path = Path(d) / "123.md"
            context_path.write_text(
                "PR #123\n\n[TITLE]\n[fx] Real PR title\n\n[DESCRIPTION]\nbody\n"
            )

            with (
                mock.patch.object(
                    mux, "context_file", return_value=context_path
                ),
                mock.patch.object(
                    mux, "_read_pr_worktree_title", return_value="commit title"
                ) as fallback,
            ):
                title = mux._read_pr_title(123)

        self.assertEqual(title, "[fx] Real PR title")
        fallback.assert_not_called()

    def test_worktree_title_fallback_stays_on_pr_first_parent(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)

            def git(*args):
                subprocess.run(
                    ["git", *args], cwd=repo, check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            git("init")
            git("config", "user.name", "Test User")
            git("config", "user.email", "test@example.com")
            (repo / "base.txt").write_text("base\n")
            git("add", "base.txt")
            git("commit", "-m", "base")
            git("checkout", "-b", "pr")
            (repo / "pr.txt").write_text("pr\n")
            git("add", "pr.txt")
            git("commit", "-m", "PR branch title")
            git("checkout", "-b", "upstream", "HEAD~1")
            (repo / "main.txt").write_text("main\n")
            git("add", "main.txt")
            git("commit", "-m", "Unrelated upstream title")
            git("checkout", "pr")
            git("merge", "--no-ff", "upstream", "-m", "[MERGEDOG] Merge main")

            with mock.patch.object(mux, "worktree_dir", return_value=repo):
                title = mux._read_pr_worktree_title(123)

        self.assertEqual(title, "PR branch title")

    def test_stack_display_layout_groups_parented_prs(self):
        jobs = [mux._pr_job(10), mux._pr_job(20), mux._pr_job(30)]
        parents = {
            10: ("C10", "C20"),
            20: ("C20", "MAIN"),
            30: ("C30", "MAIN"),
        }

        with mock.patch.object(
            mux,
            "_read_pr_commit_parent",
            side_effect=lambda pr: parents[pr],
        ):
            ordered, depths = mux._stack_display_layout(jobs)

        self.assertEqual(
            ordered,
            [mux._pr_job(20), mux._pr_job(10), mux._pr_job(30)],
        )
        self.assertEqual(depths[mux._pr_job(20)], 0)
        self.assertEqual(depths[mux._pr_job(10)], 1)
        self.assertEqual(depths[mux._pr_job(30)], 0)

    def test_stack_display_layout_uses_parent_hints_before_commit_graph(self):
        jobs = [mux._pr_job(10), mux._pr_job(20)]
        parents = {
            10: ("C10", "OLD_PARENT"),
            20: ("C20", "MAIN"),
        }

        with mock.patch.object(
            mux,
            "_read_pr_commit_parent",
            side_effect=lambda pr: parents[pr],
        ):
            ordered, depths = mux._stack_display_layout(
                jobs,
                {mux._pr_job(10): mux._pr_job(20)},
            )

        self.assertEqual(ordered, [mux._pr_job(20), mux._pr_job(10)])
        self.assertEqual(depths[mux._pr_job(10)], 1)

    def test_read_stack_parent_pr_from_log(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "10.log"
            log_path.write_text(
                "[12:29:22] stack parent PR #20 advanced; parent CI is pending\n"
            )

            parent = mux._read_stack_parent_pr_from_log(log_path)

        self.assertEqual(parent, 20)

    def test_refresh_indents_child_stack_rows(self):
        with tempfile.TemporaryDirectory() as d:
            parent_log = Path(d) / "20.log"
            child_log = Path(d) / "10.log"
            parent_log.write_text("parent\n")
            child_log.write_text("child\n")

            parent = mux._pr_job(20)
            child = mux._pr_job(10)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                child: (_FakeProc(None), object(), child_log),
                parent: (_FakeProc(None), object(), parent_log),
            }
            app._pr_titles = {parent: "Parent commit", child: "Child commit"}
            app._pr_status = {}
            table = _FakeTable()

            with (
                mock.patch.object(
                    app, "query_one", side_effect=_query_one_for(table)
                ),
                mock.patch.object(
                    mux,
                    "_stack_display_layout",
                    return_value=([parent, child], {parent: 0, child: 1}),
                ),
                mock.patch.object(mux, "read_status", return_value=None),
            ):
                app._refresh()

        self.assertEqual(table.rows[0][0].plain, "20")
        self.assertEqual(table.rows[1][0].plain, "  10")
        self.assertEqual(table.rows[0][1].plain, "Parent commit")
        self.assertEqual(table.rows[1][1].plain, "Child commit")

    def test_format_status_includes_shepherd_status_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text("[12:00:00] CI status -> pending (1/2 done)\n")
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {mux._pr_job(123): (_FakeProc(None), object(), log_path)}
            app._pr_titles = {mux._pr_job(123): "Test PR"}

            sidecar = {
                "schema_version": 1,
                "phase": "polling_ci",
                "category": "waiting",
                "message": "waiting for CI: 1/2 checks done",
            }
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["pr"], 123)
        self.assertEqual(rows[0]["state"], "running")
        self.assertEqual(rows[0]["phase"], "🟢")
        self.assertEqual(rows[0]["status"], "waiting for CI: 1/2 checks done")
        self.assertEqual(rows[0]["shepherd_status"], sidecar)

    def test_phase_label_marks_easy_operator_merge(self):
        sidecar = {
            "schema_version": 1,
            "phase": "ready",
            "category": "ready",
            "user_action": "merge when satisfied",
            "intervention_count": 0,
        }

        phase = mux._phase_label(sidecar, rc=None)

        self.assertEqual(phase, "🟡")

    def test_phase_label_marks_review_required_operator_action(self):
        sidecar = {
            "schema_version": 1,
            "phase": "ready",
            "category": "ready",
            "user_action": "review mergedog handoff and merge when satisfied",
            "intervention_count": 2,
        }

        phase = mux._phase_label(sidecar, rc=None)

        self.assertEqual(phase, "🟠")

    def test_phase_label_marks_approval_as_review_required(self):
        sidecar = {
            "schema_version": 1,
            "phase": "ready",
            "category": "ready",
            "waiting_on": "approval",
            "user_action": "approve after reviewing local mergedog log",
        }

        phase = mux._phase_label(sidecar, rc=None)

        self.assertEqual(phase, "🟠")

    def test_phase_label_marks_external_human_when_no_user_action(self):
        sidecar = {
            "schema_version": 1,
            "phase": "ready",
            "category": "waiting",
            "waiting_on": "approval",
            "message": "waiting for maintainer approval",
        }

        phase = mux._phase_label(sidecar, rc=None)

        self.assertEqual(phase, "🔵")

    def test_phase_label_marks_contributor_wait_as_external(self):
        sidecar = {
            "schema_version": 1,
            "phase": "ready",
            "category": "waiting",
            "waiting_on": "contributor",
            "message": "waiting for contributor CLA",
        }

        phase = mux._phase_label(sidecar, rc=None)

        self.assertEqual(phase, "🔵")

    def test_phase_label_marks_prior_cla_failure_as_external(self):
        sidecar = {
            "schema_version": 1,
            "phase": "ready",
            "category": "ready",
            "user_action": "merge when satisfied",
            "message": "ready for human merge; 2 suppressed failures",
        }
        body = "## Merge failed\n- EasyCLA\n"

        with mock.patch.object(
            mux, "_read_last_observed_failure_body", return_value=body
        ):
            cla_blocked = mux._status_has_cla_blocker(123, sidecar)

        phase = mux._phase_label(sidecar, rc=None, cla_blocked=cla_blocked)
        status = mux._cla_blocked_status_message(sidecar["message"])

        self.assertTrue(cla_blocked)
        self.assertEqual(phase, "🔵")
        self.assertEqual(
            status, "waiting for contributor CLA; 2 suppressed failures"
        )

    def test_help_documents_phase_meanings(self):
        app = mux.MuxApp.__new__(mux.MuxApp)

        help_text = app._dispatch_command("help")

        self.assertIn("🟢 no action", help_text)
        self.assertIn("🟡 you can merge", help_text)
        self.assertIn("🟠 review/approve first", help_text)
        self.assertIn("🔵 waiting on someone else", help_text)
        self.assertIn("🔴 halted/crashed", help_text)

    def test_completed_not_actionable_status_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "[12:00:00] PR is no longer open; shepherd complete\n"
            )
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (
                    _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                    object(),
                    log_path,
                )
            }
            app._pr_titles = {mux._pr_job(123): "Test PR"}

            with mock.patch.object(mux, "read_status", return_value=None):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["state"], "completed")
        self.assertIn("shepherd complete", rows[0]["last_log"])

    def test_refresh_keeps_completed_rows_until_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "[12:00:00] PR is no longer open; shepherd complete\n"
            )
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (
                    _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                    object(),
                    log_path,
                )
            }
            app._pr_titles = {mux._pr_job(123): "Test PR"}
            app._pr_status = {}
            table = _FakeTable()
            hint = _FakeHint()

            with (
                mock.patch.object(
                    app,
                    "query_one",
                    side_effect=_query_one_for(table, hint),
                ),
                mock.patch.object(app, "_prune_job") as prune_job,
                mock.patch.object(mux, "read_status", return_value=None),
            ):
                app._refresh()

        prune_job.assert_not_called()
        self.assertEqual(len(table.rows), 1)
        self.assertEqual(table.rows[0][2], "")
        self.assertEqual(hint.content, mux.CLEANUP_HINT)
        self.assertTrue(hint.display)

    def test_refresh_shows_cleanup_progress(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "[12:00:00] PR is no longer open; shepherd complete\n"
            )
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                job: (
                    _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                    object(),
                    log_path,
                )
            }
            app._cleanup_jobs = {job}
            app._cleanup_status = {
                job: "cleanup: removing worktree for [123] (1/1)"
            }
            app._pr_titles = {job: "Test PR"}
            app._pr_status = {}
            table = _FakeTable()
            hint = _FakeHint()

            with (
                mock.patch.object(
                    app,
                    "query_one",
                    side_effect=_query_one_for(table, hint),
                ),
                mock.patch.object(
                    mux,
                    "_stack_display_layout",
                    return_value=([job], {job: 0}),
                ),
                mock.patch.object(mux, "read_status") as read_status,
            ):
                app._refresh()

        read_status.assert_not_called()
        self.assertEqual(table.rows[0][2], "🟢")
        self.assertEqual(
            table.rows[0][3],
            "cleanup: removing worktree for [123] (1/1)",
        )
        self.assertEqual(hint.content, "")
        self.assertFalse(hint.display)

    def test_refresh_uses_sidecar_message_instead_of_log_tail(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text("[12:00:00] stale log line\n")
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {mux._pr_job(123): (_FakeProc(None), object(), log_path)}
            app._pr_titles = {mux._pr_job(123): "Test PR"}
            app._pr_status = {}
            table = _FakeTable()

            sidecar = {
                "schema_version": 1,
                "phase": "polling_ci",
                "category": "waiting",
                "message": "waiting for CI: 4/9 checks done",
            }
            with (
                mock.patch.object(
                    app, "query_one", side_effect=_query_one_for(table)
                ),
                mock.patch.object(
                    mux,
                    "_stack_display_layout",
                    return_value=([mux._pr_job(123)], {mux._pr_job(123): 0}),
                ),
                mock.patch.object(mux, "read_status", return_value=sidecar),
            ):
                app._refresh()

        self.assertEqual(table.rows[0][2], "🟢")
        self.assertEqual(table.rows[0][3], "waiting for CI: 4/9 checks done")

    def test_refresh_reports_stale_sidecar_from_prior_shepherd(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text("[12:00:00] new shepherd log line\n")
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {job: (_FakeProc(None), object(), log_path)}
            app._job_started_at = {job: 2_000_000_000.0}
            app._pr_titles = {job: "Test PR"}
            app._pr_status = {}
            table = _FakeTable()

            sidecar = {
                "schema_version": 1,
                "updated_at": "2020-01-01T00:00:00+00:00",
                "phase": "ready",
                "category": "ready",
                "message": "ready for human merge",
            }
            with (
                mock.patch.object(
                    app, "query_one", side_effect=_query_one_for(table)
                ),
                mock.patch.object(
                    mux,
                    "_stack_display_layout",
                    return_value=([job], {job: 0}),
                ),
                mock.patch.object(mux, "read_status", return_value=sidecar),
            ):
                app._refresh()

        self.assertEqual(table.rows[0][2], "🟢")
        self.assertEqual(
            table.rows[0][3],
            "starting; ignoring stale status from previous shepherd",
        )

    def test_format_status_marks_stale_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text("[12:00:00] new shepherd log line\n")
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {job: (_FakeProc(None), object(), log_path)}
            app._job_started_at = {job: 2_000_000_000.0}
            app._pr_titles = {job: "Test PR"}

            sidecar = {
                "schema_version": 1,
                "updated_at": "2020-01-01T00:00:00+00:00",
                "phase": "ready",
                "category": "ready",
                "message": "ready for human merge",
            }
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["phase"], "🟢")
        self.assertEqual(
            rows[0]["status"],
            "starting; ignoring stale status from previous shepherd",
        )
        self.assertTrue(rows[0]["shepherd_status_stale"])
        self.assertEqual(rows[0]["shepherd_status"], sidecar)

    def test_format_status_reports_cleanup_progress(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "[12:00:00] PR is no longer open; shepherd complete\n"
            )
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                job: (
                    _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                    object(),
                    log_path,
                )
            }
            app._cleanup_jobs = {job}
            app._cleanup_status = {
                job: "cleanup: removing worktree for [123] (1/1)"
            }
            app._pr_titles = {job: "Test PR"}

            with mock.patch.object(mux, "read_status") as read_status:
                rows = json.loads(app._format_status())

        read_status.assert_not_called()
        self.assertEqual(rows[0]["state"], "cleaning")
        self.assertEqual(rows[0]["phase"], "🟢")
        self.assertEqual(
            rows[0]["status"],
            "cleanup: removing worktree for [123] (1/1)",
        )

    def test_format_status_ignores_nonterminal_sidecar_after_error(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "subprocess.CalledProcessError: gh pr view failed\n"
            )
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {job: (_FakeProc(1), object(), log_path)}
            app._pr_titles = {job: "Test PR"}

            sidecar = {
                "schema_version": 1,
                "updated_at": "2026-05-31T00:51:20+00:00",
                "phase": "ready",
                "category": "ready",
                "message": "ready for human merge",
            }
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["state"], "exited_error")
        self.assertEqual(rows[0]["phase"], "🔴")
        self.assertEqual(
            rows[0]["status"],
            "HALT: shepherd exited: subprocess.CalledProcessError: "
            "gh pr view failed",
        )
        self.assertTrue(rows[0]["shepherd_status_stale"])
        self.assertEqual(rows[0]["shepherd_status"], sidecar)

    def test_format_status_summarizes_crash_traceback_after_error(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "[12:00:00] invoking codex (fix-CI mode)...\n"
                "Traceback (most recent call last):\n"
                "  File \"mergedog/claude.py\", line 274, in _run_llm_streaming\n"
                "    proc = subprocess.Popen(...)\n"
                "OSError: [Errno 7] Argument list too long: 'codex'\n"
            )
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {job: (_FakeProc(1), object(), log_path)}
            app._pr_titles = {job: "Test PR"}

            sidecar = {
                "schema_version": 1,
                "updated_at": "2026-05-31T00:51:20+00:00",
                "phase": "fixing_ci",
                "category": "action",
                "message": "fixing CI",
            }
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["state"], "exited_error")
        self.assertEqual(rows[0]["phase"], "🔴")
        self.assertEqual(
            rows[0]["status"],
            "HALT: shepherd crashed: "
            "OSError: [Errno 7] Argument list too long: 'codex'",
        )
        self.assertTrue(rows[0]["shepherd_status_stale"])

    def test_format_status_keeps_halted_sidecar_after_error(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text(
                "[12:00:00] HALT: PR head moved to untrusted commit\n"
            )
            job = mux._pr_job(123)
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {job: (_FakeProc(1), object(), log_path)}
            app._pr_titles = {job: "Test PR"}

            sidecar = {
                "schema_version": 1,
                "updated_at": "2026-05-31T00:51:20+00:00",
                "phase": "halted",
                "category": "blocked",
                "message": "HALT: PR head moved to untrusted commit",
            }
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["state"], "exited_error")
        self.assertEqual(rows[0]["phase"], "🔴")
        self.assertEqual(
            rows[0]["status"],
            "HALT: PR head moved to untrusted commit",
        )
        self.assertFalse(rows[0]["shepherd_status_stale"])
        self.assertEqual(rows[0]["shepherd_status"], sidecar)


class TestMuxCommands(unittest.TestCase):
    def test_restart_all_dispatches_bulk_restart_with_flags(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._pr_job(123): (_FakeProc(None), object(), Path("123.log")),
            mux._pr_job(456): (_FakeProc(None), object(), Path("456.log")),
        }

        with mock.patch.object(app, "_do_restart_all") as restart_all:
            result = app._dispatch_command("restart all --ignore-sev")

        self.assertEqual(result, "restarting 2 job(s)")
        restart_all.assert_called_once_with(["--ignore-sev"])

    def test_restart_all_without_prs_is_noop(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {}

        with mock.patch.object(app, "_do_restart_all") as restart_all:
            result = app._dispatch_command("restart all")

        self.assertEqual(result, "no PRs to restart")
        restart_all.assert_not_called()

    def test_shepherd_args_applies_disabled_fix_cap_default(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.ignore_sev = False
        app.max_fix_commits = 0
        app.gchat_to = None
        app.repo_slug = None

        self.assertEqual(
            app._shepherd_args([]),
            ["--max-fix-commits=0"],
        )

    def test_shepherd_args_preserves_explicit_fix_cap(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.ignore_sev = False
        app.max_fix_commits = 0
        app.gchat_to = None
        app.repo_slug = None

        self.assertEqual(
            app._shepherd_args(["--max-fix-commits=2"]),
            ["--max-fix-commits=2"],
        )

    def test_fix_cap_command_toggles_mux_default(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.max_fix_commits = mux.MAX_FIX_COMMITS

        self.assertEqual(
            app._dispatch_command("fix-cap off"),
            "fix-cap off (applies to future spawns; "
            "use `restart <pr>` or `restart all` to apply to running PRs)",
        )
        self.assertEqual(app.max_fix_commits, 0)
        self.assertEqual(app._dispatch_command("fix-cap"), "fix-cap is off")
        self.assertEqual(
            app._dispatch_command("fix-cap default"),
            "fix-cap 5 (applies to future spawns; "
            "use `restart <pr>` or `restart all` to apply to running PRs)",
        )
        self.assertEqual(app.max_fix_commits, mux.MAX_FIX_COMMITS)

    def test_ignore_sev_command_shows_mux_and_persistent_state(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.ignore_sev = False
        cfg = mock.Mock(ignored_numbers=(187193,))

        with mock.patch.object(mux, "get_ci_sev_config", return_value=cfg):
            result = app._dispatch_command("ignore-sev")

        self.assertEqual(
            result,
            "ignore-sev is off; ignored ci: sev: #187193",
        )

    def test_ignore_sev_command_adds_persistent_issue(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.ignore_sev = False
        cfg = mock.Mock(ignored_numbers=(187193,))

        with mock.patch.object(
            mux, "add_ignored_ci_sev", return_value=cfg
        ) as add:
            result = app._dispatch_command("ignore-sev add #187193")

        self.assertEqual(
            result,
            "ignored ci: sev #187193 (persistent; ignored ci: sev: #187193)",
        )
        add.assert_called_once_with(187193)

    def test_ignore_sev_command_removes_persistent_issue(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.ignore_sev = False
        cfg = mock.Mock(ignored_numbers=())

        with mock.patch.object(
            mux, "remove_ignored_ci_sev", return_value=cfg
        ) as remove:
            result = app._dispatch_command("ignore-sev remove 187193")

        self.assertEqual(
            result,
            "respecting ci: sev #187193 (persistent; ignored ci: sev: none)",
        )
        remove.assert_called_once_with(187193)

    def test_restart_dead_dispatches_only_when_dead_jobs_exist(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._pr_job(123): (_FakeProc(None), object(), Path("123.log")),
            mux._pr_job(456): (_FakeProc(1), object(), Path("456.log")),
            mux._pr_job(789): (_FakeProc(0), object(), Path("789.log")),
        }

        with mock.patch.object(app, "_do_restart_dead") as restart_dead:
            result = app._dispatch_command("restart dead --ignore-sev")

        self.assertEqual(result, "restarting dead 1 job(s)")
        restart_dead.assert_called_once_with(
            [mux._pr_job(456)], ["--ignore-sev"]
        )

    def test_restart_dead_ignores_completed_jobs(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._pr_job(123): (_FakeProc(None), object(), Path("123.log")),
            mux._pr_job(456): (_FakeProc(0), object(), Path("456.log")),
            mux._pr_job(789): (
                _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                object(),
                Path("789.log"),
            ),
        }

        with mock.patch.object(app, "_do_restart_dead") as restart_dead:
            result = app._dispatch_command("restart dead")

        self.assertEqual(result, "no dead PRs to restart")
        restart_dead.assert_not_called()

    def test_fix_command_restarts_with_operator_context(self):
        app = mux.MuxApp.__new__(mux.MuxApp)

        with mock.patch.object(app, "_do_cancel_job") as cancel, mock.patch.object(
            app, "_do_add", return_value="[123] started"
        ) as add:
            result = app._dispatch_command("fix 123 Return type should be TypeGuard")

        self.assertEqual(result, "[123] started")
        cancel.assert_called_once_with(mux._pr_job(123), keep_resumable=True)
        add.assert_called_once_with(
            123, ["--operator-fix-context=Return type should be TypeGuard"]
        )

    def test_rebase_restarts_running_job_with_rebase_flag(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            old_proc = _FakeProc(None)

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (old_proc, object(), Path("123.log"))
            }
            app._unresumable_jobs = set()
            app._pr_titles = {}
            app.ignore_sev = False
            app.gchat_to = None
            app.repo_slug = None

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group") as terminate_group,
                mock.patch.object(mux, "_spawn") as spawn,
            ):
                mux._write_mux_jobs([mux._pr_job(123)])
                terminate_group.side_effect = lambda p: setattr(p, "_rc", -15)
                spawn.return_value = (
                    _FakeProc(None),
                    object(),
                    Path("123.log"),
                )
                result = app._dispatch_command("rebase 123")
                jobs_data = json.loads(jobs_file.read_text())

        self.assertEqual(result, "[123] started")
        terminate_group.assert_called_once_with(old_proc)
        spawn.assert_called_once_with(
            mux._pr_job(123), ["--rebase"], spawn_pr=None
        )
        self.assertEqual(jobs_data, [{"kind": "pr", "pr": 123}])

    def test_cleanup_prunes_successful_completed_jobs(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        completed = mux._pr_job(123)
        failed = mux._pr_job(456)
        running = mux._pr_job(789)
        app.procs = {
            completed: (
                _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                object(),
                Path("123.log"),
            ),
            failed: (_FakeProc(1), object(), Path("456.log")),
            running: (_FakeProc(None), object(), Path("789.log")),
        }

        with mock.patch.object(app, "_cleanup_completed_jobs") as cleanup_jobs:
            result = app._dispatch_command("cleanup")

        self.assertEqual(result, "cleaning up 1 completed job(s)")
        cleanup_jobs.assert_called_once_with([completed])
        self.assertEqual(app._cleanup_jobs, {completed})
        self.assertEqual(
            app._cleanup_status[completed],
            "cleanup: queued (1/1)",
        )

    def test_clean_alias_prunes_successful_completed_jobs(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        completed = mux._pr_job(123)
        app.procs = {
            completed: (
                _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                object(),
                Path("123.log"),
            ),
        }

        with mock.patch.object(app, "_cleanup_completed_jobs") as cleanup_jobs:
            result = app._dispatch_command("clean")

        self.assertEqual(result, "cleaning up 1 completed job(s)")
        cleanup_jobs.assert_called_once_with([completed])

    def test_cleanup_without_completed_jobs_is_noop(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._pr_job(123): (_FakeProc(None), object(), Path("123.log")),
            mux._pr_job(456): (_FakeProc(1), object(), Path("456.log")),
        }

        with mock.patch.object(app, "_cleanup_completed_jobs") as cleanup_jobs:
            result = app._dispatch_command("cleanup")

        self.assertEqual(result, "no completed jobs to cleanup")
        cleanup_jobs.assert_not_called()

    def test_cleanup_while_cleanup_is_running_is_noop(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        completed = mux._pr_job(123)
        app.procs = {
            completed: (
                _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                object(),
                Path("123.log"),
            ),
        }
        app._cleanup_jobs = {completed}

        with mock.patch.object(app, "_cleanup_completed_jobs") as cleanup_jobs:
            result = app._dispatch_command("cleanup")

        self.assertEqual(result, "cleanup already in progress")
        cleanup_jobs.assert_not_called()

    def test_cancel_removes_job_from_resume_list_but_keeps_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (_FakeProc(None), object(), Path("123.log"))
            }
            app._unresumable_jobs = set()

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group") as terminate_group,
            ):
                mux._write_mux_jobs([mux._pr_job(123)])
                result = app._dispatch_command("cancel 123")
                jobs_data = json.loads(jobs_file.read_text())
                prs_data = json.loads(prs_file.read_text())

        self.assertEqual(result, "[123] terminated")
        terminate_group.assert_called_once()
        self.assertIn(mux._pr_job(123), app.procs)
        self.assertEqual(jobs_data, [])
        self.assertEqual(prs_data, [])

    def test_restart_keeps_job_resumable_while_respawning(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (_FakeProc(None), object(), Path("123.log"))
            }
            app._unresumable_jobs = set()
            app._pr_titles = {}
            app.ignore_sev = False
            app.gchat_to = None
            app.repo_slug = None

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group") as terminate_group,
                mock.patch.object(mux, "_spawn") as spawn,
            ):
                mux._write_mux_jobs([mux._pr_job(123)])
                terminate_group.side_effect = lambda p: setattr(p, "_rc", -15)
                spawn.return_value = (
                    _FakeProc(None),
                    object(),
                    Path("123.log"),
                )
                result = app._dispatch_command("restart 123")
                jobs_data = json.loads(jobs_file.read_text())

        self.assertEqual(result, "[123] started")
        self.assertEqual(jobs_data, [{"kind": "pr", "pr": 123}])

    def test_on_unmount_persists_running_and_failed_jobs_for_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (_FakeProc(None), mock.Mock(), Path("123.log")),
                mux._pr_job(456): (_FakeProc(0), mock.Mock(), Path("456.log")),
                mux._pr_job(789): (
                    _FakeProc(1),
                    mock.Mock(),
                    Path("789.log"),
                ),
                mux._pr_job(101): (
                    _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                    mock.Mock(),
                    Path("101.log"),
                ),
            }
            app._unresumable_jobs = set()
            app._ipc_server = None
            app._lock_fd = -1

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group"),
                mock.patch.object(mux.os, "killpg"),
            ):
                mux._write_mux_jobs(
                    [
                        mux._pr_job(101),
                        mux._pr_job(123),
                        mux._pr_job(456),
                        mux._pr_job(789),
                    ]
                )
                app.on_unmount()
                jobs_data = json.loads(jobs_file.read_text())
                prs_data = json.loads(prs_file.read_text())

        # Still-running 123 resumes normally on the next start; halted
        # 789 is persisted with its exit code so the next mux parks it
        # instead of re-running (and re-notifying) the halt.
        self.assertEqual(
            jobs_data,
            [{"kind": "pr", "pr": 123}, {"kind": "pr", "pr": 789, "rc": 1}],
        )
        self.assertEqual(prs_data, [123, 789])

    def test_on_unmount_does_not_repersist_cancelled_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            cancelled = mux._pr_job(123)

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                cancelled: (_FakeProc(-15), mock.Mock(), Path("123.log")),
                mux._pr_job(456): (_FakeProc(1), mock.Mock(), Path("456.log")),
            }
            app._unresumable_jobs = {cancelled}
            app._ipc_server = None
            app._lock_fd = -1

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group"),
                mock.patch.object(mux.os, "killpg"),
            ):
                mux._write_mux_jobs([cancelled, mux._pr_job(456)])
                app.on_unmount()
                jobs_data = json.loads(jobs_file.read_text())
                prs_data = json.loads(prs_file.read_text())

        self.assertEqual(jobs_data, [{"kind": "pr", "pr": 456, "rc": 1}])
        self.assertEqual(prs_data, [456])


class TestMuxJobPersistence(unittest.TestCase):
    def test_resolve_initial_jobs_resumes_known_jobs_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            jobs_file.write_text(
                json.dumps([{"kind": "pr", "pr": 123}, {"kind": "stack", "pr": 456}])
            )

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                jobs, parked, skipped = mux._resolve_initial_jobs(
                    [], resume_known=True
                )

        self.assertEqual(jobs, [mux._pr_job(123)])
        self.assertEqual(parked, {})
        self.assertEqual(skipped, [])

    def test_resolve_initial_jobs_can_skip_resume_known(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            jobs_file.write_text(json.dumps([{"kind": "pr", "pr": 123}]))

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                jobs, parked, skipped = mux._resolve_initial_jobs(
                    ["456"],
                    resume_known=False,
                )

        self.assertEqual(jobs, [mux._pr_job(456)])
        self.assertEqual(parked, {})
        self.assertEqual(skipped, [])

    def test_resolve_initial_jobs_deduplicates_known_and_explicit_prs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            jobs_file.write_text(json.dumps([{"kind": "pr", "pr": 123}]))

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                jobs, parked, skipped = mux._resolve_initial_jobs(
                    ["123", "456"],
                    resume_known=True,
                )

        self.assertEqual(jobs, [mux._pr_job(123), mux._pr_job(456)])
        self.assertEqual(parked, {})
        self.assertEqual(skipped, [])

    def test_resolve_initial_jobs_parks_previously_halted_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            jobs_file.write_text(
                json.dumps(
                    [
                        {"kind": "pr", "pr": 123},
                        {"kind": "pr", "pr": 456, "rc": 1},
                    ]
                )
            )

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                jobs, parked, skipped = mux._resolve_initial_jobs(
                    [], resume_known=True
                )

        self.assertEqual(jobs, [mux._pr_job(123)])
        self.assertEqual(parked, {mux._pr_job(456): 1})
        self.assertEqual(skipped, [])

    def test_explicit_pr_overrides_parking(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            jobs_file.write_text(
                json.dumps([{"kind": "pr", "pr": 456, "rc": 1}])
            )

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                jobs, parked, skipped = mux._resolve_initial_jobs(
                    ["456"], resume_known=True
                )

        self.assertEqual(jobs, [mux._pr_job(456)])
        self.assertEqual(parked, {})
        self.assertEqual(skipped, [])

    def test_respawn_clears_persisted_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            jobs_file.write_text(
                json.dumps([{"kind": "pr", "pr": 123, "rc": 1}])
            )

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                mux._add_mux_job(mux._pr_job(123))
                jobs_data = json.loads(jobs_file.read_text())

        self.assertEqual(jobs_data, [{"kind": "pr", "pr": 123}])

    def test_restart_unparks_job(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            # A parked job is still a tracked mux member, so it already has
            # the ``mergedog`` label. Persist it accordingly.
            jobs_file.write_text(
                json.dumps([{"kind": "pr", "pr": 123, "rc": 1}])
            )

            app = mux.MuxApp.__new__(mux.MuxApp)
            parked = mux._pr_job(123)
            app.procs = {}
            app._parked_jobs = {parked: 1}
            app._unresumable_jobs = set()
            app._pr_titles = {}
            app.ignore_sev = False
            app.gchat_to = None
            app.repo_slug = None

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_spawn") as spawn,
                mock.patch.object(mux.github, "add_label") as add_label,
            ):
                spawn.return_value = (
                    _FakeProc(None),
                    object(),
                    Path("123.log"),
                )
                result = app._dispatch_command("restart 123")

        self.assertEqual(result, "[123] started")
        self.assertEqual(app._parked_jobs, {})
        # Un-parking a job that was already tracked must not re-stamp the
        # label -- it only joins the mux once.
        add_label.assert_not_called()


class TestMergedogMuxLabel(unittest.TestCase):
    """The ``mergedog`` label tracks mux membership, not shepherd lifecycle."""

    def _bare_app(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {}
        app._parked_jobs = {}
        app._cleanup_jobs = set()
        app._unresumable_jobs = set()
        app._pr_titles = {}
        app.ignore_sev = False
        app.gchat_to = None
        app.repo_slug = None
        return app

    def test_add_mux_job_reports_first_join_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                self.assertTrue(mux._add_mux_job(mux._pr_job(123)))
                self.assertFalse(mux._add_mux_job(mux._pr_job(123)))
                jobs_file.write_text(
                    json.dumps([{"kind": "pr", "pr": 123, "rc": 1}])
                )
                # Re-adding a parked (still-tracked) job is not a fresh join.
                self.assertFalse(mux._add_mux_job(mux._pr_job(123)))

    def test_add_stamps_label_on_first_join(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            app = self._bare_app()
            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_spawn") as spawn,
                mock.patch.object(mux.github, "add_label") as add_label,
            ):
                spawn.return_value = (_FakeProc(None), object(), Path("123.log"))
                app._dispatch_command("add 123")
                # A second add of the now-running job must not re-stamp.
                app.procs[mux._pr_job(123)] = (
                    _FakeProc(0),
                    object(),
                    Path("123.log"),
                )
                app._dispatch_command("add 123")

        add_label.assert_called_once_with(123, mux.MERGEDOG_LABEL, loud=False)

    def test_remove_strips_label(self):
        app = self._bare_app()
        job = mux._pr_job(123)
        app.procs = {job: (_FakeProc(0), object(), Path("123.log"))}
        with (
            mock.patch.object(app, "_prune_job"),
            mock.patch.object(mux.github, "remove_label") as remove_label,
        ):
            result = app._do_remove(123)
        self.assertEqual(result, "[123] removed")
        remove_label.assert_called_once_with(123, mux.MERGEDOG_LABEL)

    def test_cancel_keeps_label(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            app = self._bare_app()
            job = mux._pr_job(123)
            app.procs = {job: (_FakeProc(None), object(), Path("123.log"))}
            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group"),
                mock.patch.object(mux.github, "remove_label") as remove_label,
            ):
                result = app._do_cancel(123)
        self.assertEqual(result, "[123] terminated")
        # Cancel only drops the job from the resume list; the label stays so a
        # completed/cancelled PR keeps its mux marker until explicit removal.
        remove_label.assert_not_called()

    def test_parked_jobs_count_as_dead(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {}
        app._parked_jobs = {mux._pr_job(123): 1}
        self.assertEqual(app._dead_jobs(), [mux._pr_job(123)])

    def test_read_mux_jobs_falls_back_to_legacy_prs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"
            prs_file.write_text("[123, 456]")

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                jobs = mux._read_mux_jobs()

        self.assertEqual(jobs, [mux._pr_job(123), mux._pr_job(456)])

    def test_write_mux_jobs_updates_legacy_prs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                mux._write_mux_jobs([mux._pr_job(123), mux._pr_job(456)])
                jobs_data = json.loads(jobs_file.read_text())
                prs_data = json.loads(prs_file.read_text())

        self.assertEqual(
            jobs_data,
            [{"kind": "pr", "pr": 123}, {"kind": "pr", "pr": 456}],
        )
        self.assertEqual(prs_data, [123, 456])

if __name__ == "__main__":
    unittest.main()
