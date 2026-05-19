import asyncio
import json
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
    def test_format_status_includes_shepherd_status_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text("[12:00:00] CI status -> pending (1/2 done)\n")
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {mux._pr_job(123): (_FakeProc(None), object(), log_path)}
            app._pr_titles = {mux._pr_job(123): "Test PR"}

            sidecar = {"schema_version": 1, "phase": "polling_ci"}
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["pr"], 123)
        self.assertEqual(rows[0]["state"], "running")
        self.assertEqual(rows[0]["shepherd_status"], sidecar)

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

            with (
                mock.patch.object(app, "query_one", return_value=table),
                mock.patch.object(app, "_prune_job") as prune_job,
                mock.patch.object(mux, "read_status", return_value=None),
            ):
                app._refresh()

        prune_job.assert_not_called()
        self.assertEqual(len(table.rows), 1)


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

    def test_restart_dead_dispatches_only_when_dead_jobs_exist(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._pr_job(123): (_FakeProc(None), object(), Path("123.log")),
            mux._pr_job(456): (_FakeProc(1), object(), Path("456.log")),
            mux._stack_job(789): (_FakeProc(0), object(), Path("stack-789.log")),
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
            mux._stack_job(789): (
                _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                object(),
                Path("stack-789.log"),
            ),
        }

        with mock.patch.object(app, "_do_restart_dead") as restart_dead:
            result = app._dispatch_command("restart dead")

        self.assertEqual(result, "no dead PRs to restart")
        restart_dead.assert_not_called()

    def test_stack_command_starts_stack_job(self):
        app = mux.MuxApp.__new__(mux.MuxApp)

        with mock.patch.object(app, "_do_stack_add") as stack_add:
            stack_add.return_value = "[stack 123] started"
            result = app._dispatch_command("stack 123 --force-ghstack")

        self.assertEqual(result, "[stack 123] started")
        stack_add.assert_called_once_with(123, ["--force-ghstack"])

    def test_stack_rebase_adds_rebase_flag(self):
        app = mux.MuxApp.__new__(mux.MuxApp)

        with mock.patch.object(app, "_do_stack_add") as stack_add:
            stack_add.return_value = "[stack 123] started"
            result = app._dispatch_command("stack rebase 123 --force-ghstack")

        self.assertEqual(result, "[stack 123] started")
        stack_add.assert_called_once_with(
            123, ["--rebase", "--force-ghstack"]
        )

    def test_stack_log_uses_stack_job(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._stack_job(123): (
                _FakeProc(None),
                object(),
                Path("stack-123.log"),
            ),
        }

        result = app._dispatch_command("stack log 123")

        self.assertEqual(result, "stack-123.log")

    def test_cleanup_prunes_successful_completed_jobs(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        completed = mux._pr_job(123)
        failed = mux._pr_job(456)
        running = mux._stack_job(789)
        app.procs = {
            completed: (
                _FakeProc(mux.EXIT_PR_NOT_ACTIONABLE),
                object(),
                Path("123.log"),
            ),
            failed: (_FakeProc(1), object(), Path("456.log")),
            running: (_FakeProc(None), object(), Path("stack-789.log")),
        }

        with mock.patch.object(app, "_prune_job") as prune_job:
            result = app._dispatch_command("cleanup")

        self.assertEqual(result, "cleaned up 1 completed job(s)")
        prune_job.assert_called_once_with(completed)

    def test_cleanup_without_completed_jobs_is_noop(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            mux._pr_job(123): (_FakeProc(None), object(), Path("123.log")),
            mux._pr_job(456): (_FakeProc(1), object(), Path("456.log")),
        }

        with mock.patch.object(app, "_prune_job") as prune_job:
            result = app._dispatch_command("cleanup")

        self.assertEqual(result, "no completed jobs to cleanup")
        prune_job.assert_not_called()

    def test_cancel_removes_job_from_resume_list_but_keeps_row(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (_FakeProc(None), object(), Path("123.log"))
            }

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
            app._pr_titles = {}
            app.ignore_sev = False
            app.manage_mergedog_label = False
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

    def test_on_unmount_persists_only_running_jobs_for_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {
                mux._pr_job(123): (_FakeProc(None), mock.Mock(), Path("123.log")),
                mux._pr_job(456): (_FakeProc(0), mock.Mock(), Path("456.log")),
                mux._stack_job(789): (
                    _FakeProc(1),
                    mock.Mock(),
                    Path("stack-789.log"),
                ),
            }
            app._ipc_server = None
            app._lock_fd = -1

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
                mock.patch.object(mux, "_terminate_group"),
                mock.patch.object(mux.os, "killpg"),
            ):
                mux._write_mux_jobs(
                    [mux._pr_job(123), mux._pr_job(456), mux._stack_job(789)]
                )
                app.on_unmount()
                jobs_data = json.loads(jobs_file.read_text())
                prs_data = json.loads(prs_file.read_text())

        self.assertEqual(jobs_data, [{"kind": "pr", "pr": 123}])
        self.assertEqual(prs_data, [123])


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
                jobs, skipped = mux._resolve_initial_jobs([], resume_known=True)

        self.assertEqual(jobs, [mux._pr_job(123), mux._stack_job(456)])
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
                jobs, skipped = mux._resolve_initial_jobs(
                    ["456"],
                    resume_known=False,
                )

        self.assertEqual(jobs, [mux._pr_job(456)])
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
                jobs, skipped = mux._resolve_initial_jobs(
                    ["123", "456"],
                    resume_known=True,
                )

        self.assertEqual(jobs, [mux._pr_job(123), mux._pr_job(456)])
        self.assertEqual(skipped, [])

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

    def test_write_mux_jobs_keeps_legacy_prs_regular_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prs_file = root / "mux-prs.json"
            jobs_file = root / "mux-jobs.json"

            with (
                mock.patch.object(mux, "MUX_PRS_FILE", prs_file),
                mock.patch.object(mux, "MUX_JOBS_FILE", jobs_file),
            ):
                mux._write_mux_jobs([mux._pr_job(123), mux._stack_job(456)])
                jobs_data = json.loads(jobs_file.read_text())
                prs_data = json.loads(prs_file.read_text())

        self.assertEqual(
            jobs_data,
            [{"kind": "pr", "pr": 123}, {"kind": "stack", "pr": 456}],
        )
        self.assertEqual(prs_data, [123])

if __name__ == "__main__":
    unittest.main()
