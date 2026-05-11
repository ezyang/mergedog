import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import mux


class _FakeProc:
    def __init__(self, rc):
        self._rc = rc

    def poll(self):
        return self._rc


class TestMuxStructuredStatus(unittest.TestCase):
    def test_format_status_includes_shepherd_status_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "123.log"
            log_path.write_text("[12:00:00] CI status -> pending (1/2 done)\n")
            app = mux.MuxApp.__new__(mux.MuxApp)
            app.procs = {123: (_FakeProc(None), object(), log_path)}
            app._pr_titles = {123: "Test PR"}

            sidecar = {"schema_version": 1, "phase": "polling_ci"}
            with mock.patch.object(mux, "read_status", return_value=sidecar):
                rows = json.loads(app._format_status())

        self.assertEqual(rows[0]["pr"], 123)
        self.assertEqual(rows[0]["state"], "running")
        self.assertEqual(rows[0]["shepherd_status"], sidecar)


class TestMuxCommands(unittest.TestCase):
    def test_restart_all_dispatches_bulk_restart_with_flags(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {
            123: (_FakeProc(None), object(), Path("123.log")),
            456: (_FakeProc(None), object(), Path("456.log")),
        }

        with mock.patch.object(app, "_do_restart_all") as restart_all:
            result = app._dispatch_command("restart all --ignore-sev")

        self.assertEqual(result, "restarting 2 PR(s)")
        restart_all.assert_called_once_with(["--ignore-sev"])

    def test_restart_all_without_prs_is_noop(self):
        app = mux.MuxApp.__new__(mux.MuxApp)
        app.procs = {}

        with mock.patch.object(app, "_do_restart_all") as restart_all:
            result = app._dispatch_command("restart all")

        self.assertEqual(result, "no PRs to restart")
        restart_all.assert_not_called()


if __name__ == "__main__":
    unittest.main()
