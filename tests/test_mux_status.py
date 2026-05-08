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


if __name__ == "__main__":
    unittest.main()
