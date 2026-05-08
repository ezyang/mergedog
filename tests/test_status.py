import json
import tempfile
import unittest
from pathlib import Path

from mergedog.status import read_status, write_status


class TestStructuredStatus(unittest.TestCase):
    def test_write_status_is_readable_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "123.json"

            payload = write_status(
                123,
                phase="polling_ci",
                approved=True,
                merging=False,
                ci_done=45,
                ci_total=52,
                ci_failed=2,
                fix_attempts=1,
                max_fix_attempts=5,
                path=path,
            )

            raw = json.loads(path.read_text())
            self.assertEqual(raw, payload)
            self.assertEqual(read_status(123, path=path), payload)
            self.assertEqual(raw["schema_version"], 1)
            self.assertEqual(raw["phase"], "polling_ci")
            self.assertEqual(raw["ci_done"], 45)
            self.assertEqual(raw["ci_total"], 52)
            self.assertEqual(raw["ci_failed"], 2)
            self.assertEqual(raw["approved"], True)
            self.assertEqual(raw["merging"], False)

    def test_read_status_tolerates_missing_or_bad_files(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "missing.json"
            bad = Path(d) / "bad.json"
            bad.write_text("{")
            no_phase = Path(d) / "no-phase.json"
            no_phase.write_text("{}")

            self.assertIsNone(read_status(123, path=missing))
            self.assertIsNone(read_status(123, path=bad))
            self.assertIsNone(read_status(123, path=no_phase))


if __name__ == "__main__":
    unittest.main()
