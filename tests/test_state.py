import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mergedog import state as state_mod
from mergedog.state import TrustDB


class TestTrustDBRoundTrip(unittest.TestCase):
    def _load(self, tmp: Path, pr: int) -> TrustDB:
        with mock.patch.object(
            state_mod, "state_file", return_value=tmp / f"{pr}.json"
        ):
            return TrustDB.load_or_create(pr)

    def test_save_includes_schema_version(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._load(tmp, 1)
            trust.trust("a" * 40)
            data = json.loads((tmp / "1.json").read_text())
            self.assertEqual(
                data["schema_version"], state_mod.SCHEMA_VERSION
            )

    def test_unknown_fields_survive_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "1.json").write_text(
                json.dumps(
                    {
                        "pr": 1,
                        "trusted_shas": ["a" * 40],
                        "field_from_the_future": {"x": 1},
                    }
                )
            )
            trust = self._load(tmp, 1)
            self.assertEqual(
                trust.extra_fields, {"field_from_the_future": {"x": 1}}
            )
            trust.save()
            data = json.loads((tmp / "1.json").read_text())
            self.assertEqual(data["field_from_the_future"], {"x": 1})
            self.assertEqual(data["trusted_shas"], ["a" * 40])

    def test_known_fields_not_duplicated_into_extra(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._load(tmp, 1)
            trust.spurious_check_names = ["lint / foo"]
            trust.save()
            reloaded = self._load(tmp, 1)
            self.assertEqual(reloaded.extra_fields, {})
            self.assertEqual(reloaded.spurious_check_names, ["lint / foo"])


if __name__ == "__main__":
    unittest.main()
