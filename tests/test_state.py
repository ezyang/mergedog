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

    def test_counters_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._load(tmp, 1)
            trust.fix_commits_pushed = 2
            trust.fix_budget_ack_sha = "a" * 40
            trust.merge_auto_retries = 1
            trust.save()
            reloaded = self._load(tmp, 1)
            self.assertEqual(reloaded.fix_commits_pushed, 2)
            self.assertEqual(reloaded.fix_budget_ack_sha, "a" * 40)
            self.assertEqual(reloaded.merge_auto_retries, 1)

    def test_known_fields_not_duplicated_into_extra(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._load(tmp, 1)
            trust.spurious_check_names = ["lint / foo"]
            trust.save()
            reloaded = self._load(tmp, 1)
            self.assertEqual(reloaded.extra_fields, {})
            self.assertEqual(reloaded.spurious_check_names, ["lint / foo"])


class TestFixBudget(unittest.TestCase):
    def _trust(self, tmp: Path, **kwargs) -> TrustDB:
        with mock.patch.object(
            state_mod, "state_file", return_value=tmp / "1.json"
        ):
            trust = TrustDB.load_or_create(1)
        for k, v in kwargs.items():
            setattr(trust, k, v)
        return trust

    def test_budget_survives_restart(self):
        from mergedog import shepherd

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._trust(tmp)
            shepherd._sync_fix_budget(trust, "a" * 40)
            shepherd._consume_fix_budget(trust)
            shepherd._consume_fix_budget(trust)

            # Simulate restart: reload from disk with the same ack SHA.
            reloaded = self._trust(tmp)
            count = shepherd._sync_fix_budget(reloaded, "a" * 40)
            self.assertEqual(count, 2)

    def test_new_approval_resets_budget(self):
        from mergedog import shepherd

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._trust(tmp)
            shepherd._sync_fix_budget(trust, "a" * 40)
            shepherd._consume_fix_budget(trust)

            reloaded = self._trust(tmp)
            count = shepherd._sync_fix_budget(reloaded, "b" * 40)
            self.assertEqual(count, 0)

    def test_reassess_resets_budget(self):
        from mergedog import shepherd

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._trust(tmp)
            shepherd._sync_fix_budget(trust, "a" * 40)
            shepherd._consume_fix_budget(trust)

            reloaded = self._trust(tmp)
            count = shepherd._sync_fix_budget(
                reloaded, "a" * 40, reassess=True
            )
            self.assertEqual(count, 0)

    def test_self_pr_keeps_budget_across_head_moves(self):
        from mergedog import shepherd

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._trust(tmp)
            shepherd._sync_fix_budget(trust, "a" * 40, self_pr=True)
            shepherd._consume_fix_budget(trust)

            # Self-PR ack is the (moving) head; budget must not reset.
            reloaded = self._trust(tmp)
            count = shepherd._sync_fix_budget(
                reloaded, "b" * 40, self_pr=True
            )
            self.assertEqual(count, 1)
            self.assertEqual(reloaded.fix_budget_ack_sha, "a" * 40)

    def test_self_pr_uses_first_trusted_sha_as_restart_baseline(self):
        from mergedog import shepherd

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            trust = self._trust(tmp)
            trust.trust("a" * 40)

            shepherd._sync_fix_budget(trust, "b" * 40, self_pr=True)

            self.assertEqual(trust.fix_budget_ack_sha, "a" * 40)


if __name__ == "__main__":
    unittest.main()
