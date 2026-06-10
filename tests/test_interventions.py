import unittest
from unittest.mock import patch

from mergedog import interventions
from mergedog.shepherd import _try_interventions


_GRAPHQL_504_LOG = """\
+ python3 .github/scripts/trymerge.py --check-mergeability 182443
Found 4 PRs in the stack for 182443: [182295, 182435, 182439, 182443]
stdout: fatal: ref HEAD is not a symbolic ref
stderr: Error fetching https://api.github.com/graphql HTTP Error 504: Gateway Timeout
Traceback (most recent call last):
  File "/home/runner/work/pytorch/pytorch/.github/scripts/trymerge.py", line 2644, in <module>
    main()
urllib.error.HTTPError: HTTP Error 504: Gateway Timeout
"""

_UNRELATED_FAILURE_LOG = """\
FAILED test/test_foo.py::test_bar - AssertionError: expected 1, got 2
= 1 failed, 99 passed in 12.34s =
"""


class TestFindIntervention(unittest.TestCase):
    def test_matches_graphql_5xx(self):
        match = interventions.find_intervention(_GRAPHQL_504_LOG)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertIn("graphql", match.name)

    def test_matches_503_too(self):
        log = "Error fetching https://api.github.com/graphql HTTP Error 503: x"
        self.assertIsNotNone(interventions.find_intervention(log))

    def test_does_not_match_unrelated_failure(self):
        self.assertIsNone(interventions.find_intervention(_UNRELATED_FAILURE_LOG))

    def test_does_not_match_4xx(self):
        log = "Error fetching https://api.github.com/graphql HTTP Error 404"
        self.assertIsNone(interventions.find_intervention(log))

    def test_does_not_match_5xx_from_other_host(self):
        # Anchoring on the GraphQL URL means a 5xx from some other API the
        # PR is exercising doesn't get auto-retried.
        log = "Error fetching https://example.com/api HTTP Error 504"
        self.assertIsNone(interventions.find_intervention(log))


class TestTryInterventions(unittest.TestCase):
    def _check(self, name: str, run_id: int) -> dict:
        return {
            "name": name,
            "bucket": "fail",
            "link": f"https://github.com/pytorch/pytorch/actions/runs/{run_id}/job/9999",
        }

    def _run(self, failed, checks, seen, *, rerun_ok=(True, ""), attempt=1):
        with patch(
            "mergedog.github.rerun_failed_jobs", return_value=rerun_ok
        ) as rerun, patch(
            "mergedog.github.workflow_run_attempt", return_value=attempt
        ):
            triggered = _try_interventions(failed, checks, seen)
        return triggered, rerun

    def test_triggers_rerun_on_match(self):
        failed = [("ghstack-mergeability-check", _GRAPHQL_504_LOG)]
        checks = [self._check("ghstack-mergeability-check", 12345)]
        seen: set[int] = set()
        triggered, rerun = self._run(failed, checks, seen)
        self.assertTrue(triggered)
        rerun.assert_called_once_with(12345)
        self.assertIn(12345, seen)

    def test_skips_when_run_already_intervened(self):
        failed = [("ghstack-mergeability-check", _GRAPHQL_504_LOG)]
        checks = [self._check("ghstack-mergeability-check", 12345)]
        seen: set[int] = {12345}
        triggered, rerun = self._run(failed, checks, seen)
        self.assertFalse(triggered)
        rerun.assert_not_called()

    def test_skips_when_run_already_reran_before_restart(self):
        # A previous shepherd (since killed) already retried this run;
        # GitHub's run_attempt is the restart-proof record of that.
        failed = [("ghstack-mergeability-check", _GRAPHQL_504_LOG)]
        checks = [self._check("ghstack-mergeability-check", 12345)]
        seen: set[int] = set()
        triggered, rerun = self._run(failed, checks, seen, attempt=2)
        self.assertFalse(triggered)
        rerun.assert_not_called()
        self.assertIn(12345, seen)

    def test_unknown_attempt_still_intervenes(self):
        # If the attempt lookup fails, keep the pre-existing behavior
        # (one rerun per process) rather than blocking the intervention.
        failed = [("ghstack-mergeability-check", _GRAPHQL_504_LOG)]
        checks = [self._check("ghstack-mergeability-check", 12345)]
        seen: set[int] = set()
        triggered, rerun = self._run(failed, checks, seen, attempt=None)
        self.assertTrue(triggered)
        rerun.assert_called_once_with(12345)

    def test_no_match_no_rerun(self):
        failed = [("pytest", _UNRELATED_FAILURE_LOG)]
        checks = [self._check("pytest", 22222)]
        seen: set[int] = set()
        triggered, rerun = self._run(failed, checks, seen)
        self.assertFalse(triggered)
        rerun.assert_not_called()

    def test_failed_rerun_doesnt_record_run_id(self):
        # If gh rerun fails, leave the run id out of the seen set so a
        # subsequent poll can try again (e.g. transient gh CLI hiccup).
        failed = [("ghstack-mergeability-check", _GRAPHQL_504_LOG)]
        checks = [self._check("ghstack-mergeability-check", 12345)]
        seen: set[int] = set()
        triggered, _ = self._run(
            failed, checks, seen, rerun_ok=(False, "boom")
        )
        self.assertFalse(triggered)
        self.assertNotIn(12345, seen)


if __name__ == "__main__":
    unittest.main()
