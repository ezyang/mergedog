import unittest
from pathlib import Path
from unittest import mock

from mergedog import repo


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


class TestTrunkRevertContext(unittest.TestCase):
    def test_revert_context_is_cautious_about_spurious_failures(self):
        with mock.patch.object(
            repo,
            "run",
            side_effect=[
                _Proc("base\n"),
                _Proc("abc Revert \"[CPU][Inductor] Improve cache\"\n"),
            ],
        ):
            ctx = repo.trunk_revert_context(Path("/tmp/worktree"))

        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertIn("Use this only as diagnostic context", ctx)
        self.assertIn("Do not treat a revert-area match", ctx)
        self.assertIn("choose INCONCLUSIVE instead of spurious", ctx)
        self.assertIn('Revert "[CPU][Inductor] Improve cache"', ctx)


if __name__ == "__main__":
    unittest.main()
