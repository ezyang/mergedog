import unittest
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from mergedog import repo


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


class TestPatchIdMatchesAny(unittest.TestCase):
    def test_matches_rebased_equivalent_commit(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            repo, "REPO_DIR", Path(td)
        ):
            root = Path(td)
            _git(root, "init", "-q")
            _git(root, "config", "user.name", "Tester")
            _git(root, "config", "user.email", "tester@example.com")
            (root / "base.txt").write_text("base\n")
            _git(root, "add", "base.txt")
            _git(root, "commit", "-q", "-m", "base")

            (root / "change.txt").write_text("change\n")
            _git(root, "add", "change.txt")
            _git(root, "commit", "-q", "-m", "change")
            trusted = _git(root, "rev-parse", "HEAD")

            _git(root, "checkout", "-q", "HEAD^")
            (root / "main.txt").write_text("new base\n")
            _git(root, "add", "main.txt")
            _git(root, "commit", "-q", "-m", "new base")
            _git(root, "cherry-pick", trusted)
            rebased = _git(root, "rev-parse", "HEAD")

            self.assertTrue(repo.patch_id_matches_any(rebased, [trusted]))

    def test_rejects_different_patch(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            repo, "REPO_DIR", Path(td)
        ):
            root = Path(td)
            _git(root, "init", "-q")
            _git(root, "config", "user.name", "Tester")
            _git(root, "config", "user.email", "tester@example.com")
            (root / "one.txt").write_text("one\n")
            _git(root, "add", "one.txt")
            _git(root, "commit", "-q", "-m", "one")
            trusted = _git(root, "rev-parse", "HEAD")

            (root / "two.txt").write_text("two\n")
            _git(root, "add", "two.txt")
            _git(root, "commit", "-q", "-m", "two")
            other = _git(root, "rev-parse", "HEAD")

            self.assertFalse(repo.patch_id_matches_any(other, [trusted]))


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


class TestRebaseTargetAdvances(unittest.TestCase):
    def test_true_when_target_differs_from_merge_base(self):
        with mock.patch.object(
            repo,
            "run",
            side_effect=[
                _Proc("base\n"),
                _Proc("target\n"),
            ],
        ):
            self.assertTrue(
                repo.rebase_target_advances(
                    Path("/tmp/worktree"), "origin/viable/strict"
                )
            )

    def test_false_when_target_is_current_merge_base(self):
        with mock.patch.object(
            repo,
            "run",
            side_effect=[
                _Proc("base\n"),
                _Proc("base\n"),
            ],
        ):
            self.assertFalse(
                repo.rebase_target_advances(Path("/tmp/worktree"), "base")
            )


if __name__ == "__main__":
    unittest.main()
