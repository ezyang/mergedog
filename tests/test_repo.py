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
        self.stderr = ""

    def check_returncode(self):
        if self.returncode:
            raise subprocess.CalledProcessError(
                self.returncode,
                ["cmd"],
                output=self.stdout,
                stderr=self.stderr,
            )


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        ["ghstack"],
        returncode,
        stdout,
        stderr,
    )


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


class TestGhstackRetry(unittest.TestCase):
    def test_retries_transient_proxy_failure(self):
        with mock.patch.object(
            repo,
            "run",
            side_effect=[
                _completed(
                    1,
                    stderr=(
                        "requests.exceptions.ProxyError: "
                        "Failed to establish a new connection: "
                        "[Errno 111] Connection refused"
                    ),
                ),
                _completed(0, stdout="Cherry-picked gh/u/1/orig\n"),
            ],
        ) as run, mock.patch.object(repo.time, "sleep") as sleep:
            repo.ghstack_cherry_pick(Path("/tmp/worktree"), 123)

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(repo._GHSTACK_RETRY_DELAY_SEC)

    def test_does_not_retry_non_transient_failure(self):
        with mock.patch.object(
            repo,
            "run",
            return_value=_completed(1, stderr="fatal: bad revision\n"),
        ) as run, mock.patch.object(repo.time, "sleep") as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                repo.ghstack_cherry_pick(Path("/tmp/worktree"), 123)

        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()


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


class TestWouldMergeConflict(unittest.TestCase):
    def test_detects_clean_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _git(root, "init", "-q")
            _git(root, "config", "user.name", "Tester")
            _git(root, "config", "user.email", "tester@example.com")
            (root / "base.txt").write_text("base\n")
            _git(root, "add", "base.txt")
            _git(root, "commit", "-q", "-m", "base")

            _git(root, "checkout", "-q", "-b", "upstream")
            (root / "main.txt").write_text("main\n")
            _git(root, "add", "main.txt")
            _git(root, "commit", "-q", "-m", "main")

            _git(root, "checkout", "-q", "-b", "pr", "HEAD~1")
            (root / "pr.txt").write_text("pr\n")
            _git(root, "add", "pr.txt")
            _git(root, "commit", "-q", "-m", "pr")

            self.assertFalse(repo.would_merge_conflict(root, "upstream"))

    def test_detects_conflicting_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _git(root, "init", "-q")
            _git(root, "config", "user.name", "Tester")
            _git(root, "config", "user.email", "tester@example.com")
            (root / "file.txt").write_text("base\n")
            _git(root, "add", "file.txt")
            _git(root, "commit", "-q", "-m", "base")

            _git(root, "checkout", "-q", "-b", "upstream")
            (root / "file.txt").write_text("main\n")
            _git(root, "commit", "-am", "main", "-q")

            _git(root, "checkout", "-q", "-b", "pr", "HEAD~1")
            (root / "file.txt").write_text("pr\n")
            _git(root, "commit", "-am", "pr", "-q")

            self.assertTrue(repo.would_merge_conflict(root, "upstream"))


if __name__ == "__main__":
    unittest.main()
