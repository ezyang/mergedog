import subprocess
import unittest
from unittest import mock

from mergedog import cli


class TestExternalFailureHaltMessage(unittest.TestCase):
    def test_gh_auth_failure_is_actionable(self):
        exc = subprocess.CalledProcessError(
            1,
            [
                "gh",
                "pr",
                "view",
                "186672",
                "--repo",
                "pytorch/pytorch",
                "--json",
                "number,title",
            ],
            stderr=(
                "HTTP 401: Requires authentication "
                "(https://api.github.com/graphql)\n"
                "Try authenticating with:  gh auth login\n"
            ),
        )

        msg = cli._external_failure_halt_message(exc)

        self.assertEqual(
            msg,
            "gh pr view 186672 failed: GitHub authentication is invalid; "
            "run gh auth status -h github.com, refresh or re-login, then "
            "restart the shepherd",
        )

    def test_generic_failure_uses_last_stderr_line(self):
        exc = subprocess.CalledProcessError(
            128,
            ["git", "fetch", "--prune", "origin"],
            stderr="first line\nfatal: could not read from remote repository\n",
        )

        self.assertEqual(
            cli._external_failure_halt_message(exc),
            "git fetch --prune origin failed: "
            "fatal: could not read from remote repository",
        )


class TestSingleMain(unittest.TestCase):
    def test_subprocess_failure_halts_without_traceback(self):
        exc = subprocess.CalledProcessError(
            1,
            ["gh", "pr", "view", "186672"],
            stderr="HTTP 401: Requires authentication\n",
        )

        with mock.patch("mergedog.notify.configure"), mock.patch(
            "mergedog.shepherd.shepherd", side_effect=exc
        ), mock.patch("mergedog.log.die", side_effect=SystemExit(1)) as die:
            with self.assertRaises(SystemExit):
                cli._single_main(["186672"])

        die.assert_called_once_with(
            "gh pr view 186672 failed: GitHub authentication is invalid; "
            "run gh auth status -h github.com, refresh or re-login, then "
            "restart the shepherd"
        )


if __name__ == "__main__":
    unittest.main()
