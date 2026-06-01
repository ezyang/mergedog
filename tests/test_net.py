import os
import subprocess
import unittest
from unittest import mock

from mergedog import net


class TestGithubApiEnvExtra(unittest.TestCase):
    def setUp(self):
        net._git_configured_proxy.cache_clear()

    def test_respects_existing_proxy_environment(self):
        with mock.patch.dict(
            os.environ, {"HTTPS_PROXY": "http://env-proxy"}, clear=True
        ), mock.patch.object(net.subprocess, "run") as run:
            self.assertIsNone(net.github_api_env_extra())

        run.assert_not_called()

    def test_uses_git_https_proxy_when_environment_has_none(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            net.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                ["git"], 0, "http://git-proxy\n", ""
            ),
        ) as run:
            self.assertEqual(
                net.github_api_env_extra(),
                {
                    "HTTPS_PROXY": "http://git-proxy",
                    "HTTP_PROXY": "http://git-proxy",
                },
            )

        run.assert_called_once_with(
            ["git", "config", "--get", "https.proxy"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_falls_back_to_git_http_proxy(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            net.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(["git"], 1, "", ""),
                subprocess.CompletedProcess(["git"], 0, "http://git-proxy\n", ""),
            ],
        ):
            self.assertEqual(
                net.github_api_env_extra(),
                {
                    "HTTPS_PROXY": "http://git-proxy",
                    "HTTP_PROXY": "http://git-proxy",
                },
            )
