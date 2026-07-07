import os
import subprocess
import unittest
from unittest import mock

from mergedog import net


class TestIsTransientNetworkError(unittest.TestCase):
    def test_http_5xx_codes_are_transient(self):
        self.assertTrue(net.is_transient_network_error("HTTP 502: Bad Gateway"))
        self.assertTrue(net.is_transient_network_error("HTTP 503"))
        self.assertTrue(net.is_transient_network_error("HTTP 504"))

    def test_code_inside_larger_number_is_not_transient(self):
        self.assertFalse(
            net.is_transient_network_error(
                "HTTP 404: Not Found "
                "(https://api.github.com/repos/o/r/actions/runs/9502312345/jobs)"
            )
        )

    def test_transient_messages_match_case_insensitively(self):
        self.assertTrue(net.is_transient_network_error("Connection Refused"))

    def test_permanent_error_is_not_transient(self):
        self.assertFalse(net.is_transient_network_error("HTTP 404: Not Found"))


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
