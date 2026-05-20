import unittest
from unittest import mock

from mergedog import labels


def _pr_data(*label_names: str) -> dict:
    return {
        "title": "Add a user-facing feature",
        "body": "",
        "url": "https://github.com/pytorch/pytorch/pull/1",
        "labels": [{"name": name} for name in label_names],
    }


class TestReleaseNoteAutolabelGate(unittest.TestCase):
    def test_no_release_note_labels_needs_autolabel(self):
        self.assertTrue(labels._pr_needs_autolabel(_pr_data()))

    def test_only_release_notes_label_needs_topic_label(self):
        self.assertTrue(
            labels._pr_needs_autolabel(_pr_data("release notes: nn"))
        )

    def test_only_topic_label_needs_release_notes_label(self):
        self.assertTrue(labels._pr_needs_autolabel(_pr_data("topic: nn")))

    def test_release_notes_and_topic_labels_are_sufficient(self):
        self.assertFalse(
            labels._pr_needs_autolabel(
                _pr_data("release notes: nn", "topic: nn")
            )
        )

    def test_not_user_facing_topic_is_sufficient(self):
        self.assertFalse(
            labels._pr_needs_autolabel(_pr_data("topic: not user facing"))
        )


class TestAutolabelIfNeeded(unittest.TestCase):
    def _run_autolabel(
        self,
        pr_data: dict,
        suggested: list[str],
    ) -> mock.Mock:
        repo_labels = (
            [],
            [
                {"name": "release notes: nn", "description": ""},
                {"name": "release notes: distributed", "description": ""},
            ],
            [
                {"name": "topic: nn", "description": ""},
                {"name": "topic: not user facing", "description": ""},
            ],
        )
        with mock.patch.object(
            labels, "_get_relevant_labels", return_value=repo_labels
        ), mock.patch.object(
            labels, "_get_changed_files", return_value=["torch/nn/foo.py"]
        ), mock.patch.object(
            labels, "_invoke_claude_for_labels", return_value=suggested
        ) as invoke, mock.patch.object(
            labels.github, "add_label"
        ) as add_label:
            labels.autolabel_if_needed(1, pr_data)
        invoke.assert_called_once()
        return add_label

    def test_adds_missing_topic_when_release_notes_label_exists(self):
        add_label = self._run_autolabel(
            _pr_data("release notes: nn"),
            ["release notes: nn", "topic: nn"],
        )

        add_label.assert_called_once_with(1, "topic: nn")

    def test_adds_missing_release_notes_when_topic_label_exists(self):
        add_label = self._run_autolabel(
            _pr_data("topic: nn"),
            ["release notes: nn", "topic: nn"],
        )

        add_label.assert_called_once_with(1, "release notes: nn")

    def test_skips_when_labels_are_already_sufficient(self):
        with mock.patch.object(labels, "_invoke_claude_for_labels") as invoke:
            labels.autolabel_if_needed(
                1, _pr_data("release notes: nn", "topic: nn")
            )

        invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
