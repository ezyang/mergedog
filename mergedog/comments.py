"""Disclosure text for mergedog's PR conversation comments."""

MERGEDOG_DISCLOSURE_MARKER = "<!-- mergedog:disclosure -->"
MERGEDOG_NOTICE = (
    "> [!NOTE]\n"
    "> This PR is being shepherded by "
    "[mergedog](https://github.com/ezyang/mergedog), an automation tool run "
    "by a maintainer. It watches CI and may push fixes; the PR author does "
    "not need to act unless a maintainer asks."
)
MERGEDOG_DISCLOSURE = f"{MERGEDOG_DISCLOSURE_MARKER}\n{MERGEDOG_NOTICE}"
