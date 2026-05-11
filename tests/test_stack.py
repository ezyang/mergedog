import unittest

from mergedog.stack import parse_stack_from_body, parse_stack_from_discussion


_GHSTACK_BODY = """\
Stack from [ghstack](https://github.com/ezyang/ghstack) (oldest at bottom):
* #150
* #149
* __->__ #148
* #147
* #146

This PR refactors the foo subsystem.

Test plan: CI.
"""


_GHSTACK_BODY_BLANK_AFTER_HEADER = """\
Stack from [ghstack](https://github.com/ezyang/ghstack) (oldest at bottom):

* #150
* __->__ #148
"""


_GHSTACK_BODY_SINGLE = """\
Stack from [ghstack](https://github.com/ezyang/ghstack) (oldest at bottom):
* __->__ #500

A standalone ghstack PR.
"""


_GHSTACK_DIRECT_BODY = """\
Stack from ghstack (oldest at bottom):

-> Document full test suite runtime #351
Add logging overhead metrics #350
Add repository agent instructions #349
Reuse GitHub aiohttp session #348
Catch missing awaits in script tests #347
"""


_NON_GHSTACK_BODY = """\
Fixes a bug in the bar module.

Test plan: ran the tests.
"""


class TestParseStackFromBody(unittest.TestCase):
    def test_typical_stack_returned_bottom_first(self):
        # ghstack lists topmost first; we return bottom-first so callers
        # can iterate parents-before-children naturally.
        self.assertEqual(
            parse_stack_from_body(_GHSTACK_BODY),
            [146, 147, 148, 149, 150],
        )

    def test_blank_line_between_header_and_list_is_tolerated(self):
        self.assertEqual(
            parse_stack_from_body(_GHSTACK_BODY_BLANK_AFTER_HEADER),
            [148, 150],
        )

    def test_single_entry_stack(self):
        self.assertEqual(parse_stack_from_body(_GHSTACK_BODY_SINGLE), [500])

    def test_direct_stack_format_returned_bottom_first(self):
        self.assertEqual(
            parse_stack_from_body(_GHSTACK_DIRECT_BODY),
            [347, 348, 349, 350, 351],
        )

    def test_non_ghstack_body_returns_empty(self):
        self.assertEqual(parse_stack_from_body(_NON_GHSTACK_BODY), [])

    def test_stack_comment_used_when_body_is_empty(self):
        comments = [
            {"body": "older comment"},
            {"body": _GHSTACK_BODY},
        ]
        self.assertEqual(
            parse_stack_from_discussion("", comments),
            [146, 147, 148, 149, 150],
        )

    def test_latest_parseable_stack_comment_wins(self):
        comments = [
            {"body": _GHSTACK_BODY},
            {"body": _GHSTACK_DIRECT_BODY},
        ]
        self.assertEqual(
            parse_stack_from_discussion("", comments),
            [347, 348, 349, 350, 351],
        )

    def test_empty_body(self):
        self.assertEqual(parse_stack_from_body(""), [])

    def test_text_after_list_terminates_block(self):
        body = (
            "Stack from [ghstack](...) (oldest at bottom):\n"
            "* #2\n"
            "* __->__ #1\n"
            "\n"
            "Some prose mentioning #999 should not be included.\n"
            "* #999\n"
        )
        self.assertEqual(parse_stack_from_body(body), [1, 2])


if __name__ == "__main__":
    unittest.main()
