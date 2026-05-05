import unittest

from mergedog.stack import parse_stack_from_body


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

    def test_non_ghstack_body_returns_empty(self):
        self.assertEqual(parse_stack_from_body(_NON_GHSTACK_BODY), [])

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
