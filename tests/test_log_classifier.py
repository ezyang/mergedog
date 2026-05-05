import unittest

from mergedog.log_classifier import classify


class TestClassify(unittest.TestCase):
    def test_pytest_failure(self):
        lines = [
            "collecting ...",
            "PASSED test/test_foo.py::TestFoo::test_bar",
            "FAILED test/test_overrides.py::TestTorchFunctionRedispatchOpsCPU::test_redispatch_as_strided_partial_views_cpu_bfloat16",
            "= 1 failed, 200 passed =",
        ]
        m = classify(lines)
        assert m is not None
        self.assertEqual(m.rule_name, "pytest failure")
        self.assertEqual(m.line_num, 2)
        self.assertIn("test_redispatch_as_strided", m.captures[0])

    def test_compile_error(self):
        lines = [
            "building foo.cpp",
            "foo.cpp:42:10: error: 'bar' was not declared in this scope",
            "ninja: build stopped: subcommand failed.",
        ]
        m = classify(lines)
        assert m is not None
        # "Compile error" is higher priority than "Build error"
        self.assertEqual(m.rule_name, "Compile error")
        self.assertEqual(m.line_num, 1)

    def test_python_unittest_failure(self):
        lines = [
            "test_foo (TestBar) ... ok",
            "FAIL [12.3s]: test_baz (TestQux)",
        ]
        m = classify(lines)
        assert m is not None
        self.assertEqual(m.rule_name, "Python unittest failure")

    def test_no_match(self):
        lines = ["everything is fine", "all good here"]
        self.assertIsNone(classify(lines))

    def test_priority_ordering(self):
        # "pytest failure" (priority ~12) should beat "GHA error" (priority ~76)
        lines = [
            "##[error]Process completed with exit code 1.",
            "FAILED test/test_foo.py::TestBar::test_baz",
        ]
        m = classify(lines)
        assert m is not None
        self.assertEqual(m.rule_name, "pytest failure")

    def test_last_match_wins_for_same_rule(self):
        lines = [
            "FAILED test/test_a.py::TestA::test_one",
            "lots of output here",
            "FAILED test/test_b.py::TestB::test_two",
        ]
        m = classify(lines)
        assert m is not None
        # Last occurrence of the highest-priority matching rule
        self.assertIn("test_two", m.captures[0])


if __name__ == "__main__":
    unittest.main()
