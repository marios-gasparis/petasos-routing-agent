"""Deterministic, network-free unit tests for eval/metrics.py.

Run with: python -m unittest tests.test_metrics -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from metrics import (  # noqa: E402
    containment,
    exact_match,
    grade_metric,
    grade_strategy,
    ner_f1,
    normalize_text,
    numeric_match,
    parse_entity_list,
    rouge_l,
    token_f1,
)


class TestNormalize(unittest.TestCase):
    def test_lowercase_strip_collapse_punct(self):
        self.assertEqual(normalize_text("  Hello,  World! "), "hello world")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")


class TestNumericMatch(unittest.TestCase):
    def test_exact_number(self):
        self.assertTrue(numeric_match("The answer is 42", "42"))

    def test_number_with_work_shown(self):
        self.assertTrue(numeric_match("60*3 = 180 miles", "180"))

    def test_currency_and_commas(self):
        self.assertTrue(numeric_match("$1,000.00", "1000"))

    def test_wrong_number_fails(self):
        # The Phase 4 gap the harness surfaces: a wrong value must be caught by
        # the numeric metric even though it contains a number.
        self.assertFalse(numeric_match("The total is 99", "180"))

    def test_float_tolerance(self):
        self.assertTrue(numeric_match("13.50 dollars", "13.5"))

    def test_reference_without_number_falls_back_to_exact(self):
        self.assertTrue(numeric_match("twelve", "twelve"))
        self.assertFalse(numeric_match("eleven", "twelve"))


class TestExactMatch(unittest.TestCase):
    def test_label_match_modulo_punct(self):
        self.assertTrue(exact_match("Positive.", "positive"))
        self.assertFalse(exact_match("negative", "positive"))


class TestTokenF1(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(token_f1("a b c", "a b c"), 1.0)

    def test_disjoint(self):
        self.assertEqual(token_f1("a b", "c d"), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(token_f1("a b c d", "a b"), 2 / 3)

    def test_both_empty(self):
        self.assertEqual(token_f1("", ""), 1.0)


class TestContainment(unittest.TestCase):
    def test_fact_wrapped_in_sentence(self):
        self.assertEqual(containment("The capital is Paris.", "Paris"), 1.0)

    def test_missing_fact(self):
        self.assertEqual(containment("I don't know", "Paris"), 0.0)


class TestParseEntityList(unittest.TestCase):
    def test_json_list(self):
        self.assertEqual(parse_entity_list('["A", "B"]'), ["a", "b"])

    def test_python_single_quoted_list(self):
        # The local model actually emits python-style single-quoted lists.
        self.assertEqual(parse_entity_list("['A', 'B']"), ["a", "b"])

    def test_prose_wrapped(self):
        self.assertEqual(parse_entity_list('Here: ["A", "B"] done'), ["a", "b"])

    def test_garbage(self):
        self.assertEqual(parse_entity_list("no entities here"), [])


class TestNerF1(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(
            ner_f1('["Barack Obama", "Hawaii"]', '["Hawaii", "Barack Obama"]'),
            1.0,
        )

    def test_partial(self):
        score = ner_f1('["A", "B"]', '["A", "C"]')
        self.assertAlmostEqual(score, 0.5)

    def test_disjoint(self):
        self.assertEqual(ner_f1('["X"]', '["Y"]'), 0.0)


class TestRougeL(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(rouge_l("the cat sat", "the cat sat"), 1.0)

    def test_partial_overlap(self):
        score = rouge_l("the cat sat on the mat", "the cat sat")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_no_overlap(self):
        self.assertEqual(rouge_l("alpha beta", "gamma delta"), 0.0)


class TestGradeStrategy(unittest.TestCase):
    def test_strategies(self):
        self.assertEqual(grade_strategy("math"), "metric")
        self.assertEqual(grade_strategy("sentiment"), "metric")
        self.assertEqual(grade_strategy("ner"), "metric")
        self.assertEqual(grade_strategy("summarization"), "judge_pref")
        self.assertEqual(grade_strategy("factual"), "judge_pref")
        self.assertEqual(grade_strategy("logic"), "judge_only")
        self.assertEqual(grade_strategy("code-gen"), "judge_only")
        self.assertEqual(grade_strategy("code-debug"), "judge_only")


class TestGradeMetric(unittest.TestCase):
    def test_math_pass_fail(self):
        self.assertEqual(grade_metric("math", "180", "180")[0], True)
        self.assertEqual(grade_metric("math", "99", "180")[0], False)

    def test_sentiment(self):
        self.assertEqual(grade_metric("sentiment", "positive", "positive")[0], True)

    def test_ner_threshold(self):
        p, _, _ = grade_metric("ner", '["A", "B"]', '["A", "B"]')
        self.assertTrue(p)

    def test_judge_only_returns_none(self):
        passed, _, _ = grade_metric("logic", "yes", "yes")
        self.assertIsNone(passed)

    def test_summarization_score_present(self):
        passed, score, _ = grade_metric(
            "summarization", "the cat sat on the mat", "the cat sat on the mat")
        self.assertTrue(passed)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
