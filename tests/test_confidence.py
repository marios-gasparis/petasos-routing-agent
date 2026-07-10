"""Unit tests for src/confidence.py's should_escalate escalation gate.

Covers: verifiable-check pass/fail per checkable category (math, ner,
sentiment, code-gen), the math dual-answer agreement verifier, the
degenerate-answer safety net, and the no-check-category policy (escalate on
hard difficulty OR hedge language -- tightened 2026-07-10 from hard-AND-hedge
after the live sweep measured the old gate leaking a confident
hallucination; difficulty is still ignored entirely for verifiable
categories, per project memory).

Run with: python -m unittest tests.test_confidence -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from confidence import should_escalate  # noqa: E402


class TestDegenerateAnswers(unittest.TestCase):
    def test_empty_answer_escalates_regardless_of_category(self):
        for category in ["factual", "math", "sentiment", "summarization", "ner",
                          "code-debug", "logic", "code-gen"]:
            with self.subTest(category=category):
                self.assertTrue(should_escalate(category, "easy", ""))
                self.assertTrue(should_escalate(category, "easy", "   "))

    def test_local_model_fallback_na_escalates(self):
        self.assertTrue(should_escalate("factual", "easy", "N/A"))
        self.assertTrue(should_escalate("factual", "hard", "n/a"))


class TestMathVerifiableCheck(unittest.TestCase):
    def test_numeric_answer_does_not_escalate(self):
        self.assertFalse(should_escalate("math", "hard", "The answer is 42."))
        self.assertFalse(should_escalate("math", "easy", "-3.5"))

    def test_non_numeric_answer_escalates(self):
        self.assertTrue(should_escalate("math", "easy", "I think it's a lot."))

    def test_verifiable_pass_overrides_hard_difficulty(self):
        # A structurally valid numeric answer is trusted even if the
        # heuristic router flagged the prompt as "hard" -- difficulty is a
        # signal, not a failure predictor (project memory).
        self.assertFalse(should_escalate("math", "hard", "102"))


class TestMathDualAnswerAgreement(unittest.TestCase):
    def test_agreeing_final_numbers_do_not_escalate(self):
        self.assertFalse(should_escalate(
            "math", "hard", "The total is 107.16.",
            verify_answer="Step 1: ... Step 2: ...\n107.16"))

    def test_agreement_ignores_intermediate_numbers(self):
        # Only the FINAL number of each answer is compared; work shown
        # before it must not cause a false disagreement.
        self.assertFalse(should_escalate(
            "math", "easy", "128.50 * 0.85 = 109.225, minus 10 -> 99.23",
            verify_answer="109.225 - 10 = 99.225, so 99.23"))

    def test_disagreeing_final_numbers_escalate(self):
        self.assertTrue(should_escalate(
            "math", "easy", "The answer is 100.40.",
            verify_answer="Recomputing... the answer is 107.16."))

    def test_verify_answer_without_number_escalates(self):
        # A recompute that produced no number is itself a failure signal.
        self.assertTrue(should_escalate(
            "math", "easy", "The answer is 42.",
            verify_answer="I cannot work this out."))

    def test_no_verify_answer_preserves_format_only_behavior(self):
        # Backwards compatible: without a second answer the check is the old
        # format-only one (a number present passes).
        self.assertFalse(should_escalate("math", "hard", "The answer is 42."))

    def test_small_float_tolerance_does_not_escalate(self):
        self.assertFalse(should_escalate(
            "math", "easy", "12.0001", verify_answer="12.0"))


class TestNerVerifiableCheck(unittest.TestCase):
    def test_valid_json_list_does_not_escalate(self):
        self.assertFalse(should_escalate("ner", "easy", '["Barack Obama", "Hawaii"]'))

    def test_json_list_embedded_in_extra_text_does_not_escalate(self):
        self.assertFalse(
            should_escalate("ner", "easy", 'Here you go: ["Apple Inc.", "Cupertino"]')
        )

    def test_non_list_json_escalates(self):
        self.assertTrue(should_escalate("ner", "easy", '{"entities": ["Apple"]}'))

    def test_non_json_escalates(self):
        self.assertTrue(should_escalate("ner", "easy", "Apple Inc., Cupertino"))


class TestSentimentVerifiableCheck(unittest.TestCase):
    def test_allowed_label_does_not_escalate(self):
        self.assertFalse(should_escalate("sentiment", "easy", "positive"))
        self.assertFalse(should_escalate("sentiment", "easy", "Negative."))
        self.assertFalse(should_escalate("sentiment", "easy", "  NEUTRAL  "))

    def test_disallowed_label_escalates(self):
        self.assertTrue(should_escalate("sentiment", "easy", "very positive"))
        self.assertTrue(should_escalate("sentiment", "easy", "mixed"))


class TestCodeGenVerifiableCheck(unittest.TestCase):
    def test_valid_python_does_not_escalate(self):
        code = "def add(a, b):\n    return a + b\n"
        self.assertFalse(should_escalate("code-gen", "easy", code))

    def test_valid_python_in_fenced_block_does_not_escalate(self):
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        self.assertFalse(should_escalate("code-gen", "hard", code))

    def test_syntax_error_escalates(self):
        code = "def add(a, b:\n    return a + b"
        self.assertTrue(should_escalate("code-gen", "easy", code))

    def test_empty_code_block_escalates(self):
        self.assertTrue(should_escalate("code-gen", "easy", "```python\n```"))


class TestNonCheckableCategoriesGate(unittest.TestCase):
    """No-check categories escalate on hard difficulty OR hedge language
    (2026-07-10 live-sweep tightening: +1.7% accuracy for ~858 tokens, and
    the old hard-AND-hedge gate leaked a confident hallucination). Verifiable
    categories still ignore difficulty entirely -- see
    TestMathVerifiableCheck.test_verifiable_pass_overrides_hard_difficulty."""

    NON_CHECKABLE = ["factual", "summarization", "code-debug", "logic"]

    def test_hard_difficulty_alone_escalates(self):
        for category in self.NON_CHECKABLE:
            with self.subTest(category=category):
                self.assertTrue(
                    should_escalate(category, "hard", "This is a confident, complete answer.")
                )

    def test_easy_difficulty_healthy_answer_does_not_escalate(self):
        for category in self.NON_CHECKABLE:
            with self.subTest(category=category):
                self.assertFalse(
                    should_escalate(category, "easy", "This is a confident, complete answer.")
                )

    def test_easy_difficulty_plus_hedging_language_escalates(self):
        for category in self.NON_CHECKABLE:
            with self.subTest(category=category):
                self.assertTrue(
                    should_escalate(category, "easy", "I'm not sure, but possibly this.")
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
