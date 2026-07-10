"""No-network unit tests for eval/harness.py scoring + token accounting.

Builds hand-made task records (no model / no Fireworks calls) and checks the
policy scorer, token attribution, and false-trust gap analysis. Importing the
harness pulls in src/ modules but makes no network calls at import time.

Run with: python -m unittest tests.test_harness -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import harness  # noqa: E402
from harness import (  # noqa: E402
    POLICY_MAP,
    _parse_categories,
    gap_analysis,
    score_policy,
)


def _rec(task_id, category, difficulty, local_answer, local_pass,
         remote_pass=True, remote_tokens=100, verify_answer=None):
    from confidence import should_escalate
    return {
        "task_id": task_id,
        "true_category": category,
        "routed_category": category,
        "routed_correct": True,
        "difficulty": difficulty,
        "prompt": "p",
        "reference": "r",
        "local_answer": local_answer,
        "verify_answer": verify_answer,
        "local_pass": local_pass,
        "local_via": "metric",
        "local_detail": "",
        "remote_answer": "remote",
        "remote_tokens": remote_tokens,
        "remote_pass": remote_pass,
        "remote_via": "metric",
        "shipped_escalate": should_escalate(category, difficulty, local_answer,
                                            verify_answer),
    }


class TestScorePolicy(unittest.TestCase):
    def setUp(self):
        # math with a valid-format but WRONG number: local check passes (no
        # escalation) yet the answer is graded FAIL -> the format-only gap.
        self.records = [
            _rec("math-0", "math", "easy", "99", local_pass=False),
            # sentiment correct, local passes and is graded correct.
            _rec("sent-0", "sentiment", "easy", "positive", local_pass=True),
            # ner degenerate -> escalates under shipped/verifiable; remote right.
            _rec("ner-0", "ner", "easy", "N/A", local_pass=False),
        ]

    def test_all_local_zero_tokens(self):
        r = score_policy(self.records, POLICY_MAP["all_local"])
        self.assertEqual(r["tokens"], 0)
        self.assertEqual(r["escalated"], 0)

    def test_all_remote_charges_all_escalated(self):
        r = score_policy(self.records, POLICY_MAP["all_remote"])
        self.assertEqual(r["escalated"], 3)
        self.assertEqual(r["tokens"], 300)
        # every remote answer passes in this fixture -> perfect accuracy.
        self.assertEqual(r["accuracy"], 1.0)

    def test_shipped_escalates_only_degenerate_ner(self):
        r = score_policy(self.records, POLICY_MAP["shipped"])
        # math "99" passes format check (not escalated), sentiment fine, only
        # the degenerate NER escalates.
        self.assertEqual(r["escalated"], 1)
        self.assertEqual(r["tokens"], 100)

    def test_token_nesting_monotonic(self):
        names = ["all_local", "verifiable_only", "shipped",
                 "shipped_plus_hard", "all_remote"]
        toks = [score_policy(self.records, POLICY_MAP[n])["tokens"] for n in names]
        self.assertEqual(toks, sorted(toks))


class TestGapAnalysis(unittest.TestCase):
    def test_format_only_math_leak_detected(self):
        # No verify_answer -> format-only behavior: wrong number still ships.
        records = [_rec("math-0", "math", "easy", "99", local_pass=False)]
        rows = gap_analysis(records)
        math_row = [r for r in rows if r[0] == "math"][0]
        self.assertEqual(math_row[1], 1)  # one false-trust leak
        self.assertIn("dual-answer", math_row[4])

    def test_no_leak_when_local_correct(self):
        records = [_rec("sent-0", "sentiment", "easy", "positive", local_pass=True)]
        rows = gap_analysis(records)
        sent_row = [r for r in rows if r[0] == "sentiment"][0]
        self.assertEqual(sent_row[1], 0)


class TestMathDualAnswerInPolicies(unittest.TestCase):
    def test_disagreeing_verify_answer_escalates_and_closes_the_leak(self):
        # The same wrong "99" that leaked in the format-only gate now
        # escalates when the recompute disagrees -> no false trust.
        rec = _rec("math-0", "math", "easy", "99", local_pass=False,
                   verify_answer="Recomputing: the answer is 102.")
        self.assertTrue(rec["shipped_escalate"])
        rows = gap_analysis([rec])
        math_row = [r for r in rows if r[0] == "math"][0]
        self.assertEqual(math_row[1], 0)

    def test_agreeing_verify_answer_stays_local(self):
        rec = _rec("math-1", "math", "easy", "102", local_pass=True,
                   verify_answer="102")
        self.assertFalse(rec["shipped_escalate"])


class TestParseCategories(unittest.TestCase):
    def test_none_when_no_spec(self):
        self.assertIsNone(_parse_categories(None))
        self.assertIsNone(_parse_categories(""))

    def test_parses_and_validates(self):
        self.assertEqual(_parse_categories("code-gen, logic"),
                         {"code-gen", "logic"})

    def test_unknown_names_dropped(self):
        self.assertEqual(_parse_categories("code-gen,bogus"), {"code-gen"})

    def test_all_unknown_returns_none(self):
        self.assertIsNone(_parse_categories("bogus,nope"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
