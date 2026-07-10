"""Unit tests for eval/judge.py's verdict robustness: the reasoning_content
fallback and the strict-prompt retry added after the first live sweep left
every code-gen answer ungraded (reasoning judge models truncated at the old
64-token cap before emitting a verdict). No network -- the OpenAI client is
faked. Run with: python -m unittest tests.test_judge -v
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import judge as judge_mod  # noqa: E402


def _response(content=None, reasoning_content=None, tokens=10):
    message = SimpleNamespace(content=content)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=tokens),
    )


class _FakeClient:
    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)
        outer = self

        def create(**kwargs):
            outer.calls.append(kwargs)
            return outer._responses.pop(0)

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create))


class JudgeTestBase(unittest.TestCase):
    def _judge_with(self, responses):
        client = _FakeClient(responses)
        judge_mod.reset_judge_tokens()
        with mock.patch.object(judge_mod, "judge_available", return_value=True), \
                mock.patch.object(judge_mod, "_get_client", return_value=client):
            result = judge_mod.judge("task", "reference", "candidate",
                                     model="test-model")
        return result, client


class TestVerdictParsing(JudgeTestBase):
    def test_clean_verdict_passes_first_try(self):
        (verdict, reason), client = self._judge_with(
            [_response(content="VERDICT: PASS - matches the reference")])
        self.assertEqual(verdict, "PASS")
        self.assertIn("matches", reason)
        self.assertEqual(len(client.calls), 1)

    def test_reasoning_content_fallback_when_content_empty(self):
        # Reasoning models may return empty content with the text in
        # reasoning_content (especially when truncated); the parser must
        # look there before declaring the verdict unparseable.
        (verdict, _), client = self._judge_with(
            [_response(content="",
                       reasoning_content="Thinking... VERDICT: FAIL - wrong value")])
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(len(client.calls), 1)


class TestStrictRetry(JudgeTestBase):
    def test_unparseable_first_response_retries_with_strict_prompt(self):
        (verdict, _), client = self._judge_with([
            _response(content="The candidate looks reasonable to me overall."),
            _response(content="VERDICT: PASS - correct"),
        ])
        self.assertEqual(verdict, "PASS")
        self.assertEqual(len(client.calls), 2)
        retry_system = client.calls[1]["messages"][0]["content"]
        self.assertIn("ONLY one line", retry_system)

    def test_unparseable_after_retry_returns_skip(self):
        (verdict, reason), client = self._judge_with([
            _response(content="rambling with no verdict"),
            _response(content="still rambling"),
        ])
        self.assertEqual(verdict, "SKIP")
        self.assertIn("after retry", reason)
        self.assertEqual(len(client.calls), 2)

    def test_tokens_accumulate_across_both_calls(self):
        self._judge_with([
            _response(content="no verdict here", tokens=100),
            _response(content="VERDICT: FAIL - nope", tokens=40),
        ])
        self.assertEqual(judge_mod.get_judge_tokens_used(), 140)


class TestDefaultMaxTokens(unittest.TestCase):
    def test_default_max_tokens_gives_reasoning_headroom(self):
        # Regression guard for the 0%-code-gen-coverage bug: 64 tokens
        # truncated reasoning judges before the verdict line.
        import inspect
        sig = inspect.signature(judge_mod.judge)
        self.assertGreaterEqual(sig.parameters["max_tokens"].default, 512)


if __name__ == "__main__":
    unittest.main(verbosity=2)
