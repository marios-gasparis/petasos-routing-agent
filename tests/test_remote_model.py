"""Unit tests for src/remote_model.py's pick_remote_model / Gemma-preference
logic and src/config.py's resolve_models(). No real network calls are made
-- ALLOWED_MODELS is faked via mock.patch.dict(os.environ, ...) and
remote_chat's OpenAI client is never invoked.

Run with: python -m unittest tests.test_remote_model -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import remote_model  # noqa: E402
from config import resolve_models  # noqa: E402
from remote_model import pick_remote_model, remote_chat  # noqa: E402


def _fake_response(content, finish_reason, total_tokens):
    """Build a minimal stand-in for an OpenAI chat.completions response."""
    message = mock.Mock(content=content)
    choice = mock.Mock(message=message, finish_reason=finish_reason)
    usage = mock.Mock(total_tokens=total_tokens)
    return mock.Mock(choices=[choice], usage=usage)

# A provisional model list in the shape called out by the spec's caveats
# section (compass_artifact ... "provisional model list").
SAMPLE_MODELS = (
    "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,"
    "gemma-4-31b-it-nvfp4"
)


class TestResolveModels(unittest.TestCase):
    def test_parses_comma_separated_list(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": "a, b ,c"}):
            self.assertEqual(resolve_models(), ["a", "b", "c"])

    def test_empty_env_returns_empty_list(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": ""}):
            self.assertEqual(resolve_models(), [])

    def test_missing_env_returns_empty_list(self):
        env = dict(os.environ)
        env.pop("ALLOWED_MODELS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(resolve_models(), [])

    def test_drops_blank_entries(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": "a,,b,"}):
            self.assertEqual(resolve_models(), ["a", "b"])


class TestPickRemoteModelGemmaPreference(unittest.TestCase):
    def test_non_code_category_prefers_gemma(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": SAMPLE_MODELS}):
            for category in ["factual", "math", "sentiment", "summarization",
                              "ner", "logic"]:
                with self.subTest(category=category):
                    model = pick_remote_model(category)
                    self.assertIn("gemma", model.lower())

    def test_non_code_category_picks_smallest_gemma(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": SAMPLE_MODELS}):
            model = pick_remote_model("factual")
            # gemma-4-26b-a4b-it (26b) is smaller than gemma-4-31b-it (31b)
            # and gemma-4-31b-it-nvfp4 (31b).
            self.assertEqual(model, "gemma-4-26b-a4b-it")

    def test_code_category_prefers_code_model_over_gemma(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": SAMPLE_MODELS}):
            for category in ["code-gen", "code-debug"]:
                with self.subTest(category=category):
                    model = pick_remote_model(category)
                    self.assertEqual(model, "kimi-k2p7-code")

    def test_code_category_falls_back_to_gemma_when_no_code_model(self):
        models = "minimax-m3,gemma-4-31b-it,gemma-4-26b-a4b-it"
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": models}):
            model = pick_remote_model("code-gen")
            self.assertIn("gemma", model.lower())

    def test_falls_back_to_any_model_when_no_gemma_or_code(self):
        models = "minimax-m3,some-other-model-70b"
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": models}):
            model = pick_remote_model("factual")
            self.assertIn(model, models.split(","))

    def test_empty_allowed_models_returns_none(self):
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": ""}):
            self.assertIsNone(pick_remote_model("factual"))

    def test_never_hardcodes_a_model_id(self):
        # Swapping ALLOWED_MODELS entirely changes the pick -- proves
        # selection is purely runtime-driven, no hardcoded fallback ID.
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": "only-model-9b"}):
            self.assertEqual(pick_remote_model("factual"), "only-model-9b")
        with mock.patch.dict(os.environ, {"ALLOWED_MODELS": "different-model-3b"}):
            self.assertEqual(pick_remote_model("factual"), "different-model-3b")


class TestRemoteChatTruncationGuard(unittest.TestCase):
    """The remote call gets an expanded output budget, and a length-truncated
    answer is discarded (returns the "N/A" sentinel) so callers keep the
    complete local answer. Tokens are still counted either way."""

    def setUp(self):
        self._saved_tokens = remote_model.total_tokens_used
        remote_model.total_tokens_used = 0

    def tearDown(self):
        remote_model.total_tokens_used = self._saved_tokens

    def test_expands_max_tokens_with_multiplier_and_floor(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _fake_response(
            "ok", "stop", 10
        )
        with mock.patch.object(remote_model, "_get_client", return_value=client):
            # factual cap 40 -> 40*4=160 -> floored to 512.
            remote_chat("some-model-8b", "sys", "user", max_tokens=40)
            self.assertEqual(
                client.chat.completions.create.call_args.kwargs["max_tokens"], 512
            )
            # code-gen cap 400 -> 400*4=1600 (above the floor).
            remote_chat("some-model-8b", "sys", "user", max_tokens=400)
            self.assertEqual(
                client.chat.completions.create.call_args.kwargs["max_tokens"], 1600
            )

    def test_truncated_answer_returns_na_but_counts_tokens(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _fake_response(
            "The answer begins but never fini", "length", 128
        )
        with mock.patch.object(remote_model, "_get_client", return_value=client):
            text, _usage = remote_chat("some-model-8b", "sys", "user", max_tokens=40)
        self.assertEqual(text, "N/A")
        self.assertEqual(remote_model.total_tokens_used, 128)

    def test_complete_answer_passes_through(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _fake_response(
            "  positive  ", "stop", 7
        )
        with mock.patch.object(remote_model, "_get_client", return_value=client):
            text, _usage = remote_chat("some-model-8b", "sys", "user", max_tokens=5)
        self.assertEqual(text, "positive")
        self.assertEqual(remote_model.total_tokens_used, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
