"""Unit tests for src/config.py's defensive env-var normalization: quote and
whitespace stripping, base-URL path normalization (the quoted /chat/completions
value that broke live setup), and the fail-open non-chat model filter. No
network. Run: python -m unittest tests.test_config -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (  # noqa: E402
    get_fireworks_api_key,
    get_fireworks_base_url,
    resolve_models,
)

CLEAN = "https://api.fireworks.ai/inference/v1"


def _env(**kw):
    return mock.patch.dict(os.environ, kw)


def _env_without(key):
    env = dict(os.environ)
    env.pop(key, None)
    return mock.patch.dict(os.environ, env, clear=True)


class TestBaseUrlNormalization(unittest.TestCase):
    def test_plain_url_unchanged(self):
        with _env(FIREWORKS_BASE_URL=CLEAN):
            self.assertEqual(get_fireworks_base_url(), CLEAN)

    def test_strips_surrounding_double_quotes(self):
        with _env(FIREWORKS_BASE_URL=f'"{CLEAN}"'):
            self.assertEqual(get_fireworks_base_url(), CLEAN)

    def test_strips_surrounding_single_quotes(self):
        with _env(FIREWORKS_BASE_URL=f"'{CLEAN}'"):
            self.assertEqual(get_fireworks_base_url(), CLEAN)

    def test_strips_trailing_slash(self):
        with _env(FIREWORKS_BASE_URL=CLEAN + "/"):
            self.assertEqual(get_fireworks_base_url(), CLEAN)

    def test_strips_trailing_chat_completions(self):
        with _env(FIREWORKS_BASE_URL=CLEAN + "/chat/completions"):
            self.assertEqual(get_fireworks_base_url(), CLEAN)

    def test_strips_the_exact_live_setup_bug(self):
        # quotes + trailing "/chat/completions/" -- the literal value that
        # produced a connection error then a doubled-path 404 during setup.
        with _env(FIREWORKS_BASE_URL=f'"{CLEAN}/chat/completions/"'):
            self.assertEqual(get_fireworks_base_url(), CLEAN)

    def test_missing_returns_none(self):
        with _env_without("FIREWORKS_BASE_URL"):
            self.assertIsNone(get_fireworks_base_url())


class TestApiKeyCleaning(unittest.TestCase):
    def test_strips_quotes(self):
        with _env(FIREWORKS_API_KEY='"fw-secret-123"'):
            self.assertEqual(get_fireworks_api_key(), "fw-secret-123")

    def test_missing_returns_none(self):
        with _env_without("FIREWORKS_API_KEY"):
            self.assertIsNone(get_fireworks_api_key())


class TestResolveModelsGuard(unittest.TestCase):
    def test_strips_quotes_around_whole_value(self):
        with _env(ALLOWED_MODELS='"a,b,c"'):
            self.assertEqual(resolve_models(), ["a", "b", "c"])

    def test_filters_non_chat_models(self):
        with _env(
            ALLOWED_MODELS=(
                "accounts/fireworks/models/flux-1-schnell-fp8,"
                "accounts/fireworks/models/gpt-oss-120b"
            )
        ):
            self.assertEqual(
                resolve_models(), ["accounts/fireworks/models/gpt-oss-120b"]
            )

    def test_fails_open_when_all_entries_filtered(self):
        # If every entry looks non-chat, keep them rather than return [] --
        # an empty list would silently disable all escalation.
        with _env(ALLOWED_MODELS="flux-1-schnell-fp8,whisper-v3"):
            self.assertEqual(
                resolve_models(), ["flux-1-schnell-fp8", "whisper-v3"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
