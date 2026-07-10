"""Unit tests for the Phase 6 Stage-2 learned router.

Two layers:
  - src/classifier_router.py: artifact loading + graceful degradation (a missing
    or broken artifact must never raise -- routing falls back to the heuristic).
  - src/router.py Stage-2 integration: the classifier overrides the heuristic
    only when enabled AND confident; otherwise the heuristic wins. The heuristic
    (Stage-1) is never modified.

All deterministic and network-free: the classifier is stubbed via mock so no
model file or sklearn call is required, except one guarded test that loads the
real baked artifact if present.

Run: python -m unittest tests.test_classifier_router -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import classifier_router  # noqa: E402
import router  # noqa: E402

_ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "classifier",
                         "router_clf.joblib")


class TestRouterStage2Disabled(unittest.TestCase):
    """Default (ROUTER_USE_CLASSIFIER unset) must be pure Stage-1 heuristic."""

    def test_default_matches_heuristic(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROUTER_USE_CLASSIFIER", None)
            prompt = "Write a Python function that reverses a string."
            _cat_obj, name, difficulty = router.route_task(prompt)
            h_name, h_diff = router.heuristic_route(prompt)
            self.assertEqual(name, h_name)
            self.assertEqual(difficulty, h_diff)

    def test_disabled_never_calls_classifier(self):
        with mock.patch.dict(os.environ, {"ROUTER_USE_CLASSIFIER": "0"}):
            with mock.patch.object(classifier_router, "classify_category") as m:
                router.route_task("What is the capital of France?")
                m.assert_not_called()


class TestRouterStage2Enabled(unittest.TestCase):
    """With the classifier enabled, confident predictions override; low
    confidence or unavailability falls back to the heuristic. Difficulty always
    comes from the heuristic."""

    def _route(self, prompt):
        with mock.patch.dict(os.environ, {"ROUTER_USE_CLASSIFIER": "1"}):
            return router.route_task(prompt)

    def test_confident_override_wins(self):
        # Heuristic would route this factual prompt to "factual"; a confident
        # classifier saying "math" should win.
        prompt = "What is the capital of France?"
        self.assertEqual(router.heuristic_route(prompt)[0], "factual")
        with mock.patch.object(classifier_router, "classify_category",
                               return_value=("math", 0.95)), \
             mock.patch.object(classifier_router, "min_confidence",
                               return_value=0.5):
            _obj, name, _diff = self._route(prompt)
        self.assertEqual(name, "math")

    def test_low_confidence_falls_back_to_heuristic(self):
        prompt = "What is the capital of France?"
        with mock.patch.object(classifier_router, "classify_category",
                               return_value=("math", 0.20)), \
             mock.patch.object(classifier_router, "min_confidence",
                               return_value=0.5):
            _obj, name, _diff = self._route(prompt)
        self.assertEqual(name, "factual")  # heuristic

    def test_classifier_unavailable_falls_back(self):
        prompt = "What is the capital of France?"
        with mock.patch.object(classifier_router, "classify_category",
                               return_value=None):
            _obj, name, _diff = self._route(prompt)
        self.assertEqual(name, "factual")

    def test_classifier_exception_falls_back(self):
        prompt = "What is the capital of France?"
        with mock.patch.object(classifier_router, "classify_category",
                               side_effect=RuntimeError("boom")):
            _obj, name, _diff = self._route(prompt)
        self.assertEqual(name, "factual")  # never raises

    def test_difficulty_always_from_heuristic(self):
        # A long code-gen prompt is "hard" per the heuristic; even if the
        # classifier overrides the category, difficulty is unchanged.
        prompt = ("Create a function to compute the factorial of a number "
                  "recursively, then explain how the recursion terminates, and "
                  "write a program that also handles negative inputs by raising "
                  "a ValueError, ensuring it is well documented with type hints.")
        h_diff = router.heuristic_route(prompt)[1]
        self.assertEqual(h_diff, "hard")
        with mock.patch.object(classifier_router, "classify_category",
                               return_value=("factual", 0.99)), \
             mock.patch.object(classifier_router, "min_confidence",
                               return_value=0.5):
            _obj, _name, diff = self._route(prompt)
        self.assertEqual(diff, "hard")


class TestClassifierRouterDegradation(unittest.TestCase):
    """classifier_router must degrade gracefully with no artifact / bad path."""

    def setUp(self):
        classifier_router._reset_cache_for_tests()

    def tearDown(self):
        classifier_router._reset_cache_for_tests()
        os.environ.pop("ROUTER_CLF_PATH", None)

    def test_missing_artifact_is_unavailable(self):
        with mock.patch.dict(os.environ,
                             {"ROUTER_CLF_PATH": "/no/such/file.joblib"}):
            classifier_router._reset_cache_for_tests()
            self.assertIsNone(classifier_router.get_artifact())
            self.assertFalse(classifier_router.is_available())
            self.assertIsNone(classifier_router.classify_category("hi"))
            # Unreachable gate so the classifier is never used when absent.
            self.assertEqual(classifier_router.min_confidence(), 1.0)
            self.assertIsNone(classifier_router.predict_success("hi"))
            self.assertIsNone(classifier_router.success_threshold())

    def test_success_head_absent_returns_none(self):
        # Even with the real artifact loaded, the success head is not trained
        # yet (blocked on Phase 5 labels) -> predict_success is None.
        if not os.path.exists(_ARTIFACT):
            self.skipTest("baked artifact not present")
        classifier_router._reset_cache_for_tests()
        os.environ.pop("ROUTER_CLF_PATH", None)
        self.assertTrue(classifier_router.is_available())
        self.assertIsNone(classifier_router.predict_success("hi"))
        self.assertIsNone(classifier_router.success_threshold())
        self.assertIsNone(router.predicted_local_success("hi"))


class TestRealArtifactInference(unittest.TestCase):
    """Guarded end-to-end check against the baked artifact if it exists."""

    def setUp(self):
        classifier_router._reset_cache_for_tests()
        if not os.path.exists(_ARTIFACT):
            self.skipTest("baked artifact not present")

    def tearDown(self):
        classifier_router._reset_cache_for_tests()

    def test_predicts_valid_category(self):
        valid = {"factual", "math", "sentiment", "summarization",
                 "ner", "code-debug", "logic", "code-gen"}
        result = classifier_router.classify_category(
            "Write a Python function that reverses a string.")
        self.assertIsNotNone(result)
        category, confidence = result
        self.assertIsInstance(category, str)  # not np.str_
        self.assertIn(category, valid)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_empty_prompt_returns_none(self):
        self.assertIsNone(classifier_router.classify_category("   "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
