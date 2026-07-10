"""Runtime loader/inference for the Stage-2 learned router (Phase 6).

This is the SHIPPED half of Phase 6 (unlike classifier/train_classifier.py,
which is offline/training-only). It loads classifier/router_clf.joblib -- a
scikit-learn TF-IDF + logistic-regression artifact -- and exposes cheap
inference used by src/router.py's Stage-2 path.

Design constraints (CLAUDE.md hard limits):
  - Loads in < 2 s: the artifact is ~0.6 MB; sklearn import + joblib.load is
    sub-second. No torch / sentence-transformers in the runtime path (see the
    image-budget rationale in classifier/train_classifier.py).
  - Never raises: a missing artifact, a missing scikit-learn, or an unpickle
    failure must degrade to "classifier unavailable" so src/router.py falls
    back to the Stage-1 heuristic. The accuracy gate must never be risked by an
    import error.

The artifact is a dict (see train_classifier.save_artifact):
  category_model, min_category_confidence, success_model, success_threshold,
  categories, backend, version, ...
"""

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "classifier",
    "router_clf.joblib",
)

# Load-once cache. `_load_attempted` guards against retrying (and re-logging) a
# failed load on every prompt.
_artifact = None
_load_attempted = False


def _artifact_path():
    return os.environ.get("ROUTER_CLF_PATH", _DEFAULT_PATH)


def get_artifact():
    """Lazily load and cache the classifier artifact. Returns the artifact dict
    or None if it cannot be loaded (any reason). Logged once."""
    global _artifact, _load_attempted
    if _load_attempted:
        return _artifact
    _load_attempted = True

    path = _artifact_path()
    if not os.path.exists(path):
        logger.info("classifier artifact not found at %s; Stage-2 disabled, "
                    "using Stage-1 heuristic only", path)
        return None
    try:
        import joblib  # local import so a missing dep degrades gracefully
        loaded = joblib.load(path)
    except Exception as exc:
        logger.warning("failed to load classifier artifact %s: %s; falling "
                       "back to Stage-1 heuristic", path, exc)
        return None

    if not isinstance(loaded, dict) or "category_model" not in loaded:
        logger.warning("classifier artifact %s has unexpected shape; ignoring",
                       path)
        return None

    _artifact = loaded
    logger.info("loaded Stage-2 classifier artifact %s (backend=%s, "
                "min_confidence=%s, success_head=%s)", path,
                loaded.get("backend"), loaded.get("min_category_confidence"),
                loaded.get("success_model") is not None)
    return _artifact


def is_available():
    return get_artifact() is not None


def classify_category(prompt):
    """Return (category, confidence) from the learned category head, or None if
    the classifier is unavailable or the prompt is empty. The caller decides
    whether `confidence` clears the gate (see min_confidence())."""
    art = get_artifact()
    if art is None or not prompt or not prompt.strip():
        return None
    model = art.get("category_model")
    if model is None:
        return None
    try:
        proba = model.predict_proba([prompt])[0]
        classes = list(model.named_steps["clf"].classes_)
        idx = int(proba.argmax())
        # Cast to a plain str: sklearn returns numpy str scalars, which would
        # otherwise leak np.str_ into category names / logs / JSON.
        return str(classes[idx]), float(proba[idx])
    except Exception as exc:
        logger.warning("classifier category prediction failed: %s", exc)
        return None


def min_confidence():
    """The tuned confidence gate: below this, routing defers to the heuristic.
    Falls back to a conservative 0.5 if the artifact does not carry one."""
    art = get_artifact()
    if art is None:
        return 1.0  # unreachable gate -> never use the classifier
    return float(art.get("min_category_confidence", 0.5))


def predict_success(prompt):
    """Return P(local model answers this prompt correctly) in [0, 1], or None if
    no success head is trained yet (blocked on Phase 5 harness labels) or the
    classifier is unavailable. Wired for the escalation gate but INERT until a
    success head exists -- see docs/phase6-classifier.md."""
    art = get_artifact()
    if art is None:
        return None
    model = art.get("success_model")
    if model is None or not prompt or not prompt.strip():
        return None
    try:
        classes = list(model.named_steps["clf"].classes_)
        proba = model.predict_proba([prompt])[0]
        # Probability of the positive (success == 1) class.
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba.max())
    except Exception as exc:
        logger.warning("classifier success prediction failed: %s", exc)
        return None


def success_threshold():
    """Escalate when predict_success(prompt) < this. Only meaningful once a
    success head is trained; returns None otherwise."""
    art = get_artifact()
    if art is None or art.get("success_model") is None:
        return None
    return float(art.get("success_threshold", 0.5))


def _reset_cache_for_tests():
    """Test hook: clear the load-once cache so a test can point ROUTER_CLF_PATH
    at a fixture (or simulate a missing artifact) between cases."""
    global _artifact, _load_attempted
    _artifact = None
    _load_attempted = False
