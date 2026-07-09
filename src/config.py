"""Centralized env var access and runtime constants. Fireworks model IDs
are never hardcoded anywhere in the codebase -- resolve_models() is the
only place ALLOWED_MODELS gets parsed, and everything else selects a
model from that list by substring match."""

import os

DEFAULT_VERBALIZED_CONFIDENCE_THRESHOLD = 0.6


def get_fireworks_api_key():
    return os.environ.get("FIREWORKS_API_KEY")


def get_fireworks_base_url():
    return os.environ.get("FIREWORKS_BASE_URL")


def resolve_models():
    """Parse ALLOWED_MODELS (comma-separated) at call time into a list of
    model ids, stripped of whitespace, empty entries dropped. Read at call
    time (not import time) so it always reflects the container's actual
    runtime environment."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def get_verbalized_confidence_threshold():
    """Threshold below which a self-reported confidence score should be
    treated as low (escalate). Kept here so it can be tuned without
    touching confidence.py. Literature-backed caveat (see confidence.py):
    verbalized self-confidence is weakly calibrated, so it is only ever a
    secondary signal, never the primary escalation trigger."""
    raw = os.environ.get(
        "VERBALIZED_CONFIDENCE_THRESHOLD", DEFAULT_VERBALIZED_CONFIDENCE_THRESHOLD
    )
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_VERBALIZED_CONFIDENCE_THRESHOLD
