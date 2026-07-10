"""Centralized env var access and runtime constants. Fireworks model IDs
are never hardcoded anywhere in the codebase -- resolve_models() is the
only place ALLOWED_MODELS gets parsed, and everything else selects a
model from that list by substring match.

Env values are read defensively: surrounding quotes/whitespace are stripped
and the base URL is normalized, because a malformed .env value (a quoted URL,
a stray trailing "/chat/completions") otherwise fails silently at runtime and
would tank every scored remote call."""

import os

DEFAULT_VERBALIZED_CONFIDENCE_THRESHOLD = 0.6

# Substrings marking a model as non-chat (image / audio / embedding). Such ids
# can appear in a Fireworks catalog listing (e.g. "flux-1-schnell-fp8") and
# would fail or return garbage if selected for /chat/completions.
_NON_CHAT_SUBSTRINGS = ("flux", "stable-diffusion", "sdxl", "whisper", "embed")


def _clean_env_value(value):
    """Strip surrounding whitespace and one layer of matching quotes. A
    quoted .env value (KEY="val") otherwise arrives with the quote characters
    literally attached, which silently corrupts URLs and model ids."""
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def get_fireworks_api_key():
    return _clean_env_value(os.environ.get("FIREWORKS_API_KEY")) or None


def get_fireworks_base_url():
    """Return the Fireworks API root, defensively normalized. The OpenAI
    client appends "/chat/completions" itself, so a stray trailing
    "/chat/completions" in the env value would double the path and 404 every
    remote call -- strip it (plus surrounding quotes and trailing slashes)."""
    url = _clean_env_value(os.environ.get("FIREWORKS_BASE_URL"))
    if not url:
        return None
    url = url.rstrip("/")
    if url.lower().endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    return url or None


def resolve_models():
    """Parse ALLOWED_MODELS (comma-separated) at call time into a list of
    model ids -- surrounding quotes/whitespace stripped, empty entries
    dropped, and obvious non-chat models (image/audio/embedding) filtered
    out. Read at call time (not import time) so it always reflects the
    container's actual runtime environment.

    Fails open: if the non-chat filter would drop every entry, the unfiltered
    (but still quote-cleaned) list is returned instead -- an empty list would
    disable escalation and risk the accuracy gate, a worse outcome than
    keeping an unexpected id the caller can still decline to use."""
    raw = _clean_env_value(os.environ.get("ALLOWED_MODELS", "")) or ""
    models = []
    for part in raw.split(","):
        cleaned = _clean_env_value(part)
        if cleaned:
            models.append(cleaned)
    chat_models = [
        m for m in models
        if not any(hint in m.lower() for hint in _NON_CHAT_SUBSTRINGS)
    ]
    return chat_models or models


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
