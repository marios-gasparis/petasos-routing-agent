import logging
import re

from openai import OpenAI

from config import get_fireworks_api_key, get_fireworks_base_url, resolve_models

logger = logging.getLogger(__name__)

# Code categories prefer a code-specialized allowed model (accuracy on code
# correctness matters more than the Gemma bonus); every other category
# prefers Gemma to compete for the "Best Use of Gemma via Fireworks" bonus.
_CODE_CATEGORIES = {"code-gen", "code-debug"}
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)

_client = None
total_tokens_used = 0


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=get_fireworks_base_url(), api_key=get_fireworks_api_key())
    return _client


def _size_key(model_id):
    """Best-effort 'smallest capable model' heuristic: extract a
    parameter-count token like '27b' / '8B' from the model id and use it
    as a sort key. Models with no discernible size sort last (treated as
    unknown/large) rather than being picked as "smallest" by accident."""
    match = _SIZE_RE.search(model_id)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return float("inf")


def _smallest(models):
    return min(models, key=_size_key)


def pick_remote_model(category):
    """Resolve which allowed Fireworks model to call for a category. Never
    hardcodes a model ID -- always resolves from ALLOWED_MODELS (via
    config.resolve_models()) at runtime and selects by substring match."""
    models = resolve_models()
    if not models:
        logger.error("ALLOWED_MODELS is empty; no remote model available")
        return None

    preferred = []
    if category in _CODE_CATEGORIES:
        preferred = [m for m in models if "code" in m.lower()]
    if not preferred:
        preferred = [m for m in models if "gemma" in m.lower()]
    if not preferred:
        preferred = models
    return _smallest(preferred)


def remote_chat(model, system, user, max_tokens, stop=None):
    """Answer via Fireworks (OpenAI-compatible /v1/chat/completions), routed
    through FIREWORKS_BASE_URL only -- calls bypassing this client are
    invalid for scoring. Never raises: any failure is logged and a
    fallback string returned. Unlike local_chat, usage IS accumulated
    because remote tokens count toward the score."""
    global total_tokens_used

    if not model:
        logger.error("remote_chat called with no model resolved; skipping remote call")
        return "N/A", None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
    except Exception as e:
        logger.error("remote_chat failed (model=%s): %s", model, e)
        return "N/A", None

    tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    if tokens:
        total_tokens_used += tokens
    logger.info("remote_chat model=%s tokens=%s", model, tokens)

    return text.strip(), usage


def get_total_tokens_used():
    return total_tokens_used
