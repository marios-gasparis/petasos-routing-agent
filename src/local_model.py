import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "local")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=LOCAL_BASE_URL, api_key="not-needed")
    return _client


def local_chat(system, user, max_tokens, stop=None, temperature=0.0):
    """Answer via the local llama-server (OpenAI-compatible /v1/chat/completions).
    Never raises: any failure is logged and a fallback string returned so the
    caller always has something to write. Local tokens are not counted, so
    usage is always None."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = _get_client().chat.completions.create(
            model=LOCAL_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        text = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("local_chat failed: %s", e)
        return "N/A", None

    return text.strip(), None
