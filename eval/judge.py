"""LLM-as-judge for the offline eval harness (OFFLINE ONLY).

Reference-guided, pointwise, binary PASS/FAIL grading with a one-line reason.
Research favors binary + reference-grounded grading over 1-10 scales: it
reduces position/verbosity/self-preference bias and grounds the verdict in the
case's ground truth rather than in surface plausibility.

Hard rules honored here:
  - This module is NEVER imported by src/ -- it lives under eval/ and is
    excluded from the scored container image (.dockerignore).
  - All Fireworks calls go through an OpenAI client configured with
    FIREWORKS_BASE_URL (via src/config.py). No model ID is hardcoded; the
    judge model is resolved from ALLOWED_MODELS at call time and may be a
    strong/large model because JUDGE TOKENS ARE NOT SCORED.
  - Judge tokens are accumulated in a counter SEPARATE from the agent's scored
    token counter in src/remote_model.py, so the two accounting streams never
    mix.

Degrades gracefully: with no credentials / no allowed models / any API error,
judge() returns ("SKIP", reason) instead of raising, so the harness stays
runnable offline.
"""

import logging
import os
import re
import sys

# Flat imports from src/, matching the project convention (see tests/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (  # noqa: E402
    get_fireworks_api_key,
    get_fireworks_base_url,
    resolve_models,
)

try:
    from openai import OpenAI  # noqa: E402
except Exception:  # pragma: no cover - openai always present in this project
    OpenAI = None

logger = logging.getLogger(__name__)

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)
_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.I)

_client = None
judge_tokens_used = 0


def _size_key(model_id):
    match = _SIZE_RE.search(model_id)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return -1.0  # unknown size sorts as smallest so a sized model wins "largest"


def pick_judge_model():
    """Pick the strongest allowed model to judge with. Judge tokens are not
    scored, so we prefer the LARGEST model by the same best-effort size
    heuristic used elsewhere (opposite of pick_remote_model, which minimizes
    tokens). Never hardcodes an ID; resolves from ALLOWED_MODELS at call time.
    Returns None when no model is available."""
    models = resolve_models()
    if not models:
        return None
    return max(models, key=_size_key)


def _get_client():
    global _client
    if _client is None:
        if OpenAI is None:
            return None
        _client = OpenAI(
            base_url=get_fireworks_base_url(),
            api_key=get_fireworks_api_key(),
        )
    return _client


def judge_available():
    """True if a judge call could plausibly succeed (creds + a model + base
    url present). Used by the harness to decide whether to attempt judging or
    fall back to string metrics."""
    return bool(
        get_fireworks_api_key()
        and get_fireworks_base_url()
        and resolve_models()
        and OpenAI is not None
    )


_SYSTEM_PROMPT = (
    "You are a strict, fair grader. You are given a task, a reference answer, "
    "and a candidate answer. Decide whether the candidate answer is correct "
    "with respect to the reference answer and the task. Grade only on "
    "correctness of content -- ignore wording, style, length, and formatting. "
    "Minor paraphrases that preserve the correct meaning are a PASS. Wrong, "
    "missing, or contradictory content is a FAIL. "
    "Respond on a single line in exactly this format:\n"
    "VERDICT: PASS|FAIL - <short reason>"
)


def _build_user_prompt(task, reference, candidate):
    return (
        f"TASK:\n{task}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}\n\n"
        "Grade the candidate answer."
    )


def _parse_verdict(text):
    match = _VERDICT_RE.search(text or "")
    if not match:
        return None, (text or "").strip()[:200]
    verdict = match.group(1).upper()
    reason = re.sub(r"^.*?(PASS|FAIL)\b[\s:\-]*", "", text or "", count=1,
                    flags=re.I).strip()
    return verdict, reason[:200] or "(no reason given)"


def judge(task, reference, candidate, model=None, max_tokens=64):
    """Grade one candidate answer. Returns (verdict, reason) where verdict is
    'PASS', 'FAIL', or 'SKIP' (judge unavailable / parse failure). Never
    raises. Accumulates judge tokens into the separate judge_tokens_used
    counter."""
    global judge_tokens_used

    if not judge_available():
        return "SKIP", "judge unavailable (missing creds/model)"

    client = _get_client()
    if client is None:
        return "SKIP", "openai client unavailable"

    if model is None:
        model = pick_judge_model()
    if not model:
        return "SKIP", "no allowed model to judge with"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(task, reference, candidate)},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
    except Exception as e:
        logger.error("judge call failed (model=%s): %s", model, e)
        return "SKIP", f"judge error: {e}"

    tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    if tokens:
        judge_tokens_used += tokens

    verdict, reason = _parse_verdict(text)
    if verdict is None:
        return "SKIP", f"unparseable verdict: {reason}"
    return verdict, reason


def get_judge_tokens_used():
    return judge_tokens_used


def reset_judge_tokens():
    global judge_tokens_used
    judge_tokens_used = 0
