"""Escalation gate: decide whether a local answer is likely to fail the
LLM-as-judge check and should be re-answered remotely.

Design principle (per CLAUDE.md's scoring model): the accuracy gate matters
far more than token cost, so this module is deliberately conservative --
when a category has no cheap verifiable check, it defaults to NOT
escalating only when the local answer looks healthy, and always escalates
on any sign the local answer is degenerate.

Verifiable checks (cheap, local, no LLM call) are the primary signal, per
the literature-backed guidance that raw LLM self-confidence is poorly
calibrated. Difficulty is a weak secondary signal only -- project memory
(see MEMORY.md) recorded the local 3B model solving a "hard"-labeled math
question, so difficulty must never be the sole escalation trigger.
"""

import ast
import json
import re

_SENTIMENT_LABELS = {"positive", "negative", "neutral"}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$", re.M)
_DEGENERATE_ANSWERS = {"", "n/a", "na"}
_HEDGE_RE = re.compile(
    r"\b(i'?m not sure|i don'?t know|not certain|unable to determine|"
    r"cannot determine|unclear|i cannot answer|no information|"
    r"insufficient information|as an ai)\b",
    re.I,
)

# Categories with a cheap, local, no-LLM-call verifiable check.
_VERIFIABLE_CATEGORIES = {"math", "ner", "sentiment", "code-gen"}


def _looks_degenerate(answer):
    """Catches local_model.py's own failure fallback ("N/A") and any
    empty/blank answer -- a strong, category-independent failure signal
    that should escalate regardless of verifiable checks or difficulty."""
    return (answer or "").strip().lower() in _DEGENERATE_ANSWERS


def _has_hedge_language(answer):
    return bool(_HEDGE_RE.search(answer or ""))


def _check_math(answer):
    """Passes if the answer contains a parseable number anywhere."""
    return bool(_NUMBER_RE.search(answer or ""))


def _check_ner(answer):
    """Passes if the answer is a valid JSON list. Only falls back to
    extracting a bracketed substring when the full text fails to parse at
    all (e.g. "Here you go: [...]" prose wrapping) -- if the full text
    parses successfully to something other than a list (e.g. a JSON
    object), that result is trusted and NOT overridden by a nested
    list found inside it."""
    text = (answer or "").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        return isinstance(parsed, list)

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return isinstance(json.loads(text[start : end + 1]), list)
        except (json.JSONDecodeError, ValueError):
            return False
    return False


def _check_sentiment(answer):
    """Passes if the answer is exactly one of the allowed labels, modulo
    surrounding punctuation/whitespace/case."""
    cleaned = re.sub(r"[^a-zA-Z]", "", (answer or "")).lower()
    return cleaned in _SENTIMENT_LABELS


def _check_code_gen(answer):
    """Passes if the answer parses as valid Python via ast.parse, after
    stripping markdown code fences. Best-effort: any parse failure (or a
    non-Python answer) counts as a failed check, never crashes."""
    text = _CODE_FENCE_RE.sub("", (answer or "")).strip()
    if not text:
        return False
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


_VERIFIABLE_CHECKS = {
    "math": _check_math,
    "ner": _check_ner,
    "sentiment": _check_sentiment,
    "code-gen": _check_code_gen,
}


def should_escalate(category, difficulty, local_answer):
    """Return True if the local answer should be replaced with a remote
    (Fireworks) answer.

    Precedence:
    1. Degenerate local answer (empty/N/A) -> always escalate.
    2. Category has a verifiable check -> escalate iff it fails. Difficulty
       is NOT consulted here: a structurally valid answer (e.g. a number
       for math, a JSON list for NER) is trusted regardless of the
       heuristic router's difficulty label.
    3. No verifiable check exists for this category (factual,
       summarization, code-debug, logic) -> escalate only when difficulty
       is "hard" AND the answer itself shows hedging/uncertainty language.
       Difficulty alone is deliberately insufficient to trigger escalation.
    """
    if _looks_degenerate(local_answer):
        return True

    check = _VERIFIABLE_CHECKS.get(category)
    if check is not None:
        return not check(local_answer)

    if difficulty == "hard" and _has_hedge_language(local_answer):
        return True

    return False
