"""Escalation gate: decide whether a local answer is likely to fail the
LLM-as-judge check and should be re-answered remotely.

Design principle (per CLAUDE.md's scoring model): the accuracy gate matters
far more than token cost, so this module is deliberately conservative --
when a category has no cheap verifiable check, it defaults to NOT
escalating only when the local answer looks healthy, and always escalates
on any sign the local answer is degenerate.

Verifiable checks (cheap, local, no LLM call) are the primary signal, per
the literature-backed guidance that raw LLM self-confidence is poorly
calibrated. For those categories, difficulty is ignored entirely: a
structurally valid answer is trusted even on "hard"-labeled prompts
(project memory recorded the local 3B solving a "hard" math question).

For the NO-CHECK categories (factual, summarization, code-debug, logic),
the 2026-07-10 live sweep changed the policy: hard difficulty alone now
escalates. The sweep measured the old hard-AND-hedge gate leaking a
confident hallucination, while escalating all hard no-check tasks bought
+1.7% accuracy for ~858 tokens -- cheap insurance given that failing the
accuracy gate is catastrophic (zero score) while extra tokens are only a
marginal ranking penalty.

The math check is no longer format-only: callers may pass a second,
independently generated local answer (verify_answer) and the check fails
when the two final numbers disagree -- a zero-scored-token dual-answer
agreement verifier (local tokens are free). The first live sweep measured
the format-only check leaking 2/20 wrong math answers.
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


# Same tolerances as eval/metrics.py's numeric comparison.
_NUM_REL_TOL = 1e-3
_NUM_ABS_TOL = 1e-6


def _final_number(text):
    """The LAST number in the text (the per-category system prompt asks for
    the final answer on its own line, so last = final), or None."""
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _num_close(a, b):
    return abs(a - b) <= max(_NUM_ABS_TOL, _NUM_REL_TOL * max(abs(a), abs(b)))


def _check_math(answer, verify_answer=None):
    """Format check (a parseable number must be present) plus, when a second
    independently generated answer is supplied, dual-answer agreement: the
    final numbers of both answers must match. Disagreement -- or a verify
    answer with no number at all -- fails the check (conservative: a failed
    recompute is itself a failure signal). verify_answer=None preserves the
    old format-only behavior for callers without a second answer."""
    final = _final_number(answer)
    if final is None:
        return False
    if verify_answer is not None:
        verify_final = _final_number(verify_answer)
        if verify_final is None or not _num_close(final, verify_final):
            return False
    return True


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


def should_escalate(category, difficulty, local_answer, verify_answer=None):
    """Return True if the local answer should be replaced with a remote
    (Fireworks) answer.

    verify_answer (optional): a second, independently generated LOCAL answer
    for math tasks; when provided, the math check additionally requires the
    two final numbers to agree. Zero scored-token cost (local tokens are
    free). Ignored for every other category.

    Precedence:
    1. Degenerate local answer (empty/N/A) -> always escalate.
    2. Category has a verifiable check -> escalate iff it fails. Difficulty
       is NOT consulted here: a structurally valid answer (e.g. an agreeing
       number for math, a JSON list for NER) is trusted regardless of the
       heuristic router's difficulty label.
    3. No verifiable check exists for this category (factual,
       summarization, code-debug, logic) -> escalate when difficulty is
       "hard" OR the answer shows hedging/uncertainty language. (Live-sweep
       tightening, 2026-07-10: the old hard-AND-hedge rule leaked a
       confident hallucination; hard-alone escalation on no-check
       categories measured +1.7% accuracy for ~858 tokens.)
    """
    if _looks_degenerate(local_answer):
        return True

    if category == "math":
        return not _check_math(local_answer, verify_answer)

    check = _VERIFIABLE_CHECKS.get(category)
    if check is not None:
        return not check(local_answer)

    if difficulty == "hard" or _has_hedge_language(local_answer):
        return True

    return False
