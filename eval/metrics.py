"""Deterministic, network-free grading metrics for the offline eval harness.

Everything here is pure-python (no numpy / rouge-score dependency) so the unit
tests stay hermetic and the harness can grade without any external service.
`rouge-score` / `numpy` are listed in requirements-eval.txt as optional niceties
only; this module never imports them.

Grading strategy per category (see GRADE_STRATEGY):
  - "metric" (authoritative): math (numeric match), sentiment (label exact
    match), ner (entity-set F1 >= threshold). Cheap and reliable; the judge
    is not needed.
  - "judge_pref": summarization (ROUGE-L) and factual (token-F1 containment).
    Open-ended, so the LLM judge is preferred when available; the metric is a
    fallback score used only when the judge is unavailable (offline dry runs).
  - "judge_only": logic, code-debug, code-gen. No trustworthy string metric
    exists (correctness needs reasoning / execution), so grading defers
    entirely to the judge; grade_metric() returns None for these.

This module is import-safe from tests via sys.path insertion; it does not
import anything from src/.
"""

import ast
import json
import re
from collections import Counter

# --- category grading strategy -------------------------------------------------

METRIC_AUTHORITATIVE = {"math", "sentiment", "ner"}
JUDGE_PREFERRED = {"summarization", "factual"}
JUDGE_ONLY = {"logic", "code-debug", "code-gen"}


def grade_strategy(category):
    if category in METRIC_AUTHORITATIVE:
        return "metric"
    if category in JUDGE_PREFERRED:
        return "judge_pref"
    if category in JUDGE_ONLY:
        return "judge_only"
    # Unknown category -> treat as judge-preferred (fall back to token F1).
    return "judge_pref"


# --- pass thresholds -----------------------------------------------------------

NER_F1_PASS = 0.5
ROUGE_L_PASS = 0.20
FACTUAL_F1_PASS = 0.5
NUMERIC_REL_TOL = 1e-3
NUMERIC_ABS_TOL = 1e-6

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+")


# --- text normalisation --------------------------------------------------------

def normalize_text(text):
    """Lowercase, strip surrounding whitespace, collapse internal whitespace,
    and drop punctuation -- the standard normalisation before exact/token
    comparison."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return _WORD_RE.findall((text or "").lower())


# --- numeric match (math) ------------------------------------------------------

def extract_numbers(text):
    """Return every number found in the text as floats, commas and a leading
    currency symbol stripped. Order preserved (left to right)."""
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        cleaned = raw.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _num_close(a, b):
    return abs(a - b) <= max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * max(abs(a), abs(b)))


def numeric_match(pred, ref):
    """True if the reference number appears in the prediction. The reference's
    *last* number is treated as the target (references are written as a bare
    number, optionally with units); the prediction passes if ANY number in it
    matches within tolerance -- the local model often shows brief work before
    the final figure. If the reference has no number, fall back to exact match.
    """
    ref_nums = extract_numbers(ref)
    if not ref_nums:
        return exact_match(pred, ref)
    target = ref_nums[-1]
    return any(_num_close(target, n) for n in extract_numbers(pred))


# --- exact match (sentiment / short factual) ----------------------------------

def exact_match(pred, ref):
    return normalize_text(pred) == normalize_text(ref)


# --- token-level F1 (SQuAD-style) ---------------------------------------------

def token_f1(pred, ref):
    """SQuAD-style token overlap F1 (multiset intersection). Returns 0.0 when
    either side is empty (unless both empty -> 1.0)."""
    pred_toks = tokenize(pred)
    ref_toks = tokenize(ref)
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def containment(pred, ref):
    """Fraction of reference tokens present in the prediction. Good proxy for
    short factual answers where the model wraps the fact in a sentence
    ('The capital is Paris.' contains 'paris')."""
    pred_toks = set(tokenize(pred))
    ref_toks = tokenize(ref)
    if not ref_toks:
        return 1.0 if not pred_toks else 0.0
    hit = sum(1 for t in ref_toks if t in pred_toks)
    return hit / len(ref_toks)


# --- NER entity-set F1 ---------------------------------------------------------

def parse_entity_list(text):
    """Best-effort parse of an entity list from model output. Handles proper
    JSON lists, python-literal single-quoted lists (['a', 'b'] -- what the
    local model actually emits), and prose-wrapped bracketed lists. Returns a
    list of normalized (lowercased, trimmed) entity strings; [] on failure."""
    text = (text or "").strip()
    if not text:
        return []

    def _from_obj(obj):
        if isinstance(obj, list):
            return [str(x).strip().lower() for x in obj if str(x).strip()]
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            got = _from_obj(parser(text))
        except (ValueError, SyntaxError, TypeError):
            got = None
        if got is not None:
            return got

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        inner = text[start + 1 : end]
        for parser in (json.loads, ast.literal_eval):
            try:
                got = _from_obj(parser("[" + inner + "]"))
            except (ValueError, SyntaxError, TypeError):
                got = None
            if got is not None:
                return got
        # last resort: comma split
        parts = [p.strip().strip("'\"").lower() for p in inner.split(",")]
        return [p for p in parts if p]
    return []


def ner_f1(pred, ref):
    """Set-based F1 over normalized entities parsed from pred and ref."""
    pred_set = set(parse_entity_list(pred))
    ref_set = set(parse_entity_list(ref))
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    tp = len(pred_set & ref_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(ref_set)
    return 2 * precision * recall / (precision + recall)


# --- ROUGE-L (summarization) --------------------------------------------------

def _lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(pred, ref):
    """Pure-python ROUGE-L F1 (LCS-based). No external dependency."""
    pred_toks = tokenize(pred)
    ref_toks = tokenize(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, ref_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


# --- unified metric grading ---------------------------------------------------

def grade_metric(category, pred, ref):
    """Grade a single answer with the category's string metric.

    Returns (passed, score, detail):
      - passed: True/False for categories with a usable metric, or None for
        judge_only categories (caller must defer to the judge).
      - score:  the raw metric value (0..1; for numeric/exact it is 1.0/0.0).
      - detail: short human-readable string for the report.
    """
    strategy = grade_strategy(category)
    if strategy == "judge_only":
        return None, 0.0, "judge-only (no string metric)"

    if category == "math":
        ok = numeric_match(pred, ref)
        return ok, 1.0 if ok else 0.0, f"numeric_match={ok}"
    if category == "sentiment":
        ok = exact_match(pred, ref)
        return ok, 1.0 if ok else 0.0, f"exact_match={ok}"
    if category == "ner":
        score = ner_f1(pred, ref)
        return score >= NER_F1_PASS, score, f"ner_f1={score:.2f}"
    if category == "summarization":
        score = rouge_l(pred, ref)
        return score >= ROUGE_L_PASS, score, f"rouge_l={score:.2f}"
    if category == "factual":
        score = containment(pred, ref)
        return score >= FACTUAL_F1_PASS, score, f"containment={score:.2f}"

    # Unknown -> token F1 proxy.
    score = token_f1(pred, ref)
    return score >= FACTUAL_F1_PASS, score, f"token_f1={score:.2f}"
