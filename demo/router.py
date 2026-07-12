"""Two-stage router.

Stage 1 (this module's `heuristic_route`): keyword/regex classification into
one of the 8 categories plus a cheap difficulty estimate. It is the proven
backbone (100% category accuracy on the dev set) and the always-available
fallback.

Stage 2 (Phase 6): a lightweight learned classifier (TF-IDF + logistic
regression, loaded via `classifier_router`). When enabled and confident, it can
override the heuristic's category; otherwise routing defers to Stage 1.

Stage 2 is DISABLED BY DEFAULT (env `ROUTER_USE_CLASSIFIER=1` to enable). On the
offline testset the heuristic already scores 100% on held-out variants and the
category head does not beat it, so -- per CLAUDE.md's "never sacrifice accuracy"
mandate -- the classifier is kept off until a live harness run confirms it
helps (and until the predicted-local-success head, which is the real
token-saving knob, is trained on Phase 5 labels). The wiring below is complete
so enabling it is a one-env-var flip. Difficulty is ALWAYS taken from the
heuristic (the classifier does not predict difficulty)."""

import logging
import os
import re

from categories import DEFAULT_CATEGORY, get_category

logger = logging.getLogger(__name__)


def _classifier_enabled():
    """Read at call time (like config.resolve_models) so it reflects the
    container's live environment. Default off."""
    return os.environ.get("ROUTER_USE_CLASSIFIER", "0").strip().lower() in (
        "1", "true", "yes", "on")

_CODE_BLOCK_RE = re.compile(r"```")
_BUG_RE = re.compile(r"\b(fix|bug|error|debug|traceback|exception|doesn't work|not working)\b", re.I)
# Allow adjectives between the verb and the noun ("write a Python function",
# "create a recursive method") -- the old literal "write a function" missed
# every real-world phrasing that named a language. Bounded gap ([^.?!]{0,40})
# so it stays within a single clause and can't span sentences.
_CODEGEN_RE = re.compile(
    r"\b(write|create|generate|implement)\b[^.?!]{0,40}?"
    r"\b(function|program|script|method|class|algorithm|code|snippet)\b"
    r"|\bimplement\b",
    re.I,
)
_NER_RE = re.compile(r"\b(extract .*(entities|entity)|named entit(y|ies))\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summarize|summari[sz]ation|tl;?dr)\b", re.I)
# Bare "classify" is treated as sentiment: within these 8 categories it is the
# only classification task, and this branch runs after ner/summarization so it
# can't steal those.
_SENTIMENT_RE = re.compile(r"\b(sentiment|positive or negative|is this review|classify)\b", re.I)
_MATH_KEYWORD_RE = re.compile(
    r"\b(calculate|compute|how many|how much|sum of|product of|solve for|"
    r"average of|total of|total cost|percent|percentage|square root|cube root|"
    r"multiplied by|divided by)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")
_OPERATOR_RE = re.compile(r"[+\-*/^%=]|\bplus\b|\bminus\b|\btimes\b|\bdivided by\b")
# Beyond explicit connectives, catch relational/deductive phrasings that read
# like factual questions: comparatives ("older than"), spatial relations
# ("north of"), quantified statements, and yes/no deductions. Deliberately no
# bare superlatives (e.g. "largest") -- those collide with factual questions
# like "the largest planet"; rely on the relational "... than" / "furthest"
# forms instead.
_LOGIC_RE = re.compile(
    r"\b(therefore|deduce|syllogism|puzzle|"
    r"if .+ then|either .+ or|all \w+ are|no \w+s? (can|are|is)|every .+ (is|are)|"
    r"(older|younger|taller|shorter|bigger|smaller|faster|slower|heavier|lighter|"
    r"greater|higher|lower|more|fewer|closer|further|farther) than|"
    r"north of|south of|east of|west of|furthest|farthest|closest|"
    r"answer yes or no|what day)\b",
    re.I,
)
_REASONING_CUES_RE = re.compile(r"\b(why|explain|step by step|first.*then|after that|because)\b", re.I)

_HARD_LENGTH_THRESHOLD = 300
_HARD_NUMBER_COUNT = 3


def heuristic_route(prompt):
    """Classify a prompt into (category, difficulty).

    Signal precedence deliberately puts code detection before math: code
    snippets routinely contain digits and operators that would otherwise be
    misread as a math prompt.
    """
    if not prompt or not prompt.strip():
        return DEFAULT_CATEGORY, "easy"

    text = prompt.strip()
    lower = text.lower()
    has_code_block = bool(_CODE_BLOCK_RE.search(text))

    if has_code_block and _BUG_RE.search(lower):
        category = "code-debug"
    elif _CODEGEN_RE.search(lower):
        category = "code-gen"
    elif has_code_block:
        # A bare code block with no bug/codegen wording is most often a
        # debugging request ("here's my code, what's wrong").
        category = "code-debug"
    elif _NER_RE.search(lower):
        category = "ner"
    elif _SUMMARY_RE.search(lower):
        category = "summarization"
    elif _SENTIMENT_RE.search(lower):
        category = "sentiment"
    elif _MATH_KEYWORD_RE.search(lower) or (_NUMBER_RE.search(text) and _OPERATOR_RE.search(lower)):
        category = "math"
    elif _LOGIC_RE.search(lower):
        category = "logic"
    else:
        category = DEFAULT_CATEGORY

    difficulty = _estimate_difficulty(text, lower, category)
    return category, difficulty


def _estimate_difficulty(text, lower, category):
    if len(text) > _HARD_LENGTH_THRESHOLD:
        return "hard"

    if category in ("math", "logic"):
        number_count = len(_NUMBER_RE.findall(text))
        if number_count >= _HARD_NUMBER_COUNT or _REASONING_CUES_RE.search(lower):
            return "hard"

    if category in ("code-debug", "code-gen"):
        if text.count("```") > 2 or len(text) > 150:
            return "hard"

    if _REASONING_CUES_RE.search(lower):
        return "hard"

    return "easy"


def route_task(prompt):
    """Resolve (Category object, category_name, difficulty) for a prompt.

    Category comes from Stage 2 (the learned classifier) when it is enabled AND
    confident above its tuned gate; otherwise from the Stage-1 heuristic.
    Difficulty is always the heuristic's estimate. Public shape is unchanged so
    main.py and the harness need no rewrite."""
    heuristic_category, difficulty = heuristic_route(prompt)
    category_name = heuristic_category

    if _classifier_enabled():
        category_name = _stage2_category(prompt, heuristic_category)

    return get_category(category_name), category_name, difficulty


def _stage2_category(prompt, heuristic_category):
    """Return the classifier's category if it is available and its confidence
    clears the tuned gate; otherwise the heuristic's. Any failure inside the
    classifier path degrades silently to the heuristic (never raises)."""
    try:
        import classifier_router

        result = classifier_router.classify_category(prompt)
        if result is None:
            return heuristic_category
        clf_category, confidence = result
        if confidence >= classifier_router.min_confidence():
            if clf_category != heuristic_category:
                logger.info(
                    "Stage-2 override: heuristic=%s -> classifier=%s (conf=%.3f)",
                    heuristic_category, clf_category, confidence,
                )
            return clf_category
    except Exception as exc:  # defensive: classifier must never break routing
        logger.warning("Stage-2 classifier path failed, using heuristic: %s", exc)
    return heuristic_category


def predicted_local_success(prompt):
    """Passthrough to the classifier's predicted-local-success probability, or
    None when no success head is trained yet (blocked on Phase 5 harness
    labels). Provided so the escalation gate can consult it once the head
    exists; currently INERT (returns None) and does not alter escalation. See
    docs/phase6-classifier.md."""
    try:
        import classifier_router

        return classifier_router.predict_success(prompt)
    except Exception:
        return None
