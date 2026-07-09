"""Stage-1 heuristic router: keyword/regex classification into one of the 8
categories, plus a cheap difficulty estimate. Stage-2 (MiniLM + logreg
classifier) is added in a later phase and will fall back to this heuristic
when its own confidence is low."""

import re

from categories import DEFAULT_CATEGORY, get_category

_CODE_BLOCK_RE = re.compile(r"```")
_BUG_RE = re.compile(r"\b(fix|bug|error|debug|traceback|exception|doesn't work|not working)\b", re.I)
_CODEGEN_RE = re.compile(r"\b(write a function|write a program|implement|write code|create a function)\b", re.I)
_NER_RE = re.compile(r"\b(extract .*(entities|entity)|named entit(y|ies))\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summarize|summari[sz]ation|tl;?dr)\b", re.I)
_SENTIMENT_RE = re.compile(r"\b(sentiment|positive or negative|is this review|classify.*(positive|negative))\b", re.I)
_MATH_KEYWORD_RE = re.compile(r"\b(calculate|how many|sum of|product of|solve for|average of|total of)\b", re.I)
_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")
_OPERATOR_RE = re.compile(r"[+\-*/^%=]|\bplus\b|\bminus\b|\btimes\b|\bdivided by\b")
_LOGIC_RE = re.compile(r"\b(therefore|if .+ then|either .+ or|all \w+ are|puzzle|deduce|syllogism)\b", re.I)
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
    """Convenience wrapper returning the resolved Category object alongside
    the predicted category name and difficulty."""
    category_name, difficulty = heuristic_route(prompt)
    return get_category(category_name), category_name, difficulty
