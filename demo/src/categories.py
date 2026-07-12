"""The 8 task categories and their per-category generation settings."""

from dataclasses import dataclass
from typing import List, Optional

from prompts import SYSTEM_PROMPTS

# Starting max_tokens caps (compass_artifact plan, PHASE_3).
MAX_TOKENS = {
    "factual": 40,
    "math": 256,
    "sentiment": 5,
    "summarization": 120,
    "ner": 60,
    "code-debug": 300,
    "logic": 256,
    "code-gen": 400,
}

NEEDS_COT = {"math", "logic"}

DEFAULT_CATEGORY = "factual"


@dataclass(frozen=True)
class Category:
    name: str
    system_prompt: str
    max_tokens: int
    stop: Optional[List[str]] = None
    needs_cot: bool = False


CATEGORIES = {
    name: Category(
        name=name,
        system_prompt=SYSTEM_PROMPTS[name],
        max_tokens=MAX_TOKENS[name],
        needs_cot=name in NEEDS_COT,
    )
    for name in SYSTEM_PROMPTS
}


def get_category(name):
    """Look up a Category, falling back to the default for unknown names."""
    return CATEGORIES.get(name, CATEGORIES[DEFAULT_CATEGORY])
