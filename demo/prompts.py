"""Terse per-category system prompts. Kept separate from categories.py so
prompt wording can be tuned without touching max_tokens/stop/needs_cot."""

SYSTEM_PROMPTS = {
    "factual": (
        "Answer the question factually and concisely. Answer only, no preamble."
    ),
    "math": (
        "Solve the problem. Show brief work only if needed, then end with the "
        "final numeric answer on its own line."
    ),
    "sentiment": (
        "Classify the sentiment as exactly one word: positive, negative, or "
        "neutral. Answer only, no explanation."
    ),
    "summarization": (
        "Summarize the text in 1-3 concise sentences. Answer only, no preamble."
    ),
    "ner": (
        "Extract all named entities from the text. Return them as a JSON list "
        "of strings, nothing else."
    ),
    "code-debug": (
        "Find and fix the bug in the code. Return only the corrected code."
    ),
    "logic": (
        "Solve the logic problem. Reason briefly step by step, then end with "
        "the final answer on its own line."
    ),
    "code-gen": (
        "Write the requested code. Return only the code, no explanation."
    ),
}
