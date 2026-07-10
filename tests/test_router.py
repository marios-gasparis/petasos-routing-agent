"""Hand-labeled dev set for the Stage-1 heuristic router (src/router.py).

Mirrors the PHASE_3 acceptance criterion: category accuracy >= 80% on a
hand-labeled 40-prompt dev set (5 per category), no crashes on empty/odd
prompts. Run with: python -m unittest tests.test_router -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router import heuristic_route  # noqa: E402

CATEGORY_ACCURACY_THRESHOLD = 0.8
DIFFICULTY_ACCURACY_THRESHOLD = 0.8

# (prompt, expected_category, expected_difficulty)
DATASET = [
    # -- factual (all short, no numeric/reasoning complexity -> easy) --
    ("What is the capital of France?", "factual", "easy"),
    ("Who wrote the novel 'Pride and Prejudice'?", "factual", "easy"),
    ("What year did World War II end?", "factual", "easy"),
    ("What is the boiling point of water at sea level in Celsius?", "factual", "easy"),
    ("Name the largest planet in our solar system.", "factual", "easy"),

    # -- math --
    ("Calculate the sum of 12, 34 and 56.", "math", "hard"),  # 3 numbers
    ("What is 7 times 8?", "math", "easy"),  # 2 numbers, no reasoning cue
    (
        "If a train travels 60 miles per hour for 3 hours, then continues at "
        "40 miles per hour for 2 more hours, how many total miles does it travel?",
        "math",
        "hard",
    ),  # 4 numbers
    ("How many days are in a leap year?", "math", "easy"),  # 0 numbers
    ("Solve for x: 2x + 3 = 11", "math", "hard"),  # 3 numbers

    # -- sentiment (all short -> easy) --
    ("Classify the sentiment of this review: I loved this movie!", "sentiment", "easy"),
    (
        "What is the sentiment of this sentence: 'The food was cold and the "
        "service was terrible.'",
        "sentiment",
        "easy",
    ),
    (
        "Determine if this tweet expresses positive or negative sentiment: "
        "'Best day ever!'",
        "sentiment",
        "easy",
    ),
    (
        "Is this product review positive or negative: 'It broke after one "
        "use, very disappointed.'",
        "sentiment",
        "easy",
    ),
    (
        "Sentiment analysis: 'The weather today is absolutely wonderful and "
        "I feel great.'",
        "sentiment",
        "easy",
    ),

    # -- summarization --
    (
        "Summarize the following article: The economy grew steadily over the "
        "past year, driven by strong consumer spending and low unemployment.",
        "summarization",
        "easy",
    ),
    (
        "TL;DR this paragraph: Renewable energy sources such as solar and "
        "wind have seen rapid cost declines over the last decade, making "
        "them competitive with fossil fuels.",
        "summarization",
        "easy",
    ),
    (
        "Please summarize the following: The committee met on Tuesday to "
        "discuss the annual budget, focusing on funding allocations for the "
        "upcoming fiscal year.",
        "summarization",
        "easy",
    ),
    (
        "Summarize the following passage in a few sentences: Machine "
        "learning algorithms require large amounts of data to train "
        "effectively, and the quality of that data significantly impacts "
        "model performance and generalization to new examples encountered "
        "in production environments across many different industries and "
        "use cases worldwide, including healthcare, finance and manufacturing.",
        "summarization",
        "hard",
    ),  # > 300 chars
    (
        "Summarize: Q3 revenue increased 12% year-over-year while operating "
        "costs rose only 4%, resulting in improved margins across all "
        "business units, though management cautioned that supply chain "
        "risks remain elevated heading into Q4 and could affect future "
        "profitability projections if raw material costs continue rising "
        "unexpectedly.",
        "summarization",
        "hard",
    ),  # > 300 chars

    # -- ner (all short -> easy) --
    (
        "Extract the named entities from this sentence: Barack Obama was "
        "born in Hawaii.",
        "ner",
        "easy",
    ),
    (
        "Identify all named entities in the following text: Apple Inc. is "
        "headquartered in Cupertino, California.",
        "ner",
        "easy",
    ),
    (
        "Perform named entity extraction on: The Eiffel Tower is located in "
        "Paris, France.",
        "ner",
        "easy",
    ),
    (
        "List the named entities found in this passage: Elon Musk founded "
        "SpaceX in 2002.",
        "ner",
        "easy",
    ),
    (
        "Extract entities such as people, organizations and locations from: "
        "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque.",
        "ner",
        "easy",
    ),

    # -- code-debug --
    (
        "Here is my code:\n```python\ndef f(x):\n    return x +\n```\nfix the bug",
        "code-debug",
        "easy",
    ),
    (
        "There's an error in this code:\n```python\ndef add(a, b):\n    "
        "return a - b\n```\nWhy does it give the wrong result?",
        "code-debug",
        "hard",
    ),  # "why" reasoning cue
    (
        "```python\ndef add(a, b):\n    return a - b\n```\nWhat's wrong with this?",
        "code-debug",
        "easy",
    ),  # bare code block, no bug/codegen keyword -> falls back to code-debug
    (
        "This function throws an exception:\n```python\ndef div(a, b):\n    "
        "return a / b\n```\nFix it so it handles b=0 without crashing.",
        "code-debug",
        "easy",
    ),
    (
        "I'm getting a traceback from this code:\n```python\ndef parse(s):\n"
        "    return int(s)\n```\nExplain step by step why parse('abc') fails "
        "and fix the bug so it returns None instead of raising.",
        "code-debug",
        "hard",
    ),  # "step by step" reasoning cue

    # -- code-gen --
    ("Write a function that reverses a string.", "code-gen", "easy"),
    ("Implement a binary search algorithm in Python.", "code-gen", "easy"),
    ("Write a program that checks if a number is prime.", "code-gen", "easy"),
    (
        "Create a function to compute the factorial of a number recursively, "
        "then explain how the recursion terminates, and write a program that "
        "also handles negative inputs by raising a ValueError, ensuring the "
        "function is well documented and includes type hints for clarity in "
        "a production codebase.",
        "code-gen",
        "hard",
    ),  # > 300 chars
    ("Write a function to merge two sorted lists.", "code-gen", "easy"),

    # -- logic --
    (
        "If all cats are mammals, and all mammals are animals, therefore "
        "all cats are animals. True or false?",
        "logic",
        "easy",
    ),
    (
        "Puzzle: Three friends have hats of different colors. If the first "
        "friend can see the other two hats but not his own, and reasoning "
        "step by step, who is wearing red?",
        "logic",
        "hard",
    ),  # "step by step" reasoning cue
    (
        "If it rains, then the ground gets wet. It is raining. Therefore, "
        "the ground is wet. Is this argument valid?",
        "logic",
        "easy",
    ),
    (
        "All dogs are loyal. All loyal animals are trustworthy. Therefore, "
        "all dogs are trustworthy. Deduce whether this syllogism is valid "
        "and explain your reasoning in detail with multiple supporting "
        "examples along the way to illustrate each logical step of the "
        "deduction process clearly for the reader.",
        "logic",
        "hard",
    ),  # > 300 chars
    (
        "Either the butler did it or the gardener did it. The butler has an "
        "alibi. Therefore, who did it?",
        "logic",
        "easy",
    ),

    # -- natural-phrasing regression set (Phase 5 harness found the Stage-1
    #    router at 72.5% on these before the 2026-07-10 regex broadening;
    #    lock the fixed phrasings in so a future edit can't silently regress) --
    ("Write a Python function that reverses a string.", "code-gen", "easy"),
    ("Write a Python function that checks whether a number is prime.", "code-gen", "easy"),
    ("Generate a Python script that prints the numbers 1 to 10.", "code-gen", "easy"),
    ("What is 25 percent of 200?", "math", "easy"),
    ("What is the square root of 169?", "math", "easy"),
    ("If you buy 3 items at $4.50 each, what is the total cost in dollars?", "math", "easy"),
    ("Classify: 'Best purchase I have ever made, absolutely delighted.'", "sentiment", "easy"),
    ("Classify: 'The report contains four sections and an appendix.'", "sentiment", "easy"),
    ("Tom is older than Sara. Sara is older than Mike. Who is the oldest?", "logic", "easy"),
    (
        "If it rains, the ground gets wet. It is raining. Is the ground wet? "
        "Answer yes or no.",
        "logic",
        "easy",
    ),
    ("A is north of B. B is north of C. Which is furthest south?", "logic", "easy"),
    ("If today is Monday, what day will it be in two days?", "logic", "easy"),
]

EDGE_CASES = [
    "",
    "   ",
    "\n\t\n",
    "asdf ;;;; ??? !!! ---- ===",
    "a" * 5000,
    "こんにちは、これはテストです。今日の天気について教えてください。",
    "12345 67890 !@#$% ^&*()",
]


class TestHeuristicRouteAccuracy(unittest.TestCase):
    def test_category_and_difficulty_accuracy(self):
        assert len(DATASET) == 52, f"expected 52 hand-labeled prompts, got {len(DATASET)}"

        category_correct = 0
        difficulty_correct = 0
        per_category = {}
        mismatches = []

        for prompt, expected_category, expected_difficulty in DATASET:
            category, difficulty = heuristic_route(prompt)

            stats = per_category.setdefault(expected_category, {"correct": 0, "total": 0})
            stats["total"] += 1

            cat_ok = category == expected_category
            diff_ok = difficulty == expected_difficulty
            if cat_ok:
                category_correct += 1
                stats["correct"] += 1
            if diff_ok:
                difficulty_correct += 1
            if not (cat_ok and diff_ok):
                mismatches.append(
                    (prompt[:60], expected_category, category, expected_difficulty, difficulty)
                )

        total = len(DATASET)
        category_accuracy = category_correct / total
        difficulty_accuracy = difficulty_correct / total

        print(f"\nCategory accuracy:   {category_correct}/{total} = {category_accuracy:.0%}")
        print(f"Difficulty accuracy: {difficulty_correct}/{total} = {difficulty_accuracy:.0%}")
        print("Per-category breakdown:")
        for name, stats in sorted(per_category.items()):
            print(f"  {name:15s} {stats['correct']}/{stats['total']}")

        if mismatches:
            print("Mismatches (prompt, expected_cat, got_cat, expected_diff, got_diff):")
            for m in mismatches:
                print(f"  {m}")

        self.assertGreaterEqual(
            category_accuracy,
            CATEGORY_ACCURACY_THRESHOLD,
            f"category accuracy {category_accuracy:.0%} below {CATEGORY_ACCURACY_THRESHOLD:.0%} gate",
        )
        self.assertGreaterEqual(
            difficulty_accuracy,
            DIFFICULTY_ACCURACY_THRESHOLD,
            f"difficulty accuracy {difficulty_accuracy:.0%} below {DIFFICULTY_ACCURACY_THRESHOLD:.0%} target",
        )

    def test_edge_cases_do_not_crash(self):
        valid_categories = {
            "factual", "math", "sentiment", "summarization",
            "ner", "code-debug", "logic", "code-gen",
        }
        for prompt in EDGE_CASES:
            with self.subTest(prompt=prompt[:30]):
                category, difficulty = heuristic_route(prompt)
                self.assertIn(category, valid_categories)
                self.assertIn(difficulty, {"easy", "hard"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
