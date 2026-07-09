"""Generate the offline evaluation testset.

Produces ~20 prompts per category (~160 total) across the 8 categories, each
with a known reference answer, plus lightweight paraphrase variants to mimic
the "unseen prompt variants" the scored harness must be robust to. Output is a
flat JSON array of {task_id, category, prompt, reference} written to
eval/testset/testset.json.

Design notes:
  - References are graded, never shown to the agent (no answer leakage).
  - NER references are JSON-list strings so they pass both the escalation
    check (confidence._check_ner) and the metric (metrics.ner_f1).
  - Math references are bare numbers; sentiment references are single labels.
  - Logic puzzles are written to have a single unambiguous answer (the two
    hand-written "hard" logic tasks in input/tasks.json were under-constrained;
    avoided here -- see docs/next-steps.md).
  - Paraphrase wrappers are chosen per-category so they never change the
    required answer (e.g. sentiment wrappers don't add opinionated words).

Run: python eval/gen_testset.py   [--per-category N] [--out PATH]
"""

import argparse
import json
import os

# Paraphrase wrappers that preserve the answer. {p} is the base prompt.
# Kept deliberately bland so they add no semantic content of their own.
_GENERIC_WRAPPERS = [
    "{p}",
    "Please answer the following. {p}",
    "{p} Answer concisely.",
    "Question: {p}",
    "I need help with this: {p}",
]

# For sentiment we must not inject sentiment-bearing words; keep neutral.
_SENTIMENT_WRAPPERS = [
    "{p}",
    "Classify this. {p}",
    "{p} Give the label only.",
    "Sentiment task: {p}",
]

# For NER, keep the instruction explicit so routing stays correct.
_NER_WRAPPERS = [
    "{p}",
    "{p} Return a JSON list.",
    "Entity extraction task. {p}",
    "{p} List them.",
]


# (prompt, reference) base cases. ~10 per category; expanded via wrappers.
BASE = {
    "factual": [
        ("What is the capital of France?", "Paris"),
        ("Who wrote the novel 'Pride and Prejudice'?", "Jane Austen"),
        ("What year did World War II end?", "1945"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the tallest mountain on Earth?", "Mount Everest"),
        ("What is the smallest prime number?", "2"),
        ("In which country is the Great Pyramid of Giza located?", "Egypt"),
        ("What gas do plants absorb from the atmosphere during photosynthesis?",
         "Carbon dioxide"),
    ],
    "math": [
        ("Calculate the sum of 12, 34 and 56.", "102"),
        ("What is 7 times 8?", "56"),
        ("If a train travels 60 miles per hour for 3 hours, how many miles "
         "does it travel?", "180"),
        ("Solve for x: 2x + 3 = 11.", "4"),
        ("What is 144 divided by 12?", "12"),
        ("A shirt costs $20 and is discounted by 15%. What is the sale price "
         "in dollars?", "17"),
        ("What is 25 percent of 200?", "50"),
        ("What is the square root of 169?", "13"),
        ("If you buy 3 items at $4.50 each, what is the total cost in "
         "dollars?", "13.50"),
        ("What is 15 plus 27 minus 9?", "33"),
    ],
    "sentiment": [
        ("Classify the sentiment: 'I loved this movie, it was fantastic!'",
         "positive"),
        ("What is the sentiment: 'The food was cold and the service was "
         "terrible.'", "negative"),
        ("Sentiment of: 'The package arrived on Tuesday afternoon.'",
         "neutral"),
        ("Classify: 'Best purchase I have ever made, absolutely delighted.'",
         "positive"),
        ("Sentiment: 'It broke after one use, very disappointed.'",
         "negative"),
        ("Classify the sentiment: 'The meeting is scheduled for 3pm.'",
         "neutral"),
        ("Sentiment of this review: 'Wonderful experience, highly "
         "recommend.'", "positive"),
        ("What is the sentiment: 'Worst customer service I have ever "
         "dealt with.'", "negative"),
        ("Classify: 'The report contains four sections and an appendix.'",
         "neutral"),
        ("Sentiment: 'Absolutely thrilled with how well this works!'",
         "positive"),
    ],
    "summarization": [
        ("Summarize: The company reported record quarterly profits driven by "
         "strong demand for its cloud services, and announced plans to hire "
         "an additional two thousand engineers over the next year.",
         "The company had record quarterly profits from strong cloud demand "
         "and plans to hire two thousand more engineers next year."),
        ("Summarize: Scientists have discovered a new species of frog in the "
         "Amazon rainforest that can change color to blend with its "
         "surroundings, a trait they believe helps it evade predators.",
         "Scientists found a new color-changing frog in the Amazon that uses "
         "camouflage to evade predators."),
        ("Summarize: The city council voted to expand the public transit "
         "network, adding three new bus routes and extending service hours "
         "to midnight on weekends to reduce traffic congestion.",
         "The city council expanded public transit with three new bus routes "
         "and later weekend hours to cut congestion."),
        ("Summarize: Researchers found that regular moderate exercise "
         "significantly lowers the risk of heart disease, improves mood, and "
         "can extend life expectancy by several years according to a "
         "long-term study.",
         "A long-term study found regular moderate exercise lowers heart "
         "disease risk, improves mood, and lengthens life expectancy."),
        ("Summarize: The new smartphone features a larger battery, an "
         "improved camera system with three lenses, and a faster processor, "
         "though its price has increased compared to last year's model.",
         "The new smartphone has a bigger battery, a three-lens camera, and a "
         "faster processor, but costs more than last year's model."),
        ("Summarize: Heavy rainfall over the weekend caused flooding in "
         "several low-lying neighborhoods, prompting evacuations and closing "
         "a number of major roads across the region.",
         "Weekend rainfall flooded low-lying neighborhoods, forcing "
         "evacuations and closing major roads."),
        ("Summarize: The museum unveiled a new exhibit showcasing ancient "
         "Egyptian artifacts, including jewelry, pottery, and a "
         "well-preserved sarcophagus on loan from a foreign collection.",
         "The museum opened an exhibit of ancient Egyptian artifacts, "
         "including jewelry, pottery, and a loaned sarcophagus."),
        ("Summarize: A recent survey found that most remote workers report "
         "higher productivity and better work-life balance, but some miss "
         "the social interaction of a traditional office setting.",
         "A survey found remote workers feel more productive with better "
         "work-life balance, though some miss office social interaction."),
        ("Summarize: The government announced a new policy offering tax "
         "incentives to households that install solar panels, aiming to "
         "accelerate the country's transition to renewable energy.",
         "The government introduced tax incentives for household solar panels "
         "to speed the shift to renewable energy."),
        ("Summarize: The soccer team secured the championship after a "
         "dramatic penalty shootout, capping an unbeaten season that manager "
         "and fans alike called historic.",
         "The soccer team won the championship in a penalty shootout, "
         "finishing a historic unbeaten season."),
    ],
    "ner": [
        ("Extract the named entities: Barack Obama was born in Hawaii.",
         '["Barack Obama", "Hawaii"]'),
        ("Identify all named entities: Apple Inc. is headquartered in "
         "Cupertino, California.",
         '["Apple Inc.", "Cupertino", "California"]'),
        ("Extract named entities: The Eiffel Tower is located in Paris, "
         "France.", '["Eiffel Tower", "Paris", "France"]'),
        ("List the named entities: Elon Musk founded SpaceX in 2002.",
         '["Elon Musk", "SpaceX"]'),
        ("Extract entities: Microsoft was founded by Bill Gates and Paul "
         "Allen in Albuquerque.",
         '["Microsoft", "Bill Gates", "Paul Allen", "Albuquerque"]'),
        ("Named entity extraction: Amazon opened a new office in Seattle "
         "led by Andy Jassy.",
         '["Amazon", "Seattle", "Andy Jassy"]'),
        ("Extract the named entities: Serena Williams won the tournament in "
         "London.", '["Serena Williams", "London"]'),
        ("Identify named entities: Toyota unveiled its new model in Tokyo, "
         "Japan.", '["Toyota", "Tokyo", "Japan"]'),
        ("Extract entities: Nelson Mandela became president of South Africa.",
         '["Nelson Mandela", "South Africa"]'),
        ("List named entities: Google acquired YouTube while based in "
         "Mountain View.",
         '["Google", "YouTube", "Mountain View"]'),
    ],
    "code-debug": [
        ("Fix the bug in this code:\n```python\ndef add(a, b):\n    return "
         "a - b\n```",
         "def add(a, b):\n    return a + b"),
        ("There is an error here:\n```python\ndef square(x):\n    return x * "
         "x * x\n```\nIt should return the square.",
         "def square(x):\n    return x * x"),
        ("Fix this function so it does not crash on b=0:\n```python\ndef "
         "div(a, b):\n    return a / b\n```",
         "def div(a, b):\n    if b == 0:\n        return None\n    return "
         "a / b"),
        ("Debug:\n```python\ndef greet(name)\n    return 'Hi ' + name\n```",
         "def greet(name):\n    return 'Hi ' + name"),
        ("What's wrong with this?\n```python\nfor i in range(5)\n    "
         "print(i)\n```",
         "for i in range(5):\n    print(i)"),
        ("Fix the off-by-one bug:\n```python\ndef last(lst):\n    return "
         "lst[len(lst)]\n```",
         "def last(lst):\n    return lst[len(lst) - 1]"),
        ("This raises a NameError:\n```python\ndef total(items):\n    for i "
         "in items:\n        s += i\n    return s\n```",
         "def total(items):\n    s = 0\n    for i in items:\n        s += "
         "i\n    return s"),
        ("Fix the bug:\n```python\ndef is_even(n):\n    return n % 2 == "
         "1\n```\nIt should detect even numbers.",
         "def is_even(n):\n    return n % 2 == 0"),
        ("Correct this code:\n```python\ndef concat(a, b):\n    return a + "
         "int(b)\n```\nBoth arguments are strings.",
         "def concat(a, b):\n    return a + b"),
        ("Fix:\n```python\ndef average(nums):\n    return sum(nums)\n```\nIt "
         "should return the mean.",
         "def average(nums):\n    return sum(nums) / len(nums)"),
    ],
    "logic": [
        ("If all cats are mammals, and all mammals are animals, are all cats "
         "animals? Answer yes or no.", "Yes"),
        ("If it rains, the ground gets wet. It is raining. Is the ground "
         "wet? Answer yes or no.", "Yes"),
        ("Either the butler or the gardener did it. The butler has a "
         "confirmed alibi. Who did it?", "The gardener"),
        ("All dogs are loyal. All loyal animals are trustworthy. Are all "
         "dogs trustworthy? Answer yes or no.", "Yes"),
        ("Tom is older than Sara. Sara is older than Mike. Who is the "
         "oldest?", "Tom"),
        ("A is north of B. B is north of C. Which is furthest south?", "C"),
        ("If no fish can fly, and a salmon is a fish, can a salmon fly? "
         "Answer yes or no.", "No"),
        ("There are 3 red balls and 1 blue ball in a bag. You remove the "
         "blue ball. How many balls remain?", "3"),
        ("Every book on the shelf is blue. The dictionary is on the shelf. "
         "What color is the dictionary?", "Blue"),
        ("If today is Monday, what day will it be in two days?",
         "Wednesday"),
    ],
    "code-gen": [
        ("Write a Python function that reverses a string.",
         "def reverse(s):\n    return s[::-1]"),
        ("Write a Python function that checks if a number is even.",
         "def is_even(n):\n    return n % 2 == 0"),
        ("Write a Python function that returns the maximum of a list.",
         "def maximum(lst):\n    return max(lst)"),
        ("Write a Python function that computes the factorial of n "
         "recursively.",
         "def factorial(n):\n    return 1 if n <= 1 else n * "
         "factorial(n - 1)"),
        ("Write a Python function that checks whether a number is prime.",
         "def is_prime(n):\n    if n < 2:\n        return False\n    for i "
         "in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n           "
         " return False\n    return True"),
        ("Write a Python function that merges two sorted lists into one "
         "sorted list.",
         "def merge(a, b):\n    return sorted(a + b)"),
        ("Write a Python function that counts the vowels in a string.",
         "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "
         "'aeiou')"),
        ("Write a Python function that returns the nth Fibonacci number.",
         "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b "
         "= b, a + b\n    return a"),
        ("Write a Python function that returns True if a string is a "
         "palindrome.",
         "def is_palindrome(s):\n    return s == s[::-1]"),
        ("Write a Python function that sums the integers in a list.",
         "def sum_list(lst):\n    return sum(lst)"),
    ],
}

_WRAPPERS = {
    "sentiment": _SENTIMENT_WRAPPERS,
    "ner": _NER_WRAPPERS,
}


def _wrappers_for(category):
    return _WRAPPERS.get(category, _GENERIC_WRAPPERS)


def generate(per_category=20):
    """Build the testset as a list of {task_id, category, prompt, reference}.
    Deterministic: variants are produced by cycling fixed wrappers over the
    base cases, so re-running yields identical output."""
    items = []
    for category, base in BASE.items():
        wrappers = _wrappers_for(category)
        n = 0
        # Round 0 wrapper is the identity ("{p}"); each later round applies a
        # different wrapper, so the same base case never yields a duplicate
        # prompt within a category.
        round_idx = 0
        while n < per_category:
            wrapper = wrappers[round_idx % len(wrappers)]
            for prompt, reference in base:
                if n >= per_category:
                    break
                text = wrapper.format(p=prompt)
                items.append({
                    "task_id": f"{category}-{n:03d}",
                    "category": category,
                    "prompt": text,
                    "reference": reference,
                })
                n += 1
            round_idx += 1
    return items


def main():
    parser = argparse.ArgumentParser(description="Generate the eval testset.")
    parser.add_argument("--per-category", type=int, default=20,
                        help="prompts per category (default 20 -> ~160 total)")
    default_out = os.path.join(os.path.dirname(__file__), "testset",
                               "testset.json")
    parser.add_argument("--out", default=default_out,
                        help="output JSON path")
    args = parser.parse_args()

    items = generate(args.per_category)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)

    by_cat = {}
    for it in items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
    print(f"Wrote {len(items)} prompts to {args.out}")
    for cat in sorted(by_cat):
        print(f"  {cat:15s} {by_cat[cat]}")


if __name__ == "__main__":
    main()
