import logging
import sys

from io_contract import load_tasks, write_results
from local_model import local_chat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer the user's request directly and concisely. No preamble."
MAX_TOKENS = 300


def main():
    tasks = load_tasks()
    logger.info("Loaded %d task(s)", len(tasks))

    results = []
    for t in tasks:
        answer, _usage = local_chat(SYSTEM_PROMPT, t["prompt"], max_tokens=MAX_TOKENS)
        results.append({"task_id": t["task_id"], "answer": answer})

    write_results(results)
    logger.info("Wrote %d result(s)", len(results))
    logger.info("Fireworks tokens used: 0")

    sys.exit(0)


if __name__ == "__main__":
    main()
