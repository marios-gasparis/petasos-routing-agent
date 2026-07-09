import logging
import sys

from io_contract import load_tasks, write_results
from local_model import local_chat
from router import route_task

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    tasks = load_tasks()
    logger.info("Loaded %d task(s)", len(tasks))

    results = []
    for t in tasks:
        category, category_name, difficulty = route_task(t["prompt"])
        logger.info(
            "task_id=%s category=%s difficulty=%s", t["task_id"], category_name, difficulty
        )
        answer, _usage = local_chat(
            category.system_prompt,
            t["prompt"],
            max_tokens=category.max_tokens,
            stop=category.stop,
        )
        results.append({"task_id": t["task_id"], "answer": answer})

    write_results(results)
    logger.info("Wrote %d result(s)", len(results))
    logger.info("Fireworks tokens used: 0")

    sys.exit(0)


if __name__ == "__main__":
    main()
