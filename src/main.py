import logging
import sys

from io_contract import load_tasks, write_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    tasks = load_tasks()
    logger.info("Loaded %d task(s)", len(tasks))

    results = [{"task_id": t["task_id"], "answer": "TODO"} for t in tasks]

    write_results(results)
    logger.info("Wrote %d result(s)", len(results))
    logger.info("Fireworks tokens used: 0")

    sys.exit(0)


if __name__ == "__main__":
    main()
