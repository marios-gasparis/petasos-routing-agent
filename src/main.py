import logging
import sys

from confidence import should_escalate
from io_contract import load_tasks, write_results
from local_model import local_chat
from remote_model import get_total_tokens_used, pick_remote_model, remote_chat
from router import route_task

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Math dual-answer verification: re-ask the SAME local model with this prefix
# and compare final numbers in should_escalate. Local tokens are free, so the
# second call costs only time (well within the 30s/request budget); the first
# live sweep measured the format-only math check leaking 2/20 wrong answers.
VERIFY_PREFIX = (
    "Recompute carefully step by step, then give the final numeric answer "
    "on its own line.\n\n"
)


def _math_verify_answer(category, prompt, primary_answer):
    """Second independent local answer for math tasks, or None. Skipped when
    the primary answer is already degenerate (it escalates regardless)."""
    if category.name != "math":
        return None
    primary = (primary_answer or "").strip().lower()
    if not primary or primary in ("n/a", "na"):
        return None
    verify_answer, _usage = local_chat(
        category.system_prompt,
        VERIFY_PREFIX + prompt,
        max_tokens=category.max_tokens,
        stop=category.stop,
    )
    return verify_answer


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
        verify_answer = _math_verify_answer(category, t["prompt"], answer)

        if should_escalate(category_name, difficulty, answer, verify_answer):
            model = pick_remote_model(category_name)
            logger.info(
                "task_id=%s escalating to remote model=%s", t["task_id"], model
            )
            remote_answer, _usage = remote_chat(
                model,
                category.system_prompt,
                t["prompt"],
                max_tokens=category.max_tokens,
                stop=category.stop,
            )
            if remote_answer.strip() and remote_answer.strip().lower() != "n/a":
                answer = remote_answer
            else:
                logger.warning(
                    "task_id=%s remote escalation failed, keeping local answer",
                    t["task_id"],
                )

        results.append({"task_id": t["task_id"], "answer": answer})

    write_results(results)
    logger.info("Wrote %d result(s)", len(results))
    logger.info("Fireworks tokens used: %d", get_total_tokens_used())

    sys.exit(0)


if __name__ == "__main__":
    main()
