import json
import logging
import os

logger = logging.getLogger(__name__)


def load_tasks(path="/input/tasks.json"):
    """Parse the task list. Never raises: any structural problem is logged
    and skipped so the run can proceed with whatever tasks are valid."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to read/parse %s: %s", path, e)
        return []

    if not isinstance(raw, list):
        logger.error("%s must contain a JSON array, got %s", path, type(raw).__name__)
        return []

    tasks = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning("Skipping non-object task entry: %r", item)
            continue
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if not task_id or not prompt:
            logger.warning("Skipping task missing task_id/prompt: %r", item)
            continue
        tasks.append({"task_id": task_id, "prompt": prompt})
    return tasks


def write_results(results, path="/output/results.json"):
    """Write {"task_id", "answer"} pairs atomically, guaranteeing every
    answer is a non-empty string, then re-load to validate before returning."""
    cleaned = []
    for r in results:
        answer = r.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = "N/A"
        cleaned.append({"task_id": r["task_id"], "answer": answer})

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

    with open(path, "r", encoding="utf-8") as f:
        json.load(f)

    return cleaned
