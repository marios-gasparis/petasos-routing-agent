"""Generates demo/colab/petasos_demo.ipynb from the synced routing modules in
demo/src/ (categories.py, prompts.py, router.py, confidence.py) -- the same
files the HF Space Docker demo uses. Re-run this after re-syncing demo/src/
from the main src/ so the Colab notebook never drifts from the real code:

    python demo/colab/build_notebook.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "src")
OUT_PATH = os.path.join(HERE, "petasos_demo.ipynb")

GITHUB_URL = "https://github.com/marios-gasparis/petasos-routing-agent"
REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


def _read(name):
    with open(os.path.join(SRC_DIR, name), encoding="utf-8") as f:
        return f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def writefile_cell(filename):
    return code(f"%%writefile {filename}\n" + _read(filename))


intro = f"""# Petasos — Live Routing Demo

Run every cell (Runtime -> Run all). This notebook downloads the real baked-in
**Qwen2.5-3B-Instruct** GGUF and runs the **real** `router.py` / `confidence.py`
routing and escalation-gate code from the scored competition container (synced
copies, not a reimplementation) — nothing here is mocked.

For each prompt you'll see: the category it's routed to (Stage-1 heuristic),
the difficulty estimate, the local model's answer, and whether the escalation
gate would send it to Fireworks. **No live Fireworks call is made** — an
"escalate" verdict is reported but not executed, so this public notebook can't
spend the project's remote token budget.

Full project, README, and Dockerfile: {GITHUB_URL}
"""

install_cell = code(
    "!pip -q install llama-cpp-python huggingface_hub ipywidgets\n"
)

download_cell = code(
    "from huggingface_hub import hf_hub_download\n"
    "\n"
    f'MODEL_PATH = hf_hub_download(repo_id="{REPO_ID}", filename="{FILENAME}")\n'
    "print(MODEL_PATH)\n"
)

model_and_routing_cell = code(
    "import os\n"
    "import sys\n"
    "\n"
    'sys.path.insert(0, ".")\n'
    "\n"
    "from llama_cpp import Llama\n"
    "\n"
    "from confidence import _check_math, should_escalate\n"
    "from router import route_task\n"
    "\n"
    "llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)\n"
    "\n"
    "VERIFY_PREFIX = (\n"
    '    "Recompute carefully step by step, then give the final numeric answer "\n'
    '    "on its own line.\\n\\n"\n'
    ")\n"
    "\n"
    "\n"
    "def local_chat(system, user, max_tokens, stop=None, temperature=0.0):\n"
    '    """Same contract as the project\'s src/local_model.py: never raises,\n'
    '    returns (text, usage=None)."""\n'
    "    try:\n"
    "        response = llm.create_chat_completion(\n"
    "            messages=[\n"
    '                {"role": "system", "content": system},\n'
    '                {"role": "user", "content": user},\n'
    "            ],\n"
    "            max_tokens=max_tokens,\n"
    "            temperature=temperature,\n"
    "            stop=stop,\n"
    "        )\n"
    '        text = response["choices"][0]["message"]["content"] or ""\n'
    "    except Exception as e:\n"
    '        print("local_chat failed:", e)\n'
    '        return "N/A", None\n'
    "    return text.strip(), None\n"
    "\n"
    "\n"
    "def route_and_answer(prompt):\n"
    "    category, category_name, difficulty = route_task(prompt)\n"
    "    answer, _ = local_chat(\n"
    "        category.system_prompt, prompt, max_tokens=category.max_tokens, stop=category.stop\n"
    "    )\n"
    "\n"
    "    verify_answer = None\n"
    '    verify_note = ""\n'
    '    if category_name == "math":\n'
    "        verify_answer, _ = local_chat(\n"
    "            category.system_prompt,\n"
    "            VERIFY_PREFIX + prompt,\n"
    "            max_tokens=category.max_tokens,\n"
    "            stop=category.stop,\n"
    "        )\n"
    "        agree = _check_math(answer, verify_answer)\n"
    "        verify_note = (\n"
    '            f"dual-answer verify: {verify_answer!r} -- "\n'
    '            f"{\'agrees\' if agree else \'DISAGREES\'} with the primary answer"\n'
    "        )\n"
    "\n"
    "    escalate = should_escalate(category_name, difficulty, answer, verify_answer)\n"
    "    verdict = (\n"
    '        "ESCALATE to Fireworks (not called in this demo)" if escalate else "STAYS LOCAL"\n'
    "    )\n"
    "\n"
    '    print(f"category:   {category_name}")\n'
    '    print(f"difficulty: {difficulty}")\n'
    '    print(f"answer:     {answer}")\n'
    "    if verify_note:\n"
    '        print(f"            {verify_note}")\n'
    '    print(f"verdict:    {verdict}")\n'
    "    return category_name, difficulty, answer, escalate\n"
)

ui_cell = code(
    "import ipywidgets as widgets\n"
    "from IPython.display import clear_output, display\n"
    "\n"
    "EXAMPLES = [\n"
    '    "What is the capital of Australia?",\n'
    '    "A store had 128 apples. It sold 47 in the morning and 36 in the afternoon. "\n'
    '    "How many apples are left?",\n'
    '    "The review says: \'This laptop is a disaster, it overheats constantly and "\n'
    '    "the battery dies in an hour.\' What is the sentiment?",\n'
    '    "Summarize: The city council voted 5-2 Tuesday to approve a $40 million "\n'
    '    "bond for a new public library after two years of debate over the site "\n'
    '    "and funding sources.",\n'
    '    "Extract all named entities from: \'Marie Curie won the Nobel Prize in "\n'
    '    "Physics in 1903 while working in Paris.\'",\n'
    '    "Here\'s my code, it has a bug:\\n```python\\ndef add(a, b):\\n    return a - b\\n```",\n'
    '    "All cats are mammals. Some mammals can fly. Can we deduce that some cats "\n'
    '    "can fly? Answer yes or no and explain briefly.",\n'
    '    "Write a Python function that returns the nth Fibonacci number.",\n'
    "]\n"
    "\n"
    "prompt_box = widgets.Textarea(\n"
    '    placeholder="Type a prompt...", layout=widgets.Layout(width="100%", height="80px")\n'
    ")\n"
    "example_dropdown = widgets.Dropdown(\n"
    '    options=[("-- pick an example --", "")]\n'
    '    + [(p[:60] + ("..." if len(p) > 60 else ""), p) for p in EXAMPLES],\n'
    '    description="Examples:",\n'
    ")\n"
    'run_button = widgets.Button(description="Route & Answer", button_style="primary")\n'
    "output = widgets.Output()\n"
    "\n"
    "\n"
    "def _on_example_change(change):\n"
    '    if change["new"]:\n'
    '        prompt_box.value = change["new"]\n'
    "\n"
    "\n"
    "def _on_run_click(_):\n"
    "    with output:\n"
    "        clear_output()\n"
    "        if not prompt_box.value.strip():\n"
    '            print("Enter a prompt first.")\n'
    "            return\n"
    "        route_and_answer(prompt_box.value)\n"
    "\n"
    "\n"
    'example_dropdown.observe(_on_example_change, names="value")\n'
    "run_button.on_click(_on_run_click)\n"
    "\n"
    "display(example_dropdown, prompt_box, run_button, output)\n"
)

cells = [
    md(intro),
    install_cell,
    download_cell,
    writefile_cell("prompts.py"),
    writefile_cell("categories.py"),
    writefile_cell("router.py"),
    writefile_cell("confidence.py"),
    model_and_routing_cell,
    ui_cell,
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"name": "petasos_demo.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    f.write("\n")

print(f"Wrote {OUT_PATH}")
