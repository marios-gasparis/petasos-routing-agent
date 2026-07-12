"""Gradio front end for the public HF Space demo. Runs the real local Qwen2.5-3B
model (llama.cpp) plus the real router.py / confidence.py modules -- these are
synced copies of src/router.py and src/confidence.py from the main repo, not a
reimplementation. Never calls Fireworks: an "escalate" verdict is reported but
not executed, so a public visitor can't spend the project's remote token
budget. This file is demo-only and is never imported by src/main.py."""

import time

import gradio as gr

from confidence import _check_math, should_escalate
from local_model import local_chat
from router import route_task

VERIFY_PREFIX = (
    "Recompute carefully step by step, then give the final numeric answer "
    "on its own line.\n\n"
)

EXAMPLES = [
    "What is the capital of Australia?",
    "A store had 128 apples. It sold 47 in the morning and 36 in the afternoon. "
    "How many apples are left?",
    "The review says: 'This laptop is a disaster, it overheats constantly and "
    "the battery dies in an hour.' What is the sentiment?",
    "Summarize: The city council voted 5-2 Tuesday to approve a $40 million "
    "bond for a new public library after two years of debate over the site "
    "and funding sources.",
    "Extract all named entities from: 'Marie Curie won the Nobel Prize in "
    "Physics in 1903 while working in Paris.'",
    "Here's my code, it has a bug:\n```python\ndef add(a, b):\n    return a - b\n```",
    "All cats are mammals. Some mammals can fly. Can we deduce that some cats "
    "can fly? Answer yes or no and explain briefly.",
    "Write a Python function that returns the nth Fibonacci number.",
]


def route_and_answer(prompt):
    if not prompt or not prompt.strip():
        return "-", "-", "", "Enter a prompt above.", ""

    t0 = time.time()
    category, category_name, difficulty = route_task(prompt)
    answer, _ = local_chat(
        category.system_prompt, prompt, max_tokens=category.max_tokens, stop=category.stop
    )

    verify_answer = None
    verify_note = ""
    if category_name == "math":
        verify_answer, _ = local_chat(
            category.system_prompt,
            VERIFY_PREFIX + prompt,
            max_tokens=category.max_tokens,
            stop=category.stop,
        )
        agree = _check_math(answer, verify_answer)
        verify_note = (
            f"\n\n**Dual-answer verify (2nd local call, zero cost):** `{verify_answer}` "
            f"— {'agrees' if agree else 'DISAGREES'} with the primary answer."
        )

    escalate = should_escalate(category_name, difficulty, answer, verify_answer)
    elapsed = time.time() - t0

    if escalate:
        decision = (
            "🔺 **Would escalate to Fireworks** — the local answer failed the "
            "verifiable check for this category (or the category has no check "
            "and the answer is hard/hedging)."
        )
    else:
        decision = (
            "✅ **Stays local** — the verifiable check passed (or no check "
            "exists and the answer looks healthy), so this answer ships as-is "
            "for zero remote tokens."
        )
    decision += (
        "\n\n*This public demo runs the real local model and the real "
        "routing/escalation code, but never places a live Fireworks call "
        "(so it can't spend anyone's API credits). In the scored container, "
        "an escalated task's local answer above would be replaced by the "
        "remote model's answer.*"
    )

    return category_name, difficulty, answer, decision + verify_note, f"{elapsed:.1f}s"


with gr.Blocks(title="Petasos Routing Agent — Live Demo") as demo:
    gr.Markdown(
        "# Petasos — Hybrid Local/Remote Routing Agent\n"
        "Type any prompt below. This Space runs the **real** local "
        "Qwen2.5-3B-Instruct model (served by llama.cpp) and the **real** "
        "`router.py` / `confidence.py` routing and escalation-gate code from "
        "the scored container — nothing here is mocked. You'll see which of "
        "the 8 categories it's routed to, the local model's answer, and "
        "whether the escalation gate would send it to Fireworks."
    )
    prompt_box = gr.Textbox(label="Prompt", lines=4, placeholder="Ask anything...")
    gr.Examples(examples=EXAMPLES, inputs=prompt_box)
    run_btn = gr.Button("Route & Answer", variant="primary")

    with gr.Row():
        category_out = gr.Textbox(label="Category (Stage-1 heuristic)")
        difficulty_out = gr.Textbox(label="Difficulty")
        latency_out = gr.Textbox(label="Local inference time")
    answer_out = gr.Textbox(label="Local model answer", lines=4)
    decision_out = gr.Markdown(label="Escalation decision")

    run_btn.click(
        route_and_answer,
        inputs=prompt_box,
        outputs=[category_out, difficulty_out, answer_out, decision_out, latency_out],
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)
