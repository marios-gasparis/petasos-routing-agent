"""Public Streamlit demo: type a prompt, watch the real routing pipeline run.

Runs the actual router.py / confidence.py logic (synced verbatim from src/),
against a smaller local model than the scored container (see local_model.py
for why). Never calls Fireworks -- an "escalate" verdict is reported but not
executed, so a public visitor can't spend the project's remote token budget.
This file is demo-only and is never imported by src/main.py.
"""

import time

import streamlit as st
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

st.set_page_config(page_title="Petasos Routing Agent -- Live Demo", page_icon="\U0001f9ed")

st.title("Petasos -- Hybrid Local/Remote Routing Agent")
st.markdown(
    "Type any prompt below. This demo runs the **real** `router.py` / "
    "`confidence.py` routing and escalation-gate code from the scored "
    "container -- nothing here is mocked. You'll see which of the 8 "
    "categories it's routed to, the local model's answer, and whether the "
    "escalation gate would send it to Fireworks.\n\n"
    "The scored container's local model is Qwen2.5-**3B**-Instruct; this "
    "public demo runs the smaller Qwen2.5-**0.5B**-Instruct (same family) "
    "so it fits a free hosting tier's RAM budget. Only the model size "
    "differs -- the routing/escalation logic is unchanged."
)

with st.sidebar:
    st.subheader("Examples")
    for example in EXAMPLES:
        label = example if len(example) <= 60 else example[:57] + "..."
        if st.button(label, key=example, use_container_width=True):
            st.session_state["prompt"] = example

prompt = st.text_area(
    "Prompt", key="prompt", height=120, placeholder="Ask anything..."
)
run = st.button("Route & Answer", type="primary")

if run:
    if not prompt or not prompt.strip():
        st.warning("Enter a prompt above.")
    else:
        t0 = time.time()
        with st.spinner("Routing and running the local model..."):
            category, category_name, difficulty = route_task(prompt)
            answer, _ = local_chat(
                category.system_prompt,
                prompt,
                max_tokens=category.max_tokens,
                stop=category.stop,
            )

            verify_answer = None
            agree = None
            if category_name == "math":
                verify_answer, _ = local_chat(
                    category.system_prompt,
                    VERIFY_PREFIX + prompt,
                    max_tokens=category.max_tokens,
                    stop=category.stop,
                )
                agree = _check_math(answer, verify_answer)

            escalate = should_escalate(category_name, difficulty, answer, verify_answer)
        elapsed = time.time() - t0

        col1, col2, col3 = st.columns(3)
        col1.metric("Category (Stage-1 heuristic)", category_name)
        col2.metric("Difficulty", difficulty)
        col3.metric("Local inference time", f"{elapsed:.1f}s")

        st.text_area("Local model answer", answer, height=120, disabled=True)

        if verify_answer is not None:
            st.markdown(
                f"**Dual-answer verify (2nd local call, zero cost):** `{verify_answer}` "
                f"-- {'agrees' if agree else 'DISAGREES'} with the primary answer."
            )

        if escalate:
            st.error(
                "**Would escalate to Fireworks** -- the local answer failed the "
                "verifiable check for this category (or the category has no "
                "check and the answer is hard/hedging)."
            )
        else:
            st.success(
                "**Stays local** -- the verifiable check passed (or no check "
                "exists and the answer looks healthy), so this answer ships "
                "as-is for zero remote tokens."
            )

        st.caption(
            "This public demo runs the real local model and the real "
            "routing/escalation code, but never places a live Fireworks call "
            "(so it can't spend anyone's API credits). In the scored "
            "container, an escalated task's local answer above would be "
            "replaced by the remote model's answer."
        )

st.divider()
st.caption(
    "Full project, README, and Dockerfile: "
    "https://github.com/marios-gasparis/petasos-routing-agent"
)
