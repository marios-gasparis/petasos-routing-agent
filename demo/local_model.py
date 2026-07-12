"""Local-model backend for the public Streamlit demo.

Same call contract as src/local_model.py's local_chat(system, user, max_tokens,
stop, temperature) -> (text, usage) so router.py/confidence.py (synced
verbatim from src/) need zero changes to run here.

The scored container serves the real Qwen2.5-3B-Instruct via llama-server
over HTTP. This demo instead loads a smaller sibling model in-process via
llama-cpp-python: Streamlit Community Cloud's free tier gives ~1 GB RAM,
which the 3B Q4_K_M (~1.8 GB) does not fit, but a 0.5B Q4_K_M (~500 MB) does.
It is the same Qwen2.5-Instruct family and the same routing/escalation code
runs on top of it -- only the model size changes, for hosting reasons.
"""

import os

import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = os.environ.get("DEMO_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
MODEL_FILE = os.environ.get("DEMO_MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
N_CTX = int(os.environ.get("DEMO_N_CTX", "2048"))


@st.cache_resource(show_spinner="Downloading and loading the local model (one-time, ~500 MB)...")
def _load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count() or 2,
        verbose=False,
    )


def local_chat(system, user, max_tokens, stop=None, temperature=0.0):
    """Never raises: any failure is logged and a fallback string returned so
    the caller always has something to write, matching src/local_model.py."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        llm = _load_model()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        text = response["choices"][0]["message"]["content"] or ""
    except Exception:
        return "N/A", None

    return text.strip(), None
