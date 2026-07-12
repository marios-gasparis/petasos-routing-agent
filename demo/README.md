# Petasos -- Live Routing Demo (Streamlit)

Public demo of the routing/escalation pipeline: type a prompt and watch the
real `router.py` (Stage-1 heuristic) and `confidence.py` (escalation gate)
decide the category, run the local model, and report whether the answer
would be escalated to Fireworks. Fireworks is never actually called -- a
public visitor can't spend the project's remote token budget.

`router.py`, `confidence.py`, `categories.py`, and `prompts.py` in this
folder are synced copies of the same-named files under `../src/` -- not a
reimplementation. Only `local_model.py` differs from `src/local_model.py`:
the scored container serves Qwen2.5-**3B**-Instruct via `llama-server`; this
demo loads Qwen2.5-**0.5B**-Instruct in-process via `llama-cpp-python`
because free Streamlit hosting doesn't have the RAM for the 3B model. Same
model family, same routing/escalation code, smaller local model.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. https://share.streamlit.io -> "New app" -> pick the repo/branch.
3. Main file path: `demo/streamlit_app.py`.
4. Deploy. Streamlit Cloud installs from `demo/requirements.txt`
   automatically (it looks in the app file's own directory first).
5. First load downloads the ~500 MB GGUF from Hugging Face and caches it in
   the container for the app's lifetime (`st.cache_resource`); expect a
   slower first request after each redeploy/restart.

## Run locally

```
cd demo
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Known constraints

- Streamlit Community Cloud's free tier is RAM-constrained (~1 GB). The
  0.5B Q4_K_M model (~500 MB) is sized to fit; if you see OOM restarts,
  swap to an even smaller quantization via the `DEMO_MODEL_FILE` env var
  (e.g. `qwen2.5-0.5b-instruct-q3_k_m.gguf`).
- CPU-only, single small model, no batching: expect several seconds per
  answer, more for math (dual-answer verify runs two local calls).
- This demo is not part of the scored container path and is never imported
  by `src/main.py`.
