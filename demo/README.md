---
title: Petasos Routing Agent
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Petasos — Live Routing Demo

Type any prompt and watch the actual routing pipeline run: the Stage-1
heuristic classifies it into one of 8 categories, the baked-in
**Qwen2.5-3B-Instruct** model (served locally by llama.cpp, zero token cost)
answers it, and the escalation gate decides whether that answer is good
enough to keep or should be sent to Fireworks.

This Space runs the same `router.py` / `confidence.py` / `local_model.py`
code as the scored competition container (synced copies under `src/`, not a
reimplementation). The only thing it skips is actually calling Fireworks on
an "escalate" verdict — that would let any visitor spend the project's
remote token budget, so the demo reports the decision without executing it.

Full project, README, and Dockerfile: https://github.com/marios-gasparis/petasos-routing-agent
