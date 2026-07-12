# Same base image and GGUF-baking approach as the scored container's root
# Dockerfile, but serves a Gradio UI (src/app.py) instead of the batch
# entrypoint (src/main.py). This file is the root of the HF Space repo
# (pushed via `git subtree push --prefix demo`), so all COPY paths are
# relative to demo/.
FROM ghcr.io/ggml-org/llama.cpp:server
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models \
    && dest=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    && url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
    && for i in $(seq 1 15); do \
         curl -fL --connect-timeout 30 --continue-at - -o "$dest" "$url" && break; \
         echo "download attempt $i failed, retrying in 5s..." >&2; sleep 5; \
       done \
    && test -s "$dest"

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# HF Spaces' Docker SDK expects the app on this port (see README.md's
# app_port front matter).
EXPOSE 7860

ENTRYPOINT ["./entrypoint.sh"]
