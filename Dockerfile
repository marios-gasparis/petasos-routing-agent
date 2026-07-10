# Base on the official llama.cpp server image so llama-server and its
# shared libs (built against Ubuntu 24.04's glibc) are guaranteed
# compatible with the runtime — avoids glibc mismatches from copying
# binaries onto a different distro's base image.
# Platform is set via `docker build --platform linux/amd64` at build time.
FROM ghcr.io/ggml-org/llama.cpp:server
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Bake the GGUF in at build time (startup <60s rules out downloading it
# when the container starts). Retry/resume (-C -) since a ~2GB transfer
# can stall partway through on a flaky connection.
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
# Phase 6: bake in the Stage-2 classifier artifact (src/classifier_router.py
# loads ../classifier/router_clf.joblib at runtime). Only the joblib is needed
# in the scored path; train_classifier.py is training-only but harmless (never
# imported at runtime).
COPY classifier/ ./classifier/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
