#!/usr/bin/env sh
set -e

MODEL_PATH="${MODEL_PATH:-/app/models/qwen2.5-3b-instruct-q4_k_m.gguf}"

/app/llama-server -m "$MODEL_PATH" --host 127.0.0.1 --port 8080 -c 4096 -t "$(nproc)" --no-webui &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 40); do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "llama-server failed to become healthy within 40s" >&2
    exit 1
fi

python3 src/main.py
