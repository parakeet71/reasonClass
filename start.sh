#!/bin/bash
# Start the Reason-Class router pipeline
# Usage: ./start.sh [--qwen]

MODEL="$HOME/models/SmolLM2-360M-Instruct-Q4_K_M.gguf"
CTX=2048

if [ "$1" = "--qwen" ]; then
    MODEL="$HOME/models/Qwen3.5-9B-Q4_K_M.gguf"
    CTX=4096
fi

pkill -f llama-server 2>/dev/null
pkill -f "proxy.py" 2>/dev/null
sleep 1

echo "Starting llama.cpp on :8080..."
nohup "$HOME/build/llama.cpp/build-rocm721/bin/llama-server" \
    -m "$MODEL" --port 8080 --host 127.0.0.1 \
    --n-gpu-layers 99 --ctx-size $CTX --mlock \
    > /tmp/llama-server.log 2>&1 &

sleep 8

echo "Starting proxy on :8081..."
cd "$HOME/Coding/Reason-Class/final"
nohup "$HOME/miniconda3/envs/main/bin/python" -u proxy.py \
    --port 8081 --backend http://127.0.0.1:8080 \
    > /tmp/proxy.log 2>&1 &

sleep 15

if curl -s --max-time 5 http://127.0.0.1:8081/v1/models > /dev/null 2>&1; then
    echo "Ready. Point your web UI at http://127.0.0.1:8081"
else
    echo "ERROR: proxy failed to start. Check /tmp/proxy.log"
fi
