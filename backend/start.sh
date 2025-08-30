#!/usr/bin/env bash
set -euo pipefail

# Start Ollama server in background
ollama serve >/tmp/ollama.log 2>&1 &

# Wait for Ollama to be ready
for i in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [ $i -eq 60 ]; then
    echo "Ollama failed to start" >&2
    exit 1
  fi
done

# Pull lightweight model (best-effort)
ollama pull qwen2.5:0.5b || true

# Run API on provided PORT (Render sets $PORT)
PORT_TO_USE=${PORT:-8000}
exec uvicorn main_simple:app --host 0.0.0.0 --port ${PORT_TO_USE}

