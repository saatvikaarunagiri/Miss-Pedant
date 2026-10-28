#!/usr/bin/env bash
# Start Miss Pedant: the local model server (if needed) and the web app.
set -e
cd "$(dirname "$0")"

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }
set -a; source .env; set +a

if [ -d .venv ]; then source .venv/bin/activate; fi

if [ "${LLM_PROVIDER:-mlx}" = "mlx" ]; then
  if nc -z 127.0.0.1 8080 2>/dev/null; then
    echo "mlx_lm.server already running on :8080"
  else
    echo "Starting mlx_lm.server with ${MLX_MODEL} (first run downloads ~4 GB)..."
    mlx_lm.server --model "${MLX_MODEL}" --port 8080 > mlx.log 2>&1 &
    MLX_PID=$!
    trap 'kill $MLX_PID 2>/dev/null' EXIT
    until nc -z 127.0.0.1 8080 2>/dev/null; do
      kill -0 $MLX_PID 2>/dev/null || { echo "mlx_lm.server exited — see mlx.log"; exit 1; }
      sleep 2
    done
    echo "Model server is up."
  fi
fi

cd backend
exec uvicorn app:app --host 127.0.0.1 --port 8000
