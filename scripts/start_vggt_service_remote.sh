#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="${1:-$HOME/model-lab/VAT-13/vggt-service}"
PORT="${VGGT_SERVICE_PORT:-18080}"
HOST="${VGGT_SERVICE_HOST:-0.0.0.0}"

cd "$SERVICE_ROOT"
mkdir -p logs

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt fastapi 'uvicorn[standard]' python-multipart

if pgrep -f "uvicorn scripts.vggt_inference_service:app --host ${HOST} --port ${PORT}" >/dev/null 2>&1; then
  echo "Service already running on ${HOST}:${PORT}"
  exit 0
fi

nohup python -m uvicorn scripts.vggt_inference_service:app \
  --host "$HOST" \
  --port "$PORT" \
  > "logs/vggt_service.log" 2>&1 &

sleep 2
if pgrep -f "uvicorn scripts.vggt_inference_service:app --host ${HOST} --port ${PORT}" >/dev/null 2>&1; then
  echo "VGGT service started on ${HOST}:${PORT}"
  echo "Log: ${SERVICE_ROOT}/logs/vggt_service.log"
else
  echo "Failed to start service. Check log: ${SERVICE_ROOT}/logs/vggt_service.log" >&2
  exit 1
fi
