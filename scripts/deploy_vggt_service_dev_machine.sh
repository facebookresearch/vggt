#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="${VGGT_SSH_KEY:-$HOME/.ssh/id_ed25519_linux_server}"
REMOTE_USER="${VGGT_REMOTE_USER:-vince}"
REMOTE_HOST="${VGGT_REMOTE_HOST:-192.168.1.10}"
REMOTE_ROOT="${VGGT_REMOTE_ROOT:-~/model-lab/VAT-13/vggt-service}"
REMOTE_PORT="${VGGT_SERVICE_PORT:-18080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SSH_BASE=(ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_HOST}")
RSYNC_SSH="ssh -i ${SSH_KEY}"

"${SSH_BASE[@]}" "mkdir -p ${REMOTE_ROOT}/scripts ${REMOTE_ROOT}/vggt ${REMOTE_ROOT}/logs"

rsync -av \
  -e "$RSYNC_SSH" \
  "$REPO_ROOT/requirements.txt" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/"

rsync -av \
  -e "$RSYNC_SSH" \
  "$REPO_ROOT/scripts/vggt_inference_service.py" \
  "$REPO_ROOT/scripts/start_vggt_service_remote.sh" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"

rsync -av \
  -e "$RSYNC_SSH" \
  "$REPO_ROOT/vggt/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/vggt/"

"${SSH_BASE[@]}" "chmod +x ${REMOTE_ROOT}/scripts/start_vggt_service_remote.sh && VGGT_SERVICE_PORT=${REMOTE_PORT} bash ${REMOTE_ROOT}/scripts/start_vggt_service_remote.sh ${REMOTE_ROOT}"

"${SSH_BASE[@]}" "curl -fsS http://127.0.0.1:${REMOTE_PORT}/healthz"

cat <<EOF
Deployment complete.
Remote service: http://${REMOTE_HOST}:${REMOTE_PORT}
EOF
