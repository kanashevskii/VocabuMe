#!/usr/bin/env bash

set -euo pipefail

required_vars=(
  SSH_HOST
  SSH_PORT
  SSH_USER
  SSH_KEY
  REMOTE_PATH
  SYSTEMD_SERVICE
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
done

RUNTIME_USER="${RUNTIME_USER:-eduard}"
if [[ ! "$RUNTIME_USER" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
  echo "RUNTIME_USER must be a valid Unix username" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KEY_FILE="$(mktemp)"
cleanup() {
  rm -f "$KEY_FILE"
}
trap cleanup EXIT

chmod 600 "$KEY_FILE"
printf '%s\n' "$SSH_KEY" > "$KEY_FILE"

SSH_OPTS=(
  -i "$KEY_FILE"
  -p "$SSH_PORT"
  -o StrictHostKeyChecking=accept-new
)

RSYNC_RSH="ssh ${SSH_OPTS[*]}"

cd "$ROOT_DIR/frontend"
npm ci
npm run build

cd "$ROOT_DIR"
python -m py_compile vocab/services.py

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'frontend/node_modules/' \
  --exclude 'media/' \
  --exclude 'logs/' \
  --exclude 'output/' \
  --exclude 'backups/' \
  --exclude '.playwright-cli/' \
  --exclude '__pycache__/' \
  -e "$RSYNC_RSH" \
  --rsync-path="sudo rsync" \
  "$ROOT_DIR/" "${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}/"

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${SSH_HOST}" \
  "sudo chown -R '${RUNTIME_USER}:${RUNTIME_USER}' '${REMOTE_PATH}' && cd '${REMOTE_PATH}' && if [ -x .venv/bin/python ]; then PYTHON_BIN=.venv/bin/python; elif [ -x venv/bin/python ]; then PYTHON_BIN=venv/bin/python; elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; elif command -v python >/dev/null 2>&1; then PYTHON_BIN=python; else echo 'Python interpreter not found on remote host' >&2; exit 127; fi && ( \"\$PYTHON_BIN\" -m pip uninstall --yes gTTS deep-translator >/dev/null 2>&1 || true ) && \"\$PYTHON_BIN\" -m pip install --disable-pip-version-check --no-input -r requirements-prod.lock && \"\$PYTHON_BIN\" -m pip check && \"\$PYTHON_BIN\" manage.py check --deploy && \"\$PYTHON_BIN\" manage.py migrate --noinput && for unit in vocabume-worker-high vocabume-worker-low vocabume-beat; do if [ -f \"deploy/systemd/\$unit.service\" ]; then sudo install -m 0644 \"deploy/systemd/\$unit.service\" \"/etc/systemd/system/\$unit.service\"; fi; done && sudo systemctl daemon-reload && for unit in vocabume-worker-high vocabume-worker-low vocabume-beat; do if sudo systemctl is-enabled --quiet \"\$unit.service\"; then sudo systemctl restart \"\$unit.service\" && sudo systemctl is-active \"\$unit.service\"; fi; done && sudo systemctl restart '${SYSTEMD_SERVICE}' && sudo systemctl is-active '${SYSTEMD_SERVICE}' && for attempt in \$(seq 1 15); do if curl --fail --silent --show-error --max-time 5 http://127.0.0.1:8000/api/app-config >/dev/null; then exit 0; fi; sleep 2; done; echo 'Web health check failed after restart.' >&2; sudo journalctl -u '${SYSTEMD_SERVICE}' -n 100 --no-pager >&2; exit 1"
