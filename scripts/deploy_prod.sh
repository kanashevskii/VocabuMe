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
  "sudo chown -R eduard:eduard '${REMOTE_PATH}' && cd '${REMOTE_PATH}' && python manage.py migrate --noinput && sudo systemctl restart '${SYSTEMD_SERVICE}' && sudo systemctl is-active '${SYSTEMD_SERVICE}'"
