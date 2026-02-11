#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
REMOTE_HOST="${REMOTE_HOST:-fibonacci}"
REMOTE_DIR="${REMOTE_DIR:-/users/staff/dmi-dmi/kumar0002/Collaborative-Neuron-Learning}"
TARGET_BRANCH="${TARGET_BRANCH:-master}"

cd "$ROOT_DIR"

while true; do
  TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "[$TS] collecting remote snapshot from ${REMOTE_HOST}"

  if python3 scripts/publish_cnl_status.py \
    --remote-host "$REMOTE_HOST" \
    --remote-dir "$REMOTE_DIR" \
    --output-json data/cnl_status.json \
    --output-html cnl-dashboard.html \
    --html-json-path data/cnl_status.json; then
    if ! git diff --quiet -- data/cnl_status.json cnl-dashboard.html; then
      git add data/cnl_status.json cnl-dashboard.html
      git commit -m "chore: update CNL dashboard snapshot ${TS}"
      GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=8' git push origin "$TARGET_BRANCH"
      echo "[$TS] pushed snapshot to ${TARGET_BRANCH}"
    else
      echo "[$TS] no dashboard changes to push"
    fi
  else
    echo "[$TS] snapshot collection failed; will retry"
  fi

  sleep "$INTERVAL_SECONDS"
done
