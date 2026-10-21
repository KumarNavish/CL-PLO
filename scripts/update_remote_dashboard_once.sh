#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_BRANCH="${TARGET_BRANCH:-master}"
CONFIG_PATH="${CONFIG_PATH:-configs/remote_status_projects.json}"
PROJECT_IDS="${PROJECT_IDS:-fibonacci-cnl}"
OUTPUT_JSON="${OUTPUT_JSON:-data/run_status.json}"
OUTPUT_HTML="${OUTPUT_HTML:-cnl-dashboard.html}"
HTML_JSON_PATH="${HTML_JSON_PATH:-data/run_status.json}"
COMPAT_CNL_JSON="${COMPAT_CNL_JSON:-data/cnl_status.json}"
MAX_LOG_LINES="${MAX_LOG_LINES:-80}"
LOCK_DIR="${LOCK_DIR:-/tmp/remote_dashboard_sync.lock}"

cd "$ROOT_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "sync already running; skipping"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "[$TS] collecting remote snapshots for projects: ${PROJECT_IDS}"

if python3 scripts/publish_remote_status.py \
  --config "$CONFIG_PATH" \
  --project-ids "$PROJECT_IDS" \
  --output-json "$OUTPUT_JSON" \
  --output-html "$OUTPUT_HTML" \
  --html-json-path "$HTML_JSON_PATH" \
  --compat-cnl-json "$COMPAT_CNL_JSON" \
  --max-log-lines "$MAX_LOG_LINES"; then
  if ! git diff --quiet -- "$OUTPUT_JSON" "$OUTPUT_HTML" "$COMPAT_CNL_JSON"; then
    git add "$OUTPUT_JSON" "$OUTPUT_HTML" "$COMPAT_CNL_JSON"
    git commit -m "chore: update dashboard snapshot ${TS}"
    GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=8' git push origin "$TARGET_BRANCH"
    echo "[$TS] pushed snapshot to ${TARGET_BRANCH}"
  else
    echo "[$TS] no dashboard changes to push"
  fi
else
  echo "[$TS] snapshot collection failed"
  exit 1
fi
