#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CRON_EXPR="${CRON_EXPR:-* * * * *}"
CRON_LOG="${CRON_LOG:-$ROOT_DIR/logs/remote_dashboard_cron.out}"
CRON_TAG="# remote_dashboard_sync_job"

mkdir -p "$ROOT_DIR/logs"

CMD="cd '$ROOT_DIR' && /bin/bash '$ROOT_DIR/scripts/update_remote_dashboard_once.sh' >> '$CRON_LOG' 2>&1"
LINE="$CRON_EXPR $CMD $CRON_TAG"

EXISTING="$(crontab -l 2>/dev/null || true)"
{
  printf "%s\n" "$EXISTING" | awk '!/remote_dashboard_sync_job/'
  printf "%s\n" "$LINE"
} | crontab -

echo "Installed cron entry:"
echo "$LINE"
