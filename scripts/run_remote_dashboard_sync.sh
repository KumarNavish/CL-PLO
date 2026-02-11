#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"

while true; do
  if ! "$(cd "$(dirname "$0")" && pwd)/update_remote_dashboard_once.sh"; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] snapshot collection failed; will retry"
  fi

  sleep "$INTERVAL_SECONDS"
done
