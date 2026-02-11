#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${LABEL:-com.kumarnavish.remote_dashboard_sync}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_PATH="${ROOT_DIR}/logs/remote_dashboard_launchd.out"

mkdir -p "${HOME}/Library/LaunchAgents"
mkdir -p "${ROOT_DIR}/logs"

cat > "${PLIST_PATH}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd '${ROOT_DIR}' &amp;&amp; /bin/bash '${ROOT_DIR}/scripts/update_remote_dashboard_once.sh' &gt;&gt; '${LOG_PATH}' 2&gt;&amp;1</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>${INTERVAL_SECONDS}</integer>
  <key>StandardOutPath</key>
  <string>${LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_PATH}</string>
  <key>WorkingDirectory</key>
  <string>${ROOT_DIR}</string>
</dict>
</plist>
EOF

launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl load "${PLIST_PATH}"

echo "Installed launchd agent: ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Log: ${LOG_PATH}"
