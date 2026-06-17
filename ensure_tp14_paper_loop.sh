#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="${TP14_PM2_APP_NAME:-tp14-paper-sim}"
CONFIG="${TP14_CONFIG:-config/paper_config.json}"
PYTHON="${TP14_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

cd "$ROOT"
mkdir -p runs/tp14
LOG="runs/tp14/ensure_loop.log"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

notify() {
  local message="$1"
  "$PYTHON" tp14_paper_sim.py notify-text --config "$CONFIG" --message "$message" >> "$LOG" 2>&1 || true
}

health_output="$("$PYTHON" tp14_paper_sim.py health --config "$CONFIG" --no-webhook 2>&1)"
health_exit=$?
pm2_pid="$(pm2 pid "$APP_NAME" 2>/dev/null | tr -d '[:space:]')"
if [ -z "$pm2_pid" ] || [ "$pm2_pid" = "0" ]; then
  pm2_ok=0
else
  pm2_ok=1
fi

if [ "$health_exit" -eq 0 ] && [ "$pm2_ok" -eq 1 ]; then
  log "healthy pm2_pid=$pm2_pid"
  exit 0
fi

log "restart_required pm2_pid=${pm2_pid:-none} health_exit=$health_exit"
notify "TP14 server paper health abnormal; restarting pm2. pm2_pid=${pm2_pid:-none} health_exit=$health_exit

$health_output"

pm2 restart "$APP_NAME" --update-env >> "$LOG" 2>&1
restart_exit=$?
if [ "$restart_exit" -ne 0 ]; then
  log "pm2_restart_failed exit=$restart_exit; trying startOrReload"
  pm2 startOrReload ecosystem.config.js --update-env >> "$LOG" 2>&1
  restart_exit=$?
fi

sleep 10
post_health="$("$PYTHON" tp14_paper_sim.py health --config "$CONFIG" --no-fail --no-webhook 2>&1)"
log "post_restart_health_exit=$restart_exit"
notify "TP14 server paper restart attempted. restart_exit=$restart_exit

$post_health"

exit 0
