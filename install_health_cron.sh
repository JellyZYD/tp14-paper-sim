#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p runs/tp14 config

if [ -n "${TP14_WEBHOOK_URL:-}" ]; then
  umask 077
  printf "%s" "$TP14_WEBHOOK_URL" > config/webhook_url.txt
fi

if [ ! -s config/webhook_url.txt ]; then
  echo "TP14_WEBHOOK_URL is not set and config/webhook_url.txt is missing." >&2
  echo "Run: export TP14_WEBHOOK_URL='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=...'" >&2
  exit 1
fi

repo_dir="$(pwd)"
python_bin="$repo_dir/.venv/bin/python"
if [ ! -x "$python_bin" ]; then
  echo "Missing venv python: $python_bin. Run deploy_pm2.sh first." >&2
  exit 1
fi

cron_cmd="cd $repo_dir && $python_bin tp14_paper_sim.py health --config config/paper_config.json --max-heartbeat-age-minutes 10 --max-tick-age-minutes 10 --max-complete-end-lag-minutes 30 --min-free-disk-gb 1 --min-free-disk-pct 5 --no-fail >> runs/tp14/health_cron.log 2>&1"
cron_line="*/5 * * * * TP14_HEALTH_CHECK=1 $cron_cmd # TP14_HEALTH_CHECK"

(
  crontab -l 2>/dev/null | grep -v "TP14_HEALTH_CHECK" || true
  echo "$cron_line"
) | crontab -

echo "Installed TP14 health cron:"
echo "$cron_line"
