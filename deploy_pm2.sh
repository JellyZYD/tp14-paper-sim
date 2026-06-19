#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
seed_args=(seed --config config/paper_config.json)
if [ "${TP14_FORCE_STATE:-0}" = "1" ]; then
  seed_args+=(--force-state)
fi
.venv/bin/python tp14_paper_sim.py "${seed_args[@]}"
pm2 delete tp14-paper-sim >/dev/null 2>&1 || true
pm2 start ecosystem.config.js --update-env
pm2 save
