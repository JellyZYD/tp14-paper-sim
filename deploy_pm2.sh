#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tp14_paper_sim.py seed --config config/paper_config.json
pm2 startOrReload ecosystem.config.js --update-env
pm2 save
