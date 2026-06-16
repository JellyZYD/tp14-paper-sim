#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 tp14_paper_sim.py seed --config config/paper_config.json
pm2 startOrReload ecosystem.config.js --update-env
pm2 save
