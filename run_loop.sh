#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python tp14_paper_sim.py loop --config config/paper_config.json
