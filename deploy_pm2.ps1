$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pip install -r requirements.txt
python tp14_paper_sim.py seed --config config/paper_config.json
pm2 startOrReload ecosystem.config.js --update-env
pm2 save
