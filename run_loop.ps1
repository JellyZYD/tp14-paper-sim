$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python tp14_paper_sim.py loop --config config/paper_config.json
