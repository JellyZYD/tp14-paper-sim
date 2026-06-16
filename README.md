# TP14 Paper Simulation Bot

This is the deploy-only paper trading version of the TP14 squeeze strategy.

It does not run universe selection, optimization, or backtests on the server. The symbol list is fixed in `config/paper_config.json`.

## Server Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tp14_paper_sim.py bootstrap --config config/paper_config.json
python tp14_paper_sim.py loop --config config/paper_config.json
```

For Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tp14_paper_sim.py bootstrap --config config/paper_config.json
python tp14_paper_sim.py loop --config config/paper_config.json
```

## Runtime Model

- Signal data: 15m OHLCV plus funding, OI, global long/short ratio, top-account ratio, top-position ratio.
- Execution data: 1m OHLCV only, used for fixed 14% TP and protective stop checks.
- Entry: `OI_Expansion_48` gate plus `whale_fade` resolver.
- Signal lag: 1 closed 15m bar. No future bar is used.
- Accounts: robust 5% margin at 10x and high-risk 10% margin at 10x.

## Important Files

- `runs/tp14/paper_state.json`: current account state and open positions.
- `runs/tp14/events.jsonl`: entry/exit event log.
- `runs/tp14/last_tick.json`: latest tick diagnostics.
- `runs/tp14/paper_loop.log`: loop log.

## Commands

Run one tick:

```bash
python tp14_paper_sim.py tick --config config/paper_config.json
```

Run forever:

```bash
python tp14_paper_sim.py loop --config config/paper_config.json
```

Bootstrap or refresh 60 days of history:

```bash
python tp14_paper_sim.py bootstrap --config config/paper_config.json --workers 4
```
