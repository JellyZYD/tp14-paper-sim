# TP14 Paper Simulation Bot

This is the deploy-only paper trading version of the TP14 squeeze strategy.

It does not run universe selection, optimization, or backtests on the server. The symbol list is fixed in `config/paper_config.json`.

## Server Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tp14_paper_sim.py start --config config/paper_config.json --workers 4
```

With pm2 on Linux:

```bash
# Optional: enable WeCom/DingTalk-compatible text webhook notifications.
export TP14_WEBHOOK_URL='https://...'
bash deploy_pm2.sh
```

For Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tp14_paper_sim.py start --config config/paper_config.json
```

## Runtime Model

- Signal data: 15m OHLCV plus funding, OI, global long/short ratio, top-account ratio, top-position ratio.
- Execution data: 1m OHLCV only, used for fixed 14% TP and protective stop checks.
- Entry: `OI_Expansion_48` gate plus `whale_fade` resolver.
- Signal lag: 1 closed 15m bar. No future bar is used.
- Thresholds: fixed from `runs/tp14/paper_state.json` `preflight.training_start` / `training_end` for the current paper cycle. They are not recomputed from a rolling window on every tick.
- Accounts: robust 5% margin at 10x and high-risk 10% margin at 10x.

## Binance Data Limit

Binance public `/futures/data/*` endpoints for OI and long/short ratios only allow recent history. The bootstrap command caps those datasets at 30 days while downloading 60 days of 15m klines and funding. If the bot keeps running, local OI/ratio history will accumulate beyond the initial public-API window.

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

Bootstrap from packaged seed data. If the seed archive is missing, this falls back to downloading history:

```bash
python tp14_paper_sim.py bootstrap --config config/paper_config.json --workers 4
```

Use the packaged seed data instead of a long bootstrap:

```bash
python tp14_paper_sim.py seed --config config/paper_config.json
```

One command that seeds if needed, performs a short incremental refresh, then runs forever:

```bash
python tp14_paper_sim.py start --config config/paper_config.json --workers 4
```
