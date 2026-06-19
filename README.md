# TP14 V2 Paper Simulation Bot

Deploy-only paper trading runner for the TP14 V2 deployable rank-fixed strategy.

This repository does not run universe discovery, parameter search, or backtests on the server. Those steps are done locally, then the fixed artifacts are committed here.

## Current Package

- Candidate source: Binance USD-M current top contracts snapshot, 497 eligible contracts after BTC/ETH/large-cap exclusions.
- Signal data update: 482/497 contracts downloaded successfully for 15m OHLCV, funding, OI, global account ratio, top account ratio, and top position ratio.
- Execution data update: 494/497 contracts downloaded successfully for 1m OHLCV.
- Final quality pool: 130 contracts passed both 15m/state and 1m coverage filters.
- Live universe: 120 fixed symbols in `config/paper_config.json`.
- Training window: `2026-04-20T16:15:00Z` to `2026-06-19T16:15:00Z`.
- Model artifact: `artifacts/tp14_v2_artifacts.pkl`.
- Seed data: split archive `bootstrap_seed/tp14_seed.zip.part001` and `bootstrap_seed/tp14_seed.zip.part002`.

## Strategy Accounts

The runner has four paper accounts, all starting at 100 USDT:

- `tp14_v2_stable_tp10_sl08_size05`: stable filter, 10x, 5% margin, 10% TP, 8% stop.
- `tp14_v2_stable_tp10_sl08_size10`: stable filter, 10x, 10% margin, 10% TP, 8% stop.
- `tp14_v2_highret_tp18_sl08_size05`: high-return filter, 10x, 5% margin, 18% TP, 8% stop.
- `tp14_v2_highret_tp18_sl08_size10`: high-return filter, 10x, 10% margin, 18% TP, 8% stop.

WeCom notifications are sent in Chinese and include the paper account, strategy id, leg, score, profile viable rate, margin, notional, TP, stop, PnL, and equity after exit.

## Signal Logic

- Mode: `tp14_v2_rankfixed`.
- Candidate event: OI expansion gate plus whale-fade resolver.
- Sequence filters: threshold persistence and price-position pattern legs.
- Score normalization: train-fitted CDF/rank mapping only. The live/test frame is not ranked against future events.
- Model scores: fixed LightGBM models serialized in `artifacts/tp14_v2_artifacts.pkl`.
- Profile filters: symbol-side historical profile from the fixed training window.
- Signal clock: completed 15m bars only; `signal_lag_bars=0` is safe because incomplete 15m klines are not used.
- Entry fill guard: paper entries and reverse fills require a completed 1m execution bar whose open time is not earlier than the confirmed 15m signal close.
- Exit: 1m intrabar hard stop first, then take-profit, then time exit at 24h max hold.
- Execution data guard: if an open position has a consecutive 1m data gap greater than 2 minutes, the runner sends a Chinese risk alert and pauses time/reverse exit for that position until the missing 1m path can be replayed.

## Server Setup

```bash
cd /root/tp14-paper-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TP14_WEBHOOK_URL='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4b76e995-c816-4e0d-81cb-3e07c7ae140e'
TP14_FORCE_STATE=1 bash deploy_pm2.sh
```

`deploy_pm2.sh` performs:

- install/update Python dependencies,
- extract the split seed archive if needed,
- initialize paper state without server-side selection/backtest,
- start or reload pm2.

Use `TP14_FORCE_STATE=1` for the first V2 upgrade so old TP14 account state is replaced by the four new 100U V2 paper accounts. For later restarts or code-only upgrades, omit `TP14_FORCE_STATE=1` to preserve open positions and trade history.

## PM2 Commands

```bash
pm2 status
pm2 logs tp14-paper-sim --lines 100
pm2 restart tp14-paper-sim --update-env
pm2 save
```

## Manual Commands

Run one tick:

```bash
python3 tp14_paper_sim.py tick --config config/paper_config.json
```

Seed from packaged split archive:

```bash
python3 tp14_paper_sim.py seed --config config/paper_config.json --force-state
```

Start forever:

```bash
python3 tp14_paper_sim.py start --config config/paper_config.json --workers 4
```

Silent health check:

```bash
python3 tp14_paper_sim.py health --config config/paper_config.json --no-webhook
```

## Important Files

- `config/paper_config.json`: fixed 120-symbol universe and paper accounts.
- `artifacts/tp14_v2_artifacts.pkl`: fixed thresholds, profiles, score CDFs, and models.
- `artifacts/tp14_v2_artifact_manifest.json`: human-readable artifact summary.
- `bootstrap_seed/seed_manifest.json`: seed data manifest.
- `runs/tp14/paper_state.json`: current account state and open positions.
- `runs/tp14/events.jsonl`: entry/exit event log.
- `runs/tp14/last_tick.json`: latest tick diagnostics.
- `runs/tp14/paper_loop.log`: loop log.

## Notes

- The server refreshes only live market data and simulates paper orders. It does not change universe, thresholds, profiles, or models.
- Single-symbol data lag does not freeze the whole universe. V2 uses the latest available completed 15m end and stale symbols simply do not emit signals for that tick.
- Health webhook checks are optional and abnormal-only. Entry/exit notifications are sent by the trading loop itself.
