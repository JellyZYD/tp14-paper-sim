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

- `tp14_v2_stable_scoretpv2_fadep3_sl08_size05`: stable filter, 10x, 5% margin, score-v2 dynamic TP, 8% stop, 3%+ emotion-fade exit.
- `tp14_v2_stable_scoretpv2_fadep3_sl08_size10`: stable filter, 10x, 10% margin, score-v2 dynamic TP, 8% stop, 3%+ emotion-fade exit.
- `tp14_v2_highret_tp20_fadep3_sl08_size05`: high-return filter, 10x, 5% margin, 20% TP, 8% stop, 3%+ emotion-fade exit.
- `tp14_v2_highret_tp20_fadep3_sl08_size10`: high-return filter, 10x, 10% margin, 20% TP, 8% stop, 3%+ emotion-fade exit.

The current config keeps the previous leverage and position sizing. Existing state is migrated in place from the old `tp10/tp18` account names by `legacy_paper_accounts`; equity, open positions, closed trades, and processed signal keys are preserved. Accounts that are not present in the active config are moved to `archived_accounts` and no longer trade.

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
- Exit: 1m intrabar hard stop first, then fixed/dynamic take-profit, then 3%+ emotion-fade exit, then time exit at 24h max hold.
- Dynamic TP: stable accounts use score-v2 TP buckets from live `score_lgbm_combo/score_profile`: 12%, 16%, 20%, or 24%.
- Emotion-fade exit: after a position has at least 3% raw favorable move, exit on the next 1m close when the 5m whale-vs-retail favorable raw signal fades back to zero or worse.
- Execution data guard: if an open position has a consecutive 1m data gap greater than 2 minutes, the runner sends a Chinese risk alert and pauses time/reverse exit for that position until the missing 1m path can be replayed.
- Binance rate-limit guard: while a 418/429 backoff is active, the runner does not actively request new execution klines for entries or reverses. Open positions are checked only with cached 1m data; if the cache has a gap, the execution-gap alert path takes over.

## Server Setup

```bash
cd /root/tp14-paper-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TP14_WEBHOOK_URL='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4b76e995-c816-4e0d-81cb-3e07c7ae140e'
TP14_FORCE_STATE=1 bash deploy_pm2.sh
bash install_health_cron.sh
```

`deploy_pm2.sh` performs:

- install/update Python dependencies,
- extract the split seed archive only when local market data is missing or `TP14_FORCE_STATE=1`,
- initialize paper state without server-side selection/backtest,
- start or reload pm2.

`install_health_cron.sh` installs a 5-minute health monitor. Normal checks stay silent. It sends Chinese WeCom alerts only when health turns abnormal, and sends one recovery message when the system returns to normal. It checks PM2 status, stale ticks, recent traceback logs, Binance rate-limit/backoff, unreadable state, execution data gaps, and low disk space.

Use `TP14_FORCE_STATE=1` only when intentionally resetting the paper accounts to fresh 100U state. For this exit-model upgrade and normal restarts, omit `TP14_FORCE_STATE=1`; the runner migrates old account names and applies the new exit settings without deleting open positions, trade history, or the local market-data cache. If old PM2 logs contain historical tracebacks, run `pm2 flush tp14-paper-sim` after the upgrade to avoid a one-time stale health alert.

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
