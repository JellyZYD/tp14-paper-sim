from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BINANCE_FAPI = "https://fapi.binance.com"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def config_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    data_root = Path(config["paths"]["data_root"])
    run_dir = Path(config["paths"]["run_dir"])
    return data_root, run_dir


def closed_end_ms(interval: str, now: pd.Timestamp | None = None) -> int:
    ts = now or utc_now()
    step = INTERVAL_MS[interval]
    current_open = int(ts.timestamp() * 1000) // step * step
    return current_open - 1


def binance_get(path: str, params: dict[str, Any], retries: int = 3, timeout: int = 20) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(f"{BINANCE_FAPI}{path}", params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("code"):
                raise RuntimeError(payload)
            return payload
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Binance request failed: {path} {params} {last_exc!r}")


def read_csv_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_timestamp_ms(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except Exception:
        return None
    if df.empty:
        return None
    return int(pd.to_numeric(df["timestamp"], errors="coerce").max())


def append_csv(path: Path, rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    rows.to_csv(path, mode="a", header=not exists, index=False)
    return int(len(rows))


def normalize_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    raw = out.pop("timestamp") if "timestamp" in out.columns else pd.Series(out.index)
    unit = "ms" if pd.to_numeric(raw, errors="coerce").dropna().median() > 10_000_000_000 else None
    out.index = pd.to_datetime(raw, unit=unit, utc=True)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def table_path(root: Path, dataset: str, symbol: str) -> Path:
    if dataset == "signal_klines":
        return root / "klines" / f"{symbol}.csv"
    if dataset == "exec_klines":
        return root / "klines_1m" / f"{symbol}.csv"
    if dataset == "funding":
        return root / "funding" / f"{symbol}.csv"
    if dataset == "oi":
        return root / "market_state_hist" / "oi" / f"{symbol}.csv"
    if dataset == "global_acct_ratio":
        return root / "market_state_hist" / "global_acct_ratio" / f"{symbol}.csv"
    if dataset == "top_acct_ratio":
        return root / "market_state_hist" / "top_acct_ratio" / f"{symbol}.csv"
    if dataset == "top_pos_ratio":
        return root / "market_state_hist" / "top_pos_ratio" / f"{symbol}.csv"
    raise ValueError(f"Unknown dataset: {dataset}")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list[Any]] = []
    cursor = start_ms
    step = INTERVAL_MS[interval]
    while cursor < end_ms:
        page = binance_get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500},
        )
        if not page:
            break
        rows.extend(page)
        last_open = int(page[-1][0])
        cursor = last_open + step
        if len(page) < 1500:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    keep = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        page = binance_get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1]["fundingTime"]) + 1
        if len(page) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"fundingTime": "timestamp", "fundingRate": "FundingRate"})
    df = df[["timestamp", "FundingRate"]]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce")
    return df


def fetch_hist_ratio(endpoint: str, symbol: str, start_ms: int, end_ms: int, rename: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        page = binance_get(
            endpoint,
            {"symbol": symbol, "period": "5m", "startTime": cursor, "endTime": end_ms, "limit": 500},
        )
        if not page:
            break
        rows.extend(page)
        cursor = int(page[-1]["timestamp"]) + 1
        if len(page) < 500:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns=rename)
    keep = ["timestamp"] + [value for value in rename.values() if value != "timestamp"]
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def update_one_dataset(root: Path, symbol: str, dataset: str, lookback_days: int) -> dict[str, Any]:
    path = table_path(root, dataset, symbol)
    if dataset == "signal_klines":
        interval = "15m"
        step_ms = INTERVAL_MS[interval]
        end_ms = closed_end_ms(interval)
        fetcher = lambda start: fetch_klines(symbol, interval, start, end_ms)
    elif dataset == "exec_klines":
        interval = "1m"
        step_ms = INTERVAL_MS[interval]
        end_ms = closed_end_ms(interval)
        fetcher = lambda start: fetch_klines(symbol, interval, start, end_ms)
    elif dataset == "funding":
        step_ms = 1
        end_ms = int(utc_now().timestamp() * 1000)
        fetcher = lambda start: fetch_funding(symbol, start, end_ms)
    elif dataset == "oi":
        step_ms = INTERVAL_MS["5m"]
        end_ms = int(utc_now().timestamp() * 1000)
        fetcher = lambda start: fetch_hist_ratio(
            "/futures/data/openInterestHist",
            symbol,
            start,
            end_ms,
            {"sumOpenInterest": "OpenInterest", "sumOpenInterestValue": "OpenInterestValue"},
        )
    elif dataset == "global_acct_ratio":
        step_ms = INTERVAL_MS["5m"]
        end_ms = int(utc_now().timestamp() * 1000)
        fetcher = lambda start: fetch_hist_ratio(
            "/futures/data/globalLongShortAccountRatio",
            symbol,
            start,
            end_ms,
            {"longShortRatio": "LS_Ratio"},
        )
    elif dataset == "top_acct_ratio":
        step_ms = INTERVAL_MS["5m"]
        end_ms = int(utc_now().timestamp() * 1000)
        fetcher = lambda start: fetch_hist_ratio(
            "/futures/data/topLongShortAccountRatio",
            symbol,
            start,
            end_ms,
            {"longShortRatio": "TopAccount_LS_Ratio"},
        )
    elif dataset == "top_pos_ratio":
        step_ms = INTERVAL_MS["5m"]
        end_ms = int(utc_now().timestamp() * 1000)
        fetcher = lambda start: fetch_hist_ratio(
            "/futures/data/topLongShortPositionRatio",
            symbol,
            start,
            end_ms,
            {"longShortRatio": "TopPosition_LS_Ratio"},
        )
    else:
        raise ValueError(dataset)

    latest = latest_timestamp_ms(path)
    default_start = end_ms - lookback_days * 24 * 60 * 60 * 1000
    start_ms = max((latest + step_ms) if latest is not None else default_start, default_start)
    if start_ms >= end_ms:
        return {"dataset": dataset, "added": 0, "status": "up_to_date"}
    rows = fetcher(start_ms)
    added = append_csv(path, rows)
    return {"dataset": dataset, "added": added, "status": "ok"}


def update_symbol(root: Path, symbol: str, datasets: set[str], lookback_days: int) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": symbol, "ok": True}
    for dataset in sorted(datasets):
        try:
            row = update_one_dataset(root, symbol, dataset, lookback_days)
            result[f"{dataset}_added"] = row["added"]
            result[f"{dataset}_status"] = row["status"]
        except Exception as exc:
            result["ok"] = False
            result[f"{dataset}_status"] = f"error: {exc!r}"
    return result


def update_many(config: dict[str, Any], datasets: set[str], lookback_days: int, workers: int) -> list[dict[str, Any]]:
    root, _ = config_paths(config)
    symbols = list(config["symbols"])
    rows: list[dict[str, Any]] = []
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(update_symbol, root, symbol, datasets, lookback_days): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"symbol": symbol, "ok": False, "error": repr(exc)})
    return rows


def read_frame(root: Path, dataset: str, symbol: str) -> pd.DataFrame:
    return normalize_timestamp_index(read_csv_table(table_path(root, dataset, symbol)))


def resample_ohlcv(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for col in ("quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume"):
        if col in df.columns:
            agg[col] = "sum"
    return df.resample(freq).agg(agg).dropna(subset=["open", "high", "low", "close"])


def resample_last(df: pd.DataFrame, target_index: pd.DatetimeIndex, limit: int = 192) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=target_index)
    base = df.resample("15min").last()
    aligned = base.reindex(base.index.union(target_index)).ffill(limit=limit).reindex(target_index)
    return aligned


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(3, int(window * 0.5)))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def build_features(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    k = read_frame(root, "signal_klines", symbol)
    if k.empty:
        return pd.DataFrame()
    k = k.loc[(k.index >= start) & (k.index < end)]
    base = resample_ohlcv(k, "15min")
    for dataset in ("funding", "oi", "global_acct_ratio", "top_acct_ratio", "top_pos_ratio"):
        frame = read_frame(root, dataset, symbol)
        if not frame.empty:
            frame = frame.loc[(frame.index >= start) & (frame.index < end)]
        aligned = resample_last(frame, base.index)
        if not aligned.empty:
            base = base.join(aligned, how="left")

    if "OpenInterest" not in base.columns:
        base["OpenInterest"] = np.nan
    if "LS_Ratio" not in base.columns:
        base["LS_Ratio"] = np.nan
    if "TopPosition_LS_Ratio" not in base.columns:
        base["TopPosition_LS_Ratio"] = np.nan

    for window in (48, 96):
        delta = base["OpenInterest"].replace(0, np.nan).pct_change(window)
        base[f"OI_Delta_{window}"] = delta
        base[f"OI_Expansion_{window}"] = delta.clip(lower=0)

    base["WhalePosition_Retail_Divergence"] = safe_log(base["TopPosition_LS_Ratio"]) - safe_log(base["LS_Ratio"])
    base["Log_LS_Ratio"] = safe_log(base["LS_Ratio"])

    if "taker_buy_volume" in base.columns:
        base["TakerBuyImbalance"] = base["taker_buy_volume"] / base["volume"].replace(0, np.nan) * 2.0 - 1.0
    else:
        base["TakerBuyImbalance"] = np.nan

    body_high = base[["open", "close"]].max(axis=1)
    body_low = base[["open", "close"]].min(axis=1)
    bar_range = (base["high"] - base["low"]).replace(0, np.nan)
    base["Range_Pct"] = bar_range / base["close"].replace(0, np.nan)
    base["Volume_ZScore"] = rolling_zscore(base["volume"], 96)
    base["Range_ZScore"] = rolling_zscore(base["Range_Pct"], 96)
    base["UpperWick_RangeShare"] = (base["high"] - body_high).clip(lower=0) / bar_range
    base["LowerWick_RangeShare"] = (body_low - base["low"]).clip(lower=0) / bar_range
    base["LiquidationProxy_Max"] = (
        pd.concat([base["UpperWick_RangeShare"], base["LowerWick_RangeShare"]], axis=1).max(axis=1)
        * base["Volume_ZScore"].clip(lower=0)
        * base["Range_ZScore"].clip(lower=0)
    )
    return base.replace([np.inf, -np.inf], np.nan)


def apply_cooldown_positions(mask: np.ndarray, cooldown_bars: int) -> np.ndarray:
    kept: list[int] = []
    last = -cooldown_bars - 1
    for pos in np.flatnonzero(mask):
        if pos - last > cooldown_bars:
            kept.append(int(pos))
            last = int(pos)
    return np.asarray(kept, dtype=int)


def resolver_raw(df: pd.DataFrame) -> pd.Series:
    return -df["WhalePosition_Retail_Divergence"]


def compute_signal_context(
    config: dict[str, Any],
    complete_15m_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]], float]:
    root, _ = config_paths(config)
    entry = config["entry_model"]
    train_start = complete_15m_end - pd.Timedelta(days=int(entry["train_days"]))
    data: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, dict[str, float]] = {}
    entry_gates: list[float] = []
    for symbol in config["symbols"]:
        df = build_features(root, symbol, train_start, complete_15m_end)
        if df.empty:
            continue
        data[symbol] = df
        gate = df["OI_Expansion_48"].replace([np.inf, -np.inf], np.nan).dropna()
        raw_abs = resolver_raw(df).abs().replace([np.inf, -np.inf], np.nan).dropna()
        if gate.empty or raw_abs.empty:
            continue
        gate_th = float(gate.quantile(float(entry["gate_quantile"])))
        resolver_th = float(raw_abs.quantile(float(entry["resolver_quantile"])))
        thresholds[symbol] = {"gate_threshold": gate_th, "resolver_threshold": resolver_th}
        raw = resolver_raw(df).to_numpy(dtype=float)
        gates = df["OI_Expansion_48"].to_numpy(dtype=float)
        base_mask = np.isfinite(gates) & np.isfinite(raw) & (gates >= gate_th) & (np.abs(raw) >= resolver_th)
        kept = apply_cooldown_positions(base_mask, int(entry["cooldown_bars"]))
        entry_gates.extend(gates[kept].tolist())
    extra_gate = float(np.quantile(np.asarray(entry_gates, dtype=float), float(entry["gate_entry_quantile"]))) if entry_gates else math.inf
    return data, thresholds, extra_gate


def latest_common_complete_15m_end(config: dict[str, Any]) -> pd.Timestamp:
    root, _ = config_paths(config)
    ends = []
    for symbol in config["symbols"]:
        df = read_frame(root, "signal_klines", symbol)
        if not df.empty:
            ends.append(df.index.max() + pd.Timedelta(minutes=15))
    if not ends:
        raise RuntimeError("No local 15m signal data. Run bootstrap first.")
    return min(ends).floor("15min")


def latest_fill_price(config: dict[str, Any], symbol: str, as_of: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    root, _ = config_paths(config)
    df = read_frame(root, "exec_klines", symbol)
    eligible = df.loc[df.index <= as_of]
    if eligible.empty:
        raise RuntimeError(f"No 1m execution data for {symbol}")
    row = eligible.iloc[-1]
    return pd.Timestamp(eligible.index[-1]), float(row["close"])


def side_name(direction: float) -> str:
    return "long" if direction > 0 else "short"


def side_float(side: str) -> float:
    return 1.0 if side == "long" else -1.0


def take_profit_price(entry_price: float, side: float, pct: float) -> float:
    return float(entry_price * (1.0 + pct)) if side > 0 else float(entry_price * (1.0 - pct))


def protective_stop_price(entry_price: float, side: float, config: dict[str, Any]) -> float:
    execution = config["execution"]
    leverage = float(config["accounts"][0]["leverage"])
    max_stop_pct = float(execution["max_stop_pct"])
    adverse_pct = (1.0 / leverage) - float(execution["maintenance_margin_rate"]) - float(execution["liquidation_buffer_pct"])
    if side > 0:
        return float(max(entry_price * (1.0 - max_stop_pct), entry_price * (1.0 - adverse_pct)))
    return float(min(entry_price * (1.0 + max_stop_pct), entry_price * (1.0 + adverse_pct)))


def init_state(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    _, run_dir = config_paths(config)
    state_path = run_dir / "paper_state.json"
    if state_path.exists() and not force:
        return load_json(state_path)
    complete_end = latest_common_complete_15m_end(config)
    train_start = complete_end - pd.Timedelta(days=int(config["entry_model"]["train_days"]))
    state = {
        "created_at": utc_now().isoformat(),
        "status": "ready_for_official_paper",
        "universe": {"official_symbols": list(config["symbols"])},
        "preflight": {
            "training_start": train_start.isoformat(),
            "training_end": complete_end.isoformat(),
            "source": "fixed_deploy_config",
        },
        "accounts": [
            {
                "paper_account": account["paper_account"],
                "initial_equity_usdt": float(account["initial_equity_usdt"]),
                "equity_usdt": float(account["initial_equity_usdt"]),
                "cash_usdt": float(account["initial_equity_usdt"]),
                "position_margin_pct": float(account["position_margin_pct"]),
                "leverage": float(account["leverage"]),
                "take_profit_pct": float(account["take_profit_pct"]),
                "positions": [],
                "orders": [],
                "trades": [],
                "processed_signal_keys": [],
            }
            for account in config["accounts"]
        ],
    }
    write_json(state_path, state)
    return state


def fill_stop_price(side: float, stop_price: float, open_price: float, low_price: float, high_price: float) -> float | None:
    if side > 0:
        if not np.isfinite(low_price) or low_price > stop_price:
            return None
        return float(open_price) if np.isfinite(open_price) and open_price <= stop_price else float(stop_price)
    if not np.isfinite(high_price) or high_price < stop_price:
        return None
    return float(open_price) if np.isfinite(open_price) and open_price >= stop_price else float(stop_price)


def close_position(
    config: dict[str, Any],
    account: dict[str, Any],
    position: dict[str, Any],
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: str,
) -> dict[str, Any]:
    side = side_float(position["side"])
    fee = float(config["execution"]["tx_cost_bps"]) / 10_000.0
    gross_return = side * (exit_price / float(position["entry_price"]) - 1.0)
    net_return = gross_return - 2.0 * fee
    leveraged = float(position["leverage"]) * net_return
    pnl = float(position["margin_usdt"]) * leveraged
    account["equity_usdt"] = float(account["equity_usdt"]) + pnl
    account["cash_usdt"] = account["equity_usdt"]
    trade = {
        **position,
        "exit_time": exit_time.isoformat(),
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "leveraged_net_return": float(leveraged),
        "pnl_usdt": float(pnl),
        "account_equity_after": float(account["equity_usdt"]),
    }
    account.setdefault("trades", []).append(trade)
    return trade


def check_exits(config: dict[str, Any], state: dict[str, Any], tick_time: pd.Timestamp) -> list[dict[str, Any]]:
    root, _ = config_paths(config)
    events: list[dict[str, Any]] = []
    max_hold_hours = int(config["entry_model"]["max_hold_bars"]) * 15 / 60
    for account in state["accounts"]:
        remaining = []
        for position in account.get("positions", []):
            symbol = position["symbol"]
            side = side_float(position["side"])
            last_checked = pd.Timestamp(position.get("last_checked_time", position["entry_fill_time"]))
            m1 = read_frame(root, "exec_klines", symbol)
            window = m1.loc[(m1.index > last_checked) & (m1.index <= tick_time)]
            exit_event: dict[str, Any] | None = None
            for ts, row in window.iterrows():
                bar_open = float(row["open"])
                bar_high = float(row["high"])
                bar_low = float(row["low"])
                stop_fill = fill_stop_price(side, float(position["protective_stop_price"]), bar_open, bar_low, bar_high)
                if stop_fill is not None:
                    exit_event = {"time": pd.Timestamp(ts) + pd.Timedelta(minutes=1), "price": stop_fill, "reason": "protective_stop"}
                    break
                tp_fill = None
                if side > 0 and bar_high >= float(position["take_profit_price"]):
                    tp_fill = bar_open if bar_open >= float(position["take_profit_price"]) else float(position["take_profit_price"])
                elif side < 0 and bar_low <= float(position["take_profit_price"]):
                    tp_fill = bar_open if bar_open <= float(position["take_profit_price"]) else float(position["take_profit_price"])
                if tp_fill is not None:
                    exit_event = {"time": pd.Timestamp(ts) + pd.Timedelta(minutes=1), "price": float(tp_fill), "reason": "intrabar_take_profit"}
                    break
            if exit_event is None and tick_time - pd.Timestamp(position["entry_fill_time"]) >= pd.Timedelta(hours=max_hold_hours):
                fill_time, fill_price = latest_fill_price(config, symbol, tick_time)
                exit_event = {"time": fill_time, "price": fill_price, "reason": "time_exit"}
            if exit_event is None:
                position["last_checked_time"] = tick_time.isoformat()
                remaining.append(position)
            else:
                trade = close_position(config, account, position, pd.Timestamp(exit_event["time"]), float(exit_event["price"]), str(exit_event["reason"]))
                events.append({"event": "exit", "paper_account": account["paper_account"], **trade})
        account["positions"] = remaining
    return events


def signal_key(signal: dict[str, Any]) -> str:
    return f"{signal['symbol']}|{signal['entry_bar_end']}|{signal['side']}"


def on_cooldown(account: dict[str, Any], symbol: str, now: pd.Timestamp, cooldown_bars: int) -> bool:
    cooldown = pd.Timedelta(minutes=15 * cooldown_bars)
    for trade in account.get("trades", []):
        if trade.get("symbol") == symbol and now - pd.Timestamp(trade["entry_fill_time"]) <= cooldown:
            return True
    return False


def compute_signals(config: dict[str, Any], complete_end: pd.Timestamp) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entry = config["entry_model"]
    data, thresholds, extra_gate = compute_signal_context(config, complete_end)
    signals: list[dict[str, Any]] = []
    for symbol in config["symbols"]:
        df = data.get(symbol)
        threshold = thresholds.get(symbol)
        if df is None or threshold is None or df.empty:
            continue
        df = df.loc[df.index + pd.Timedelta(minutes=15) <= complete_end]
        if len(df) <= int(entry["signal_lag_bars"]):
            continue
        i = len(df) - 1
        signal_idx = i - int(entry["signal_lag_bars"])
        gate = float(df["OI_Expansion_48"].iloc[signal_idx])
        raw = float(resolver_raw(df).iloc[signal_idx])
        if not np.isfinite(gate) or not np.isfinite(raw):
            continue
        if gate < threshold["gate_threshold"] or gate < extra_gate or abs(raw) < threshold["resolver_threshold"]:
            continue
        direction = float(np.sign(raw))
        signals.append(
            {
                "symbol": symbol,
                "direction": direction,
                "side": side_name(direction),
                "entry_bar_start": df.index[i].isoformat(),
                "entry_bar_end": (df.index[i] + pd.Timedelta(minutes=15)).isoformat(),
                "signal_bar_start": df.index[signal_idx].isoformat(),
                "signal_bar_end": (df.index[signal_idx] + pd.Timedelta(minutes=15)).isoformat(),
                "signal_gate": gate,
                "signal_raw": raw,
                "gate_threshold": threshold["gate_threshold"],
                "resolver_threshold": threshold["resolver_threshold"],
                "extra_entry_gate": extra_gate,
                "theoretical_entry_close": float(df["close"].iloc[i]),
            }
        )
    return signals, {"extra_entry_gate": extra_gate, "threshold_symbols": len(thresholds)}


def open_entries(config: dict[str, Any], state: dict[str, Any], signals: list[dict[str, Any]], tick_time: pd.Timestamp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cooldown_bars = int(config["entry_model"]["cooldown_bars"])
    for account in state["accounts"]:
        account.setdefault("processed_signal_keys", [])
        open_symbols = {position["symbol"] for position in account.get("positions", [])}
        for signal in signals:
            symbol = signal["symbol"]
            key = signal_key(signal)
            if key in account["processed_signal_keys"] or symbol in open_symbols:
                continue
            if on_cooldown(account, symbol, tick_time, cooldown_bars):
                continue
            fill_time, fill_price = latest_fill_price(config, symbol, tick_time)
            direction = float(signal["direction"])
            margin = float(account["equity_usdt"]) * float(account["position_margin_pct"])
            notional = margin * float(account["leverage"])
            position = {
                "symbol": symbol,
                "side": side_name(direction),
                "direction": direction,
                "entry_fill_time": fill_time.isoformat(),
                "entry_decision_time": tick_time.isoformat(),
                "entry_bar_start": signal["entry_bar_start"],
                "entry_bar_end": signal["entry_bar_end"],
                "signal_bar_start": signal["signal_bar_start"],
                "signal_bar_end": signal["signal_bar_end"],
                "entry_price": float(fill_price),
                "theoretical_entry_close": float(signal["theoretical_entry_close"]),
                "margin_usdt": float(margin),
                "notional_usdt": float(notional),
                "quantity": float(notional / fill_price) if fill_price > 0 else 0.0,
                "leverage": float(account["leverage"]),
                "position_margin_pct": float(account["position_margin_pct"]),
                "take_profit_pct": float(account["take_profit_pct"]),
                "take_profit_price": take_profit_price(fill_price, direction, float(account["take_profit_pct"])),
                "protective_stop_price": protective_stop_price(fill_price, direction, config),
                "tx_cost_bps": float(config["execution"]["tx_cost_bps"]),
                "last_checked_time": fill_time.isoformat(),
                "signal_gate": signal["signal_gate"],
                "signal_raw": signal["signal_raw"],
            }
            account.setdefault("positions", []).append(position)
            account["processed_signal_keys"].append(key)
            open_symbols.add(symbol)
            events.append({"event": "entry", "paper_account": account["paper_account"], **position})
    return events


def acquire_lock(path: Path, stale_seconds: float = 300.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if path.exists():
        try:
            payload = load_json(path)
            if now - float(payload.get("timestamp", 0.0)) < stale_seconds:
                return False
        except Exception:
            pass
        path.unlink(missing_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "timestamp": now, "started_at": utc_now().isoformat()}, handle, indent=2)
    return True


def release_lock(path: Path) -> None:
    try:
        payload = load_json(path)
    except Exception:
        payload = {}
    if payload.get("pid") == os.getpid():
        path.unlink(missing_ok=True)


def run_tick(config: dict[str, Any]) -> dict[str, Any]:
    _, run_dir = config_paths(config)
    lock = run_dir / "paper_tick.lock"
    if not acquire_lock(lock):
        return {"skipped": "tick_lock_active"}
    try:
        state_path = run_dir / "paper_state.json"
        state = init_state(config, force=False) if not state_path.exists() else load_json(state_path)
        tick_time = utc_now()
        complete_end = latest_common_complete_15m_end(config)
        exits = check_exits(config, state, tick_time)
        signals, meta = compute_signals(config, complete_end)
        entries = open_entries(config, state, signals, tick_time)
        state["last_tick"] = {
            "tick_time": tick_time.isoformat(),
            "complete_15m_end": complete_end.isoformat(),
            "signals": len(signals),
            "entries": len(entries),
            "exits": len(exits),
            **meta,
        }
        events = exits + entries
        for event in events:
            append_jsonl(run_dir / "events.jsonl", {"tick_time": tick_time.isoformat(), **event})
        write_json(state_path, state)
        last_tick = {"last_tick": state["last_tick"], "signals": signals, "events": events}
        write_json(run_dir / "last_tick.json", last_tick)
        return last_tick
    finally:
        release_lock(lock)


def bootstrap(config: dict[str, Any], workers: int | None = None, force_state: bool = False) -> None:
    execution = config["execution"]
    max_workers = int(workers or execution["workers"])
    datasets = {
        "signal_klines",
        "exec_klines",
        "funding",
        "oi",
        "global_acct_ratio",
        "top_acct_ratio",
        "top_pos_ratio",
    }
    rows = update_many(config, datasets, int(config["entry_model"]["train_days"]), max_workers)
    _, run_dir = config_paths(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "bootstrap_update_report.csv", index=False)
    init_state(config, force=force_state)


def loop(config: dict[str, Any], max_iterations: int = 0) -> None:
    _, run_dir = config_paths(config)
    lock = run_dir / "paper_loop.lock"
    log_path = run_dir / "paper_loop.log"
    if not acquire_lock(lock, stale_seconds=900.0):
        raise RuntimeError(f"Loop lock is active: {lock}")
    append_log(log_path, f"START {utc_now().isoformat()} pid={os.getpid()}")
    last_state_update = 0.0
    last_funding_update = 0.0
    iteration = 0
    try:
        while True:
            iteration += 1
            started = time.time()
            datasets = {"signal_klines", "exec_klines"}
            if started - last_state_update >= float(config["execution"]["state_update_interval_minutes"]) * 60:
                datasets.update({"oi", "global_acct_ratio", "top_acct_ratio", "top_pos_ratio"})
                last_state_update = started
            if started - last_funding_update >= float(config["execution"]["funding_update_interval_minutes"]) * 60:
                datasets.add("funding")
                last_funding_update = started
            rows = update_many(config, datasets, 2, int(config["execution"]["workers"]))
            failed = [row for row in rows if not row.get("ok")]
            append_log(log_path, f"{utc_now().isoformat()} UPDATE datasets={','.join(sorted(datasets))} failed={len(failed)}")
            if failed:
                append_log(log_path, f"{utc_now().isoformat()} UPDATE_FAILED {failed}")
            result = run_tick(config)
            last = result.get("last_tick", {})
            append_log(
                log_path,
                f"{utc_now().isoformat()} TICK signals={last.get('signals')} entries={last.get('entries')} exits={last.get('exits')}",
            )
            write_json(run_dir / "paper_loop.heartbeat.json", {"pid": os.getpid(), "iteration": iteration, "updated_at": utc_now().isoformat()})
            if max_iterations and iteration >= max_iterations:
                append_log(log_path, f"STOP max_iterations={max_iterations}")
                break
            elapsed = time.time() - started
            time.sleep(max(float(config["execution"]["poll_interval_seconds"]) - elapsed, 1.0))
    finally:
        release_lock(lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TP14 deploy-only paper simulation bot.")
    parser.add_argument("command", choices=("bootstrap", "tick", "loop", "init-state"))
    parser.add_argument("--config", default="config/paper_config.json")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force-state", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config))
    if args.command == "bootstrap":
        bootstrap(config, args.workers, args.force_state)
    elif args.command == "init-state":
        init_state(config, args.force_state)
    elif args.command == "tick":
        result = run_tick(config)
        print(json.dumps(result.get("last_tick", result), ensure_ascii=False, indent=2, default=str))
    elif args.command == "loop":
        loop(config, args.max_iterations)


if __name__ == "__main__":
    main()
