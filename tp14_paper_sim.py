from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
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
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            raise
        payload = json.loads(backup.read_text(encoding="utf-8-sig"))
        write_json(path, payload)
        return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    backup = path.with_suffix(path.suffix + ".bak")
    tmp_backup = backup.with_name(f"{backup.name}.tmp.{os.getpid()}")
    tmp_backup.write_text(text, encoding="utf-8")
    os.replace(tmp_backup, backup)


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
    old = read_csv_table(path)
    before = len(old) if not old.empty else 0
    combined = pd.concat([old, rows], ignore_index=True) if not old.empty else rows.copy()
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_numeric(combined["timestamp"], errors="coerce").astype("Int64")
        combined = combined.dropna(subset=["timestamp"])
        combined["timestamp"] = combined["timestamp"].astype("int64")
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    combined.to_csv(tmp, index=False)
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    os.replace(tmp, path)
    return max(int(len(combined) - before), 0)


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
    period_ms = INTERVAL_MS["5m"]
    limit = 500
    while cursor < end_ms:
        page_end = min(end_ms, cursor + (limit - 1) * period_ms)
        page = binance_get(
            endpoint,
            {"symbol": symbol, "period": "5m", "startTime": cursor, "endTime": page_end, "limit": limit},
        )
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1]["timestamp"]) + period_ms
        if next_cursor <= cursor:
            break
        if page_end >= end_ms and len(page) < limit:
            break
        cursor = next_cursor
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
        lookback_days = min(lookback_days, 30)
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
        lookback_days = min(lookback_days, 30)
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
        lookback_days = min(lookback_days, 30)
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
        lookback_days = min(lookback_days, 30)
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


def is_rate_limited(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        text = " ".join(str(value) for value in row.values())
        if "HTTPError" in text and (" 418 " in text or " 429 " in text or "I'm a teapot" in text or "Too Many Requests" in text):
            return True
    return False


def rate_limit_backoff_path(config: dict[str, Any]) -> Path:
    _, run_dir = config_paths(config)
    return run_dir / "rate_limit_backoff.json"


def read_backoff_until(config: dict[str, Any]) -> float:
    path = rate_limit_backoff_path(config)
    if not path.exists():
        return 0.0
    try:
        payload = load_json(path)
        return float(payload.get("backoff_until", 0.0))
    except Exception:
        return 0.0


def write_backoff_until(config: dict[str, Any], backoff_until: float, reason: str) -> None:
    write_json(
        rate_limit_backoff_path(config),
        {
            "backoff_until": backoff_until,
            "backoff_until_utc": pd.Timestamp.fromtimestamp(backoff_until, tz="UTC").isoformat(),
            "reason": reason,
            "updated_at": utc_now().isoformat(),
        },
    )


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


def coalesce_columns(frame: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    present = [col for col in (target, *aliases) if col in frame.columns]
    if not present:
        return frame
    out = frame.copy()
    out[target] = out[present].bfill(axis=1).iloc[:, 0]
    drop_cols = [col for col in aliases if col in out.columns]
    return out.drop(columns=drop_cols)


def build_features(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    k = read_frame(root, "signal_klines", symbol)
    if k.empty:
        return pd.DataFrame()
    k = k.loc[(k.index >= start) & (k.index < end)]
    base = resample_ohlcv(k, "15min")
    for dataset in ("funding", "oi", "global_acct_ratio", "top_acct_ratio", "top_pos_ratio"):
        frame = read_frame(root, dataset, symbol)
        if dataset == "global_acct_ratio":
            frame = coalesce_columns(frame, "LS_Ratio", ("ratio",))
        elif dataset == "top_acct_ratio":
            frame = coalesce_columns(frame, "TopAccount_LS_Ratio", ("ratio",))
        elif dataset == "top_pos_ratio":
            frame = coalesce_columns(frame, "TopPosition_LS_Ratio", ("ratio",))
        elif dataset == "funding":
            frame = coalesce_columns(frame, "FundingRate", ("funding_rate",))
        elif dataset == "oi":
            frame = coalesce_columns(frame, "OpenInterest", ("oi",))
            frame = coalesce_columns(frame, "OpenInterestValue", ("oi_value",))
        frame = frame.loc[:, ~frame.columns.duplicated(keep="last")]
        expected_cols = {
            "funding": ("FundingRate",),
            "oi": ("OpenInterest", "OpenInterestValue"),
            "global_acct_ratio": ("LS_Ratio",),
            "top_acct_ratio": ("TopAccount_LS_Ratio",),
            "top_pos_ratio": ("TopPosition_LS_Ratio",),
        }[dataset]
        frame = frame[[col for col in expected_cols if col in frame.columns]]
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


def state_training_window(state: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    preflight = state.get("preflight", {})
    if "training_start" not in preflight or "training_end" not in preflight:
        raise RuntimeError("paper_state preflight.training_start/training_end are required for fixed-window thresholds")
    train_start = pd.Timestamp(preflight["training_start"])
    train_end = pd.Timestamp(preflight["training_end"])
    if train_end <= train_start:
        raise RuntimeError(f"Invalid fixed training window: {train_start} -> {train_end}")
    return train_start, train_end


def normalize_strategy_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def compute_signal_context(
    config: dict[str, Any],
    state: dict[str, Any],
    complete_15m_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]], float, pd.Timestamp, pd.Timestamp]:
    root, _ = config_paths(config)
    entry = config["entry_model"]
    train_start, train_end = state_training_window(state)
    if train_end > complete_15m_end:
        raise RuntimeError(f"Fixed training_end {train_end} is after complete_15m_end {complete_15m_end}")
    data: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, dict[str, float]] = {}
    entry_gates: list[float] = []
    for symbol in config["symbols"]:
        df = build_features(root, symbol, train_start, complete_15m_end)
        if df.empty:
            continue
        data[symbol] = df
        train_df = df.loc[(df.index >= train_start) & (df.index < train_end)]
        gate = train_df["OI_Expansion_48"].replace([np.inf, -np.inf], np.nan).dropna()
        raw_abs = resolver_raw(train_df).abs().replace([np.inf, -np.inf], np.nan).dropna()
        if gate.empty or raw_abs.empty:
            continue
        gate_th = float(gate.quantile(float(entry["gate_quantile"])))
        resolver_th = float(raw_abs.quantile(float(entry["resolver_quantile"])))
        thresholds[symbol] = {"gate_threshold": gate_th, "resolver_threshold": resolver_th}
        raw = resolver_raw(train_df).to_numpy(dtype=float)
        gates = train_df["OI_Expansion_48"].to_numpy(dtype=float)
        base_mask = np.isfinite(gates) & np.isfinite(raw) & (gates >= gate_th) & (np.abs(raw) >= resolver_th)
        kept = apply_cooldown_positions(base_mask, int(entry["cooldown_bars"]))
        entry_gates.extend(gates[kept].tolist())
    extra_gate = float(np.quantile(np.asarray(entry_gates, dtype=float), float(entry["gate_entry_quantile"]))) if entry_gates else math.inf
    return data, thresholds, extra_gate, train_start, train_end


def latest_common_complete_15m_end(config: dict[str, Any], as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    root, _ = config_paths(config)
    ends = []
    for symbol in config["symbols"]:
        df = read_frame(root, "signal_klines", symbol)
        if not df.empty:
            ends.append(df.index.max() + pd.Timedelta(minutes=15))
    if not ends:
        raise RuntimeError("No local 15m signal data. Run bootstrap first.")
    if str(config.get("entry_model", {}).get("mode", "")).lower() == "tp14_v2_rankfixed":
        data_complete_end = max(ends).floor("15min")
    else:
        data_complete_end = min(ends).floor("15min")
    if as_of is None:
        return data_complete_end
    time_complete_end = pd.Timestamp(as_of).floor("15min")
    return min(data_complete_end, time_complete_end)


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


def funding_return_between(
    config: dict[str, Any],
    position: dict[str, Any],
    exit_time: pd.Timestamp,
    funding_cutoff_time: pd.Timestamp | None = None,
) -> float:
    root, _ = config_paths(config)
    rates_frame = read_frame(root, "funding", str(position["symbol"]))
    if rates_frame.empty:
        return 0.0
    rate_col = "FundingRate" if "FundingRate" in rates_frame.columns else "funding_rate"
    if rate_col not in rates_frame.columns:
        return 0.0
    rates = pd.to_numeric(rates_frame[rate_col], errors="coerce").dropna()
    if rates.empty:
        return 0.0
    entry_time = pd.Timestamp(position["entry_fill_time"])
    cutoff = pd.Timestamp(funding_cutoff_time) if funding_cutoff_time is not None else pd.Timestamp(exit_time)
    side = side_float(str(position["side"]))
    window = rates.loc[(rates.index >= entry_time) & (rates.index < cutoff)]
    if window.empty:
        return 0.0
    return float((-side * window).sum())


def protective_stop_price(entry_price: float, side: float, config: dict[str, Any], account: dict[str, Any] | None = None) -> float:
    execution = config["execution"]
    account = account or config["accounts"][0]
    leverage = float(account["leverage"])
    max_stop_pct = float(account.get("stop_loss_pct", execution["max_stop_pct"]))
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
    train_end = complete_end
    if str(config.get("entry_model", {}).get("mode", "")).lower() == "tp14_v2_rankfixed":
        from tp14_v2_live_core import load_artifact

        artifact = load_artifact(config)
        train_start = pd.Timestamp(artifact["training_start"])
        train_end = pd.Timestamp(artifact["training_end"])
    state = {
        "created_at": utc_now().isoformat(),
        "status": "ready_for_official_paper",
        "universe": {"official_symbols": list(config["symbols"])},
        "preflight": {
            "training_start": train_start.isoformat(),
            "training_end": train_end.isoformat(),
            "source": "tp14_v2_artifact" if str(config.get("entry_model", {}).get("mode", "")).lower() == "tp14_v2_rankfixed" else "fixed_deploy_config",
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
                "stop_loss_pct": float(account.get("stop_loss_pct", config["execution"]["max_stop_pct"])),
                "strategy_ids": normalize_strategy_ids(account.get("strategy_ids", [])),
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
    funding_return: float = 0.0,
) -> dict[str, Any]:
    side = side_float(position["side"])
    fee = float(config["execution"]["tx_cost_bps"]) / 10_000.0
    gross_return = side * (exit_price / float(position["entry_price"]) - 1.0)
    fee_return = 2.0 * fee
    net_return = gross_return - fee_return + funding_return
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
        "fee_return": float(fee_return),
        "funding_return": float(funding_return),
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
                    exit_event = {
                        "time": pd.Timestamp(ts) + pd.Timedelta(minutes=1),
                        "price": stop_fill,
                        "reason": "protective_stop",
                        "funding_cutoff_time": pd.Timestamp(ts),
                    }
                    break
                tp_fill = None
                if side > 0 and bar_high >= float(position["take_profit_price"]):
                    tp_fill = bar_open if bar_open >= float(position["take_profit_price"]) else float(position["take_profit_price"])
                elif side < 0 and bar_low <= float(position["take_profit_price"]):
                    tp_fill = bar_open if bar_open <= float(position["take_profit_price"]) else float(position["take_profit_price"])
                if tp_fill is not None:
                    exit_event = {
                        "time": pd.Timestamp(ts) + pd.Timedelta(minutes=1),
                        "price": float(tp_fill),
                        "reason": "intrabar_take_profit",
                        "funding_cutoff_time": pd.Timestamp(ts),
                    }
                    break
            if exit_event is None and tick_time - pd.Timestamp(position["entry_fill_time"]) >= pd.Timedelta(hours=max_hold_hours):
                fill_time, fill_price = latest_fill_price(config, symbol, tick_time)
                exit_event = {"time": fill_time, "price": fill_price, "reason": "time_exit"}
            if exit_event is None:
                position["last_checked_time"] = tick_time.isoformat()
                remaining.append(position)
            else:
                funding_return = funding_return_between(
                    config,
                    position,
                    pd.Timestamp(exit_event["time"]),
                    pd.Timestamp(exit_event.get("funding_cutoff_time", exit_event["time"])),
                )
                trade = close_position(
                    config,
                    account,
                    position,
                    pd.Timestamp(exit_event["time"]),
                    float(exit_event["price"]),
                    str(exit_event["reason"]),
                    funding_return,
                )
                events.append({"event": "exit", "paper_account": account["paper_account"], **trade})
        account["positions"] = remaining
    return events


def signal_key(signal: dict[str, Any]) -> str:
    return f"{signal.get('strategy_id', 'legacy')}|{signal['symbol']}|{signal['entry_bar_end']}|{signal['side']}"


def account_allows_signal(account: dict[str, Any], signal: dict[str, Any]) -> bool:
    strategy_ids = account.get("strategy_ids") or account.get("strategy_id")
    if not strategy_ids:
        return True
    if isinstance(strategy_ids, str):
        allowed = {strategy_ids}
    else:
        allowed = {str(item) for item in strategy_ids}
    return str(signal.get("strategy_id", "legacy")) in allowed


def on_cooldown(account: dict[str, Any], symbol: str, now: pd.Timestamp, cooldown_bars: int) -> bool:
    cooldown = pd.Timedelta(minutes=15 * cooldown_bars)
    for trade in account.get("trades", []):
        if trade.get("symbol") == symbol and now - pd.Timestamp(trade["entry_fill_time"]) <= cooldown:
            return True
    return False


def compute_signals(config: dict[str, Any], state: dict[str, Any], complete_end: pd.Timestamp) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(config.get("entry_model", {}).get("mode", "")).lower() == "tp14_v2_rankfixed":
        from tp14_v2_live_core import compute_v2_signals

        return compute_v2_signals(config, state, complete_end)

    entry = config["entry_model"]
    data, thresholds, extra_gate, train_start, train_end = compute_signal_context(config, state, complete_end)
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
    return signals, {
        "extra_entry_gate": extra_gate,
        "threshold_symbols": len(thresholds),
        "training_start": train_start.isoformat(),
        "training_end": train_end.isoformat(),
        "threshold_mode": "fixed_preflight_window",
    }


def build_position_from_signal(
    config: dict[str, Any],
    account: dict[str, Any],
    signal: dict[str, Any],
    tick_time: pd.Timestamp,
    fill_time: pd.Timestamp,
    fill_price: float,
) -> dict[str, Any]:
    direction = float(signal["direction"])
    margin = float(account["equity_usdt"]) * float(account["position_margin_pct"])
    notional = margin * float(account["leverage"])
    return {
        "symbol": signal["symbol"],
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
        "stop_loss_pct": float(account.get("stop_loss_pct", config["execution"]["max_stop_pct"])),
        "protective_stop_price": protective_stop_price(fill_price, direction, config, account),
        "tx_cost_bps": float(config["execution"]["tx_cost_bps"]),
        "last_checked_time": fill_time.isoformat(),
        "signal_gate": signal["signal_gate"],
        "signal_raw": signal["signal_raw"],
        "strategy_id": signal.get("strategy_id", "legacy"),
        "strategy_name": signal.get("strategy_name", signal.get("strategy_id", "legacy")),
        "leg": signal.get("leg", ""),
        "score": signal.get("score", np.nan),
        "profile_viable_rate": signal.get("profile_viable_rate", np.nan),
    }


def apply_reverse_signals(config: dict[str, Any], state: dict[str, Any], signals: list[dict[str, Any]], tick_time: pd.Timestamp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cooldown = pd.Timedelta(minutes=15 * int(config["entry_model"]["cooldown_bars"]))
    for account in state["accounts"]:
        account.setdefault("processed_signal_keys", [])
        account_signals = [signal for signal in signals if account_allows_signal(account, signal)]
        signal_by_symbol = {str(signal["symbol"]): signal for signal in account_signals}
        remaining: list[dict[str, Any]] = []
        for position in account.get("positions", []):
            symbol = str(position["symbol"])
            signal = signal_by_symbol.get(symbol)
            if signal is None or float(signal["direction"]) != -side_float(str(position["side"])):
                remaining.append(position)
                continue
            key = signal_key(signal)
            if key in account["processed_signal_keys"]:
                remaining.append(position)
                continue
            if tick_time - pd.Timestamp(position["entry_fill_time"]) <= cooldown:
                remaining.append(position)
                continue
            try:
                fill_time, fill_price = latest_fill_price(config, symbol, tick_time)
            except RuntimeError:
                remaining.append(position)
                continue
            funding_return = funding_return_between(config, position, fill_time, fill_time)
            trade = close_position(config, account, position, fill_time, fill_price, "reverse", funding_return)
            events.append({"event": "exit", "paper_account": account["paper_account"], **trade})
            new_position = build_position_from_signal(config, account, signal, tick_time, fill_time, fill_price)
            remaining.append(new_position)
            account["processed_signal_keys"].append(key)
            events.append({"event": "entry", "paper_account": account["paper_account"], **new_position})
        account["positions"] = remaining
    return events


def open_entries(config: dict[str, Any], state: dict[str, Any], signals: list[dict[str, Any]], tick_time: pd.Timestamp) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cooldown_bars = int(config["entry_model"]["cooldown_bars"])
    for account in state["accounts"]:
        account.setdefault("processed_signal_keys", [])
        open_symbols = {position["symbol"] for position in account.get("positions", [])}
        for signal in signals:
            if not account_allows_signal(account, signal):
                continue
            symbol = signal["symbol"]
            key = signal_key(signal)
            if key in account["processed_signal_keys"] or symbol in open_symbols:
                continue
            if on_cooldown(account, symbol, tick_time, cooldown_bars):
                continue
            try:
                fill_time, fill_price = latest_fill_price(config, symbol, tick_time)
            except RuntimeError:
                continue
            position = build_position_from_signal(config, account, signal, tick_time, fill_time, fill_price)
            account.setdefault("positions", []).append(position)
            account["processed_signal_keys"].append(key)
            open_symbols.add(symbol)
            events.append({"event": "entry", "paper_account": account["paper_account"], **position})
    return events


def resolve_webhook_url(config: dict[str, Any]) -> str:
    notifications = config.get("notifications", {})
    if notifications and not bool(notifications.get("enabled", True)):
        return ""
    direct = str(notifications.get("webhook_url", "")).strip()
    if direct:
        return direct
    env_name = str(notifications.get("webhook_url_env", "TP14_WEBHOOK_URL")).strip()
    if env_name and os.environ.get(env_name):
        return str(os.environ[env_name]).strip()
    file_name = str(notifications.get("webhook_url_file", "")).strip()
    if file_name:
        path = Path(file_name)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return ""


def account_snapshot(state: dict[str, Any], paper_account: str) -> dict[str, Any]:
    for account in state.get("accounts", []):
        if account.get("paper_account") == paper_account:
            return account
    return {}


def format_event_message(event: dict[str, Any], state: dict[str, Any], tick_time: pd.Timestamp) -> str:
    account = account_snapshot(state, str(event.get("paper_account", "")))
    equity = float(account.get("equity_usdt", event.get("account_equity_after", np.nan)))
    open_count = len(account.get("positions", []))
    lines = [
        f"TP14 纸面交易 {event.get('event')} | {event.get('paper_account')} | {event.get('symbol')} {event.get('side')}",
        f"tick_time_utc: {tick_time.isoformat()}",
        f"equity_usdt: {equity:.4f}" if np.isfinite(equity) else "equity_usdt: n/a",
        f"open_positions: {open_count}",
    ]
    if event.get("strategy_id"):
        lines.append(f"strategy: {event.get('strategy_id')} | {event.get('strategy_name', '')}")
    if event.get("leg"):
        lines.append(f"leg: {event.get('leg')}")
    if event.get("event") == "entry":
        lines.extend(
            [
                f"entry_time: {event.get('entry_fill_time')}",
                f"entry_price: {float(event.get('entry_price', np.nan)):.8g}",
                f"margin_usdt: {float(event.get('margin_usdt', np.nan)):.4f}",
                f"notional_usdt: {float(event.get('notional_usdt', np.nan)):.4f}",
                f"take_profit_price: {float(event.get('take_profit_price', np.nan)):.8g}",
                f"protective_stop_price: {float(event.get('protective_stop_price', np.nan)):.8g}",
                f"score: {float(event.get('score', np.nan)):.6f}" if np.isfinite(float(event.get("score", np.nan))) else "score: n/a",
                f"profile_viable_rate: {float(event.get('profile_viable_rate', np.nan)):.4f}" if np.isfinite(float(event.get("profile_viable_rate", np.nan))) else "profile_viable_rate: n/a",
            ]
        )
    elif event.get("event") == "exit":
        lines.extend(
            [
                f"exit_time: {event.get('exit_time')}",
                f"exit_reason: {event.get('exit_reason')}",
                f"exit_price: {float(event.get('exit_price', np.nan)):.8g}",
                f"pnl_usdt: {float(event.get('pnl_usdt', np.nan)):.4f}",
                f"leveraged_net_return: {float(event.get('leveraged_net_return', np.nan)):.4%}",
            ]
        )
    return "\n".join(lines)


def notify_events(config: dict[str, Any], state: dict[str, Any], tick_time: pd.Timestamp, events: list[dict[str, Any]]) -> tuple[int, str | None]:
    webhook_url = resolve_webhook_url(config)
    if not webhook_url or not events:
        return 0, None
    timeout = float(config.get("notifications", {}).get("timeout_seconds", 10.0))
    sent = 0
    try:
        for event in events:
            payload = {"msgtype": "text", "text": {"content": format_event_message(event, state, tick_time)}}
            response = requests.post(webhook_url, json=payload, timeout=timeout)
            response.raise_for_status()
            sent += 1
    except Exception as exc:
        return sent, repr(exc)
    return sent, None


def notify_text(config: dict[str, Any], content: str) -> tuple[bool, str | None]:
    webhook_url = resolve_webhook_url(config)
    if not webhook_url:
        return False, "webhook_not_configured"
    timeout = float(config.get("notifications", {}).get("timeout_seconds", 10.0))
    try:
        response = requests.post(webhook_url, json={"msgtype": "text", "text": {"content": content}}, timeout=timeout)
        response.raise_for_status()
        return True, None
    except Exception as exc:
        return False, repr(exc)


def load_json_optional(path: Path) -> dict[str, Any]:
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None


def tail_text_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:]


def health_report(config: dict[str, Any], max_heartbeat_age_minutes: float, max_tick_age_minutes: float) -> tuple[dict[str, Any], str]:
    _, run_dir = config_paths(config)
    now = utc_now()
    state = load_json_optional(run_dir / "paper_state.json")
    heartbeat = load_json_optional(run_dir / "paper_loop.heartbeat.json")
    last_tick_payload = load_json_optional(run_dir / "last_tick.json").get("last_tick", {})
    backoff = load_json_optional(run_dir / "rate_limit_backoff.json")
    log_tail = tail_text_lines(run_dir / "paper_loop.log", 200)

    hb_ts = parse_timestamp(heartbeat.get("updated_at"))
    tick_ts = parse_timestamp(last_tick_payload.get("tick_time"))
    hb_age = float((now - hb_ts).total_seconds() / 60.0) if hb_ts is not None else None
    tick_age = float((now - tick_ts).total_seconds() / 60.0) if tick_ts is not None else None
    backoff_until = float(backoff.get("backoff_until", 0.0)) if backoff else 0.0
    backoff_remaining = max((backoff_until - now.timestamp()) / 60.0, 0.0) if backoff_until else 0.0

    recent_errors = [line for line in log_tail if " ERROR " in line or "Traceback" in line or "JSONDecodeError" in line]
    recent_warnings = [
        line
        for line in log_tail
        if "UPDATE_FAILED" in line or "RATE_LIMIT" in line or " 418 " in line or " 429 " in line or "Too Many Requests" in line
    ]

    accounts = []
    total_open = 0
    total_trades = 0
    for account in state.get("accounts", []):
        positions = account.get("positions", [])
        trades = account.get("trades", [])
        total_open += len(positions)
        total_trades += len(trades)
        accounts.append(
            {
                "paper_account": account.get("paper_account"),
                "equity_usdt": float(account.get("equity_usdt", 0.0)),
                "open_positions": len(positions),
                "closed_trades": len(trades),
                "open_symbols": [position.get("symbol") for position in positions],
            }
        )

    healthy = True
    reasons: list[str] = []
    if not heartbeat:
        healthy = False
        reasons.append("missing_heartbeat")
    elif hb_age is None or hb_age > max_heartbeat_age_minutes:
        healthy = False
        reasons.append(f"stale_heartbeat_minutes={hb_age}")
    if not last_tick_payload:
        healthy = False
        reasons.append("missing_last_tick")
    elif tick_age is None or tick_age > max_tick_age_minutes:
        healthy = False
        reasons.append(f"stale_last_tick_minutes={tick_age}")
    if recent_errors:
        reasons.append(f"recent_error_lines={len(recent_errors)}")
    if backoff_remaining > 0:
        reasons.append(f"rate_limit_backoff_remaining_minutes={backoff_remaining:.1f}")

    summary = {
        "checked_at": now.isoformat(),
        "healthy": healthy,
        "reasons": reasons,
        "heartbeat_age_minutes": hb_age,
        "last_tick_age_minutes": tick_age,
        "last_tick": last_tick_payload,
        "accounts": accounts,
        "total_open_positions": total_open,
        "total_closed_trades": total_trades,
        "recent_log_errors": recent_errors[-5:],
        "recent_log_warnings": recent_warnings[-5:],
        "rate_limit_backoff_remaining_minutes": backoff_remaining,
    }

    lines = [
        f"TP14 server paper health: {'healthy' if healthy else 'unhealthy'}",
        f"checked_at_utc: {summary['checked_at']}",
        f"heartbeat_age_min: {hb_age:.2f}" if hb_age is not None else "heartbeat_age_min: n/a",
        f"last_tick_age_min: {tick_age:.2f}" if tick_age is not None else "last_tick_age_min: n/a",
        (
            "last_tick: "
            f"signals={last_tick_payload.get('signals')} "
            f"entries={last_tick_payload.get('entries')} "
            f"exits={last_tick_payload.get('exits')}"
        ),
        f"open_positions: {total_open}",
        f"closed_trades: {total_trades}",
    ]
    if reasons:
        lines.append("reasons: " + "; ".join(reasons))
    for account in accounts:
        lines.append(
            f"{account['paper_account']}: equity={account['equity_usdt']:.4f} "
            f"open={account['open_positions']} trades={account['closed_trades']} "
            f"symbols={','.join(str(symbol) for symbol in account['open_symbols'])}"
        )
    return summary, "\n".join(lines)


def acquire_lock(path: Path, stale_seconds: float = 300.0) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if path.exists():
        try:
            payload = load_json(path)
            timestamp = float(payload.get("timestamp", 0.0))
            pid = int(payload.get("pid", 0))
            if now - timestamp < stale_seconds and pid_is_running(pid):
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


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
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
        complete_end = latest_common_complete_15m_end(config, tick_time)
        exits = check_exits(config, state, tick_time)
        signals, meta = compute_signals(config, state, complete_end)
        reverses = apply_reverse_signals(config, state, signals, tick_time)
        entries = open_entries(config, state, signals, tick_time)
        state["last_tick"] = {
            "tick_time": tick_time.isoformat(),
            "complete_15m_end": complete_end.isoformat(),
            "signals": len(signals),
            "entries": len([event for event in reverses + entries if event.get("event") == "entry"]),
            "exits": len([event for event in exits + reverses if event.get("event") == "exit"]),
            **meta,
        }
        events = exits + reverses + entries
        for event in events:
            append_jsonl(run_dir / "events.jsonl", {"tick_time": tick_time.isoformat(), **event})
        write_json(state_path, state)
        notified, webhook_error = notify_events(config, state, tick_time, events)
        last_tick = {"last_tick": state["last_tick"], "signals": signals, "events": events}
        last_tick["last_tick"]["notified"] = notified
        if webhook_error:
            last_tick["last_tick"]["webhook_error"] = webhook_error
            append_log(run_dir / "paper_loop.log", f"{utc_now().isoformat()} WEBHOOK_ERROR {webhook_error}")
        write_json(run_dir / "last_tick.json", last_tick)
        return last_tick
    finally:
        release_lock(lock)


def bootstrap(config: dict[str, Any], workers: int | None = None, force_state: bool = False) -> None:
    seed_archive = Path("bootstrap_seed") / "tp14_seed.zip"
    if not has_signal_data(config) and seed_archive.exists():
        seed_from_archive(config, seed_archive, force_state=force_state)
        return
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


def seed_from_archive(config: dict[str, Any], archive: Path | None = None, force_state: bool = False) -> None:
    archive_path = archive or Path("bootstrap_seed") / "tp14_seed.zip"
    extract_path = archive_path
    temp_path: Path | None = None
    if not extract_path.exists():
        parts = sorted(archive_path.parent.glob(f"{archive_path.name}.part*"))
        if not parts:
            raise FileNotFoundError(f"Missing seed archive: {archive_path}")
        temp_path = archive_path.parent / f"{archive_path.name}.assembled.tmp"
        with temp_path.open("wb") as out:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, out)
        extract_path = temp_path
    try:
        with zipfile.ZipFile(extract_path, "r") as zf:
            zf.extractall(Path("."))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    init_state(config, force=force_state)


def has_signal_data(config: dict[str, Any]) -> bool:
    root, _ = config_paths(config)
    return any(table_path(root, "signal_klines", symbol).exists() for symbol in config["symbols"])


def start(config: dict[str, Any], workers: int | None = None, force_state: bool = False, max_iterations: int = 0) -> None:
    if not has_signal_data(config):
        seed_from_archive(config, force_state=force_state)
    else:
        init_state(config, force=force_state)
    backoff_until = read_backoff_until(config)
    if time.time() < backoff_until:
        _, run_dir = config_paths(config)
        append_log(run_dir / "paper_loop.log", f"{utc_now().isoformat()} START_BACKOFF_SKIP_REFRESH remaining_seconds={backoff_until - time.time():.0f}")
    else:
        rows = update_many(
            config,
            {"signal_klines", "exec_klines", "funding", "oi", "global_acct_ratio", "top_acct_ratio", "top_pos_ratio"},
            2,
            int(workers or config["execution"]["workers"]),
        )
        if is_rate_limited(rows):
            backoff_minutes = float(config["execution"].get("rate_limit_backoff_minutes", 60.0))
            write_backoff_until(config, time.time() + backoff_minutes * 60.0, "binance_418_or_429_start")
    loop(config, max_iterations=max_iterations)


def loop(config: dict[str, Any], max_iterations: int = 0) -> None:
    _, run_dir = config_paths(config)
    lock = run_dir / "paper_loop.lock"
    log_path = run_dir / "paper_loop.log"
    if not acquire_lock(lock, stale_seconds=900.0):
        raise RuntimeError(f"Loop lock is active: {lock}")
    append_log(log_path, f"START {utc_now().isoformat()} pid={os.getpid()}")
    last_state_update = 0.0
    last_funding_update = 0.0
    rate_limit_backoff_until = read_backoff_until(config)
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
            if time.time() < rate_limit_backoff_until:
                remaining = rate_limit_backoff_until - time.time()
                rows = []
                append_log(log_path, f"{utc_now().isoformat()} RATE_LIMIT_BACKOFF remaining_seconds={remaining:.0f}")
            else:
                rows = update_many(config, datasets, 2, int(config["execution"]["workers"]))
                if is_rate_limited(rows):
                    backoff_minutes = float(config["execution"].get("rate_limit_backoff_minutes", 60.0))
                    rate_limit_backoff_until = time.time() + backoff_minutes * 60.0
                    write_backoff_until(config, rate_limit_backoff_until, "binance_418_or_429_loop")
                    append_log(log_path, f"{utc_now().isoformat()} RATE_LIMIT_DETECTED backoff_minutes={backoff_minutes}")
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
    parser.add_argument("command", choices=("bootstrap", "seed", "start", "tick", "loop", "init-state", "health", "notify-text"))
    parser.add_argument("--config", default="config/paper_config.json")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force-state", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--seed-archive", default=None)
    parser.add_argument("--max-heartbeat-age-minutes", type=float, default=5.0)
    parser.add_argument("--max-tick-age-minutes", type=float, default=20.0)
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--no-webhook", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--message", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config))
    if args.command == "bootstrap":
        bootstrap(config, args.workers, args.force_state)
    elif args.command == "seed":
        seed_from_archive(config, Path(args.seed_archive) if args.seed_archive else None, args.force_state)
    elif args.command == "start":
        start(config, args.workers, args.force_state, args.max_iterations)
    elif args.command == "init-state":
        init_state(config, args.force_state)
    elif args.command == "tick":
        result = run_tick(config)
        print(json.dumps(result.get("last_tick", result), ensure_ascii=False, indent=2, default=str))
    elif args.command == "loop":
        loop(config, args.max_iterations)
    elif args.command == "health":
        summary, text = health_report(config, args.max_heartbeat_age_minutes, args.max_tick_age_minutes)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        else:
            print(text)
        if not args.no_webhook and not summary["healthy"]:
            ok, error = notify_text(config, text)
            if not ok:
                print(f"WEBHOOK_ERROR {error}", file=sys.stderr)
        raise SystemExit(0 if args.no_fail or summary["healthy"] else 2)
    elif args.command == "notify-text":
        if not args.message:
            raise SystemExit("--message is required for notify-text")
        ok, error = notify_text(config, args.message)
        if not ok:
            print(f"WEBHOOK_ERROR {error}", file=sys.stderr)
            raise SystemExit(2)
        print("notified=1")


if __name__ == "__main__":
    main()
