from __future__ import annotations

import pickle
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GATE_COLUMN = "OI_Expansion_48"
RESOLVER_COLUMN = "WhalePosition_Retail_Divergence"
LIVE_LEGS: tuple[tuple[str, str], ...] = (
    ("long_p33_inf_pos60_100", "score_lgbm_combo"),
    ("long_p5_8_pos20_40", "score_lgbm_close"),
    ("short_p9_16_pos60_80", "score_not_overextended"),
    ("long_p129_pos80_100", "score_profile"),
)


@lru_cache(maxsize=4)
def load_artifact_file(artifact_path: str) -> dict[str, Any]:
    with Path(artifact_path).open("rb") as handle:
        return pickle.load(handle)


def load_artifact(config: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(config["entry_model"].get("artifact_path", "artifacts/tp14_v2_artifacts.pkl"))
    if not artifact_path.is_absolute():
        artifact_path = Path(__file__).resolve().parent / artifact_path
    return load_artifact_file(str(artifact_path.resolve()))


def normalize_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    raw = out.pop("timestamp") if "timestamp" in out.columns else pd.Series(out.index)
    if pd.api.types.is_numeric_dtype(raw):
        unit = "ms" if pd.to_numeric(raw, errors="coerce").dropna().median() > 10_000_000_000 else "s"
        idx = pd.to_datetime(raw, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(raw, utc=True)
    out.index = pd.DatetimeIndex(idx).tz_convert("UTC")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_csv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return normalize_timestamp_index(pd.read_csv(path))


def table_path(root: Path, dataset: str, symbol: str) -> Path:
    if dataset == "klines":
        return root / "klines" / f"{symbol}.csv"
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
    raise ValueError(dataset)


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
    return base.reindex(base.index.union(target_index)).ffill(limit=limit).reindex(target_index)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(3, int(window * 0.5)))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.replace(0, np.nan).pct_change(periods=periods)


def safe_log(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def coalesce_columns(frame: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    present = [col for col in (target, *aliases) if col in frame.columns]
    if not present:
        return frame
    out = frame.copy()
    out[target] = out[present].bfill(axis=1).iloc[:, 0]
    return out.drop(columns=[col for col in aliases if col in out.columns])


def build_feature_frame(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    k = read_csv_frame(table_path(root, "klines", symbol))
    if k.empty:
        return pd.DataFrame()
    k = k.loc[(k.index >= start) & (k.index < end)]
    base = resample_ohlcv(k, "15min")
    if base.empty:
        return pd.DataFrame()

    specs = {
        "funding": ("FundingRate", ("funding_rate", "fundingRate")),
        "oi": ("OpenInterest", ("oi", "sumOpenInterest")),
        "global_acct_ratio": ("LS_Ratio", ("ratio", "longShortRatio")),
        "top_acct_ratio": ("TopAccount_LS_Ratio", ("ratio", "longShortRatio")),
        "top_pos_ratio": ("TopPosition_LS_Ratio", ("ratio", "longShortRatio")),
    }
    for dataset, (target, aliases) in specs.items():
        frame = read_csv_frame(table_path(root, dataset, symbol))
        if dataset == "oi":
            frame = coalesce_columns(frame, "OpenInterestValue", ("oi_value", "sumOpenInterestValue"))
        frame = coalesce_columns(frame, target, aliases)
        keep = [target]
        if dataset == "oi":
            keep.append("OpenInterestValue")
        frame = frame[[col for col in keep if col in frame.columns]] if not frame.empty else frame
        if not frame.empty:
            frame = frame.loc[(frame.index >= start) & (frame.index < end)]
        aligned = resample_last(frame, base.index)
        if not aligned.empty:
            base = base.join(aligned, how="left")

    close = pd.to_numeric(base["close"], errors="coerce")
    for span in (20, 50, 100, 200):
        ema = close.ewm(span=span, adjust=False, min_periods=max(3, int(span * 0.5))).mean()
        base[f"EMA_{span}"] = ema
        base[f"Close_EMA{span}_Spread"] = close / ema - 1.0
    for fast, slow in ((20, 50), (50, 100), (100, 200), (20, 100), (20, 200)):
        spread = base[f"EMA_{fast}"] / base[f"EMA_{slow}"] - 1.0
        base[f"EMA_{fast}_{slow}_Spread"] = spread
        base[f"EMA_{fast}_{slow}_Cross"] = np.sign(spread)
    for window in (4, 16, 48, 96):
        base[f"ROC_{window}"] = close.pct_change(window)
    base["Trend_EMA_Breadth"] = (base[[f"Close_EMA{s}_Spread" for s in (20, 50, 100, 200)]] > 0).mean(axis=1)
    base["Trend_ROC_Breadth"] = (base[[f"ROC_{w}" for w in (4, 16, 48, 96)]] > 0).mean(axis=1)

    if "FundingRate" not in base.columns:
        base["FundingRate"] = np.nan
    for window in (32, 96, 192):
        base[f"FR_ZScore_{window}"] = rolling_zscore(base["FundingRate"], window)
    base["FR_ZScore"] = base["FR_ZScore_96"]

    if "OpenInterest" not in base.columns:
        base["OpenInterest"] = np.nan
    for window in (4, 16, 48, 96):
        delta = safe_pct_change(base["OpenInterest"], window)
        base[f"OI_Delta_{window}"] = delta
        base[f"OI_Delta_ZScore_{window}"] = rolling_zscore(delta, window)
        base[f"OI_Expansion_{window}"] = delta.clip(lower=0)
        base[f"OI_Crush_{window}"] = (-delta).clip(lower=0)
    base["OI_Delta"] = base["OI_Delta_16"]

    for col in ("LS_Ratio", "TopAccount_LS_Ratio", "TopPosition_LS_Ratio"):
        if col not in base.columns:
            base[col] = np.nan
        base[f"{col}_ZScore_96"] = rolling_zscore(base[col], 96)
    base["WhaleAccount_Retail_Divergence"] = safe_log(base["TopAccount_LS_Ratio"]) - safe_log(base["LS_Ratio"])
    base["WhalePosition_Retail_Divergence"] = safe_log(base["TopPosition_LS_Ratio"]) - safe_log(base["LS_Ratio"])
    base["Log_LS_Ratio"] = safe_log(base["LS_Ratio"])
    base["RetailShortCrowding"] = (-base["Log_LS_Ratio"]).clip(lower=0)
    base["RetailLongCrowding"] = base["Log_LS_Ratio"].clip(lower=0)

    if "taker_buy_volume" in base.columns:
        base["TakerBuyVolumeShare"] = base["taker_buy_volume"] / base["volume"].replace(0, np.nan)
        base["TakerBuyImbalance"] = base["TakerBuyVolumeShare"] * 2.0 - 1.0
    else:
        base["TakerBuyVolumeShare"] = np.nan
        base["TakerBuyImbalance"] = np.nan
    if "taker_buy_quote_volume" in base.columns and "quote_volume" in base.columns:
        base["TakerBuyQuoteShare"] = base["taker_buy_quote_volume"] / base["quote_volume"].replace(0, np.nan)
    base["Abs_Taker_Imbalance"] = base["TakerBuyImbalance"].abs()

    body_high = base[["open", "close"]].max(axis=1)
    body_low = base[["open", "close"]].min(axis=1)
    bar_range = (base["high"] - base["low"]).replace(0, np.nan)
    base["UpperWick_RangeShare"] = (base["high"] - body_high).clip(lower=0) / bar_range
    base["LowerWick_RangeShare"] = (body_low - base["low"]).clip(lower=0) / bar_range
    base["MaxWick_RangeShare"] = base[["UpperWick_RangeShare", "LowerWick_RangeShare"]].max(axis=1)
    base["Body_RangeShare"] = (body_high - body_low) / bar_range
    base["Range_Pct"] = bar_range / close.replace(0, np.nan)
    base["Volume_ZScore_96"] = rolling_zscore(base["volume"], 96)
    base["Volume_ZScore"] = base["Volume_ZScore_96"]
    base["Range_ZScore_96"] = rolling_zscore(base["Range_Pct"], 96)
    base["Range_ZScore"] = base["Range_ZScore_96"]
    volume_pressure = base["Volume_ZScore"].clip(lower=0)
    range_pressure = base["Range_ZScore"].clip(lower=0)
    base["UpperWick_VolumeSpike_Score"] = base["UpperWick_RangeShare"] * volume_pressure * range_pressure
    base["LowerWick_VolumeSpike_Score"] = base["LowerWick_RangeShare"] * volume_pressure * range_pressure
    base["WickVolumeSpikeFlag"] = (base["MaxWick_RangeShare"] >= 0.55) & (base["Volume_ZScore"] >= 2.0) & (base["Range_ZScore"] >= 1.0)
    base["LiquidationProxy_Max"] = base[["UpperWick_VolumeSpike_Score", "LowerWick_VolumeSpike_Score"]].max(axis=1)
    base["LiquidationProxy_ZScore"] = rolling_zscore(base["LiquidationProxy_Max"], 96)
    components = [
        base["OI_Expansion_96"].rank(pct=True),
        base["Volume_ZScore"].clip(lower=0).rank(pct=True),
        base["Range_ZScore"].clip(lower=0).rank(pct=True),
        base["LiquidationProxy_Max"].clip(lower=0).rank(pct=True),
    ]
    base["VolRegimeScore"] = pd.concat(components, axis=1).mean(axis=1)
    return base.replace([np.inf, -np.inf], np.nan)


def resolver_raw(frame: pd.DataFrame) -> pd.Series:
    return -pd.to_numeric(frame[RESOLVER_COLUMN], errors="coerce")


def add_candidate_features(frame: pd.DataFrame, threshold: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    raw = resolver_raw(out)
    gate = pd.to_numeric(out[GATE_COLUMN], errors="coerce")
    out["candidate_raw"] = raw
    out["candidate_gate"] = gate
    out["candidate_direction"] = np.sign(raw).replace(0.0, np.nan)
    out["raw_abs"] = raw.abs()
    out["gate_strength"] = gate / abs(float(threshold["gate_threshold"])) if threshold.get("gate_threshold") else np.nan
    out["raw_strength"] = raw.abs() / abs(float(threshold["resolver_threshold"])) if threshold.get("resolver_threshold") else np.nan
    for col in ("OI_Expansion_48", "OI_Delta", "FR_ZScore", "WhalePosition_Retail_Divergence", "TakerBuyImbalance", "LS_Ratio", "Volume_ZScore", "Range_Pct", "ROC_16"):
        if col in out.columns:
            value = pd.to_numeric(out[col], errors="coerce")
            for lag in (1, 4, 16):
                out[f"{col}_slope_{lag}"] = value - value.shift(lag)
            out[f"{col}_accel_4_16"] = out[f"{col}_slope_4"] - out[f"{col}_slope_16"]
    close = pd.to_numeric(out["close"], errors="coerce")
    for window in (96, 192):
        rolling_high = pd.to_numeric(out["high"], errors="coerce").rolling(window, min_periods=max(8, window // 4)).max()
        rolling_low = pd.to_numeric(out["low"], errors="coerce").rolling(window, min_periods=max(8, window // 4)).min()
        denom = (rolling_high - rolling_low).replace(0.0, np.nan)
        out[f"close_pos_{window}"] = (close - rolling_low) / denom
        out[f"dist_rolling_high_{window}"] = rolling_high / close - 1.0
        out[f"dist_rolling_low_{window}"] = close / rolling_low - 1.0
    for ema in (20, 50, 100, 200):
        col = f"EMA_{ema}"
        if col in out.columns:
            ema_value = pd.to_numeric(out[col], errors="coerce")
            out[f"EMA_{ema}_slope_4"] = ema_value.pct_change(4)
            out[f"EMA_{ema}_slope_16"] = ema_value.pct_change(16)
    out["range_stretch_96"] = out["Range_Pct"] / out["Range_Pct"].rolling(96, min_periods=24).median().replace(0.0, np.nan)
    if "quote_volume" in out.columns:
        qv = pd.to_numeric(out["quote_volume"], errors="coerce")
        out["quote_volume_z_96"] = (qv - qv.rolling(96, min_periods=24).mean()) / qv.rolling(96, min_periods=24).std(ddof=0)
    base = (
        np.isfinite(gate)
        & np.isfinite(raw)
        & (gate >= float(threshold["gate_threshold"]))
        & (raw.abs() >= float(threshold["resolver_threshold"]))
    )
    persist = np.zeros(len(out), dtype=float)
    current = 0
    for i, is_base in enumerate(base.to_numpy(dtype=bool)):
        current = current + 1 if is_base else 0
        persist[i] = current
    out["threshold_persist_bars"] = persist
    prior_base = base.shift(1, fill_value=False)
    out["just_crossed_threshold"] = base.astype(int) - prior_base.astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def apply_cooldown_positions(mask: np.ndarray, cooldown_bars: int) -> np.ndarray:
    kept: list[int] = []
    last = -cooldown_bars - 1
    for pos in np.flatnonzero(mask):
        if pos - last > cooldown_bars:
            kept.append(int(pos))
            last = int(pos)
    return np.asarray(kept, dtype=int)


def candidate_positions(features: pd.DataFrame, threshold: dict[str, float], cooldown_bars: int) -> np.ndarray:
    gate = pd.to_numeric(features[GATE_COLUMN], errors="coerce").to_numpy(dtype=float)
    raw = pd.to_numeric(features["candidate_raw"], errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(gate)
        & np.isfinite(raw)
        & (gate >= float(threshold["gate_threshold"]))
        & (np.abs(raw) >= float(threshold["resolver_threshold"]))
        & (raw != 0.0)
    )
    return apply_cooldown_positions(mask, cooldown_bars)


def event_from_row(symbol: str, features: pd.DataFrame, pos: int, entry_pos: int, threshold: dict[str, float]) -> dict[str, Any] | None:
    row = features.iloc[pos]
    direction = float(np.sign(row["candidate_raw"]))
    if direction == 0.0 or not np.isfinite(direction):
        return None
    signal_bar_start = pd.Timestamp(features.index[pos])
    entry_bar_start = pd.Timestamp(features.index[entry_pos])
    close_pos_96 = float(row.get("close_pos_96", np.nan))
    close_pos_192 = float(row.get("close_pos_192", np.nan))
    event: dict[str, Any] = {
        "symbol": symbol,
        "signal_bar_start": signal_bar_start,
        "signal_time": signal_bar_start + pd.Timedelta(minutes=15),
        "feature_time": signal_bar_start,
        "entry_bar_start": entry_bar_start,
        "entry_bar_end": entry_bar_start + pd.Timedelta(minutes=15),
        "side": "long" if direction > 0 else "short",
        "direction": direction,
        "candidate_gate_threshold": float(threshold["gate_threshold"]),
        "candidate_resolver_threshold": float(threshold["resolver_threshold"]),
        "directional_close_pos_96": close_pos_96 if direction > 0 else 1.0 - close_pos_96,
        "directional_close_pos_192": close_pos_192 if direction > 0 else 1.0 - close_pos_192,
        "abs_ROC_16": abs(float(row.get("ROC_16", np.nan))),
        "abs_Range_ZScore": abs(float(row.get("Range_ZScore", np.nan))),
        "abs_Volume_ZScore": abs(float(row.get("Volume_ZScore", np.nan))),
        "trend_align_ema20": bool(direction * float(row.get("Close_EMA20_Spread", np.nan)) > 0.0),
        "trend_align_roc16": bool(direction * float(row.get("ROC_16", np.nan)) > 0.0),
        "theoretical_entry_close": float(features["close"].iloc[entry_pos]),
        "signal_gate": float(row.get(GATE_COLUMN, np.nan)),
        "signal_raw": float(row.get("candidate_raw", np.nan)),
    }
    for col in features.columns:
        if col in event:
            continue
        if col.startswith(("EMA_", "close_pos_", "dist_rolling_")) or col.endswith(("_slope_1", "_slope_4", "_slope_16", "_accel_4_16")):
            event[col] = row.get(col, np.nan)
        elif col in {
            "gate_strength",
            "raw_strength",
            "threshold_persist_bars",
            "just_crossed_threshold",
            "range_stretch_96",
            "quote_volume_z_96",
            "candidate_raw",
            "candidate_gate",
            "raw_abs",
        }:
            event[col] = row.get(col, np.nan)
    return event


def add_sequence_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True)
    out = out.sort_values(["symbol", "signal_time", "side"]).reset_index(drop=True)
    out["persist_log1p"] = np.log1p(pd.to_numeric(out.get("threshold_persist_bars", 0.0), errors="coerce").fillna(0.0))
    out["fresh_cross"] = pd.to_numeric(out.get("just_crossed_threshold", 0.0), errors="coerce").fillna(0.0).gt(0).astype(float)
    out["directional_stretch_x_persist"] = pd.to_numeric(out.get("directional_close_pos_96", 0.5), errors="coerce").fillna(0.5) * out["persist_log1p"]
    for (symbol, side), idx in out.groupby(["symbol", "side"], sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        times = out.loc[loc, "signal_time"].astype("int64").to_numpy()
        out.loc[loc, "same_side_seq_count"] = np.arange(len(loc), dtype=float)
        prev = np.empty(len(loc), dtype="float64")
        prev[:] = np.nan
        if len(loc) > 1:
            prev[1:] = times[:-1]
        out.loc[loc, "hours_since_same_side"] = pd.Series((times - prev) / 1e9 / 3600.0).fillna(9999.0).to_numpy()
        for hours in (4, 12, 24, 48, 72):
            start = times - int(hours * 3600 * 1e9)
            counts = np.searchsorted(times, start, side="left")
            out.loc[loc, f"prior_same_side_{hours}h"] = np.arange(len(loc)) - counts
    for symbol, idx in out.groupby("symbol", sort=False).groups.items():
        symbol_loc = np.asarray(list(idx), dtype=int)
        part = out.loc[symbol_loc, ["signal_time", "side"]].copy()
        for side_name in ("long", "short"):
            this_loc = symbol_loc[part["side"].eq(side_name).to_numpy()]
            opp_times = part.loc[~part["side"].eq(side_name), "signal_time"].astype("int64").to_numpy()
            this_times = out.loc[this_loc, "signal_time"].astype("int64").to_numpy()
            if len(this_loc) == 0:
                continue
            if len(opp_times) == 0:
                out.loc[this_loc, "hours_since_opposite_side"] = 9999.0
                out.loc[this_loc, "prior_opposite_side_24h"] = 0.0
                out.loc[this_loc, "prior_opposite_side_48h"] = 0.0
                continue
            prev_idx = np.searchsorted(opp_times, this_times, side="left") - 1
            prev_time = np.where(prev_idx >= 0, opp_times[np.maximum(prev_idx, 0)], np.nan)
            out.loc[this_loc, "hours_since_opposite_side"] = pd.Series((this_times - prev_time) / 1e9 / 3600.0).fillna(9999.0).to_numpy()
            for hours in (24, 48):
                start = this_times - int(hours * 3600 * 1e9)
                left = np.searchsorted(opp_times, start, side="left")
                right = np.searchsorted(opp_times, this_times, side="left")
                out.loc[this_loc, f"prior_opposite_side_{hours}h"] = right - left
    out["same_side_burst_24h"] = pd.to_numeric(out.get("prior_same_side_24h", 0.0), errors="coerce").fillna(0.0) / (
        1.0 + pd.to_numeric(out.get("hours_since_same_side", 9999.0), errors="coerce").fillna(9999.0).clip(lower=0.0)
    )
    out["isolated_same_side_24h"] = pd.to_numeric(out.get("prior_same_side_24h", 0.0), errors="coerce").fillna(0.0).eq(0).astype(float)
    out["reversal_pressure_24h"] = pd.to_numeric(out.get("prior_opposite_side_24h", 0.0), errors="coerce").fillna(0.0) - pd.to_numeric(
        out.get("prior_same_side_24h", 0.0), errors="coerce"
    ).fillna(0.0)
    return out


def apply_profiles(events: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    if events.empty:
        return events
    defaults = {
        "profile_events": 0.0,
        "profile_win_rate": 0.5,
        "profile_avg_mfe": 0.0,
        "profile_avg_mae": 0.0,
        "profile_avg_path_score": 0.0,
        "profile_viable_rate": 0.0,
    }
    profiles = artifact.get("profiles", {})
    rows = []
    for _, row in events.iterrows():
        key = f"{row['symbol']}|{row['side']}"
        profile = {**defaults, **profiles.get(key, {})}
        rows.append(profile)
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def cdf(values: pd.Series, reference: np.ndarray, default: float = 0.5) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(reference) == 0:
        return pd.Series(default, index=series.index, dtype=float)
    raw = series.to_numpy(dtype=float)
    out = np.full(len(raw), default, dtype=float)
    ok = np.isfinite(raw)
    out[ok] = np.searchsorted(reference, raw[ok], side="right") / float(len(reference))
    return pd.Series(out, index=series.index)


def add_scores(events: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    refs = {key: np.asarray(value, dtype=float) for key, value in artifact.get("cdf_refs", {}).items()}

    def ref(col: str) -> pd.Series:
        return cdf(pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index), refs.get(col, np.asarray([], dtype=float)))

    out["score_raw_strength"] = 0.35 * ref("gate_strength") + 0.35 * ref("raw_strength") + 0.10 * ref("profile_avg_path_score") + 0.10 * ref("profile_viable_rate") - 0.05 * ref("abs_Range_ZScore") - 0.05 * ref("directional_close_pos_96")
    out["score_profile"] = 0.35 * ref("profile_avg_path_score") + 0.25 * ref("profile_viable_rate") + 0.20 * ref("profile_win_rate") + 0.10 * ref("gate_strength") + 0.10 * ref("raw_strength") - 0.20 * ref("profile_avg_mae")
    out["score_not_overextended"] = 0.35 * ref("gate_strength") + 0.25 * ref("raw_strength") - 0.20 * ref("directional_close_pos_96") - 0.10 * ref("abs_Range_ZScore") - 0.10 * ref("abs_Volume_ZScore")
    out["score_breakout_follow"] = 0.25 * ref("gate_strength") + 0.25 * ref("raw_strength") + 0.20 * ref("directional_close_pos_96") + 0.10 * ref("abs_Volume_ZScore") + 0.10 * ref("threshold_persist_bars") + 0.10 * out.get("trend_align_roc16", pd.Series(False, index=out.index)).fillna(False).astype(float)

    models = artifact.get("models", {})
    feature_cols = list(artifact.get("feature_cols", []))
    medians = artifact.get("feature_medians", {})
    if feature_cols and models:
        matrix = out.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            matrix[col] = matrix[col].fillna(float(medians.get(col, 0.0)))
        for name, model in models.items():
            if model is None:
                out[f"score_lgbm_{name}"] = np.nan
                continue
            if isinstance(model, dict) and model.get("type") == "lightgbm_booster":
                raw_pred = model["booster"].predict(matrix)
            elif name in ("viable", "close"):
                raw_pred = model.predict_proba(matrix)[:, 1]
            else:
                raw_pred = model.predict(matrix)
            out[f"score_lgbm_{name}"] = cdf(pd.Series(raw_pred, index=out.index), refs.get(f"model_{name}", np.asarray([], dtype=float)))
        out["score_lgbm_combo"] = 0.40 * out["score_lgbm_viable"] + 0.25 * out["score_lgbm_close"] + 0.25 * out["score_lgbm_path"] + 0.10 * ref("profile_avg_path_score") - 0.10 * ref("abs_Range_ZScore")
    else:
        for col in ("score_lgbm_viable", "score_lgbm_close", "score_lgbm_path", "score_lgbm_combo"):
            out[col] = np.nan
    return out


def pattern_mask(events: pd.DataFrame, rule_name: str) -> pd.Series:
    specs = {
        "long_p33_inf_pos60_100": ("long", 33, np.inf, 0.60, 1.00),
        "long_p5_8_pos20_40": ("long", 5, 8, 0.20, 0.40),
        "long_p129_pos80_100": ("long", 129, np.inf, 0.80, 1.00),
        "short_p9_16_pos60_80": ("short", 9, 16, 0.60, 0.80),
    }
    side, min_persist, max_persist, min_pos, max_pos = specs[rule_name]
    persist = pd.to_numeric(events.get("threshold_persist_bars", 0.0), errors="coerce").fillna(0.0)
    dpos = pd.to_numeric(events.get("directional_close_pos_96", 0.5), errors="coerce").fillna(0.5)
    return events["side"].eq(side) & persist.ge(min_persist) & persist.le(max_persist) & dpos.ge(min_pos) & dpos.le(max_pos)


def strategy_mask(events: pd.DataFrame, strategy: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=events.index)
    if strategy.get("require_ema20_align", False):
        mask &= events.get("trend_align_ema20", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    profile = pd.to_numeric(events.get("profile_viable_rate", np.nan), errors="coerce")
    mode = strategy.get("profile_filter", "none")
    if mode == "outside_034_045":
        mask &= profile.le(0.34) | profile.ge(0.45)
    elif mode == "outside_034_050":
        mask &= profile.le(0.34) | profile.ge(0.50)
    elif mode == "side_specific":
        long_mask = events["side"].eq("long") & (profile.le(0.34) | profile.ge(0.50))
        short_mask = events["side"].eq("short") & (profile.le(0.34) | (profile.ge(0.35) & profile.le(0.40)) | profile.ge(0.50))
        mask &= long_mask | short_mask
    return mask


def compute_v2_signals(config: dict[str, Any], state: dict[str, Any], complete_end: pd.Timestamp) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = load_artifact(config)
    root = Path(config["paths"]["data_root"])
    train_start = pd.Timestamp(artifact["training_start"])
    train_end = pd.Timestamp(artifact["training_end"])
    signal_lag = int(config.get("entry_model", {}).get("signal_lag_bars", 0))
    cooldown_bars = int(config.get("entry_model", {}).get("cooldown_bars", artifact.get("cooldown_bars", 16)))
    target_signal_time = complete_end - pd.Timedelta(minutes=15 * signal_lag)
    events: list[dict[str, Any]] = []
    threshold_count = 0
    for symbol in config["symbols"]:
        threshold = artifact.get("thresholds", {}).get(symbol)
        if not threshold:
            continue
        threshold_count += 1
        features = build_feature_frame(root, symbol, train_start, complete_end)
        if features.empty or len(features) <= signal_lag:
            continue
        features = add_candidate_features(features, threshold)
        positions = candidate_positions(features, threshold, cooldown_bars)
        for pos in positions:
            entry_pos = pos + signal_lag
            if entry_pos >= len(features):
                continue
            event = event_from_row(symbol, features, int(pos), int(entry_pos), threshold)
            if event is not None:
                events.append(event)
    frame = pd.DataFrame(events)
    if frame.empty:
        return [], {
            "threshold_symbols": threshold_count,
            "training_start": train_start.isoformat(),
            "training_end": train_end.isoformat(),
            "threshold_mode": "tp14_v2_rankfixed_artifact",
            "strategy_mode": artifact.get("strategy_mode", "deployable_rankfixed"),
        }
    frame = add_sequence_features(frame)
    frame = apply_profiles(frame, artifact)
    frame = add_scores(frame, artifact)
    frame = frame[pd.to_datetime(frame["signal_time"], utc=True).eq(target_signal_time)].copy()
    if frame.empty:
        return [], {
            "threshold_symbols": threshold_count,
            "training_start": train_start.isoformat(),
            "training_end": train_end.isoformat(),
            "threshold_mode": "tp14_v2_rankfixed_artifact",
            "strategy_mode": artifact.get("strategy_mode", "deployable_rankfixed"),
        }

    selected: list[pd.DataFrame] = []
    leg_thresholds = artifact.get("leg_thresholds", {})
    for rule_name, score_col in LIVE_LEGS:
        threshold = float(leg_thresholds.get(f"{rule_name}|{score_col}", np.inf))
        if not np.isfinite(threshold) or score_col not in frame.columns:
            continue
        mask = pattern_mask(frame, rule_name) & pd.to_numeric(frame[score_col], errors="coerce").ge(threshold)
        part = frame[mask].copy()
        if part.empty:
            continue
        part["leg"] = rule_name
        part["leg_score_col"] = score_col
        part["leg_threshold"] = threshold
        part["score"] = pd.to_numeric(part[score_col], errors="coerce")
        selected.append(part)
    if not selected:
        return [], {
            "threshold_symbols": threshold_count,
            "training_start": train_start.isoformat(),
            "training_end": train_end.isoformat(),
            "threshold_mode": "tp14_v2_rankfixed_artifact",
            "strategy_mode": artifact.get("strategy_mode", "deployable_rankfixed"),
        }
    base = pd.concat(selected, ignore_index=True).drop_duplicates(["symbol", "signal_time", "side"], keep="first")
    signals: list[dict[str, Any]] = []
    strategies = artifact.get("strategies", [])
    for strategy in strategies:
        part = base[strategy_mask(base, strategy)].copy()
        for _, row in part.iterrows():
            signals.append(
                {
                    "strategy_id": strategy["strategy_id"],
                    "strategy_name": strategy.get("strategy_name", strategy["strategy_id"]),
                    "symbol": row["symbol"],
                    "direction": float(row["direction"]),
                    "side": row["side"],
                    "entry_bar_start": pd.Timestamp(row["entry_bar_start"]).isoformat(),
                    "entry_bar_end": pd.Timestamp(row["entry_bar_end"]).isoformat(),
                    "signal_bar_start": pd.Timestamp(row["signal_bar_start"]).isoformat(),
                    "signal_bar_end": pd.Timestamp(row["signal_time"]).isoformat(),
                    "signal_gate": float(row.get("signal_gate", np.nan)),
                    "signal_raw": float(row.get("signal_raw", np.nan)),
                    "gate_threshold": float(row.get("candidate_gate_threshold", np.nan)),
                    "resolver_threshold": float(row.get("candidate_resolver_threshold", np.nan)),
                    "extra_entry_gate": np.nan,
                    "theoretical_entry_close": float(row.get("theoretical_entry_close", np.nan)),
                    "leg": row.get("leg", ""),
                    "leg_score_col": row.get("leg_score_col", ""),
                    "score": float(row.get("score", np.nan)),
                    "score_lgbm_combo": float(row.get("score_lgbm_combo", np.nan)),
                    "score_profile": float(row.get("score_profile", np.nan)),
                    "score_lgbm_close": float(row.get("score_lgbm_close", np.nan)),
                    "score_not_overextended": float(row.get("score_not_overextended", np.nan)),
                    "profile_avg_mfe": float(row.get("profile_avg_mfe", np.nan)),
                    "profile_avg_mae": float(row.get("profile_avg_mae", np.nan)),
                    "raw_strength": float(row.get("raw_strength", np.nan)),
                    "gate_strength": float(row.get("gate_strength", np.nan)),
                    "profile_viable_rate": float(row.get("profile_viable_rate", np.nan)),
                }
            )
    return signals, {
        "threshold_symbols": threshold_count,
        "training_start": train_start.isoformat(),
        "training_end": train_end.isoformat(),
        "threshold_mode": "tp14_v2_rankfixed_artifact",
        "strategy_mode": artifact.get("strategy_mode", "deployable_rankfixed"),
        "candidate_events_at_signal_time": int(len(frame)),
        "base_selected_events": int(len(base)),
    }
