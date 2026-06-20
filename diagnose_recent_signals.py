from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tp14_v2_live_core as core


def latest_15m_end(root: Path, symbols: list[str]) -> pd.Timestamp:
    ends: list[pd.Timestamp] = []
    for symbol in symbols:
        path = root / "klines" / f"{symbol}.csv"
        if not path.exists():
            continue
        frame = core.read_csv_frame(path)
        if not frame.empty:
            ends.append(pd.Timestamp(frame.index.max()))
    if not ends:
        raise RuntimeError("No local klines data found.")
    return max(ends).floor("15min") + pd.Timedelta(minutes=15)


def diagnose(config: dict, hours: float) -> None:
    root = Path(config["paths"]["data_root"])
    artifact = core.load_artifact(config)
    train_start = pd.Timestamp(artifact["training_start"])
    end = latest_15m_end(root, list(config["symbols"]))
    start = end - pd.Timedelta(hours=float(hours))
    cooldown = int(config["entry_model"].get("cooldown_bars", artifact.get("cooldown_bars", 16)))

    events: list[dict] = []
    per_symbol: list[tuple[str, int, int, str]] = []
    for symbol in config["symbols"]:
        threshold = artifact.get("thresholds", {}).get(symbol)
        if not threshold:
            per_symbol.append((symbol, 0, 0, "missing_threshold"))
            continue
        features = core.build_feature_frame(root, symbol, train_start, end)
        if features.empty:
            per_symbol.append((symbol, 0, 0, "empty_features"))
            continue
        features = core.add_candidate_features(features, threshold)
        positions = core.candidate_positions(features, threshold, cooldown)
        in_window = 0
        for pos in positions:
            pos = int(pos)
            if pos >= len(features):
                continue
            event = core.event_from_row(symbol, features, pos, pos, threshold)
            if event is None:
                continue
            signal_time = pd.Timestamp(event["signal_time"])
            if start <= signal_time <= end:
                events.append(event)
                in_window += 1
        per_symbol.append((symbol, int(len(positions)), in_window, "ok"))

    raw = pd.DataFrame(events)
    print(f"window_utc={start.isoformat()} -> {end.isoformat()}")
    print(f"symbols={len(config['symbols'])}")
    print(f"stage_1_gate_resolver_candidates={len(raw)}")
    if raw.empty:
        print("final_signals=0")
        print("top_symbols_by_all_period_threshold_candidates:")
        for row in sorted(per_symbol, key=lambda item: item[1], reverse=True)[:20]:
            print(row)
        return

    raw = core.add_sequence_features(raw)
    raw = core.apply_profiles(raw, artifact)
    raw = core.add_scores(raw, artifact)
    print(f"stage_1_symbols={raw['symbol'].nunique()}")
    print(f"stage_1_side_counts={raw['side'].value_counts().to_dict()}")

    selected: list[pd.DataFrame] = []
    for rule_name, score_col in core.LIVE_LEGS:
        threshold = float(artifact.get("leg_thresholds", {}).get(f"{rule_name}|{score_col}", np.inf))
        if np.isfinite(threshold) and score_col in raw.columns:
            pattern = core.pattern_mask(raw, rule_name)
            passed = pattern & pd.to_numeric(raw[score_col], errors="coerce").ge(threshold)
        else:
            pattern = pd.Series(False, index=raw.index)
            passed = pd.Series(False, index=raw.index)
        print(
            "leg",
            rule_name,
            "score_col",
            score_col,
            "threshold",
            threshold,
            "pattern_pool",
            int(pattern.sum()),
            "pass_score",
            int(passed.sum()),
        )
        if passed.any():
            part = raw[passed].copy()
            part["leg"] = rule_name
            part["score_col"] = score_col
            selected.append(part)

    base = pd.concat(selected, ignore_index=True).drop_duplicates(["symbol", "signal_time", "side"], keep="first") if selected else pd.DataFrame()
    print(f"stage_2_live_leg_selected={len(base)}")
    if not base.empty:
        cols = [
            "signal_time",
            "symbol",
            "side",
            "leg",
            "score_col",
            "score_lgbm_combo",
            "score_profile",
            "profile_viable_rate",
            "threshold_persist_bars",
            "directional_close_pos_96",
        ]
        print(base[[col for col in cols if col in base.columns]].sort_values("signal_time").tail(50).to_string(index=False))

    for strategy in artifact.get("strategies", []):
        signals = int(core.strategy_mask(base, strategy).sum()) if not base.empty else 0
        print("final_strategy", strategy.get("strategy_id"), "signals", signals)

    if base.empty:
        cols = [
            "signal_time",
            "symbol",
            "side",
            "candidate_gate",
            "candidate_gate_threshold",
            "candidate_raw",
            "candidate_resolver_threshold",
            "threshold_persist_bars",
            "directional_close_pos_96",
            "trend_align_ema20",
            "profile_viable_rate",
            "score_not_overextended",
            "score_lgbm_combo",
            "score_profile",
        ]
        for col in cols:
            if col not in raw.columns:
                raw[col] = np.nan
        print("stage_1_candidates_tail:")
        print(raw[cols].sort_values(["signal_time", "symbol"]).tail(50).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose recent TP14 V2 signal filtering stages.")
    parser.add_argument("--config", default="config/paper_config.json")
    parser.add_argument("--hours", type=float, default=24.0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    diagnose(config, args.hours)


if __name__ == "__main__":
    sys.exit(main())
