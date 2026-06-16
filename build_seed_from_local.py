from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        raw = out.pop("timestamp")
    else:
        raw = pd.Series(out.index)
    if pd.api.types.is_datetime64_any_dtype(raw):
        idx = pd.to_datetime(raw, utc=True)
    else:
        numeric = pd.to_numeric(raw, errors="coerce")
        median = numeric.dropna().median()
        unit = "ms" if median > 10_000_000_000 else "s"
        idx = pd.to_datetime(numeric, unit=unit, utc=True)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return normalize_timestamp_index(read_table(path))


def write_csv_with_ms_timestamp(df: pd.DataFrame, path: Path) -> int:
    if df.empty:
        return 0
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    timestamps = [int(pd.Timestamp(ts).timestamp() * 1000) for ts in out.index]
    out.insert(0, "timestamp", timestamps)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return int(len(out))


def coalesce_columns(frame: pd.DataFrame, target: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    present = [col for col in (target, *aliases) if col in frame.columns]
    if not present:
        return frame
    out = frame.copy()
    out[target] = out[present].bfill(axis=1).iloc[:, 0]
    return out.drop(columns=[col for col in aliases if col in out.columns])


def standardize_columns(df: pd.DataFrame, dest_parts: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    dataset = "/".join(dest_parts)
    if dataset == "data/funding":
        out = coalesce_columns(out, "FundingRate", ("funding_rate",))
        keep = ["FundingRate"]
    elif dataset == "data/market_state_hist/oi":
        out = coalesce_columns(out, "OpenInterest", ("oi",))
        out = coalesce_columns(out, "OpenInterestValue", ("oi_value",))
        keep = ["OpenInterest", "OpenInterestValue"]
    elif dataset == "data/market_state_hist/global_acct_ratio":
        out = coalesce_columns(out, "LS_Ratio", ("ratio",))
        keep = ["LS_Ratio"]
    elif dataset == "data/market_state_hist/top_acct_ratio":
        out = coalesce_columns(out, "TopAccount_LS_Ratio", ("ratio",))
        keep = ["TopAccount_LS_Ratio"]
    elif dataset == "data/market_state_hist/top_pos_ratio":
        out = coalesce_columns(out, "TopPosition_LS_Ratio", ("ratio",))
        keep = ["TopPosition_LS_Ratio"]
    else:
        keep = list(out.columns)
    keep_existing = [col for col in keep if col in out.columns]
    return out[keep_existing]


def copy_dataset(
    source_root: Path,
    stage_root: Path,
    symbols: list[str],
    source_parts: tuple[str, ...],
    dest_parts: tuple[str, ...],
    lookback_days: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        source_path = source_root.joinpath(*source_parts, f"{symbol}.parquet")
        if not source_path.exists():
            source_path = source_root.joinpath(*source_parts, f"{symbol}.csv")
        df = load_frame(source_path)
        if lookback_days is not None and not df.empty:
            end = df.index.max()
            df = df.loc[df.index >= end - pd.Timedelta(days=lookback_days)]
        df = standardize_columns(df, dest_parts)
        dest_path = stage_root.joinpath(*dest_parts, f"{symbol}.csv")
        rows.append(
            {
                "symbol": symbol,
                "dataset": "/".join(dest_parts),
                "source": str(source_path),
                "rows": write_csv_with_ms_timestamp(df, dest_path),
                "start": df.index.min().isoformat() if not df.empty else None,
                "end": df.index.max().isoformat() if not df.empty else None,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deploy seed archive from local research data.")
    parser.add_argument("--config", default="config/paper_config.json")
    parser.add_argument("--signal-root", default="../data/top200_15m")
    parser.add_argument("--exec-root", default="../data")
    parser.add_argument("--output-dir", default="bootstrap_seed")
    parser.add_argument("--signal-lookback-days", type=int, default=60)
    parser.add_argument("--state-lookback-days", type=int, default=30)
    parser.add_argument("--exec-lookback-days", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    symbols = list(config["symbols"])
    output_dir = Path(args.output_dir)
    stage = output_dir / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    signal_root = Path(args.signal_root)
    exec_root = Path(args.exec_root)
    rows: list[dict[str, Any]] = []
    rows += copy_dataset(signal_root, stage, symbols, ("klines",), ("data", "klines"), args.signal_lookback_days)
    rows += copy_dataset(exec_root, stage, symbols, ("klines",), ("data", "klines_1m"), args.exec_lookback_days)
    rows += copy_dataset(signal_root, stage, symbols, ("funding",), ("data", "funding"), args.signal_lookback_days)
    rows += copy_dataset(signal_root, stage, symbols, ("market_state_hist", "oi"), ("data", "market_state_hist", "oi"), args.state_lookback_days)
    rows += copy_dataset(
        signal_root,
        stage,
        symbols,
        ("market_state_hist", "global_acct_ratio"),
        ("data", "market_state_hist", "global_acct_ratio"),
        args.state_lookback_days,
    )
    rows += copy_dataset(
        signal_root,
        stage,
        symbols,
        ("market_state_hist", "top_acct_ratio"),
        ("data", "market_state_hist", "top_acct_ratio"),
        args.state_lookback_days,
    )
    rows += copy_dataset(
        signal_root,
        stage,
        symbols,
        ("market_state_hist", "top_pos_ratio"),
        ("data", "market_state_hist", "top_pos_ratio"),
        args.state_lookback_days,
    )

    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbols": symbols,
        "signal_lookback_days": args.signal_lookback_days,
        "state_lookback_days": args.state_lookback_days,
        "exec_lookback_days": args.exec_lookback_days,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output_dir / "tp14_seed.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage).as_posix())
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(list(stage.rglob('*.csv')))}, indent=2))


if __name__ == "__main__":
    main()
