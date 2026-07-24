"""Inspect ingested raw data.

Used before building crosswalks and features, to see what a source actually
contains rather than what its documentation implies. Reads the most recent
retrieval for a source and reports column names, coverage and the distribution
of the fields the panel depends on.

    python -m hsre.ingest.inspect ucdp_ged
    python -m hsre.ingest.inspect ucdp_ged --column adm_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hsre.config import REPO_ROOT, load_config
from hsre.ingest.adapters import filter_to_country, load_frame

RAW_ROOT = REPO_ROOT / "data" / "raw"


def latest_retrieval(source_name: str) -> Path | None:
    """Most recent stored file for a source."""
    root = RAW_ROOT / source_name
    if not root.exists():
        return None
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix in {".csv", ".json"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def summarise(frame: pd.DataFrame, source_name: str, in_scope: int, total: int) -> None:
    print(f"source: {source_name}")
    print(f"rows: {in_scope} in scope of {total}")
    print(f"columns: {len(frame.columns)}")
    print()

    print("columns present:")
    for column in frame.columns:
        non_null = frame[column].notna().sum()
        pct = 100 * non_null / len(frame) if len(frame) else 0
        print(f"  {column:24s} {non_null:>8d} non-null  ({pct:5.1f}%)")
    print()

    if "date_start" in frame.columns:
        dates = pd.to_datetime(frame["date_start"], errors="coerce")
        print(f"date range: {dates.min().date()} to {dates.max().date()}")
        by_year = dates.dt.year.value_counts().sort_index()
        recent = by_year[by_year.index >= 2016]
        print("events per year from 2016:")
        for year, count in recent.items():
            print(f"  {int(year)}  {count:>6d}")
        print()


def show_column(frame: pd.DataFrame, column: str, limit: int = 60) -> None:
    if column not in frame.columns:
        print(f"column '{column}' not present", file=sys.stderr)
        print(f"available: {list(frame.columns)}", file=sys.stderr)
        return
    counts = frame[column].value_counts(dropna=False)
    print(f"distinct values in '{column}': {len(counts)}")
    print()
    for value, count in counts.head(limit).items():
        label = "<missing>" if pd.isna(value) else str(value)
        print(f"  {count:>6d}  {label}")
    if len(counts) > limit:
        print(f"  ... {len(counts) - limit} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an ingested source")
    parser.add_argument("source", help="source name from config/sources.yml")
    parser.add_argument("--column", help="show the value distribution for one column")
    parser.add_argument("--limit", type=int, default=60, help="rows to show for --column")
    parser.add_argument(
        "--unfiltered", action="store_true", help="skip the country restriction"
    )
    args = parser.parse_args(argv)

    config = load_config()
    source = config.sources.get(args.source)
    if source is None:
        print(f"unknown source: {args.source}", file=sys.stderr)
        return 1

    path = latest_retrieval(args.source)
    if path is None:
        print(f"no stored retrieval for {args.source}. Run ingestion first.", file=sys.stderr)
        return 1

    frame = load_frame(path, source)
    total = len(frame)
    if not args.unfiltered and source.country_id:
        frame = filter_to_country(frame, source.country_id)

    if args.column:
        show_column(frame, args.column, args.limit)
    else:
        summarise(frame, args.source, len(frame), total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
