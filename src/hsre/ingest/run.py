"""Ingestion entry point.

Run one source or all sources for a country panel. Each run fetches, checks
volume against the trailing baseline, stores immutably, and records health.

    python -m hsre.ingest.run --source ucdp_ged --start 2016-01-01 --end 2024-12-31
    python -m hsre.ingest.run --setting nigeria
    python -m hsre.ingest.run --list

Required sources that fail cause a non-zero exit, because a forecast built on
a missing outcome source is not a degraded forecast but an invalid one.
Optional sources that fail are reported and the run continues, with the
affected outputs marked degraded downstream.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

from hsre.config import REPO_ROOT, Config, load_config
from hsre.ingest.adapters import build_adapter, load_frame
from hsre.ingest.adapters import MissingToken, filter_to_country
from hsre.ingest.base import SourceDown
from hsre.monitoring import ledger
from hsre.validate import schema

INBOX_ROOT = REPO_ROOT / "data" / "raw" / "_inbox"
ENV_FILE = REPO_ROOT / ".env"


def archive_ledger() -> Path | None:
    """Move the current ledger aside rather than deleting it.

    Reliability figures reported in the manuscript should come from runs of
    the finished pipeline, not from development attempts. Archiving rather
    than deleting keeps the development history auditable.
    """
    current = ledger.LEDGER_PATH
    if not current.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archived = current.with_name(f"run_ledger_{stamp}.jsonl")
    current.rename(archived)
    return archived


def load_env(path: Path | None = None) -> int:
    """Read KEY=VALUE pairs from .env into the environment.

    Kept deliberately small rather than adding a dependency. Existing
    environment variables win, so an exported token overrides the file.
    """
    target = path or ENV_FILE
    if not target.exists():
        return 0
    loaded = 0
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def ingest_source(
    config: Config,
    name: str,
    window_start: date | None,
    window_end: date | None,
) -> bool:
    """Fetch and validate one source. Returns True on success."""
    source = config.sources.get(name)
    if source is None:
        print(f"unknown source: {name}", file=sys.stderr)
        return False

    kwargs = {}
    if source.route == "export":
        inbox = INBOX_ROOT / name
        inbox.mkdir(parents=True, exist_ok=True)
        kwargs["inbox"] = inbox

    adapter = build_adapter(source, **kwargs)

    try:
        result = adapter.run(window_start, window_end)
    except MissingToken as exc:
        print(f"  {name}: {exc}", file=sys.stderr)
        return False
    except SourceDown as exc:
        marker = "REQUIRED" if source.required else "optional"
        print(f"  {name}: FAILED ({marker}) {exc}", file=sys.stderr)
        return False

    baseline = schema.trailing_median_rows(name)
    volume_status, detail = schema.check_volume(source, result.row_count, baseline)

    note = ""
    if volume_status == ledger.STATUS_THIN:
        note = (
            f"  THIN: {detail['ratio']:.2f} of trailing median "
            f"({detail['observed']} vs {detail['trailing_median']:.0f})"
        )

    try:
        frame = load_frame(result.raw_path, source)
        fetched_rows = len(frame)
        # The bulk archive is global; the API applies this server-side.
        if source.effective_route == "bulk" and source.country_id:
            frame = filter_to_country(frame, source.country_id)
        surviving, report = schema.validate(frame, source)
        scope = ""
        if len(frame) != fetched_rows:
            scope = f" ({len(frame)} in scope of {fetched_rows})"
        print(
            f"  {name}: {result.row_count} fetched{scope}, "
            f"{report.rows_out} valid, {report.quarantined} quarantined"
            f"{note}"
        )
        if report.quarantined:
            print(f"    reasons: {report.reasons}")
    except schema.SchemaDrift as exc:
        print(f"  {name}: SCHEMA DRIFT {exc}", file=sys.stderr)
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HSRE source ingestion")
    parser.add_argument("--source", help="single source name from config/sources.yml")
    parser.add_argument("--setting", choices=["nigeria", "usa"], help="ingest a whole panel")
    parser.add_argument("--start", help="window start, YYYY-MM-DD")
    parser.add_argument("--end", help="window end, YYYY-MM-DD")
    parser.add_argument("--list", action="store_true", help="list configured sources")
    parser.add_argument(
        "--reset-ledger",
        action="store_true",
        help="archive the run ledger before starting, for a clean reliability baseline",
    )
    args = parser.parse_args(argv)

    load_env()
    config = load_config()

    if args.reset_ledger:
        archived = archive_ledger()
        if archived:
            print(f"ledger archived to {archived.name}")
        else:
            print("no ledger to archive")

    if args.list:
        for name, source in sorted(config.sources.items()):
            flag = "required" if source.required else "optional"
            route = source.effective_route
            alt = sorted(set(source.routes) - {route})
            note = f"  (also: {', '.join(alt)})" if alt else ""
            token = source.route_setting("token_env")
            auth = "  token required" if token else ""
            print(
                f"{name:16s} {source.setting:8s} {route:7s} "
                f"{source.role:11s} {flag}{auth}{note}"
            )
        return 0

    if not args.source and not args.setting:
        parser.error("give either --source or --setting")

    window_start = _parse_date(args.start)
    window_end = _parse_date(args.end)

    if args.source:
        targets = {args.source: config.sources.get(args.source)}
    else:
        targets = config.sources_for(args.setting)

    print(f"ingesting {len(targets)} source(s)")
    failures_required: list[str] = []
    failures_optional: list[str] = []

    for name, source in targets.items():
        if source is None:
            failures_required.append(name)
            continue
        ok = ingest_source(config, name, window_start, window_end)
        if not ok:
            (failures_required if source.required else failures_optional).append(name)

    if failures_optional:
        print(f"optional sources unavailable, forecasts will be degraded: {failures_optional}")
    if failures_required:
        print(f"required sources unavailable: {failures_required}", file=sys.stderr)
        return 1

    rate = ledger.pipeline_success_rate("ingest")
    print(f"ingest success rate to date: {rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
