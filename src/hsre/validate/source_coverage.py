"""Compare outcome sources for coverage.

Conventional conflict datasets code organised political violence. Youth and
community violence is largely cultism, communal, criminal and interpersonal
killing, which falls outside those inclusion criteria. This module measures the
gap directly rather than asserting it, producing a comparison suitable for the
manuscript.

The gap is not a data-cleaning problem to be resolved before analysis. It is
substantive: which violence a state can see determines which violence a state
can respond to, and a locality invisible to the outcome source is invisible to
the warning service built on it.

    python -m hsre.validate.source_coverage
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from hsre.config import load_config
from hsre.ingest.adapters import filter_to_country, load_frame
from hsre.ingest.inspect import latest_retrieval

# Nigeria Watch cause categories that carry youth and community violence.
# Kept explicit so the operationalisation is auditable rather than implied.
COMMUNITY_VIOLENCE_CAUSES = {
    "crime",
    "cultism",
    "communal",
    "ethno-religious",
    "land",
    "political",
    "witchcraft",
    "ritual",
    "mob",
    "lynching",
    "vigilante",
}


def normalise_state(series: pd.Series) -> pd.Series:
    """Strip source-specific suffixes so state names compare across datasets."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+(state|State)$", "", regex=True)
        .str.replace(r"^Federal Capital [Tt]erritory$", "FCT", regex=True)
        .str.title()
    )
    return cleaned


def load_outcome(source_name: str, config) -> pd.DataFrame | None:
    source = config.sources.get(source_name)
    if source is None:
        return None
    path = latest_retrieval(source_name)
    if path is None:
        return None
    frame = load_frame(path, source)
    if source.country_id:
        frame = filter_to_country(frame, source.country_id)
    return frame


def summarise_by_state(
    frame: pd.DataFrame,
    state_col: str,
    date_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.loc[working[date_col].between(start, end)]
    working["state"] = normalise_state(working[state_col])
    counts = (
        working.groupby("state").size().rename(label).sort_values(ascending=False)
    )
    return counts.to_frame()


def compare(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    primary_label: str,
    secondary_label: str,
) -> pd.DataFrame:
    merged = primary.join(secondary, how="outer").fillna(0).astype(int)
    merged["ratio"] = merged[primary_label] / merged[secondary_label].replace(0, pd.NA)
    return merged.sort_values(primary_label, ascending=False)


def report(merged: pd.DataFrame, primary_label: str, secondary_label: str) -> None:
    total_primary = merged[primary_label].sum()
    total_secondary = merged[secondary_label].sum()

    print(f"{primary_label} events:   {total_primary}")
    print(f"{secondary_label} events: {total_secondary}")
    if total_secondary:
        print(f"ratio: {total_primary / total_secondary:.1f}x")
    print()

    invisible = merged.loc[merged[secondary_label] == 0]
    if len(invisible):
        print(f"states with events in {primary_label} but none in {secondary_label}:")
        for state, row in invisible.iterrows():
            print(f"  {state:22s} {int(row[primary_label]):>7d}  vs  0")
        print()

    print(f"concentration in {secondary_label}:")
    top_secondary = merged.sort_values(secondary_label, ascending=False)
    share = top_secondary[secondary_label].head(1).sum() / max(total_secondary, 1)
    print(f"  single largest state holds {share * 100:.1f}% of all events")

    print(f"concentration in {primary_label}:")
    share_primary = merged[primary_label].head(1).sum() / max(total_primary, 1)
    print(f"  single largest state holds {share_primary * 100:.1f}% of all events")
    print()

    print("per-state comparison (top 15 by primary source):")
    header = f"  {'state':22s} {primary_label:>10s} {secondary_label:>10s} {'ratio':>8s}"
    print(header)
    for state, row in merged.head(15).iterrows():
        ratio = row["ratio"]
        ratio_text = f"{ratio:8.1f}" if pd.notna(ratio) else "       -"
        print(
            f"  {state:22s} {int(row[primary_label]):>10d} "
            f"{int(row[secondary_label]):>10d} {ratio_text}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare outcome source coverage")
    parser.add_argument("--primary", default="nigeria_watch")
    parser.add_argument("--secondary", default="ucdp_ged")
    args = parser.parse_args(argv)

    config = load_config()
    period = config.thresholds["study_period"]
    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])

    secondary_frame = load_outcome(args.secondary, config)
    if secondary_frame is None:
        print(f"no ingested data for {args.secondary}", file=sys.stderr)
        return 1

    secondary = summarise_by_state(
        secondary_frame, "adm_1", "date_start", start, end, args.secondary
    )

    primary_frame = load_outcome(args.primary, config)
    if primary_frame is None:
        print(
            f"no ingested data for {args.primary}.\n"
            f"Download from nigeriawatch.org and place the export in\n"
            f"  data/raw/_inbox/{args.primary}/\n"
            f"then run: python -m hsre.ingest.run --source {args.primary}\n",
            file=sys.stderr,
        )
        print("Showing the secondary source alone for now.\n")
        total = secondary[args.secondary].sum()
        print(f"{args.secondary} events {start.date()} to {end.date()}: {total}")
        share = secondary[args.secondary].head(1).sum() / max(total, 1)
        print(f"single largest state holds {share * 100:.1f}% of all events")
        zero = secondary.loc[secondary[args.secondary] < 20]
        print(f"\nstates with fewer than 20 events across the whole period: {len(zero)}")
        for state, row in zero.iterrows():
            print(f"  {state:22s} {int(row[args.secondary]):>5d}")
        return 0

    primary = summarise_by_state(
        primary_frame, "state", "date", start, end, args.primary
    )
    merged = compare(primary, secondary, args.primary, args.secondary)
    report(merged, args.primary, args.secondary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
