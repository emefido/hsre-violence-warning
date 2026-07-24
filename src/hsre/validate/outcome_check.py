"""Outcome definition diagnostics.

Run before freezing the preregistration. Builds the locality-week outcome from
real ingested data under the current settings in config/thresholds.yml and
reports whether the definition is viable.

The specific failure this catches: where events are sparse, a trailing median
of zero makes any single event an escalation, so the outcome stops measuring
change in level and starts measuring presence of violence. That is a different
research question from the one the study asks.

    python -m hsre.validate.outcome_check ucdp_ged
    python -m hsre.validate.outcome_check ucdp_ged --min-events 2
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from hsre.config import load_config
from hsre.ingest.adapters import filter_to_country, load_frame
from hsre.ingest.inspect import latest_retrieval


def build_locality_weeks(
    frame: pd.DataFrame,
    locality_col: str,
    date_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Complete locality-week grid with event counts and fatalities.

    The grid is complete rather than observed-only, because a week with no
    recorded event is a real zero that the model must be able to see.
    """
    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.loc[working[date_col].between(start, end)]
    working = working.loc[working[locality_col].notna()]

    working["week"] = working[date_col].dt.to_period("W-SUN").dt.start_time
    deaths = working["best"] if "best" in working.columns else 0

    grouped = (
        working.assign(deaths=deaths)
        .groupby([locality_col, "week"])
        .agg(events=("id", "count"), deaths=("deaths", "sum"))
        .reset_index()
    )

    localities = sorted(working[locality_col].unique())
    weeks = pd.date_range(start, end, freq="W-MON")
    grid = pd.MultiIndex.from_product(
        [localities, weeks], names=[locality_col, "week"]
    ).to_frame(index=False)

    panel = grid.merge(grouped, on=[locality_col, "week"], how="left")
    panel[["events", "deaths"]] = panel[["events", "deaths"]].fillna(0).astype(int)
    return panel


def escalation_outcome(
    panel: pd.DataFrame,
    locality_col: str,
    horizon_weeks: int,
    baseline_weeks: int,
    percentile: float,
    min_events: int,
) -> pd.DataFrame:
    """Label escalation per the manuscript definition, with a count floor.

    min_events is the addition. Without it, a trailing median of zero turns any
    single event into an escalation, which in a sparse panel is most of the
    positive class.
    """
    out = panel.sort_values([locality_col, "week"]).copy()
    grouped = out.groupby(locality_col)["events"]

    out["trailing_median"] = grouped.transform(
        lambda s: s.shift(1).rolling(baseline_weeks, min_periods=baseline_weeks).median()
    )
    out["locality_pct"] = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=baseline_weeks).quantile(percentile)
    )
    out["future_events"] = grouped.transform(
        lambda s: s.shift(-1).rolling(horizon_weeks, min_periods=1).sum().shift(-(horizon_weeks - 1))
    )
    future_deaths = out.groupby(locality_col)["deaths"].transform(
        lambda s: s.shift(-1).rolling(horizon_weeks, min_periods=1).sum().shift(-(horizon_weeks - 1))
    )
    out["future_deaths"] = future_deaths

    exceeds_baseline = out["future_events"] > out["trailing_median"]
    reaches_percentile = out["future_events"] >= out["locality_pct"]
    clears_floor = out["future_events"] >= min_events
    lethal = out["future_deaths"] > 0

    out["escalation"] = (
        (exceeds_baseline | reaches_percentile) & clears_floor & lethal
    ).astype("Int64")
    out.loc[out["trailing_median"].isna() | out["future_events"].isna(), "escalation"] = pd.NA
    return out


def report(out: pd.DataFrame, locality_col: str, min_events: int) -> None:
    labelled = out.loc[out["escalation"].notna()]
    n = len(labelled)
    positives = int(labelled["escalation"].sum())
    rate = positives / n if n else 0.0

    print(f"labelled locality-weeks: {n}")
    print(f"escalation weeks:        {positives}  ({rate * 100:.2f}%)")
    print(f"minimum event floor:     {min_events}")
    print()

    zero_median = (labelled["trailing_median"] == 0).mean()
    print(f"trailing median is zero in {zero_median * 100:.1f}% of labelled weeks")
    if zero_median > 0.5:
        print(
            "  WARNING: with a zero baseline the outcome measures presence of\n"
            "  violence rather than escalation. Raise the event floor or\n"
            "  aggregate to a coarser locality."
        )
    print()

    per_locality = (
        labelled.groupby(locality_col)["escalation"]
        .agg(["sum", "count"])
        .assign(rate=lambda d: d["sum"] / d["count"])
        .sort_values("sum", ascending=False)
    )
    print("escalation weeks by locality (top 10):")
    for name, row in per_locality.head(10).iterrows():
        print(f"  {str(name):28s} {int(row['sum']):>5d}  ({row['rate'] * 100:5.2f}%)")
    print()

    never = per_locality.loc[per_locality["sum"] == 0]
    print(f"localities with no escalation week at all: {len(never)} of {len(per_locality)}")
    if len(never):
        print(f"  {', '.join(str(x) for x in never.index[:10])}")
        if len(never) > 10:
            print(f"  ... and {len(never) - 10} more")
    print()

    concentration = per_locality["sum"].head(1).sum() / max(positives, 1)
    print(f"share of all escalation weeks in the single worst locality: {concentration * 100:.1f}%")
    if concentration > 0.4:
        print(
            "  WARNING: the positive class is dominated by one locality. A model\n"
            "  can score well by learning that locality alone. Report performance\n"
            "  with it held out."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check outcome definition viability")
    parser.add_argument("source", help="ingested outcome source, e.g. ucdp_ged")
    parser.add_argument("--locality-col", default="adm_1")
    parser.add_argument("--date-col", default="date_start")
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="override the event floor for the escalation definition",
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
    if source.country_id:
        frame = filter_to_country(frame, source.country_id)

    thresholds = config.thresholds
    period = thresholds["study_period"]
    outcome = thresholds["outcome"]
    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    min_events = (
        args.min_events
        if args.min_events is not None
        else outcome.get("min_future_events", 1)
    )

    panel = build_locality_weeks(
        frame, args.locality_col, args.date_col, start, end
    )
    print(f"panel: {panel[args.locality_col].nunique()} localities, "
          f"{panel['week'].nunique()} weeks, {len(panel)} locality-weeks")
    print(f"mean events per locality-week: {panel['events'].mean():.3f}")
    print(f"locality-weeks with no event:  {(panel['events'] == 0).mean() * 100:.1f}%")
    print()

    out = escalation_outcome(
        panel,
        args.locality_col,
        horizon_weeks=outcome["horizon_days"] // 7,
        baseline_weeks=outcome["baseline_window_weeks"],
        percentile=outcome["percentile_cut"],
        min_events=min_events,
    )
    report(out, args.locality_col, min_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
