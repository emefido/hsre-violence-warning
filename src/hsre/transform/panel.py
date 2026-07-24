"""Locality-week panel construction.

Builds a complete state-week grid from ACLED events and labels two outcomes.

Two properties matter more than anything else here.

The grid is complete, not observed-only. A week with no recorded event is a
real zero the model must be able to see. Building only from observed rows
would delete every quiet week and make the outcome meaningless.

Every feature reads only data available as of the forecast date. Escalation is
labelled from weeks t+1 and t+2, while all predictors come from t and earlier.
Temporal leakage is the commonest defect in applied forecasting of this kind,
and it is prevented structurally rather than by discipline.

A third property is specific to ACLED: its weeks begin on Saturday. A
Monday-anchored grid produces zero overlap on merge, which yields a silently
empty panel rather than an error. The anchor is read from config and asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hsre.monitoring import ledger
from hsre.transform.nigeria_states import CANONICAL_STATES, harmonise

# Column names in the ACLED aggregated file. The disaggregated API uses
# lowercase names, handled by `normalise_acled_columns`.
AGG_COLUMNS = {
    "WEEK": "week",
    "ADMIN1": "admin1",
    "EVENT_TYPE": "event_type",
    "SUB_EVENT_TYPE": "sub_event_type",
    "EVENTS": "events",
    "FATALITIES": "fatalities",
}


class PanelError(RuntimeError):
    """Raised when the panel cannot be built correctly."""


@dataclass
class PanelReport:
    localities: int
    weeks: int
    rows: int
    labelled: int
    positives: int
    zero_baseline_share: float
    top_locality: str
    top_locality_share: float
    localities_without_escalation: int

    @property
    def base_rate(self) -> float:
        return self.positives / self.labelled if self.labelled else 0.0


def normalise_acled_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the aggregated file or the disaggregated API output."""
    working = frame.copy()
    if "WEEK" in working.columns:
        working = working.rename(columns=AGG_COLUMNS)
        return working

    # Disaggregated events: one row per event, so an events column is derived.
    lower = {c: c.lower() for c in working.columns}
    working = working.rename(columns=lower)
    if "event_date" not in working.columns:
        raise PanelError(
            "frame has neither WEEK (aggregated) nor event_date (disaggregated)"
        )
    working["events"] = 1
    working = working.rename(columns={"event_date": "week"})
    return working


def _week_floor(dates: pd.Series, anchor: str) -> pd.Series:
    """Snap dates to the start of their week under the configured anchor."""
    period = anchor.replace("W-", "")
    return pd.to_datetime(dates).dt.to_period(f"W-{period}").dt.start_time


def build_panel(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    week_anchor: str,
    state_column: str = "admin1",
    source_name: str = "acled",
    localities: list[str] | None = None,
) -> pd.DataFrame:
    """Complete locality-week grid with event and fatality counts.

    Also carries the per-week breakdown needed to label both outcomes, so the
    grid is built once rather than per outcome.
    """
    working = normalise_acled_columns(frame)
    working["week"] = pd.to_datetime(working["week"], errors="coerce")
    working = working.loc[working["week"].between(start, end)].copy()
    if working.empty:
        raise PanelError(f"no rows fall between {start.date()} and {end.date()}")

    harmonised, match_rate = harmonise(working, state_column, source_name)
    if harmonised.empty:
        raise PanelError("state harmonisation matched no rows")

    grid_states = localities or CANONICAL_STATES
    weeks = pd.date_range(start, end, freq=week_anchor)
    if len(weeks) == 0:
        raise PanelError(f"week anchor {week_anchor} produced no weeks")

    # The alignment check that would otherwise fail silently.
    observed_weeks = set(harmonised["week"].unique())
    overlap = observed_weeks & set(weeks)
    if not overlap:
        observed_days = sorted({pd.Timestamp(w).day_name() for w in observed_weeks})
        raise PanelError(
            f"week anchor {week_anchor} does not align with the data. "
            f"Observed week start days: {observed_days}. "
            f"A misaligned anchor yields an empty panel rather than an error."
        )

    grid = pd.MultiIndex.from_product(
        [sorted(grid_states), weeks], names=["state", "week"]
    ).to_frame(index=False)

    ledger.record(
        stage="panel",
        status=ledger.STATUS_OK,
        source=source_name,
        localities=len(grid_states),
        weeks=len(weeks),
        rows=len(grid),
        state_match_rate=round(match_rate, 4),
        week_anchor=week_anchor,
        week_overlap=len(overlap),
    )
    return grid, harmonised


def aggregate_outcome(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    event_types: list[str] | None = None,
    sub_event_types: list[str] | None = None,
) -> pd.DataFrame:
    """Count events and fatalities per locality-week for one outcome subset."""
    subset = events
    if event_types:
        subset = subset.loc[subset["event_type"].isin(event_types)]
    if sub_event_types:
        subset = subset.loc[subset["sub_event_type"].isin(sub_event_types)]

    counts = (
        subset.groupby(["state", "week"])
        .agg(events=("events", "sum"), fatalities=("fatalities", "sum"))
        .reset_index()
    )
    panel = grid.merge(counts, on=["state", "week"], how="left")
    panel[["events", "fatalities"]] = panel[["events", "fatalities"]].fillna(0)
    return panel.sort_values(["state", "week"]).reset_index(drop=True)


def label_escalation(
    panel: pd.DataFrame,
    horizon_weeks: int,
    baseline_weeks: int,
    percentile_cut: float,
    min_future_events: int,
    require_lethal: bool,
    outcome_name: str = "escalation",
) -> pd.DataFrame:
    """Label escalation using only past information for the baseline.

    Escalation requires the future window to exceed the locality's trailing
    median, reach its own historical percentile, and clear an absolute floor.
    The floor matters because a trailing median of zero would otherwise make
    any single event an escalation, which measures presence of violence rather
    than change in level.
    """
    out = panel.sort_values(["state", "week"]).copy()
    grouped = out.groupby("state")["events"]

    # Baselines shift by one so the labelled week never informs its own label.
    out["trailing_median"] = grouped.transform(
        lambda s: s.shift(1).rolling(baseline_weeks, min_periods=baseline_weeks).median()
    )
    out["locality_percentile"] = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=baseline_weeks).quantile(percentile_cut)
    )

    # Future window is weeks t+1 .. t+horizon.
    out["future_events"] = grouped.transform(
        lambda s: s.shift(-1).rolling(horizon_weeks, min_periods=1).sum().shift(-(horizon_weeks - 1))
    )
    out["future_fatalities"] = out.groupby("state")["fatalities"].transform(
        lambda s: s.shift(-1).rolling(horizon_weeks, min_periods=1).sum().shift(-(horizon_weeks - 1))
    )

    condition = (
        (out["future_events"] > out["trailing_median"])
        & (out["future_events"] >= out["locality_percentile"])
        & (out["future_events"] >= min_future_events)
    )
    if require_lethal:
        condition = condition & (out["future_fatalities"] > 0)

    out[outcome_name] = condition.astype("Int64")
    unlabelled = (
        out["trailing_median"].isna()
        | out["locality_percentile"].isna()
        | out["future_events"].isna()
    )
    out.loc[unlabelled, outcome_name] = pd.NA
    return out


def summarise(panel: pd.DataFrame, outcome_name: str = "escalation") -> PanelReport:
    labelled = panel.loc[panel[outcome_name].notna()]
    per_locality = labelled.groupby("state")[outcome_name].sum().sort_values(ascending=False)
    positives = int(labelled[outcome_name].sum())

    return PanelReport(
        localities=panel["state"].nunique(),
        weeks=panel["week"].nunique(),
        rows=len(panel),
        labelled=len(labelled),
        positives=positives,
        zero_baseline_share=float((labelled["trailing_median"] == 0).mean()),
        top_locality=str(per_locality.index[0]) if len(per_locality) else "",
        top_locality_share=float(per_locality.iloc[0] / positives) if positives else 0.0,
        localities_without_escalation=int((per_locality == 0).sum()),
    )


def build_both_outcomes(
    frame: pd.DataFrame,
    thresholds: dict[str, Any],
    source_name: str = "acled",
) -> dict[str, pd.DataFrame]:
    """Construct the primary and youth panels from one ACLED extract."""
    period = thresholds["study_period"]
    outcome = thresholds["outcome"]
    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    horizon = outcome["horizon_days"] // 7
    baseline = outcome["baseline_window_weeks"]

    grid, events = build_panel(
        frame,
        start=start,
        end=end,
        week_anchor=outcome["week_anchor"],
        source_name=source_name,
    )

    panels: dict[str, pd.DataFrame] = {}
    for name, spec in outcome["nigeria"].items():
        counted = aggregate_outcome(
            grid,
            events,
            event_types=spec.get("event_types"),
            sub_event_types=spec.get("sub_event_types"),
        )
        labelled = label_escalation(
            counted,
            horizon_weeks=horizon,
            baseline_weeks=baseline,
            percentile_cut=spec["percentile_cut"],
            min_future_events=spec["min_future_events"],
            require_lethal=spec["require_lethal"],
        )
        report = summarise(labelled)
        ledger.record(
            stage="panel",
            status=ledger.STATUS_OK,
            source=source_name,
            outcome=name,
            rows=report.rows,
            labelled=report.labelled,
            positives=report.positives,
            base_rate=round(report.base_rate, 4),
            expected_base_rate=spec.get("observed_base_rate"),
            zero_baseline_share=round(report.zero_baseline_share, 4),
            top_locality=report.top_locality,
            top_locality_share=round(report.top_locality_share, 4),
        )
        panels[name] = labelled
    return panels
