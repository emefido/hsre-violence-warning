"""Stage 5 verification: locality-week panel and dual outcomes."""

from __future__ import annotations

import pandas as pd
import pytest

from hsre.config import load_config
from hsre.monitoring import ledger
from hsre.transform.panel import (
    PanelError,
    aggregate_outcome,
    build_panel,
    build_both_outcomes,
    label_escalation,
    normalise_acled_columns,
    summarise,
)


def _agg_frame(weeks, states=("Borno", "Lagos"), events=3, fatalities=2):
    rows = []
    for state in states:
        for week in weeks:
            rows.append(
                {
                    "WEEK": week,
                    "ADMIN1": state,
                    "EVENT_TYPE": "Riots",
                    "SUB_EVENT_TYPE": "Mob violence",
                    "EVENTS": events,
                    "FATALITIES": fatalities,
                }
            )
    return pd.DataFrame(rows)


def test_misaligned_week_anchor_raises_rather_than_emptying(tmp_path, monkeypatch):
    """ACLED weeks start Saturday. A Monday anchor yields zero overlap, which
    produces a silently empty panel unless it is caught."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    saturdays = pd.date_range("2016-01-02", periods=30, freq="W-SAT")
    frame = _agg_frame(saturdays)
    with pytest.raises(PanelError, match="does not align"):
        build_panel(
            frame,
            start=pd.Timestamp("2016-01-01"),
            end=pd.Timestamp("2016-12-31"),
            week_anchor="W-MON",
        )


def test_correct_anchor_builds_a_complete_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    saturdays = pd.date_range("2016-01-02", periods=30, freq="W-SAT")
    grid, events = build_panel(
        _agg_frame(saturdays),
        start=pd.Timestamp("2016-01-02"),
        end=pd.Timestamp("2016-07-30"),
        week_anchor="W-SAT",
    )
    # All 37 canonical states appear, not only the two with events.
    assert grid["state"].nunique() == 37
    assert len(grid) == 37 * grid["week"].nunique()


def test_quiet_weeks_are_real_zeros(tmp_path, monkeypatch):
    """A week with no event must be visible to the model as a zero rather
    than deleted from the panel."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    saturdays = pd.date_range("2016-01-02", periods=20, freq="W-SAT")
    frame = _agg_frame(saturdays[:5], states=("Borno",))
    grid, events = build_panel(
        frame,
        start=saturdays[0],
        end=saturdays[-1],
        week_anchor="W-SAT",
    )
    panel = aggregate_outcome(grid, events, event_types=["Riots"])
    borno = panel.loc[panel["state"] == "Borno"]
    assert (borno["events"] == 0).sum() > 0
    assert (borno["events"] > 0).sum() == 5


def test_labels_use_only_past_information(tmp_path, monkeypatch):
    """The trailing baseline must never see the week it labels."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    saturdays = pd.date_range("2016-01-02", periods=40, freq="W-SAT")
    grid, events = build_panel(
        _agg_frame(saturdays, states=("Borno",)),
        start=saturdays[0],
        end=saturdays[-1],
        week_anchor="W-SAT",
    )
    panel = aggregate_outcome(grid, events, event_types=["Riots"])
    out = label_escalation(
        panel, horizon_weeks=2, baseline_weeks=12,
        percentile_cut=0.90, min_future_events=2, require_lethal=True,
    )
    borno = out.loc[out["state"] == "Borno"].reset_index(drop=True)
    # First 12 weeks cannot be labelled: no baseline exists yet.
    assert borno.loc[:11, "escalation"].isna().all()


def test_event_floor_prevents_presence_masquerading_as_escalation(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    saturdays = pd.date_range("2016-01-02", periods=40, freq="W-SAT")
    # A near-silent locality with one lone event.
    rows = []
    for i, week in enumerate(saturdays):
        rows.append({
            "WEEK": week, "ADMIN1": "Ekiti", "EVENT_TYPE": "Riots",
            "SUB_EVENT_TYPE": "Mob violence",
            "EVENTS": 1 if i == 25 else 0, "FATALITIES": 1 if i == 25 else 0,
        })
    grid, events = build_panel(
        pd.DataFrame(rows), start=saturdays[0], end=saturdays[-1], week_anchor="W-SAT"
    )
    panel = aggregate_outcome(grid, events, event_types=["Riots"])

    permissive = label_escalation(
        panel, 2, 12, 0.90, min_future_events=1, require_lethal=True
    )
    strict = label_escalation(
        panel, 2, 12, 0.90, min_future_events=4, require_lethal=True
    )
    ekiti_p = permissive.loc[permissive["state"] == "Ekiti", "escalation"].sum()
    ekiti_s = strict.loc[strict["state"] == "Ekiti", "escalation"].sum()
    assert ekiti_p >= 1
    assert ekiti_s == 0


def test_youth_outcome_does_not_require_lethality():
    """Mob violence, abduction and demonstration policing are frequently
    non-fatal. Requiring a death would discard most of the outcome."""
    config = load_config()
    youth = config.thresholds["outcome"]["nigeria"]["youth"]
    assert youth["require_lethal"] is False
    assert "Mob violence" in youth["sub_event_types"]
    assert "Abduction/forced disappearance" in youth["sub_event_types"]


def test_primary_outcome_excludes_remote_violence():
    """Air strikes and IEDs are counter-insurgency, which is the bias the
    study is designed to avoid."""
    config = load_config()
    primary = config.thresholds["outcome"]["nigeria"]["primary"]
    assert "Explosions/Remote violence" not in primary["event_types"]
    assert "Protests" not in primary["event_types"]


def test_protests_are_excluded_with_a_stated_reason():
    config = load_config()
    excluded = config.thresholds["outcome"]["excluded_event_types"]
    assert "Protests" in excluded
    assert "predictor" in excluded["Protests"]


def test_disaggregated_frame_is_accepted():
    """The API returns one row per event rather than pre-aggregated counts."""
    frame = pd.DataFrame({
        "event_date": ["2016-01-02", "2016-01-09"],
        "admin1": ["Borno", "Lagos"],
        "event_type": ["Riots", "Riots"],
        "sub_event_type": ["Mob violence", "Mob violence"],
        "fatalities": [1, 0],
    })
    out = normalise_acled_columns(frame)
    assert "week" in out.columns
    assert (out["events"] == 1).all()


def test_frame_without_recognised_date_column_raises():
    with pytest.raises(PanelError, match="neither WEEK"):
        normalise_acled_columns(pd.DataFrame({"foo": [1]}))


def test_build_both_outcomes_produces_two_panels(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    config = load_config()
    saturdays = pd.date_range("2016-01-02", periods=60, freq="W-SAT")
    rows = []
    for week in saturdays:
        rows.append({"WEEK": week, "ADMIN1": "Borno", "EVENT_TYPE": "Battles",
                     "SUB_EVENT_TYPE": "Armed clash", "EVENTS": 6, "FATALITIES": 9})
        rows.append({"WEEK": week, "ADMIN1": "Lagos", "EVENT_TYPE": "Riots",
                     "SUB_EVENT_TYPE": "Mob violence", "EVENTS": 4, "FATALITIES": 0})
    thresholds = dict(config.thresholds)
    thresholds["study_period"] = {
        **thresholds["study_period"],
        "start": saturdays[0].date(),
        "end": saturdays[-1].date(),
    }
    panels = build_both_outcomes(pd.DataFrame(rows), thresholds)
    assert set(panels) == {"primary", "youth"}
    # Lagos mob violence is non-fatal, so it appears only in the youth outcome.
    youth_lagos = panels["youth"].loc[panels["youth"]["state"] == "Lagos", "escalation"]
    assert youth_lagos.sum() > 0
    primary_lagos = panels["primary"].loc[panels["primary"]["state"] == "Lagos", "escalation"]
    assert primary_lagos.sum() == 0


def test_summary_reports_concentration(tmp_path, monkeypatch):
    """Concentration in one locality is the failure mode that made the UCDP
    design unusable, so it is reported rather than discovered later."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    saturdays = pd.date_range("2016-01-02", periods=40, freq="W-SAT")
    grid, events = build_panel(
        _agg_frame(saturdays, states=("Borno",), events=5),
        start=saturdays[0], end=saturdays[-1], week_anchor="W-SAT",
    )
    panel = aggregate_outcome(grid, events, event_types=["Riots"])
    out = label_escalation(panel, 2, 12, 0.90, 2, True)
    report = summarise(out)
    assert report.localities == 37
    assert report.top_locality == "Borno"
    assert report.localities_without_escalation == 36
