"""Stage 9 verification: alert budgets and the volume-detection trade-off."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsre.alerts.budget import (
    REGIME_ACCURACY,
    REGIME_BUDGET,
    REGIME_RECALL,
    accuracy_optimised_threshold,
    apply_regime,
    budget_curve,
    budget_threshold,
    compare_regimes,
    recall_optimised_threshold,
)
from hsre.monitoring import ledger


def _scored_frame(n_weeks=50, n_states=37, seed=13, base_rate=0.20):
    """Panel with scores that rank escalation weeks better than chance."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2023-01-07", periods=n_weeks, freq="W-SAT")
    rows = []
    for week in weeks:
        for state in range(n_states):
            escalated = int(rng.random() < base_rate)
            rows.append({
                "state": f"S{state:02d}", "week": week,
                "escalation": escalated,
                # Severe events are a subset of escalations.
                "severe": int(escalated and rng.random() < 0.3),
            })
    frame = pd.DataFrame(rows)
    # Overlapping score distributions. A perfectly separable fixture yields
    # precision of 1.0 at every volume and hides the trade-off entirely,
    # which is the opposite of the real panel where AP is around 0.45 to 0.68.
    signal = frame["escalation"].to_numpy() * 0.30
    scores = np.clip(signal + rng.normal(0.5, 0.25, len(frame)), 0, 1)
    return frame, scores


def test_budget_threshold_admits_exactly_the_declared_volume():
    scores = np.linspace(0, 1, 1000)
    threshold = budget_threshold(scores, 50)
    assert (scores >= threshold).sum() == 50


def test_budget_threshold_handles_capacity_above_the_data():
    scores = np.array([0.1, 0.5, 0.9])
    threshold = budget_threshold(scores, 100)
    assert (scores >= threshold).sum() == 3


def test_alert_budget_caps_volume_at_capacity(tmp_path, monkeypatch):
    """The defining property of the regime: the threshold is a function of
    institutional capacity rather than of model output."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    frame, scores = _scored_frame()
    result = apply_regime(
        frame, scores, REGIME_BUDGET, "test", capacity_per_week=4
    )
    assert result.alerts_per_week == pytest.approx(4.0, abs=0.05)


def test_recall_regime_buys_detection_with_volume(tmp_path, monkeypatch):
    """Alert fatigue made numerical: chasing recall costs alert volume, and
    the exchange rate between the two is what the study measures."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    frame, scores = _scored_frame()
    budgeted = apply_regime(frame, scores, REGIME_BUDGET, "t", capacity_per_week=4)
    recall_led = apply_regime(frame, scores, REGIME_RECALL, "t", target_recall=0.80)

    assert recall_led.alerts_per_week > budgeted.alerts_per_week
    assert recall_led.recall > budgeted.recall
    # Volume buys detection, but at falling marginal precision.
    assert recall_led.precision < budgeted.precision
    assert recall_led.false_alerts_per_week > budgeted.false_alerts_per_week


def test_accuracy_regime_ignores_capacity(tmp_path, monkeypatch):
    """The field default optimises a classification metric and says nothing
    about how many alerts a responder can handle."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    frame, scores = _scored_frame()
    small = apply_regime(frame, scores, REGIME_ACCURACY, "t", capacity_per_week=2)
    large = apply_regime(frame, scores, REGIME_ACCURACY, "t", capacity_per_week=20)
    assert small.alerts_total == large.alerts_total


def test_error_budget_has_two_linked_constraints(tmp_path, monkeypatch):
    """A burden ceiling alone permits alerting on everything; a recall floor
    alone permits alerting on nothing. Both are required."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    frame, scores = _scored_frame()

    # Recall-led alerting breaches the review burden.
    heavy = apply_regime(
        frame, scores, REGIME_RECALL, "t",
        severe_column="severe", max_review_burden_per_week=5,
    )
    assert heavy.review_burden_breached
    assert not heavy.passes_error_budget

    # A very tight budget breaches the severe recall floor.
    tight = apply_regime(
        frame, scores, REGIME_BUDGET, "t", capacity_per_week=1,
        severe_column="severe", min_severe_recall=0.90,
    )
    assert tight.severe_recall_breached
    assert not tight.passes_error_budget


def test_severe_recall_is_measured_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    frame, scores = _scored_frame()
    result = apply_regime(
        frame, scores, REGIME_BUDGET, "t", capacity_per_week=8,
        severe_column="severe",
    )
    assert result.severe_recall is not None
    assert 0.0 <= result.severe_recall <= 1.0


def test_curve_recall_rises_monotonically_with_volume():
    """More alerts can never catch fewer escalations."""
    frame, scores = _scored_frame()
    curve = budget_curve(frame, scores, "t")
    recalls = curve.points["recall"].to_numpy()
    assert np.all(np.diff(recalls) >= -1e-12)


def test_curve_precision_falls_as_volume_rises():
    """The trade-off the study measures: volume buys detection at the cost of
    precision."""
    frame, scores = _scored_frame()
    curve = budget_curve(frame, scores, "t")
    top = curve.points.loc[curve.points["alerts_per_week"] <= 3, "precision"].mean()
    bottom = curve.points.loc[curve.points["alerts_per_week"] >= 25, "precision"].mean()
    assert top > bottom


def test_curve_reports_volume_needed_for_a_target_recall():
    frame, scores = _scored_frame()
    curve = budget_curve(frame, scores, "t")
    modest = curve.volume_for_recall(0.30)
    ambitious = curve.volume_for_recall(0.80)
    assert modest < ambitious


def test_curve_covers_every_locality_at_full_volume():
    frame, scores = _scored_frame()
    curve = budget_curve(frame, scores, "t", max_alerts_per_week=37)
    assert curve.points["recall"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_comparison_table_covers_all_three_regimes(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    frame, scores = _scored_frame()
    table = compare_regimes(frame, scores, "t", capacity_per_week=4, severe_column="severe")
    assert set(table["regime"]) == {REGIME_ACCURACY, REGIME_RECALL, REGIME_BUDGET}
    assert "passes_error_budget" in table.columns
    assert "false_alerts_per_week" in table.columns


def test_random_scores_give_precision_near_the_base_rate():
    """A model with no skill should alert no better than chance, which is the
    null the curve is read against."""
    frame, _ = _scored_frame(base_rate=0.20)
    rng = np.random.default_rng(5)
    noise = rng.random(len(frame))
    curve = budget_curve(frame, noise, "t")
    mid = curve.points.loc[curve.points["alerts_per_week"] == 10, "precision"].iloc[0]
    assert abs(mid - 0.20) < 0.06
