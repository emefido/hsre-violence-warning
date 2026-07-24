"""Stage 7 verification: temporal splitting, baselines and evaluation.

The dominant risks here are random splitting (which leaks the future) and
reporting accuracy on a rare outcome (which flatters a useless model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsre.features.build import LAG_PREFIX, build_features
from hsre.models.baselines import (
    CountBaseline,
    LogisticBaseline,
    PersistenceBaseline,
    TrailingRateBaseline,
    default_baselines,
    lag_only_features,
)
from hsre.models.evaluate import (
    bootstrap_ap,
    comparison_table,
    evaluate,
    geographic_split,
    precision_recall_at_k,
    temporal_split,
)
from hsre.monitoring import ledger
from hsre.transform.panel import label_escalation


def _labelled_panel(states=("Borno", "Lagos", "Kano"), n_weeks=200, seed=3):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2016-01-02", periods=n_weeks, freq="W-SAT")
    rows = []
    for i, state in enumerate(states):
        rate = 4.0 - i
        for week in weeks:
            rows.append({
                "state": state, "week": week,
                "events": int(rng.poisson(max(rate, 0.4))),
                "fatalities": int(rng.poisson(1)),
            })
    panel = pd.DataFrame(rows)
    panel = build_features(panel)
    return label_escalation(
        panel, horizon_weeks=2, baseline_weeks=12,
        percentile_cut=0.90, min_future_events=2, require_lethal=False,
    )


def test_temporal_split_never_puts_later_weeks_in_train():
    """Random splitting would leak the future. Neighbouring weeks in the same
    locality are strongly dependent."""
    panel = _labelled_panel()
    split = temporal_split(
        panel,
        train_end=pd.Timestamp("2017-12-30"),
        validation_end=pd.Timestamp("2018-12-29"),
        test_start=pd.Timestamp("2019-01-05"),
    )
    assert split.train["week"].max() < split.validation["week"].min()
    assert split.validation["week"].max() < split.test["week"].min()


def test_temporal_split_rejects_an_empty_partition():
    panel = _labelled_panel()
    with pytest.raises(ValueError, match="empty"):
        temporal_split(
            panel,
            train_end=pd.Timestamp("2015-01-01"),
            validation_end=pd.Timestamp("2015-06-01"),
            test_start=pd.Timestamp("2015-07-01"),
        )


def test_geographic_split_holds_out_whole_localities():
    """A model can score well by learning one dominant locality. Performance
    on unseen localities is the honest measure."""
    panel = _labelled_panel()
    split = geographic_split(
        panel, holdout_localities=["Borno"], train_end=pd.Timestamp("2018-12-29")
    )
    assert "Borno" not in set(split.train["state"])
    assert set(split.test["state"]) == {"Borno"}


def test_precision_at_k_matches_a_hand_computed_case():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # Top 20% is 2 rows: scores 0.9 (positive) and 0.8 (negative).
    precision, recall = precision_recall_at_k(y, scores, 0.2)
    assert precision == 0.5
    assert recall == 0.5


def test_precision_at_k_always_alerts_at_least_one():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.1, 0.1])
    precision, _ = precision_recall_at_k(y, scores, 0.001)
    assert precision == 1.0


def test_lift_exposes_a_no_skill_model():
    """Average precision equal to the base rate means no skill, whatever the
    absolute value looks like."""
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    noise = rng.random(2000)
    metrics = evaluate(y, noise, "noise", "test", bootstrap_draws=0)
    assert abs(metrics.lift() - 1.0) < 0.2


def test_perfect_scores_reach_lift_above_one():
    y = np.array([1] * 100 + [0] * 400)
    scores = y.astype(float)
    metrics = evaluate(y, scores, "oracle", "test", bootstrap_draws=0)
    assert metrics.average_precision > 0.99
    assert metrics.lift() > 4


def test_evaluate_rejects_a_test_set_with_no_positives():
    with pytest.raises(ValueError, match="no positive"):
        evaluate(np.zeros(50), np.random.random(50), "m", "o", bootstrap_draws=0)


def test_bootstrap_interval_brackets_the_estimate():
    """Signal mixed with noise, so the scores overlap between classes. A
    perfectly separable fixture would give a degenerate interval at 1.0 and
    test nothing."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.3).astype(int)
    scores = y * 0.25 + rng.random(1000)

    point = average_precision_score(y, scores)
    assert 0.3 < point < 0.95, "fixture should be informative but imperfect"
    low, high = bootstrap_ap(y, scores, draws=300)
    assert low < point < high
    assert high - low < 0.3, "interval should be reasonably tight at n=1000"


def test_baselines_see_only_lag_features():
    """The distinction between baseline and main model is the feature set,
    not the algorithm. That is what isolates the contribution of the extra
    sources for H1."""
    panel = _labelled_panel()
    features = lag_only_features(panel)
    assert features
    assert all(f.startswith(LAG_PREFIX) for f in features)
    assert not any(f.startswith(("sig_", "sp_", "hlth_")) for f in features)


def test_persistence_needs_no_fitting():
    panel = _labelled_panel()
    split = temporal_split(
        panel, pd.Timestamp("2017-12-30"), pd.Timestamp("2018-12-29"),
        pd.Timestamp("2019-01-05"),
    )
    model = PersistenceBaseline().fit(split.train, [], "escalation")
    scores = model.predict_scores(split.test, [])
    assert len(scores) == len(split.test)
    assert not np.isnan(scores).any()


def test_every_baseline_fits_and_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _labelled_panel()
    split = temporal_split(
        panel, pd.Timestamp("2017-12-30"), pd.Timestamp("2018-12-29"),
        pd.Timestamp("2019-01-05"),
    )
    features = lag_only_features(panel)
    for model in default_baselines():
        model.fit(split.train, features, "escalation")
        scores = model.predict_scores(split.test, features)
        assert len(scores) == len(split.test)
        assert np.isfinite(scores).all(), f"{model.name} produced non-finite scores"


def test_logistic_beats_random_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _labelled_panel()
    split = temporal_split(
        panel, pd.Timestamp("2017-12-30"), pd.Timestamp("2018-12-29"),
        pd.Timestamp("2019-01-05"),
    )
    features = lag_only_features(panel)
    model = LogisticBaseline().fit(split.train, features, "escalation")
    scores = model.predict_scores(split.test, features)
    y = split.test["escalation"].astype(int).to_numpy()

    fitted = evaluate(y, scores, "logistic", "test", bootstrap_draws=0)
    rng = np.random.default_rng(7)
    noise = evaluate(y, rng.random(len(y)), "noise", "test", bootstrap_draws=0)
    assert fitted.average_precision > noise.average_precision


def test_comparison_table_orders_by_performance(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.25).astype(int)
    good = evaluate(y, y * 0.8 + rng.random(500) * 0.2, "good", "youth", bootstrap_draws=0)
    bad = evaluate(y, rng.random(500), "bad", "youth", bootstrap_draws=0)
    table = comparison_table([bad, good])
    assert table.iloc[0]["model"] == "good"
    assert "average_precision" in table.columns
    assert "accuracy" not in table.columns


def test_metrics_row_reports_operational_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    rng = np.random.default_rng(4)
    y = (rng.random(800) < 0.2).astype(int)
    metrics = evaluate(y, rng.random(800), "m", "primary", bootstrap_draws=50)
    row = metrics.to_row()
    assert "precision_at_2pct" in row
    assert "recall_at_2pct" in row
    assert "ap_ci_low" in row
