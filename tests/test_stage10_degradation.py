"""Stage 10 verification: source failure experiments and geographic transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsre.features.build import SIGNAL_PREFIX, SPATIAL_PREFIX, build_features, feature_columns
from hsre.models.baselines import LogisticBaseline, lag_only_features
from hsre.models.degradation import (
    FAILURE_LOCALITY_MASK,
    FAILURE_NONE,
    FAILURE_SIGNAL_DELAY,
    FAILURE_SIGNAL_REMOVED,
    FAILURE_SPATIAL_REMOVED,
    FAILURE_VOLUME_TRUNCATION,
    apply_failure,
    degradation_table,
    geographic_transfer,
    is_graceful,
    run_degradation,
    transfer_table,
)
from hsre.models.evaluate import temporal_split
from hsre.monitoring import ledger
from hsre.transform.panel import label_escalation


def _panel(n_weeks=200, seed=17):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2016-01-02", periods=n_weeks, freq="W-SAT")
    rows, signal_rows = [], []
    for i, state in enumerate(("Alpha", "Beta", "Gamma", "Delta")):
        protest = rng.poisson(1.5, n_weeks)
        for j, week in enumerate(weeks):
            driver = protest[j - 2] if j >= 2 else 0
            rows.append({
                "state": state, "week": week,
                "events": int(rng.poisson(1.0 + 0.8 * driver + 0.3 * i)),
                "fatalities": int(rng.poisson(0.5)),
            })
            signal_rows.append({"state": state, "week": week, "events": int(protest[j])})
    centroids = pd.DataFrame({
        "state": ["Alpha", "Beta", "Gamma", "Delta"],
        "latitude": [6.0, 7.0, 11.0, 12.0],
        "longitude": [3.0, 4.0, 12.0, 13.0],
    })
    featured = build_features(
        pd.DataFrame(rows), centroids=centroids, signals=pd.DataFrame(signal_rows)
    )
    return label_escalation(
        featured, horizon_weeks=2, baseline_weeks=12,
        percentile_cut=0.90, min_future_events=2, require_lethal=False,
    )


def _split(panel):
    # The synthetic panel spans 200 weeks from January 2016, ending in
    # October 2019, so the boundaries sit inside that range.
    return temporal_split(
        panel, pd.Timestamp("2018-06-30"), pd.Timestamp("2018-12-29"),
        pd.Timestamp("2019-01-05"),
    )


def test_no_failure_leaves_the_panel_untouched():
    panel = _panel()
    assert apply_failure(panel, FAILURE_NONE).equals(panel)


def test_signal_delay_shifts_only_signal_features():
    """A delayed source still reports, just late."""
    panel = _panel()
    degraded = apply_failure(panel, FAILURE_SIGNAL_DELAY, severity=2)
    signal_cols = [c for c in panel.columns if c.startswith(SIGNAL_PREFIX)]
    lag_cols = [c for c in panel.columns if c.startswith("lag_")]

    assert not panel[signal_cols].equals(degraded[signal_cols])
    pd.testing.assert_frame_equal(panel[lag_cols], degraded[lag_cols])


def test_signal_removal_zeroes_the_family():
    panel = _panel()
    degraded = apply_failure(panel, FAILURE_SIGNAL_REMOVED)
    signal_cols = [c for c in panel.columns if c.startswith(SIGNAL_PREFIX)]
    assert (degraded[signal_cols] == 0).all().all()


def test_spatial_removal_zeroes_only_spatial():
    panel = _panel()
    degraded = apply_failure(panel, FAILURE_SPATIAL_REMOVED)
    spatial = [c for c in panel.columns if c.startswith(SPATIAL_PREFIX)]
    signal = [c for c in panel.columns if c.startswith(SIGNAL_PREFIX)]
    assert (degraded[spatial] == 0).all().all()
    pd.testing.assert_frame_equal(panel[signal], degraded[signal])


def test_locality_masking_silences_a_share_of_states():
    """The dangerous failure: a locality stops reporting, and its history
    collapses to zero, which is indistinguishable from peace."""
    panel = _panel()
    degraded = apply_failure(panel, FAILURE_LOCALITY_MASK, severity=0.5, seed=1)
    per_state = degraded.groupby("state")["lag_events_sum_12w"].sum()
    silenced = (per_state == 0).sum()
    assert silenced == 2, f"expected 2 of 4 states silenced, got {silenced}"


def test_locality_masking_leaves_labels_intact():
    """Only the information available to the model changes, never the
    outcome being scored against."""
    panel = _panel()
    degraded = apply_failure(panel, FAILURE_LOCALITY_MASK, severity=0.5)
    pd.testing.assert_series_equal(panel["escalation"], degraded["escalation"])


def test_volume_truncation_scales_counts_down():
    panel = _panel()
    degraded = apply_failure(panel, FAILURE_VOLUME_TRUNCATION, severity=0.5)
    column = "lag_events_sum_12w"
    ratio = (degraded[column].sum() / panel[column].sum())
    assert ratio == pytest.approx(0.5, abs=0.01)


def test_unknown_failure_mode_raises():
    with pytest.raises(ValueError, match="unknown failure"):
        apply_failure(_panel(), "cosmic_rays")


def test_degradation_measures_loss_against_a_clean_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel()
    split = _split(panel)
    features = feature_columns(panel)
    model = LogisticBaseline().fit(split.train, features, "escalation")

    results = run_degradation(model, split, "synthetic", features)
    assert results
    for result in results:
        assert result.baseline_average_precision > 0
        # Degradation should not improve performance beyond noise.
        assert result.relative_ap_loss < 0.05


def test_worse_failures_cost_more(tmp_path, monkeypatch):
    """Degradation must be monotone in severity. Masking half the localities
    cannot cost less than masking a tenth, and a two-week delay cannot cost
    less than a one-week delay. Which failure mode hurts most overall depends
    on which source the model relies on, and that is a finding rather than
    something to assert in advance."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel()
    split = _split(panel)
    features = feature_columns(panel)
    model = LogisticBaseline().fit(split.train, features, "escalation")

    results = {
        (r.failure, r.severity): abs(r.relative_ap_loss)
        for r in run_degradation(
            model, split, "synthetic", features,
            failures=(
                (FAILURE_LOCALITY_MASK, 0.10),
                (FAILURE_LOCALITY_MASK, 0.50),
                (FAILURE_SIGNAL_DELAY, 1),
                (FAILURE_SIGNAL_DELAY, 2),
            ),
        )
    }
    assert results[(FAILURE_LOCALITY_MASK, 0.50)] >= results[(FAILURE_LOCALITY_MASK, 0.10)]
    assert results[(FAILURE_SIGNAL_DELAY, 2)] >= results[(FAILURE_SIGNAL_DELAY, 1)]


def test_removing_a_source_the_model_relies_on_is_measurable(tmp_path, monkeypatch):
    """In this fixture protest genuinely drives violence, so removing the
    signal family must cost something. A degradation harness that cannot
    detect the loss of a known-important source cannot be trusted on real
    data."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel()
    split = _split(panel)
    features = feature_columns(panel)
    model = LogisticBaseline().fit(split.train, features, "escalation")

    results = run_degradation(
        model, split, "synthetic", features,
        failures=((FAILURE_SIGNAL_REMOVED, 1.0),),
    )
    assert results[0].relative_ap_loss < -0.01


def test_graceful_degradation_is_detectable():
    from hsre.models.degradation import DegradationResult

    gentle = [
        DegradationResult("f", 1.0, "o", 0.48, 0.24, 0.50, 0.25),
        DegradationResult("g", 1.0, "o", 0.47, 0.23, 0.50, 0.25),
    ]
    cliff = [DegradationResult("h", 1.0, "o", 0.10, 0.05, 0.50, 0.25)]
    assert is_graceful(gentle)
    assert not is_graceful(cliff)


def test_degradation_table_reports_relative_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel()
    split = _split(panel)
    features = feature_columns(panel)
    model = LogisticBaseline().fit(split.train, features, "escalation")
    table = degradation_table(run_degradation(model, split, "s", features))
    assert "relative_ap_loss" in table.columns
    assert "recall_at_capacity" in table.columns


def test_geographic_transfer_scores_unseen_localities(tmp_path, monkeypatch):
    """A model can look competent by learning one dominant locality. Transfer
    to held-out states is the honest measure."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel()
    features = lag_only_features(panel)
    results = geographic_transfer(
        LogisticBaseline, panel, features,
        holdout_sets=(("Alpha",), ("Gamma",)),
        train_end=pd.Timestamp("2018-06-30"),
        outcome_name="synthetic",
    )
    assert len(results) == 2
    for result in results:
        assert result.n > 0
        assert 0 <= result.average_precision <= 1


def test_transfer_table_reports_lift(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel()
    results = geographic_transfer(
        LogisticBaseline, panel, lag_only_features(panel),
        holdout_sets=(("Beta",),),
        train_end=pd.Timestamp("2018-06-30"), outcome_name="s",
    )
    table = transfer_table(results)
    assert "lift" in table.columns
    assert "held_out" in table.columns
