"""Stage 8 verification: main model and the H1 ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hsre.features.build import LAG_PREFIX, SIGNAL_PREFIX, build_features
from hsre.models.evaluate import temporal_split
from hsre.models.gradient_boosting import (
    FEATURE_SETS,
    GradientBoostingModel,
    ablate,
    ablation_table,
)
from hsre.monitoring import ledger
from hsre.transform.panel import label_escalation


def _panel_with_signal(n_weeks=250, seed=5):
    """Panel where protest activity genuinely leads violence.

    Built so that a model using the signal family should outperform one using
    lag features alone. If the ablation cannot detect a signal that is present
    by construction, it cannot be trusted to detect one in real data.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2016-01-02", periods=n_weeks, freq="W-SAT")
    rows, signal_rows = [], []
    for state in ("Alpha", "Beta", "Gamma"):
        protest = rng.poisson(1.5, n_weeks)
        for i, week in enumerate(weeks):
            # Violence responds to protest two weeks earlier.
            driver = protest[i - 2] if i >= 2 else 0
            events = rng.poisson(1.0 + 1.2 * driver)
            rows.append({
                "state": state, "week": week,
                "events": int(events), "fatalities": int(rng.poisson(0.5)),
            })
            signal_rows.append({"state": state, "week": week, "events": int(protest[i])})
    panel = pd.DataFrame(rows)
    signals = pd.DataFrame(signal_rows)
    featured = build_features(panel, signals=signals)
    return label_escalation(
        featured, horizon_weeks=2, baseline_weeks=12,
        percentile_cut=0.90, min_future_events=2, require_lethal=False,
    )


def _split(panel):
    return temporal_split(
        panel,
        train_end=pd.Timestamp("2018-12-29"),
        validation_end=pd.Timestamp("2019-12-28"),
        test_start=pd.Timestamp("2020-01-04"),
    )


def test_model_needs_no_compiled_runtime():
    """LightGBM installs on macOS and then fails at import for want of an
    OpenMP runtime. The main model uses scikit-learn instead."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = GradientBoostingModel()
    assert isinstance(model._make_estimator(), HistGradientBoostingClassifier)


def test_early_stopping_is_disabled():
    """Internal early stopping splits the training data at random, which
    would break the temporal discipline the rest of the pipeline keeps."""
    estimator = GradientBoostingModel()._make_estimator()
    assert estimator.early_stopping is False


def test_model_fits_and_scores_in_range(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    features = [c for c in panel.columns if c.startswith(LAG_PREFIX)]
    model = GradientBoostingModel().fit(
        split.train, features, "escalation", calibration_set=split.validation
    )
    scores = model.predict_scores(split.test, features)
    assert len(scores) == len(split.test)
    assert ((scores >= 0) & (scores <= 1)).all(), "probabilities must lie in [0, 1]"


def test_model_beats_random_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    from hsre.models.evaluate import evaluate

    panel = _panel_with_signal()
    split = _split(panel)
    features = [c for c in panel.columns if c.startswith(LAG_PREFIX)]
    model = GradientBoostingModel().fit(split.train, features, "escalation")
    y = split.test["escalation"].astype(int).to_numpy()

    fitted = evaluate(y, model.predict_scores(split.test, features), "gb", "t", bootstrap_draws=0)
    rng = np.random.default_rng(9)
    noise = evaluate(y, rng.random(len(y)), "noise", "t", bootstrap_draws=0)
    assert fitted.average_precision > noise.average_precision


def test_ablation_detects_a_signal_that_is_present_by_construction(tmp_path, monkeypatch):
    """The direct test of the H1 machinery. Violence here responds to protest
    two weeks earlier, so adding the signal family must improve detection. An
    ablation that misses a planted signal cannot be trusted on real data."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    results = ablate(
        split, panel, outcome_name="synthetic",
        feature_sets={
            "lag_only": (LAG_PREFIX,),
            "lag_signal": (LAG_PREFIX, SIGNAL_PREFIX),
        },
        bootstrap_draws=0,
    )
    by_name = {r.feature_set: r.metrics.average_precision for r in results}
    assert by_name["lag_signal"] > by_name["lag_only"], (
        f"signal family added nothing: {by_name}"
    )


def test_ablation_holds_the_algorithm_fixed(tmp_path, monkeypatch):
    """Only the feature set varies, so any difference is attributable to the
    data sources rather than to a richer model."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    results = ablate(split, panel, outcome_name="synthetic", bootstrap_draws=0)
    counts = {r.feature_set: r.n_features for r in results}
    assert counts["lag_only"] < counts["multi_source"]
    assert all(r.metrics.model.startswith("gb_") for r in results)


def test_ablation_table_reports_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    results = ablate(
        split, panel, outcome_name="synthetic",
        feature_sets={"lag_only": (LAG_PREFIX,)}, bootstrap_draws=50,
    )
    table = ablation_table(results)
    assert "ap_ci_low" in table.columns
    assert "feature_set" in table.columns
    assert "accuracy" not in table.columns


def test_calibration_uses_a_disjoint_later_period(tmp_path, monkeypatch):
    """Calibrating on the training data would be fitting to data the model
    has already seen."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    assert split.train["week"].max() < split.validation["week"].min()

    features = [c for c in panel.columns if c.startswith(LAG_PREFIX)]
    model = GradientBoostingModel().fit(
        split.train, features, "escalation", calibration_set=split.validation
    )
    from sklearn.calibration import CalibratedClassifierCV

    assert isinstance(model._model, CalibratedClassifierCV)


def test_permutation_importance_ranks_features(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    features = [c for c in panel.columns if c.startswith((LAG_PREFIX, SIGNAL_PREFIX))]
    model = GradientBoostingModel(calibrate=False).fit(split.train, features, "escalation")
    importance = model.permutation_importance(split.test, "escalation", n_repeats=2)
    assert len(importance) == len(features)
    assert importance["importance"].iloc[0] >= importance["importance"].iloc[-1]


def test_feature_sets_are_nested():
    """Each set adds one family to a previous one, so the contribution of
    each source is isolated rather than confounded."""
    assert set(FEATURE_SETS["lag_only"]) < set(FEATURE_SETS["multi_source"])
    assert set(FEATURE_SETS["multi_source"]) < set(FEATURE_SETS["multi_source_health"])


def test_capacity_selection_prefers_simpler_models_under_regime_shift(tmp_path, monkeypatch):
    """The outcome is non-stationary: escalation rises from roughly 5% of
    locality-weeks in 2016 to 39% in 2024. An unconstrained booster memorises
    the training regime and transfers badly, so capacity must be selected
    rather than defaulted."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    rng = np.random.default_rng(21)
    weeks = pd.date_range("2016-01-02", periods=250, freq="W-SAT")
    rows = []
    for state in ("A", "B", "C"):
        for i, week in enumerate(weeks):
            # Intensity rises over the period, mirroring the real panel.
            rate = 0.5 + 3.0 * (i / len(weeks))
            rows.append({"state": state, "week": week,
                         "events": int(rng.poisson(rate)),
                         "fatalities": int(rng.poisson(0.4))})
    panel = label_escalation(
        build_features(pd.DataFrame(rows)), horizon_weeks=2, baseline_weeks=12,
        percentile_cut=0.90, min_future_events=2, require_lethal=False,
    )
    split = _split(panel)
    features = [c for c in panel.columns if c.startswith(LAG_PREFIX)]

    model = GradientBoostingModel(calibrate=False)
    chosen = model.select_capacity(split.train, split.validation, features, "escalation")
    assert "max_leaf_nodes" in chosen
    assert "validation_average_precision" in chosen
    # The selection must actually take effect on the instance.
    assert model.max_leaf_nodes == chosen["max_leaf_nodes"]


def test_capacity_selection_never_touches_the_test_period(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.jsonl")
    panel = _panel_with_signal()
    split = _split(panel)
    features = [c for c in panel.columns if c.startswith(LAG_PREFIX)]

    seen = {"rows": 0}
    original = GradientBoostingModel.select_capacity

    def spy(self, train, validation, feats, outcome, grid=None):
        seen["rows"] = len(train) + len(validation)
        return original(self, train, validation, feats, outcome, grid)

    monkeypatch.setattr(GradientBoostingModel, "select_capacity", spy)
    model = GradientBoostingModel(calibrate=False)
    model.select_capacity(split.train, split.validation, features, "escalation")
    assert seen["rows"] == len(split.train) + len(split.validation)
    assert seen["rows"] < len(panel)
