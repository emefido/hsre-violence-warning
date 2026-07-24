"""Degradation experiments and geographic transfer.

Two validation regimes that conventional forecasting evaluation omits.

**Source failure.** H4 claims forecast errors are associated with missing,
delayed or uneven source data. Testing that without a live deployment requires
simulating the failures against real data: delay a source, remove it, mask a
share of reporting localities, truncate its volume. The question is whether
the service degrades gradually or falls off a cliff, and whether it degrades
evenly or concentrates the damage somewhere.

This matters operationally because a system that fails gracefully can keep
running in a flag-raised state, while one that fails abruptly must stop. An
institution cannot know which it has without testing.

**Geographic transfer.** A model can score well by learning one dominant
locality. Holding out entire states tests whether it has learned anything that
transfers, which is the honest measure when the outcome is concentrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hsre.features.build import (
    HEALTH_PREFIX,
    SIGNAL_PREFIX,
    SPATIAL_PREFIX,
    feature_columns,
)
from hsre.monitoring import ledger

FAILURE_NONE = "none"
FAILURE_SIGNAL_DELAY = "signal_delayed"
FAILURE_SIGNAL_REMOVED = "signal_removed"
FAILURE_SPATIAL_REMOVED = "spatial_removed"
FAILURE_LOCALITY_MASK = "localities_masked"
FAILURE_VOLUME_TRUNCATION = "volume_truncated"


@dataclass
class DegradationResult:
    """Performance under one simulated source failure."""

    failure: str
    severity: float
    outcome: str
    average_precision: float
    recall_at_capacity: float
    baseline_average_precision: float
    baseline_recall_at_capacity: float

    @property
    def ap_change(self) -> float:
        return self.average_precision - self.baseline_average_precision

    @property
    def recall_change(self) -> float:
        return self.recall_at_capacity - self.baseline_recall_at_capacity

    @property
    def relative_ap_loss(self) -> float:
        if self.baseline_average_precision == 0:
            return 0.0
        return self.ap_change / self.baseline_average_precision

    def to_row(self) -> dict:
        return {
            "outcome": self.outcome,
            "failure": self.failure,
            "severity": self.severity,
            "average_precision": round(self.average_precision, 4),
            "ap_change": round(self.ap_change, 4),
            "relative_ap_loss": round(self.relative_ap_loss, 4),
            "recall_at_capacity": round(self.recall_at_capacity, 4),
            "recall_change": round(self.recall_change, 4),
        }


def apply_failure(
    frame: pd.DataFrame,
    failure: str,
    severity: float = 1.0,
    seed: int = 20260724,
    group_column: str = "state",
) -> pd.DataFrame:
    """Simulate one source failure on an already-featured panel.

    Failures are applied to the features rather than re-run through ingestion,
    which keeps the comparison clean: the same rows, the same labels, only the
    information available to the model changes.
    """
    out = frame.copy()
    rng = np.random.default_rng(seed)

    if failure == FAILURE_NONE:
        return out

    if failure == FAILURE_SIGNAL_DELAY:
        # The source still reports, but late. Features shift by the delay.
        weeks = int(severity)
        signal_columns = [c for c in out.columns if c.startswith(SIGNAL_PREFIX)]
        for column in signal_columns:
            out[column] = out.groupby(group_column)[column].transform(
                lambda s: s.shift(weeks)
            )
        return out

    if failure == FAILURE_SIGNAL_REMOVED:
        # The source stops entirely. Its features go to a constant, which is
        # what a model sees when a feed dies and imputation fills the gap.
        for column in [c for c in out.columns if c.startswith(SIGNAL_PREFIX)]:
            out[column] = 0.0
        return out

    if failure == FAILURE_SPATIAL_REMOVED:
        for column in [c for c in out.columns if c.startswith(SPATIAL_PREFIX)]:
            out[column] = 0.0
        return out

    if failure == FAILURE_LOCALITY_MASK:
        # A share of localities stop reporting. Their event history collapses
        # to zero, which is the dangerous case: indistinguishable from peace.
        localities = out[group_column].unique()
        n_masked = max(int(len(localities) * severity), 1)
        masked = rng.choice(localities, size=n_masked, replace=False)
        rows = out[group_column].isin(masked)
        history = [c for c in feature_columns(out) if not c.startswith(HEALTH_PREFIX)]
        out.loc[rows, history] = 0.0
        return out

    if failure == FAILURE_VOLUME_TRUNCATION:
        # The source reports a reduced share of events. Counts scale down
        # while the outcome does not, so the model sees a quieter world than
        # the one it is being scored against.
        countable = [
            c
            for c in feature_columns(out)
            if ("events" in c or "fatalities" in c) and not c.startswith(HEALTH_PREFIX)
        ]
        for column in countable:
            out[column] = out[column] * (1 - severity)
        return out

    raise ValueError(f"unknown failure mode: {failure}")


def run_degradation(
    model,
    split,
    outcome_name: str,
    features: list[str],
    outcome_column: str = "escalation",
    capacity_per_week: int = 4,
    failures: tuple[tuple[str, float], ...] | None = None,
    seed: int = 20260724,
) -> list[DegradationResult]:
    """Score the fitted model under each simulated failure.

    The model is fitted once on clean training data and then scored on
    degraded test data. That mirrors deployment: a model trained in good
    conditions meets a feed that later fails.
    """
    from sklearn.metrics import average_precision_score

    from hsre.alerts.budget import budget_curve

    cases = failures or (
        (FAILURE_SIGNAL_DELAY, 1),
        (FAILURE_SIGNAL_DELAY, 2),
        (FAILURE_SIGNAL_REMOVED, 1.0),
        (FAILURE_SPATIAL_REMOVED, 1.0),
        (FAILURE_LOCALITY_MASK, 0.10),
        (FAILURE_LOCALITY_MASK, 0.25),
        (FAILURE_LOCALITY_MASK, 0.50),
        (FAILURE_VOLUME_TRUNCATION, 0.25),
        (FAILURE_VOLUME_TRUNCATION, 0.50),
    )

    y_true = split.test[outcome_column].astype(int).to_numpy()

    clean_scores = model.predict_scores(split.test, features)
    baseline_ap = float(average_precision_score(y_true, clean_scores))
    baseline_curve = budget_curve(split.test, clean_scores, outcome_name)
    baseline_recall = baseline_curve.detection_at(capacity_per_week)

    results = []
    for failure, severity in cases:
        degraded = apply_failure(split.test, failure, severity, seed=seed)
        scores = model.predict_scores(degraded, features)
        ap = float(average_precision_score(y_true, scores))
        curve = budget_curve(degraded.assign(**{outcome_column: y_true}), scores, outcome_name)

        result = DegradationResult(
            failure=failure,
            severity=severity,
            outcome=outcome_name,
            average_precision=ap,
            recall_at_capacity=curve.detection_at(capacity_per_week),
            baseline_average_precision=baseline_ap,
            baseline_recall_at_capacity=baseline_recall,
        )
        results.append(result)

        ledger.record(
            stage="degradation",
            status=ledger.STATUS_OK,
            outcome=outcome_name,
            failure=failure,
            severity=severity,
            average_precision=round(ap, 4),
            relative_ap_loss=round(result.relative_ap_loss, 4),
        )
    return results


def degradation_table(results: list[DegradationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def is_graceful(
    results: list[DegradationResult], cliff_threshold: float = 0.25
) -> bool:
    """Whether degradation is gradual rather than abrupt.

    A service that fails gracefully can keep running with a flag raised. One
    that falls off a cliff must stop. The distinction determines whether
    abstention or continuation is the correct failure behaviour.
    """
    return all(abs(r.relative_ap_loss) < cliff_threshold for r in results)


@dataclass
class TransferResult:
    """Performance on localities the model never saw in training."""

    held_out: tuple[str, ...]
    outcome: str
    average_precision: float
    base_rate: float
    n: int

    @property
    def lift(self) -> float:
        return self.average_precision / self.base_rate if self.base_rate else 0.0

    def to_row(self) -> dict:
        return {
            "outcome": self.outcome,
            "held_out": ", ".join(self.held_out),
            "n": self.n,
            "base_rate": round(self.base_rate, 4),
            "average_precision": round(self.average_precision, 4),
            "lift": round(self.lift, 3),
        }


def geographic_transfer(
    model_factory,
    panel: pd.DataFrame,
    features: list[str],
    holdout_sets: tuple[tuple[str, ...], ...],
    train_end: pd.Timestamp,
    outcome_name: str,
    outcome_column: str = "escalation",
) -> list[TransferResult]:
    """Train without certain localities, then score on them.

    Reported because the outcome is concentrated: a model can appear
    competent by learning the dominant locality alone. Transfer to unseen
    localities is the honest test.
    """
    from sklearn.metrics import average_precision_score

    labelled = panel.loc[panel[outcome_column].notna()].copy()
    labelled["week"] = pd.to_datetime(labelled["week"])

    results = []
    for holdout in holdout_sets:
        seen = labelled.loc[~labelled["state"].isin(holdout)]
        held = labelled.loc[labelled["state"].isin(holdout)]
        train = seen.loc[seen["week"] <= train_end]
        test = held.loc[held["week"] > train_end]
        if test.empty or test[outcome_column].sum() == 0:
            continue

        model = model_factory()
        model.fit(train, features, outcome_column)
        scores = model.predict_scores(test, features)
        y = test[outcome_column].astype(int).to_numpy()

        result = TransferResult(
            held_out=holdout,
            outcome=outcome_name,
            average_precision=float(average_precision_score(y, scores)),
            base_rate=float(y.mean()),
            n=len(test),
        )
        results.append(result)

        ledger.record(
            stage="transfer",
            status=ledger.STATUS_OK,
            outcome=outcome_name,
            held_out=list(holdout),
            average_precision=round(result.average_precision, 4),
            lift=round(result.lift, 3),
        )
    return results


def transfer_table(results: list[TransferResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])
