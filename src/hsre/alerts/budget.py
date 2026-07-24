"""Alert allocation under a capacity constraint.

The central operational question of the study. A model produces a risk score
for every locality-week, but an institution can investigate only a handful.
Which localities it alerts on is therefore a decision about capacity, not
about the model.

Three regimes are compared.

    accuracy_optimised   threshold maximising F1, the field default. Ignores
                         how many alerts the responder can actually handle.
    recall_optimised     threshold catching most escalations. Maximises alert
                         volume and produces alert fatigue.
    alert_budgeted       volume fixed at declared capacity, detection reported
                         at that volume. The threshold becomes a function of
                         institutional capacity rather than model output.

The error budget takes two linked constraints rather than one. A false alarm
and a missed death are not equivalent, so a maximum review burden alone would
permit a service that never misses anything by alerting on everything, and a
minimum recall alone would permit one that alerts on nothing. A release fails
when either constraint is breached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hsre.monitoring import ledger

REGIME_ACCURACY = "accuracy_optimised"
REGIME_RECALL = "recall_optimised"
REGIME_BUDGET = "alert_budgeted"


@dataclass
class AlertOutcome:
    """Result of applying one alerting regime to one test period."""

    regime: str
    outcome: str
    threshold: float
    alerts_total: int
    alerts_per_week: float
    true_alerts: int
    false_alerts: int
    false_alerts_per_week: float
    missed: int
    precision: float
    recall: float
    severe_recall: float | None = None
    review_burden_breached: bool = False
    severe_recall_breached: bool = False

    @property
    def passes_error_budget(self) -> bool:
        return not (self.review_burden_breached or self.severe_recall_breached)

    def to_row(self) -> dict:
        return {
            "outcome": self.outcome,
            "regime": self.regime,
            "threshold": round(self.threshold, 4),
            "alerts_total": self.alerts_total,
            "alerts_per_week": round(self.alerts_per_week, 2),
            "false_alerts_per_week": round(self.false_alerts_per_week, 2),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "severe_recall": (
                round(self.severe_recall, 4) if self.severe_recall is not None else None
            ),
            "missed": self.missed,
            "review_burden_breached": self.review_burden_breached,
            "severe_recall_breached": self.severe_recall_breached,
            "passes_error_budget": self.passes_error_budget,
        }


def _confusion(y_true: np.ndarray, alerted: np.ndarray) -> tuple[int, int, int]:
    true_alerts = int((alerted & (y_true == 1)).sum())
    false_alerts = int((alerted & (y_true == 0)).sum())
    missed = int(((~alerted) & (y_true == 1)).sum())
    return true_alerts, false_alerts, missed


def accuracy_optimised_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1, the conventional default.

    Included because it is what the field reports, and because comparing it
    against a capacity-aware threshold is the point of the exercise.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.999, 200)))
    best_f1, best_threshold = -1.0, float(candidates[0])
    for threshold in candidates:
        alerted = scores >= threshold
        tp, fp, fn = _confusion(y_true, alerted)
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else 0.0
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def recall_optimised_threshold(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.80
) -> float:
    """Lowest threshold reaching a target recall.

    The regime that produces alert fatigue: it catches most escalations by
    alerting on a large share of localities.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_true = y_true[order]
    total_positives = sorted_true.sum()
    if total_positives == 0:
        return float(scores.max())

    cumulative = np.cumsum(sorted_true) / total_positives
    reached = np.searchsorted(cumulative, target_recall)
    index = min(int(reached), len(scores) - 1)
    return float(scores[order][index])


def budget_threshold(scores: np.ndarray, alerts_allowed: int) -> float:
    """Threshold admitting exactly the declared number of alerts."""
    if alerts_allowed >= len(scores):
        return float(scores.min())
    ordered = np.sort(scores)[::-1]
    return float(ordered[alerts_allowed - 1])


def apply_regime(
    frame: pd.DataFrame,
    scores: np.ndarray,
    regime: str,
    outcome_name: str,
    outcome_column: str = "escalation",
    severe_column: str | None = None,
    capacity_per_week: int = 4,
    target_recall: float = 0.80,
    max_review_burden_per_week: int = 10,
    min_severe_recall: float = 0.70,
) -> AlertOutcome:
    """Apply one alerting regime and evaluate it against the error budget."""
    y_true = frame[outcome_column].astype(int).to_numpy()
    n_weeks = frame["week"].nunique()

    if regime == REGIME_ACCURACY:
        threshold = accuracy_optimised_threshold(y_true, scores)
    elif regime == REGIME_RECALL:
        threshold = recall_optimised_threshold(y_true, scores, target_recall)
    elif regime == REGIME_BUDGET:
        threshold = budget_threshold(scores, capacity_per_week * n_weeks)
    else:
        raise ValueError(f"unknown regime: {regime}")

    alerted = scores >= threshold
    true_alerts, false_alerts, missed = _confusion(y_true, alerted)
    total_alerts = int(alerted.sum())

    precision = true_alerts / total_alerts if total_alerts else 0.0
    recall = true_alerts / y_true.sum() if y_true.sum() else 0.0

    severe_recall = None
    if severe_column and severe_column in frame.columns:
        severe = frame[severe_column].astype(int).to_numpy() == 1
        if severe.sum():
            severe_recall = float((alerted & severe).sum() / severe.sum())

    alerts_per_week = total_alerts / n_weeks if n_weeks else 0.0

    outcome = AlertOutcome(
        regime=regime,
        outcome=outcome_name,
        threshold=threshold,
        alerts_total=total_alerts,
        alerts_per_week=alerts_per_week,
        true_alerts=true_alerts,
        false_alerts=false_alerts,
        false_alerts_per_week=false_alerts / n_weeks if n_weeks else 0.0,
        missed=missed,
        precision=precision,
        recall=recall,
        severe_recall=severe_recall,
        review_burden_breached=alerts_per_week > max_review_burden_per_week,
        severe_recall_breached=(
            severe_recall is not None and severe_recall < min_severe_recall
        ),
    )

    ledger.record(
        stage="alerts",
        status=ledger.STATUS_OK if outcome.passes_error_budget else ledger.STATUS_THIN,
        outcome=outcome_name,
        regime=regime,
        alerts_per_week=round(alerts_per_week, 2),
        precision=round(precision, 4),
        recall=round(recall, 4),
        passes_error_budget=outcome.passes_error_budget,
    )
    return outcome


@dataclass
class BudgetCurve:
    """Detection achieved across the full range of alert volumes.

    The central empirical exhibit. If detection degrades slowly as volume
    falls, the field's implicit preference for sensitivity is expensive and
    hard to justify.
    """

    outcome: str
    points: pd.DataFrame = field(default_factory=pd.DataFrame)

    def detection_at(self, alerts_per_week: float) -> float:
        if self.points.empty:
            return 0.0
        idx = (self.points["alerts_per_week"] - alerts_per_week).abs().idxmin()
        return float(self.points.loc[idx, "recall"])

    def volume_for_recall(self, target: float) -> float:
        """Alerts per week needed to reach a target recall."""
        reached = self.points.loc[self.points["recall"] >= target]
        if reached.empty:
            return float("inf")
        return float(reached["alerts_per_week"].min())


def budget_curve(
    frame: pd.DataFrame,
    scores: np.ndarray,
    outcome_name: str,
    outcome_column: str = "escalation",
    severe_column: str | None = None,
    max_alerts_per_week: int = 37,
) -> BudgetCurve:
    """Detection and precision at every feasible alert volume."""
    y_true = frame[outcome_column].astype(int).to_numpy()
    n_weeks = frame["week"].nunique()
    total_positives = int(y_true.sum())

    severe = None
    if severe_column and severe_column in frame.columns:
        severe = frame[severe_column].astype(int).to_numpy() == 1

    order = np.argsort(-scores, kind="stable")
    sorted_true = y_true[order]
    cumulative_true = np.cumsum(sorted_true)

    rows = []
    for per_week in range(1, max_alerts_per_week + 1):
        k = min(per_week * n_weeks, len(scores))
        captured = int(cumulative_true[k - 1])
        row = {
            "alerts_per_week": per_week,
            "alerts_total": k,
            "true_alerts": captured,
            "false_alerts": k - captured,
            "false_alerts_per_week": (k - captured) / n_weeks,
            "precision": captured / k,
            "recall": captured / total_positives if total_positives else 0.0,
        }
        if severe is not None and severe.sum():
            sorted_severe = severe[order]
            row["severe_recall"] = float(sorted_severe[:k].sum() / severe.sum())
        rows.append(row)

    return BudgetCurve(outcome=outcome_name, points=pd.DataFrame(rows))


def compare_regimes(
    frame: pd.DataFrame,
    scores: np.ndarray,
    outcome_name: str,
    capacity_per_week: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """Run all three regimes and return the comparison table."""
    outcomes = [
        apply_regime(
            frame, scores, regime, outcome_name,
            capacity_per_week=capacity_per_week, **kwargs,
        )
        for regime in (REGIME_ACCURACY, REGIME_RECALL, REGIME_BUDGET)
    ]
    return pd.DataFrame([o.to_row() for o in outcomes])
