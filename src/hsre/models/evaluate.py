"""Evaluation.

Two decisions here determine whether the results mean anything.

Splits are temporal, never random. Neighbouring weeks in the same locality are
strongly dependent, so random splitting leaks the future into the training set
and inflates performance substantially. Train, validation and test boundaries
come from config and are fixed before any modelling.

Accuracy is not reported. With a 17% base rate a model predicting "no
escalation" everywhere scores 83%, which is worse than useless as a summary.
Average precision is the headline metric, following the convention in
subnational conflict forecasting, and it is reported alongside precision and
recall at a fixed alert budget because that is what an institution can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hsre.monitoring import ledger


@dataclass
class Split:
    """One temporal partition of the panel."""

    name: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.name,
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "test_rows": len(self.test),
            "train_positives": int(self.train["escalation"].sum()),
            "test_positives": int(self.test["escalation"].sum()),
            "test_base_rate": round(float(self.test["escalation"].mean()), 4),
        }


def temporal_split(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    outcome: str = "escalation",
) -> Split:
    """Partition by date. The test period stays untouched until selection ends.

    A gap between validation_end and test_start is preserved if the config
    defines one, so that the forecast horizon of the last validation week does
    not overlap the first test week.
    """
    labelled = panel.loc[panel[outcome].notna()].copy()
    labelled["week"] = pd.to_datetime(labelled["week"])

    train = labelled.loc[labelled["week"] <= train_end]
    validation = labelled.loc[
        (labelled["week"] > train_end) & (labelled["week"] <= validation_end)
    ]
    test = labelled.loc[labelled["week"] >= test_start]

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if part.empty:
            raise ValueError(f"temporal split produced an empty {name} set")

    return Split("temporal", train, validation, test)


def geographic_split(
    panel: pd.DataFrame,
    holdout_localities: list[str],
    train_end: pd.Timestamp,
    outcome: str = "escalation",
) -> Split:
    """Hold out entire localities to test spatial transfer.

    Reported because a model can score well by learning one dominant locality.
    Performance on unseen localities is the honest measure of whether the
    model has learned anything transferable.
    """
    labelled = panel.loc[panel[outcome].notna()].copy()
    labelled["week"] = pd.to_datetime(labelled["week"])

    seen = labelled.loc[~labelled["state"].isin(holdout_localities)]
    held = labelled.loc[labelled["state"].isin(holdout_localities)]
    if held.empty:
        raise ValueError(f"no rows for holdout localities: {holdout_localities}")

    train = seen.loc[seen["week"] <= train_end]
    validation = seen.loc[seen["week"] > train_end]
    return Split("geographic", train, validation, held)


@dataclass
class Metrics:
    """Evaluation of one model on one test set."""

    model: str
    outcome: str
    n: int
    positives: int
    base_rate: float
    average_precision: float
    roc_auc: float
    brier: float
    precision_at_k: dict[float, float] = field(default_factory=dict)
    recall_at_k: dict[float, float] = field(default_factory=dict)
    ap_ci: tuple[float, float] | None = None

    def lift(self) -> float:
        """Average precision relative to the base rate.

        A model with no skill scores average precision equal to the base rate,
        so lift of 1.0 means no skill regardless of the absolute number.
        """
        return self.average_precision / self.base_rate if self.base_rate else 0.0

    def to_row(self) -> dict[str, Any]:
        row = {
            "model": self.model,
            "outcome": self.outcome,
            "n": self.n,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 4),
            "average_precision": round(self.average_precision, 4),
            "lift": round(self.lift(), 3),
            "roc_auc": round(self.roc_auc, 4),
            "brier": round(self.brier, 4),
        }
        if self.ap_ci:
            row["ap_ci_low"] = round(self.ap_ci[0], 4)
            row["ap_ci_high"] = round(self.ap_ci[1], 4)
        for k, value in sorted(self.precision_at_k.items()):
            row[f"precision_at_{int(k * 100)}pct"] = round(value, 4)
        for k, value in sorted(self.recall_at_k.items()):
            row[f"recall_at_{int(k * 100)}pct"] = round(value, 4)
        return row


def precision_recall_at_k(
    y_true: np.ndarray, scores: np.ndarray, k_share: float
) -> tuple[float, float]:
    """Precision and recall when only the top k share of rows are alerted.

    This is the operational metric. An institution that can review 2% of
    localities per week needs to know what it catches at 2%, not what the
    model achieves across every threshold it will never use.
    """
    n = len(scores)
    k = max(int(np.ceil(n * k_share)), 1)
    order = np.argsort(-scores, kind="stable")[:k]
    selected = y_true[order]
    total_positives = y_true.sum()

    precision = float(selected.sum() / k) if k else 0.0
    recall = float(selected.sum() / total_positives) if total_positives else 0.0
    return precision, recall


def bootstrap_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
    draws: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260724,
) -> tuple[float, float]:
    """Percentile bootstrap interval for average precision.

    Resamples prediction-outcome pairs, which is the convention in this
    literature for comparing models on the same test set.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        sample_true = y_true[idx]
        if sample_true.sum() == 0:
            values[i] = np.nan
            continue
        values[i] = average_precision_score(sample_true, scores[idx])
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1 - alpha)),
    )


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    outcome_name: str,
    k_shares: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    bootstrap_draws: int = 1000,
    seed: int = 20260724,
) -> Metrics:
    """Full evaluation of one score vector."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores have different lengths")
    if y_true.sum() == 0:
        raise ValueError("test set contains no positive cases")

    metrics = Metrics(
        model=model_name,
        outcome=outcome_name,
        n=len(y_true),
        positives=int(y_true.sum()),
        base_rate=float(y_true.mean()),
        average_precision=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        brier=float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
    )

    for share in k_shares:
        precision, recall = precision_recall_at_k(y_true, scores, share)
        metrics.precision_at_k[share] = precision
        metrics.recall_at_k[share] = recall

    if bootstrap_draws:
        metrics.ap_ci = bootstrap_ap(y_true, scores, draws=bootstrap_draws, seed=seed)

    ledger.record(
        stage="evaluate",
        status=ledger.STATUS_OK,
        model=model_name,
        outcome=outcome_name,
        n=metrics.n,
        average_precision=round(metrics.average_precision, 4),
        lift=round(metrics.lift(), 3),
        base_rate=round(metrics.base_rate, 4),
    )
    return metrics


def comparison_table(results: list[Metrics]) -> pd.DataFrame:
    """Assemble model results into the table reported in the manuscript."""
    frame = pd.DataFrame([m.to_row() for m in results])
    return frame.sort_values(
        ["outcome", "average_precision"], ascending=[True, False]
    ).reset_index(drop=True)
