"""Main model and feature ablation.

The main model is scikit-learn's histogram gradient boosting: the same
algorithm class as LightGBM, without an external OpenMP runtime. That choice
is deliberate. A replication package that fails to import on a reviewer's
machine is a reliability failure of exactly the kind this study examines.

H1 claims that combining event history, public signals and structural
vulnerability outperforms event history alone. Testing it requires holding the
algorithm fixed and varying only the feature set, which is what `ablate` does.
Any improvement is then attributable to the additional sources rather than to
a richer functional form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from hsre.features.build import (
    HEALTH_PREFIX,
    LAG_PREFIX,
    SEASONAL_PREFIX,
    SIGNAL_PREFIX,
    SPATIAL_PREFIX,
    feature_columns,
)
from hsre.models.evaluate import Metrics, Split, evaluate

# Feature sets for the ablation. Each adds one family to the one before, so
# the contribution of each source is isolated.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "lag_only": (LAG_PREFIX,),
    "lag_seasonal": (LAG_PREFIX, SEASONAL_PREFIX),
    "lag_spatial": (LAG_PREFIX, SEASONAL_PREFIX, SPATIAL_PREFIX),
    "lag_signal": (LAG_PREFIX, SEASONAL_PREFIX, SIGNAL_PREFIX),
    "multi_source": (
        LAG_PREFIX,
        SEASONAL_PREFIX,
        SPATIAL_PREFIX,
        SIGNAL_PREFIX,
    ),
    "multi_source_health": (
        LAG_PREFIX,
        SEASONAL_PREFIX,
        SPATIAL_PREFIX,
        SIGNAL_PREFIX,
        HEALTH_PREFIX,
    ),
}


class GradientBoostingModel:
    """Histogram gradient boosting with optional probability calibration.

    Calibration matters here because the alert-budget analysis in the next
    stage ranks localities by predicted probability and reports precision at a
    fixed volume. Miscalibrated scores still rank correctly, but the Brier
    score and any probabilistic statement about risk would be misleading.
    """

    def __init__(
        self,
        name: str = "gradient_boosting",
        seed: int = 20260724,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        calibrate: bool = True,
    ):
        self.name = name
        self.seed = seed
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.calibrate = calibrate
        self._model = None
        self._features: list[str] = []

    def _make_estimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.seed,
            # Early stopping is disabled: it would split the training data
            # internally and at random, which breaks the temporal discipline
            # the rest of the pipeline maintains.
            early_stopping=False,
        )

    def fit(
        self,
        train: pd.DataFrame,
        features: list[str],
        outcome: str,
        calibration_set: pd.DataFrame | None = None,
    ) -> "GradientBoostingModel":
        self._features = [f for f in features if f in train.columns]
        if not self._features:
            raise KeyError("gradient boosting model received no usable features")

        X = train[self._features].astype(float)
        y = train[outcome].astype(int)

        estimator = self._make_estimator()

        if self.calibrate and calibration_set is not None and len(calibration_set) > 50:
            # Fit on train, then calibrate on the validation period. Using a
            # later, disjoint period keeps the calibration honest rather than
            # fitting it on data the model has already seen.
            estimator.fit(X, y)
            self._model = self._calibrate(estimator, calibration_set, outcome)
        else:
            estimator.fit(X, y)
            self._model = estimator
        return self

    def select_capacity(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        features: list[str],
        outcome: str,
        grid: tuple[dict, ...] | None = None,
    ) -> dict:
        """Choose model capacity on the validation period.

        Necessary because the outcome is non-stationary: escalation rises from
        roughly 5% of locality-weeks in 2016 to 39% in 2024, so the training
        period is a materially different regime from the test period. An
        unconstrained booster memorises the training regime and transfers
        badly, while a constrained one generalises. Capacity is therefore a
        parameter to select rather than a default to accept, and it is
        selected on a period the test set does not include.
        """
        from sklearn.metrics import average_precision_score

        candidates = grid or (
            {"max_leaf_nodes": 4, "min_samples_leaf": 100, "max_iter": 200},
            {"max_leaf_nodes": 8, "min_samples_leaf": 100, "max_iter": 200},
            {"max_leaf_nodes": 15, "min_samples_leaf": 60, "max_iter": 250},
            {"max_leaf_nodes": 31, "min_samples_leaf": 40, "max_iter": 300},
        )

        usable = [f for f in features if f in train.columns]
        X_train = train[usable].astype(float)
        y_train = train[outcome].astype(int)
        X_val = validation[usable].astype(float)
        y_val = validation[outcome].astype(int)

        best_score, best_params = -np.inf, dict(candidates[0])
        for params in candidates:
            estimator = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate,
                l2_regularization=self.l2_regularization,
                random_state=self.seed,
                early_stopping=False,
                **params,
            )
            estimator.fit(X_train, y_train)
            score = average_precision_score(
                y_val, estimator.predict_proba(X_val)[:, 1]
            )
            if score > best_score:
                best_score, best_params = score, dict(params)

        self.max_leaf_nodes = best_params["max_leaf_nodes"]
        self.min_samples_leaf = best_params["min_samples_leaf"]
        self.max_iter = best_params["max_iter"]
        return {**best_params, "validation_average_precision": float(best_score)}

    def _calibrate(
        self,
        fitted: HistGradientBoostingClassifier,
        calibration_set: pd.DataFrame,
        outcome: str,
    ):
        """Wrap a fitted estimator in an isotonic calibrator.

        scikit-learn replaced `cv="prefit"` with `FrozenEstimator` in 1.6, so
        both paths are supported. Pinning a version instead would make the
        replication package fail on whichever side of that boundary a
        reviewer's environment happens to sit.
        """
        X_cal = calibration_set[self._features].astype(float)
        y_cal = calibration_set[outcome].astype(int)

        try:
            from sklearn.frozen import FrozenEstimator

            calibrator = CalibratedClassifierCV(
                FrozenEstimator(fitted), method="isotonic"
            )
        except ImportError:
            calibrator = CalibratedClassifierCV(fitted, method="isotonic", cv="prefit")

        calibrator.fit(X_cal, y_cal)
        return calibrator

    def predict_scores(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:  # noqa: ARG002
        if self._model is None:
            raise RuntimeError("model is not fitted")
        X = frame[self._features].astype(float)
        return self._model.predict_proba(X)[:, 1]

    def permutation_importance(
        self,
        frame: pd.DataFrame,
        outcome: str,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Drop in average precision when each feature is shuffled.

        Reported as a predictive association rather than a causal effect.
        """
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(self.seed)
        y = frame[outcome].astype(int).to_numpy()
        baseline = average_precision_score(y, self.predict_scores(frame, self._features))

        rows = []
        for feature in self._features:
            drops = []
            for _ in range(n_repeats):
                shuffled = frame.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                score = average_precision_score(
                    y, self.predict_scores(shuffled, self._features)
                )
                drops.append(baseline - score)
            rows.append(
                {
                    "feature": feature,
                    "importance": float(np.mean(drops)),
                    "std": float(np.std(drops)),
                }
            )
        return pd.DataFrame(rows).sort_values("importance", ascending=False)


@dataclass
class AblationResult:
    feature_set: str
    n_features: int
    metrics: Metrics

    def to_row(self) -> dict:
        row = self.metrics.to_row()
        row["feature_set"] = self.feature_set
        row["n_features"] = self.n_features
        return row


def ablate(
    split: Split,
    panel: pd.DataFrame,
    outcome_name: str,
    outcome_column: str = "escalation",
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    seed: int = 20260724,
    bootstrap_draws: int = 1000,
) -> list[AblationResult]:
    """Fit the same algorithm on progressively richer feature sets.

    This is the direct test of H1. The algorithm, hyperparameters and split
    are identical across runs, so differences are attributable to the feature
    families alone.
    """
    sets = feature_sets or FEATURE_SETS
    y_test = split.test[outcome_column].astype(int).to_numpy()

    results = []
    for name, prefixes in sets.items():
        features = feature_columns(panel, families=prefixes)
        if not features:
            continue
        model = GradientBoostingModel(name=f"gb_{name}", seed=seed)
        # Capacity is selected on the validation period for every feature set,
        # so no set is advantaged by a hyperparameter chosen for another.
        model.select_capacity(split.train, split.validation, features, outcome_column)
        model.fit(
            split.train,
            features,
            outcome_column,
            calibration_set=split.validation,
        )
        scores = model.predict_scores(split.test, features)
        metrics = evaluate(
            y_test,
            scores,
            model_name=f"gb_{name}",
            outcome_name=outcome_name,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        )
        results.append(AblationResult(name, len(features), metrics))
    return results


def ablation_table(results: list[AblationResult]) -> pd.DataFrame:
    frame = pd.DataFrame([r.to_row() for r in results])
    ordered = [
        "outcome",
        "feature_set",
        "n_features",
        "average_precision",
        "lift",
        "ap_ci_low",
        "ap_ci_high",
        "brier",
        "precision_at_2pct",
        "recall_at_2pct",
    ]
    present = [c for c in ordered if c in frame.columns]
    return frame[present + [c for c in frame.columns if c not in present]]
