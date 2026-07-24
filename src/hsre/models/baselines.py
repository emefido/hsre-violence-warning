"""Baseline models.

These exist to make H1 falsifiable. The hypothesis claims that combining event
history, public signals and structural vulnerability outperforms event history
alone, and that claim is only meaningful against a baseline that is genuinely
hard to beat.

Conflict-history baselines are notoriously strong in this literature, so the
bar is set deliberately high: the strongest baseline here uses the full lag
feature family, not a token comparison.

Four baselines, in increasing sophistication:

    persistence      last week's activity, no fitting at all
    trailing_rate    the locality's recent mean, a naive but stubborn benchmark
    count            negative binomial on lagged counts, for over-dispersion
    logistic         regularised logistic regression on the full lag family

Every one of them sees only `lag_` features. The distinction from the main
model is the feature set, not the algorithm, which is what isolates the
contribution of the additional sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hsre.features.build import LAG_PREFIX, feature_columns


class Model(Protocol):
    """Minimal interface every model in this study satisfies."""

    name: str

    def fit(self, train: pd.DataFrame, features: list[str], outcome: str) -> "Model": ...

    def predict_scores(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray: ...


def lag_only_features(panel: pd.DataFrame) -> list[str]:
    """Event-history features alone.

    The baseline feature set. H1 is supported only if adding signal and
    structural families improves on this.
    """
    return feature_columns(panel, families=(LAG_PREFIX,))


@dataclass
class PersistenceBaseline:
    """Last week's event count as the score.

    No fitting. Included because violence is strongly autocorrelated and a
    model that cannot beat this has learned nothing.
    """

    name: str = "persistence"
    column: str = f"{LAG_PREFIX}events_1w"

    def fit(self, train, features, outcome):  # noqa: ARG002 - no fitting needed
        return self

    def predict_scores(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:  # noqa: ARG002
        if self.column not in frame.columns:
            raise KeyError(f"persistence baseline requires '{self.column}'")
        return frame[self.column].fillna(0).to_numpy(dtype=float)


@dataclass
class TrailingRateBaseline:
    """The locality's mean activity over the trailing window.

    Smoother than persistence and correspondingly harder to beat, because it
    captures the locality's level rather than one week's noise.
    """

    name: str = "trailing_rate"
    column: str = f"{LAG_PREFIX}events_mean_12w"

    def fit(self, train, features, outcome):  # noqa: ARG002
        return self

    def predict_scores(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:  # noqa: ARG002
        if self.column not in frame.columns:
            raise KeyError(f"trailing rate baseline requires '{self.column}'")
        return frame[self.column].fillna(0).to_numpy(dtype=float)


class CountBaseline:
    """Negative binomial regression on lagged counts.

    Violence counts are over-dispersed, so a Poisson assumption understates
    the variance. Implemented through statsmodels where available, falling
    back to a Poisson fit when the dispersion estimate fails to converge.
    """

    name = "count_nb"

    def __init__(self, columns: tuple[str, ...] | None = None):
        self.columns = columns or (
            f"{LAG_PREFIX}events_sum_4w",
            f"{LAG_PREFIX}events_sum_12w",
            f"{LAG_PREFIX}fatalities_sum_12w",
        )
        self._result = None
        self._used: list[str] = []

    def fit(self, train: pd.DataFrame, features: list[str], outcome: str) -> "CountBaseline":  # noqa: ARG002
        import statsmodels.api as sm

        self._used = [c for c in self.columns if c in train.columns]
        if not self._used:
            raise KeyError("count baseline found none of its required columns")

        X = sm.add_constant(train[self._used].fillna(0).astype(float), has_constant="add")
        y = train[outcome].astype(int)

        # Estimate the dispersion parameter from the data rather than taking
        # the library default of 1.0. Violence counts are over-dispersed and
        # the degree varies by outcome, so a fixed alpha misstates the
        # variance and the library warns about exactly this.
        alpha = self._estimate_alpha(y)
        try:
            self._result = sm.GLM(
                y, X, family=sm.families.NegativeBinomial(alpha=alpha)
            ).fit()
        except Exception:  # noqa: BLE001 - fall back rather than fail the run
            self._result = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        return self

    @staticmethod
    def _estimate_alpha(y: pd.Series) -> float:
        """Method-of-moments dispersion estimate.

        For a negative binomial, variance = mean + alpha * mean^2. Solving for
        alpha gives (variance - mean) / mean^2. Values at or below zero mean
        the data are not over-dispersed, so a small positive floor is used.
        """
        mean = float(y.mean())
        variance = float(y.var())
        if mean <= 0:
            return 1.0
        alpha = (variance - mean) / (mean**2)
        return max(alpha, 1e-6)

    def predict_scores(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:  # noqa: ARG002
        import statsmodels.api as sm

        if self._result is None:
            raise RuntimeError("count baseline is not fitted")
        X = sm.add_constant(frame[self._used].fillna(0).astype(float), has_constant="add")
        return np.asarray(self._result.predict(X), dtype=float)


class LogisticBaseline:
    """Regularised logistic regression on the full lag family.

    The strongest baseline. Because it sees every event-history feature, any
    improvement from the main model is attributable to the additional data
    sources rather than to a richer functional form.
    """

    name = "logistic_lag"

    def __init__(self, C: float = 1.0, seed: int = 20260724):
        self.C = C
        self.seed = seed
        self._pipeline: Pipeline | None = None
        self._features: list[str] = []

    def fit(self, train: pd.DataFrame, features: list[str], outcome: str) -> "LogisticBaseline":
        self._features = [f for f in features if f in train.columns]
        if not self._features:
            raise KeyError("logistic baseline received no usable features")

        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=self.C,
                        max_iter=2000,
                        random_state=self.seed,
                        # Class weighting rather than resampling: the rare
                        # class is preserved rather than duplicated.
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        X = train[self._features].fillna(0).astype(float)
        self._pipeline.fit(X, train[outcome].astype(int))
        return self

    def predict_scores(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:  # noqa: ARG002
        if self._pipeline is None:
            raise RuntimeError("logistic baseline is not fitted")
        X = frame[self._features].fillna(0).astype(float)
        return self._pipeline.predict_proba(X)[:, 1]


def default_baselines() -> list[Model]:
    return [
        PersistenceBaseline(),
        TrailingRateBaseline(),
        CountBaseline(),
        LogisticBaseline(),
    ]
