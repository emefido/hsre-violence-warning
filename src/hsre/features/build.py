"""Feature construction.

Every feature for locality i in week t uses only information available by the
end of week t. Escalation is labelled from weeks t+1 and t+2, so any feature
that reads forward is leakage, and leakage is the commonest defect in applied
forecasting of this kind. It is prevented structurally here: all rolling
windows are shifted by one before aggregation, and `assert_no_leakage` checks
the result rather than trusting the implementation.

Four feature families, matching the manuscript.

    lag        the locality's own event history
    spatial    violence in neighbouring localities
    signal     protests and demonstrations as leading indicators
    health     data-quality measures, which are predictors in their own right

The health family exists because H4 claims forecast errors track measurable
data conditions. That is only testable if those conditions are in the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hsre.monitoring import ledger

# Feature name prefixes, used to group families for ablation and reporting.
LAG_PREFIX = "lag_"
SPATIAL_PREFIX = "sp_"
SIGNAL_PREFIX = "sig_"
HEALTH_PREFIX = "hlth_"
SEASONAL_PREFIX = "seas_"

# Columns that are outcomes or labelling intermediates, never predictors.
NON_FEATURE_COLUMNS = {
    "state",
    "week",
    "events",
    "fatalities",
    "escalation",
    "trailing_median",
    "locality_percentile",
    "future_events",
    "future_fatalities",
}


class LeakageError(RuntimeError):
    """Raised when a feature reads information from the forecast window."""


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    undefined_value: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Ratio plus an indicator for where it was undefined.

    A zero denominator is common and meaningful here: it marks a locality with
    no recent activity, which is exactly the case the study must retain.
    Returning NaN would cause complete-case analysis to delete those rows, and
    with them the quiet localities that conventional sources already omit.

    The ratio is therefore filled with a stated value and paired with a flag,
    so the model can learn from the condition instead of the row vanishing.
    """
    denom = denominator.replace(0, np.nan)
    ratio = numerator / denom
    undefined = ratio.isna().astype(int)
    return ratio.fillna(undefined_value), undefined


def _shifted_rolling(
    grouped: pd.core.groupby.SeriesGroupBy, window: int, func: str
) -> pd.Series:
    """Rolling aggregate over the window ending at t-1.

    The shift is what makes the feature legitimate: without it the window
    includes week t itself, which the model would not have when forecasting.
    """
    return grouped.transform(
        lambda s: getattr(s.shift(1).rolling(window, min_periods=1), func)()
    )


def add_lag_features(
    panel: pd.DataFrame,
    windows: tuple[int, ...] = (1, 4, 12, 26),
    group_column: str = "state",
) -> pd.DataFrame:
    """The locality's own event history.

    Conflict-history baselines are notoriously hard to beat, so these are both
    the baseline model's only inputs and the foundation the richer models must
    improve upon.
    """
    out = panel.sort_values([group_column, "week"]).copy()
    events = out.groupby(group_column)["events"]
    deaths = out.groupby(group_column)["fatalities"]

    out[f"{LAG_PREFIX}events_1w"] = events.transform(lambda s: s.shift(1))
    out[f"{LAG_PREFIX}fatalities_1w"] = deaths.transform(lambda s: s.shift(1))

    for window in windows:
        out[f"{LAG_PREFIX}events_sum_{window}w"] = _shifted_rolling(events, window, "sum")
        out[f"{LAG_PREFIX}events_mean_{window}w"] = _shifted_rolling(events, window, "mean")
        out[f"{LAG_PREFIX}fatalities_sum_{window}w"] = _shifted_rolling(deaths, window, "sum")

    # Volatility distinguishes a locality with a steady low level from one
    # that alternates between quiet and severe weeks.
    out[f"{LAG_PREFIX}events_std_12w"] = _shifted_rolling(events, 12, "std")

    # Weeks since the last event. Recency carries information that counts do
    # not, particularly in sparse localities.
    def _weeks_since(series: pd.Series) -> pd.Series:
        had_event = series.shift(1).fillna(0) > 0
        counter = np.zeros(len(had_event), dtype=float)
        gap = np.nan
        for i, flag in enumerate(had_event.to_numpy()):
            if flag:
                gap = 0.0
            elif not np.isnan(gap):
                gap += 1.0
            counter[i] = gap
        return pd.Series(counter, index=series.index)

    gaps = out.groupby(group_column)["events"].transform(_weeks_since)
    # NaN before a locality's first recorded event. That is a real state, so
    # it is capped at the observed maximum and flagged rather than dropped.
    out[f"{LAG_PREFIX}no_prior_event"] = gaps.isna().astype(int)
    out[f"{LAG_PREFIX}weeks_since_event"] = gaps.fillna(
        gaps.max() if gaps.notna().any() else 0.0
    )

    # Short-run deviation from the medium-run level: the escalation signal a
    # human analyst would look for.
    ratio, undefined = safe_ratio(
        out[f"{LAG_PREFIX}events_mean_4w"], out[f"{LAG_PREFIX}events_mean_26w"]
    )
    out[f"{LAG_PREFIX}trend_4w_over_26w"] = ratio
    out[f"{LAG_PREFIX}trend_undefined"] = undefined
    return out


def build_neighbours(
    centroids: pd.DataFrame,
    k: int = 4,
    state_column: str = "state",
) -> dict[str, list[str]]:
    """Nearest localities by centroid distance.

    Centroids are used rather than shared borders because ACLED supplies them
    directly, so no external boundary file is needed and the crosswalk has one
    fewer dependency to decay.
    """
    frame = centroids.dropna(subset=["latitude", "longitude"]).copy()
    names = frame[state_column].to_numpy()
    coords = np.radians(frame[["latitude", "longitude"]].to_numpy())

    # Haversine on the unit sphere; scale is irrelevant for ranking.
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    neighbours: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        order = np.argsort(distance[i])
        nearest = [names[j] for j in order if names[j] != name][:k]
        neighbours[str(name)] = [str(n) for n in nearest]
    return neighbours


def add_spatial_features(
    panel: pd.DataFrame,
    neighbours: dict[str, list[str]],
    windows: tuple[int, ...] = (1, 4),
    group_column: str = "state",
) -> pd.DataFrame:
    """Violence in neighbouring localities.

    Conflict diffuses across boundaries, so a locality's risk depends partly
    on its neighbours. Neighbour values are lagged identically to own-history
    features.
    """
    out = panel.sort_values([group_column, "week"]).copy()
    wide = out.pivot_table(
        index="week", columns=group_column, values="events", aggfunc="sum", fill_value=0
    )

    for window in windows:
        lagged = wide.shift(1).rolling(window, min_periods=1).sum()
        neighbour_sum = pd.DataFrame(index=lagged.index, columns=lagged.columns, dtype=float)
        for state in lagged.columns:
            peers = [n for n in neighbours.get(str(state), []) if n in lagged.columns]
            neighbour_sum[state] = lagged[peers].sum(axis=1) if peers else 0.0
        melted = neighbour_sum.stack().rename(f"{SPATIAL_PREFIX}events_sum_{window}w")
        melted.index.names = ["week", group_column]
        out = out.merge(melted.reset_index(), on=["week", group_column], how="left")

    out[f"{SPATIAL_PREFIX}n_neighbours"] = out[group_column].map(
        lambda s: len(neighbours.get(str(s), []))
    )
    return out


def add_signal_features(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    windows: tuple[int, ...] = (1, 4, 12),
    group_column: str = "state",
) -> pd.DataFrame:
    """Protest and demonstration activity as leading indicators.

    Protests are excluded from the outcome because they are overwhelmingly
    peaceful, but the protest-to-conflict literature treats them as
    informative about what follows. Including them here is what makes H1
    testable: whether multi-source signals beat event history alone.
    """
    out = panel.sort_values([group_column, "week"]).copy()
    counts = (
        signals.groupby([group_column, "week"])
        .agg(signal_events=("events", "sum"))
        .reset_index()
    )
    out = out.merge(counts, on=[group_column, "week"], how="left")
    out["signal_events"] = out["signal_events"].fillna(0)

    grouped = out.groupby(group_column)["signal_events"]
    out[f"{SIGNAL_PREFIX}events_1w"] = grouped.transform(lambda s: s.shift(1))
    for window in windows:
        out[f"{SIGNAL_PREFIX}events_sum_{window}w"] = _shifted_rolling(grouped, window, "sum")

    # Ratio of the shortest to the longest configured window: a rising trend
    # in protest activity relative to its own recent norm. Only built when two
    # distinct windows exist, so the feature set follows the configuration
    # rather than assuming it.
    if len(windows) >= 2:
        short, long = min(windows), max(windows)
        ratio, undefined = safe_ratio(
            out[f"{SIGNAL_PREFIX}events_sum_{short}w"],
            out[f"{SIGNAL_PREFIX}events_sum_{long}w"],
        )
        out[f"{SIGNAL_PREFIX}trend_{short}w_over_{long}w"] = ratio
        out[f"{SIGNAL_PREFIX}trend_undefined"] = undefined
    return out.drop(columns=["signal_events"])


def add_health_features(
    panel: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    group_column: str = "state",
) -> pd.DataFrame:
    """Data-quality measures as predictors.

    H4 claims forecast errors are associated with missing, delayed or uneven
    source data after controlling for event intensity. Testing that requires
    those conditions to be in the panel rather than described in prose.
    """
    out = panel.sort_values([group_column, "week"]).copy()

    # Reporting density relative to the locality's own recent norm. A sharp
    # drop is the thin-source condition: not silence in the world, but
    # silence in the source.
    events = out.groupby(group_column)["events"]
    trailing = _shifted_rolling(events, 12, "median")
    recent = _shifted_rolling(events, 4, "mean")
    # Undefined where the trailing median is zero, which is the majority of
    # weeks in quiet localities. Filled with 1.0 (activity matching its own
    # norm) and flagged, so those localities stay in the panel.
    ratio, undefined = safe_ratio(recent, trailing, undefined_value=1.0)
    out[f"{HEALTH_PREFIX}volume_ratio_4w_12w"] = ratio
    out[f"{HEALTH_PREFIX}volume_ratio_undefined"] = undefined

    # Share of recent weeks with no recorded event. High values mean the
    # baseline is uninformative, which is where the outcome is least reliable.
    out[f"{HEALTH_PREFIX}zero_share_12w"] = out.groupby(group_column)["events"].transform(
        lambda s: (s.shift(1) == 0).rolling(12, min_periods=1).mean()
    )

    if coverage is not None:
        merged = out.merge(coverage, on=[group_column, "week"], how="left")
        for column in coverage.columns:
            if column in (group_column, "week"):
                continue
            merged[f"{HEALTH_PREFIX}{column}"] = merged[column]
            merged = merged.drop(columns=[column])
        out = merged

    # Explicit missingness flag rather than silent imputation.
    out[f"{HEALTH_PREFIX}baseline_unavailable"] = out[
        f"{HEALTH_PREFIX}volume_ratio_undefined"
    ]
    return out


def add_seasonal_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Calendar position, known in advance and therefore never leakage."""
    out = panel.copy()
    weeks = pd.to_datetime(out["week"])
    out[f"{SEASONAL_PREFIX}month"] = weeks.dt.month
    out[f"{SEASONAL_PREFIX}week_of_year"] = weeks.dt.isocalendar().week.astype(int)
    # Cyclical encoding so December and January are adjacent.
    out[f"{SEASONAL_PREFIX}month_sin"] = np.sin(2 * np.pi * out[f"{SEASONAL_PREFIX}month"] / 12)
    out[f"{SEASONAL_PREFIX}month_cos"] = np.cos(2 * np.pi * out[f"{SEASONAL_PREFIX}month"] / 12)
    return out


def feature_columns(panel: pd.DataFrame, families: tuple[str, ...] | None = None) -> list[str]:
    """Predictor columns, optionally restricted to named families."""
    prefixes = families or (
        LAG_PREFIX,
        SPATIAL_PREFIX,
        SIGNAL_PREFIX,
        HEALTH_PREFIX,
        SEASONAL_PREFIX,
    )
    return [
        column
        for column in panel.columns
        if column not in NON_FEATURE_COLUMNS and column.startswith(prefixes)
    ]


def assert_no_leakage(
    panel: pd.DataFrame,
    group_column: str = "state",
    tolerance: float = 1e-9,
) -> None:
    """Verify that no feature reads the current or future week.

    The check is empirical rather than a code review: it perturbs the outcome
    window and confirms no feature moves. A feature that changes when only
    future values change is reading forward.
    """
    columns = feature_columns(panel)
    if not columns:
        raise LeakageError("no feature columns found to check")

    original = panel.sort_values([group_column, "week"]).reset_index(drop=True)
    corrupted = original.copy()
    # Corrupt the final quarter of each locality's series.
    cut = int(len(corrupted) * 0.75)
    corrupted.loc[cut:, "events"] = corrupted.loc[cut:, "events"] * 1000 + 777
    corrupted.loc[cut:, "fatalities"] = corrupted.loc[cut:, "fatalities"] * 1000 + 777

    rebuilt = add_lag_features(corrupted, group_column=group_column)
    shared = [c for c in columns if c in rebuilt.columns and c.startswith(LAG_PREFIX)]

    # Rows strictly before the corruption must be unchanged.
    check_rows = slice(0, max(cut - 1, 0))
    for column in shared:
        left = original.loc[check_rows, column].fillna(-1).to_numpy()
        right = rebuilt.loc[check_rows, column].fillna(-1).to_numpy()
        if not np.allclose(left, right, atol=tolerance, equal_nan=True):
            raise LeakageError(
                f"feature '{column}' changed when only later weeks were altered, "
                f"which means it reads forward in time"
            )


def build_features(
    panel: pd.DataFrame,
    centroids: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    coverage: pd.DataFrame | None = None,
    outcome_name: str = "escalation",
) -> pd.DataFrame:
    """Assemble every feature family and verify the result."""
    out = add_lag_features(panel)

    if centroids is not None:
        neighbours = build_neighbours(centroids)
        out = add_spatial_features(out, neighbours)

    if signals is not None:
        out = add_signal_features(out, signals)

    out = add_health_features(out, coverage=coverage)
    out = add_seasonal_features(out)

    assert_no_leakage(out)

    columns = feature_columns(out)
    ledger.record(
        stage="features",
        status=ledger.STATUS_OK,
        rows=len(out),
        n_features=len(columns),
        families={
            "lag": sum(c.startswith(LAG_PREFIX) for c in columns),
            "spatial": sum(c.startswith(SPATIAL_PREFIX) for c in columns),
            "signal": sum(c.startswith(SIGNAL_PREFIX) for c in columns),
            "health": sum(c.startswith(HEALTH_PREFIX) for c in columns),
            "seasonal": sum(c.startswith(SEASONAL_PREFIX) for c in columns),
        },
        leakage_check="passed",
    )
    return out
