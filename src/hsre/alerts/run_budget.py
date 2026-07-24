"""Alert budget results.

    python -m hsre.alerts.run_budget --data path/to/acled.xlsx

Produces the central empirical exhibit: how much detection is lost by
constraining alert volume to what a responding institution can actually
review.

The model used is the strongest performer from stage 8, which is regularised
logistic regression on event history. That choice follows the evidence rather
than preference: gradient boosting on the full feature set did not beat it,
and using a weaker model here would understate what a capacity-constrained
service can achieve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hsre.alerts.budget import budget_curve, compare_regimes
from hsre.config import REPO_ROOT, load_config
from hsre.features.build import build_features
from hsre.models.baselines import LogisticBaseline, lag_only_features
from hsre.models.evaluate import temporal_split
from hsre.models.run_baselines import load_acled, prepare_inputs
from hsre.transform.panel import build_both_outcomes

TABLES_DIR = REPO_ROOT / "reports" / "tables"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def plot_curve(curves: dict[str, pd.DataFrame], capacity: int, path: Path) -> None:
    """Detection against alert volume, with the capacity constraint marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (name, points) in zip(axes, curves.items()):
        ax.plot(points["alerts_per_week"], points["recall"], label="recall", lw=2)
        ax.plot(
            points["alerts_per_week"], points["precision"],
            label="precision", lw=2, ls="--",
        )
        ax.axvline(capacity, color="grey", ls=":", lw=1.5)
        ax.annotate(
            f"capacity = {capacity}/week",
            xy=(capacity, 0.95), xytext=(capacity + 1.2, 0.95),
            fontsize=8, color="grey", va="top",
        )
        ax.set_title(name)
        ax.set_xlabel("alerts per week")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("share")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Detection against alert volume, Nigeria state-weeks 2023-2024",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alert budget analysis")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="alerts per week the responding institution can review",
    )
    args = parser.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1

    config = load_config()
    period = config.thresholds["study_period"]
    budget_config = config.thresholds["alert_budget"]
    capacity = args.capacity or budget_config["illustrative_absolute"]["nigeria"]
    max_burden = budget_config["max_review_burden_per_week"]
    min_severe = budget_config["min_severe_recall"]

    frame = load_acled(path)
    centroids, signals = prepare_inputs(frame)
    panels = build_both_outcomes(frame, config.thresholds)

    curves: dict[str, pd.DataFrame] = {}
    regime_tables = []

    for outcome_name, panel in panels.items():
        featured = build_features(panel, centroids=centroids, signals=signals)
        split = temporal_split(
            featured,
            train_end=pd.Timestamp(period["train_end"]),
            validation_end=pd.Timestamp(period["validation_end"]),
            test_start=pd.Timestamp(period["test_start"]),
        )

        features = lag_only_features(featured)
        model = LogisticBaseline().fit(split.train, features, "escalation")
        scores = model.predict_scores(split.test, features)

        test = split.test.copy()
        # Severe weeks feed the recall floor in the two-part error budget.
        #
        # The primary outcome already requires a fatality, so defining severe
        # as "escalation with a death" would make every escalation severe and
        # the constraint vacuous. Severity there is therefore a higher
        # fatality threshold. The youth outcome does not require lethality,
        # so any fatal escalation counts as severe.
        min_fatalities = config.thresholds["outcome"]["severe_event"]["min_fatalities"]
        if config.thresholds["outcome"]["nigeria"][outcome_name]["require_lethal"]:
            test["severe"] = (
                (test["escalation"] == 1)
                & (test["future_fatalities"] >= min_fatalities)
            ).astype(int)
        else:
            test["severe"] = (
                (test["escalation"] == 1) & (test["future_fatalities"] > 0)
            ).astype(int)

        print(f"\n=== {outcome_name} ===")
        print(f"  test: {len(test)} locality-weeks over {test['week'].nunique()} weeks")
        print(f"  escalation weeks: {int(test['escalation'].sum())}")
        print(f"  severe weeks:     {int(test['severe'].sum())}")

        table = compare_regimes(
            test, scores, outcome_name,
            capacity_per_week=capacity,
            severe_column="severe",
            max_review_burden_per_week=max_burden,
            min_severe_recall=min_severe,
        )
        regime_tables.append(table)

        print("\n  regime comparison:")
        for _, row in table.iterrows():
            verdict = "pass" if row["passes_error_budget"] else "FAIL"
            print(
                f"    {row['regime']:20s} {row['alerts_per_week']:6.1f}/wk "
                f"P={row['precision']:.3f} R={row['recall']:.3f} "
                f"false={row['false_alerts_per_week']:5.1f}/wk  budget:{verdict}"
            )

        curve = budget_curve(test, scores, outcome_name, severe_column="severe")
        curves[outcome_name] = curve.points

        print("\n  detection against volume:")
        for per_week in (1, 2, 4, 8, 16, 37):
            row = curve.points.loc[curve.points["alerts_per_week"] == per_week]
            if row.empty:
                continue
            row = row.iloc[0]
            print(
                f"    {per_week:2d}/wk: recall={row['recall']:.3f} "
                f"precision={row['precision']:.3f} "
                f"false={row['false_alerts_per_week']:5.1f}/wk"
            )

        at_capacity = curve.detection_at(capacity)
        at_full = curve.points["recall"].iloc[-1]
        volume_for_full = curve.volume_for_recall(0.95)
        print(
            f"\n  at capacity ({capacity}/wk) the service detects "
            f"{at_capacity * 100:.1f}% of escalation weeks"
        )
        print(
            f"  reaching 95% detection would require "
            f"{volume_for_full:.0f} alerts per week, "
            f"{volume_for_full / capacity:.0f}x the stated capacity"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(regime_tables).to_csv(TABLES_DIR / "alert_regimes.csv", index=False)
    pd.concat(
        [points.assign(outcome=name) for name, points in curves.items()]
    ).to_csv(TABLES_DIR / "alert_budget_curve.csv", index=False)

    figure = FIGURES_DIR / "alert_budget_curve.png"
    plot_curve(curves, capacity, figure)

    print(
        f"\nwritten: reports/tables/alert_regimes.csv, "
        f"reports/tables/alert_budget_curve.csv, "
        f"reports/figures/alert_budget_curve.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
