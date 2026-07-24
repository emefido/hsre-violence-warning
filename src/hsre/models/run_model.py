"""Main model results and the H1 ablation.

    python -m hsre.models.run_model --data path/to/acled.xlsx

Fits the same algorithm on progressively richer feature sets. The algorithm,
hyperparameters and temporal split are identical across runs, so differences
are attributable to the feature families rather than to model capacity. That
is what makes H1 falsifiable: a multi-source model that fails to beat event
history alone is a null result, not a tuning problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hsre.config import REPO_ROOT, load_config
from hsre.features.build import build_features, feature_columns
from hsre.models.baselines import default_baselines, lag_only_features
from hsre.models.evaluate import comparison_table, evaluate, temporal_split
from hsre.models.gradient_boosting import (
    FEATURE_SETS,
    GradientBoostingModel,
    ablate,
    ablation_table,
)
from hsre.models.run_baselines import load_acled, prepare_inputs
from hsre.transform.panel import build_both_outcomes

TABLES_DIR = REPO_ROOT / "reports" / "tables"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the main model and run the H1 ablation")
    parser.add_argument("--data", required=True, help="ACLED file, xlsx or csv")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--importance",
        action="store_true",
        help="compute permutation importance, which is slow",
    )
    args = parser.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1

    config = load_config()
    period = config.thresholds["study_period"]

    frame = load_acled(path)
    centroids, signals = prepare_inputs(frame)
    panels = build_both_outcomes(frame, config.thresholds)

    ablation_results = []
    all_metrics = []

    for outcome_name, panel in panels.items():
        featured = build_features(panel, centroids=centroids, signals=signals)
        split = temporal_split(
            featured,
            train_end=pd.Timestamp(period["train_end"]),
            validation_end=pd.Timestamp(period["validation_end"]),
            test_start=pd.Timestamp(period["test_start"]),
        )

        print(f"\n=== {outcome_name} ===")
        print(f"  train {len(split.train)} | validation {len(split.validation)} "
              f"| test {len(split.test)}")
        print(f"  test base rate {split.test['escalation'].mean():.4f}")

        # The strongest baseline sets the bar.
        y_test = split.test["escalation"].astype(int).to_numpy()
        lag_features = lag_only_features(featured)
        print("\n  baselines (lag features only):")
        for model in default_baselines():
            model.fit(split.train, lag_features, "escalation")
            metrics = evaluate(
                y_test,
                model.predict_scores(split.test, lag_features),
                model_name=model.name,
                outcome_name=outcome_name,
                bootstrap_draws=args.bootstrap,
                seed=config.seed,
            )
            all_metrics.append(metrics)
            print(f"    {model.name:18s} AP={metrics.average_precision:.4f} "
                  f"lift={metrics.lift():.2f}")

        print("\n  ablation (same algorithm, varying feature set):")
        results = ablate(
            split,
            featured,
            outcome_name=outcome_name,
            seed=config.seed,
            bootstrap_draws=args.bootstrap,
        )
        for result in results:
            m = result.metrics
            interval = ""
            if m.ap_ci:
                interval = f" [{m.ap_ci[0]:.3f}, {m.ap_ci[1]:.3f}]"
            print(f"    {result.feature_set:22s} n={result.n_features:2d} "
                  f"AP={m.average_precision:.4f}{interval} lift={m.lift():.2f}")
            all_metrics.append(m)
        ablation_results.extend(results)

        # H1 verdict for this outcome.
        by_name = {r.feature_set: r.metrics for r in results}
        if "lag_only" in by_name and "multi_source" in by_name:
            base = by_name["lag_only"]
            multi = by_name["multi_source"]
            delta = multi.average_precision - base.average_precision
            overlap = (
                base.ap_ci and multi.ap_ci and multi.ap_ci[0] <= base.ap_ci[1]
            )
            print(f"\n  H1: multi-source minus lag-only = {delta:+.4f} AP")
            if overlap:
                print("      confidence intervals overlap; difference is not clear-cut")
            elif delta > 0:
                print("      multi-source higher with non-overlapping intervals")
            else:
                print("      lag-only higher; H1 not supported for this outcome")

        if args.importance:
            features = feature_columns(featured, families=FEATURE_SETS["multi_source_health"])
            model = GradientBoostingModel(calibrate=False).fit(
                split.train, features, "escalation"
            )
            importance = model.permutation_importance(split.test, "escalation", n_repeats=3)
            TABLES_DIR.mkdir(parents=True, exist_ok=True)
            out = TABLES_DIR / f"importance_{outcome_name}.csv"
            importance.to_csv(out, index=False)
            print(f"\n  top features by permutation importance:")
            for _, row in importance.head(8).iterrows():
                print(f"    {row['feature']:32s} {row['importance']:+.4f}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ablation_table(ablation_results).to_csv(TABLES_DIR / "ablation.csv", index=False)
    comparison_table(all_metrics).to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    print(f"\nwritten: reports/tables/ablation.csv, reports/tables/model_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
