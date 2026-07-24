"""Degradation and transfer results.

    python -m hsre.models.run_validation --data path/to/acled.xlsx

Two validation regimes that conventional forecasting evaluation omits.

Source failure tests H4 directly: whether forecast errors track measurable
data conditions. The model is fitted once on clean training data and then
scored on degraded test data, which mirrors deployment, where a model trained
in good conditions meets a feed that later fails.

Geographic transfer tests whether the model has learned anything beyond the
dominant locality, which matters because the outcome is concentrated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from hsre.config import REPO_ROOT, load_config
from hsre.features.build import build_features, feature_columns
from hsre.models.baselines import LogisticBaseline, lag_only_features
from hsre.models.degradation import (
    degradation_table,
    geographic_transfer,
    is_graceful,
    run_degradation,
    transfer_table,
)
from hsre.models.evaluate import temporal_split
from hsre.models.run_baselines import load_acled, prepare_inputs
from hsre.transform.panel import build_both_outcomes

TABLES_DIR = REPO_ROOT / "reports" / "tables"

# Held out one at a time. Borno dominates the primary outcome and the FCT the
# youth outcome, so both are tested; the remainder cover other regions.
HOLDOUT_SETS = (
    ("Borno",),
    ("FCT",),
    ("Lagos",),
    ("Zamfara",),
    ("Katsina",),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Degradation and transfer validation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--capacity", type=int, default=None)
    args = parser.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1

    config = load_config()
    period = config.thresholds["study_period"]
    capacity = args.capacity or config.thresholds["alert_budget"][
        "illustrative_absolute"
    ]["nigeria"]

    frame = load_acled(path)
    centroids, signals = prepare_inputs(frame)
    panels = build_both_outcomes(frame, config.thresholds)

    degradation_results = []
    transfer_results = []

    for outcome_name, panel in panels.items():
        featured = build_features(panel, centroids=centroids, signals=signals)
        split = temporal_split(
            featured,
            train_end=pd.Timestamp(period["train_end"]),
            validation_end=pd.Timestamp(period["validation_end"]),
            test_start=pd.Timestamp(period["test_start"]),
        )
        features = feature_columns(featured)
        model = LogisticBaseline().fit(split.train, features, "escalation")

        print(f"\n=== {outcome_name} ===")
        print("\n  source failure experiments (H4):")
        results = run_degradation(
            model, split, outcome_name, features, capacity_per_week=capacity
        )
        baseline = results[0].baseline_average_precision
        print(f"    clean baseline AP = {baseline:.4f}")
        for result in results:
            print(
                f"    {result.failure:22s} sev={result.severity:<5} "
                f"AP={result.average_precision:.4f} "
                f"({result.relative_ap_loss * 100:+6.1f}%) "
                f"R@{capacity}/wk={result.recall_at_capacity:.3f}"
            )
        degradation_results.extend(results)

        verdict = "gradual" if is_graceful(results) else "abrupt"
        worst = max(results, key=lambda r: abs(r.relative_ap_loss))
        print(f"\n    degradation is {verdict}")
        print(
            f"    worst case: {worst.failure} at severity {worst.severity}, "
            f"{worst.relative_ap_loss * 100:+.1f}% AP"
        )

        print("\n  geographic transfer:")
        lag_features = lag_only_features(featured)
        transfers = geographic_transfer(
            LogisticBaseline,
            featured,
            lag_features,
            holdout_sets=HOLDOUT_SETS,
            train_end=pd.Timestamp(period["train_end"]),
            outcome_name=outcome_name,
        )
        for result in transfers:
            print(
                f"    held out {result.held_out[0]:10s} n={result.n:5d} "
                f"base={result.base_rate:.3f} AP={result.average_precision:.4f} "
                f"lift={result.lift:.2f}"
            )
        transfer_results.extend(transfers)

        if transfers:
            lifts = [t.lift for t in transfers]
            print(
                f"    transfer lift ranges {min(lifts):.2f} to {max(lifts):.2f} "
                f"(1.00 means no skill)"
            )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    degradation_table(degradation_results).to_csv(
        TABLES_DIR / "degradation.csv", index=False
    )
    transfer_table(transfer_results).to_csv(
        TABLES_DIR / "geographic_transfer.csv", index=False
    )
    print(
        "\nwritten: reports/tables/degradation.csv, "
        "reports/tables/geographic_transfer.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
