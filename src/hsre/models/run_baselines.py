"""Baseline results.

Fits every baseline on both Nigerian outcomes and writes the comparison table
reported in the manuscript.

    python -m hsre.models.run_baselines --data path/to/acled.xlsx

Baselines see only `lag_` features. The main model adds signal, spatial and
health families, so any improvement is attributable to those sources rather
than to a richer algorithm. That separation is what makes H1 falsifiable.
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
from hsre.transform.nigeria_states import normalise
from hsre.transform.panel import build_both_outcomes

TABLES_DIR = REPO_ROOT / "reports" / "tables"


def load_acled(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    if "COUNTRY" in frame.columns:
        frame = frame.loc[frame["COUNTRY"] == "Nigeria"]
    return frame.copy()


def prepare_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Centroids for spatial features and protests as the leading signal."""
    working = frame.copy()
    working["state"] = working["ADMIN1"].map(normalise)
    working = working.dropna(subset=["state"])

    centroids = (
        working.groupby("state")[["CENTROID_LATITUDE", "CENTROID_LONGITUDE"]]
        .first()
        .reset_index()
        .rename(
            columns={
                "CENTROID_LATITUDE": "latitude",
                "CENTROID_LONGITUDE": "longitude",
            }
        )
    )

    protests = working.loc[working["EVENT_TYPE"] == "Protests"].copy()
    protests["week"] = pd.to_datetime(protests["WEEK"])
    signals = (
        protests.groupby(["state", "week"])["EVENTS"]
        .sum()
        .reset_index()
        .rename(columns={"EVENTS": "events"})
    )
    return centroids, signals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate baseline models")
    parser.add_argument("--data", required=True, help="ACLED file, xlsx or csv")
    parser.add_argument("--bootstrap", type=int, default=1000)
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

    results = []
    for outcome_name, panel in panels.items():
        featured = build_features(panel, centroids=centroids, signals=signals)
        split = temporal_split(
            featured,
            train_end=pd.Timestamp(period["train_end"]),
            validation_end=pd.Timestamp(period["validation_end"]),
            test_start=pd.Timestamp(period["test_start"]),
        )
        print(f"\n=== {outcome_name} ===")
        for key, value in split.describe().items():
            print(f"  {key}: {value}")

        features = lag_only_features(featured)
        print(f"  baseline features: {len(features)} (lag family only)")
        print(f"  full feature set:  {len(feature_columns(featured))}")

        y_test = split.test["escalation"].astype(int).to_numpy()
        for model in default_baselines():
            model.fit(split.train, features, "escalation")
            scores = model.predict_scores(split.test, features)
            metrics = evaluate(
                y_test,
                scores,
                model_name=model.name,
                outcome_name=outcome_name,
                bootstrap_draws=args.bootstrap,
                seed=config.seed,
            )
            results.append(metrics)
            print(
                f"    {model.name:16s} AP={metrics.average_precision:.4f} "
                f"lift={metrics.lift():.2f} "
                f"P@2%={metrics.precision_at_k[0.02]:.3f} "
                f"R@2%={metrics.recall_at_k[0.02]:.3f}"
            )

    table = comparison_table(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLES_DIR / "baseline_comparison.csv"
    table.to_csv(output, index=False)
    print(f"\nwritten: {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
