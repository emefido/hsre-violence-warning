# hsre-violence-warning

Analysis code for a study of youth and community violence early warning in
Nigeria and the United States, evaluated as an operational service rather than
as a prediction problem alone.

The pipeline builds a locality-week panel from free public sources, forecasts
short-horizon escalation, and measures the reliability of the whole chain from
ingestion through to alert: data freshness, source completeness, alert burden
against realistic review capacity, geographic performance disparities, and the
effect of simulated source failures on detection.

## Scope

The repository produces five outputs, which correspond to the results reported
in the accompanying manuscript:

1. Panel construction summary: locality-weeks per country, coverage by year and
   geography, share meeting the completeness threshold, most common
   source-health failure.
2. Model comparison: persistence, event-history regression, multi-source
   regression, gradient boosting and pooled model, reporting average precision,
   Brier score, calibration slope, precision at k and severe-event recall at k,
   with 95% bootstrap intervals.
3. Alert-budget curve: detection against alert volume with accuracy-optimised,
   recall-optimised and alert-budgeted thresholds marked, per country.
4. Geographic disparity: recall and calibration by region, urbanicity and
   deprivation stratum, before and after controlling for coverage and reporting
   continuity.
5. Source-failure experiments: marginal effect of each simulated failure mode
   on detection at capacity.

What the repository deliberately does not do: ingest private communications,
score individuals, identify alleged perpetrators, or route anything to
enforcement. These are design constraints, not omissions.

## Layout

```
config/         source registry, preregistered thresholds, geography settings
src/hsre/
  config.py     single entry point for all settings, validated at load
  ingest/       one adapter per source, retry and quarantine
  validate/     schema contracts, volume baselines, quarantine handling
  transform/    geographic crosswalks and locality-week harmonisation
  features/     leakage-safe feature construction
  models/       baselines, main model, calibration
  alerts/       ranking and the three threshold regimes
  monitoring/   run ledger and health metrics
data/
  raw/          immutable, timestamped, checksummed. Not committed.
  interim/      validated and harmonised. Not committed.
  processed/    the analysis panel. Not committed.
  metadata/     crosswalks and source registry state. Committed.
reports/        generated tables and figures
logs/           run ledger
tests/          stage verification
```

Raw data is never committed. Several sources prohibit redistribution, so the
repository ships retrieval scripts, pinned versions and checksums instead.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # then fill in credentials
```

If you already have an environment from an earlier stage, reinstall after
pulling so newly declared dependencies are picked up:

```bash
pip install -e ".[dev]"
```

`tests/test_dependencies.py` checks that every declared dependency imports and
that no third-party import in `src/` is undeclared, so a package that happens
to be present on the author's machine cannot silently become a requirement.

### No compiled extensions required

The pipeline deliberately avoids libraries that link external compiled
runtimes. Gradient boosting uses scikit-learn's `HistGradientBoostingClassifier`
rather than LightGBM: the same histogram-based algorithm, but with no OpenMP
dependency. The LightGBM wheel installs cleanly on macOS and then fails at
import, which would break replication for reasons unrelated to the research.

Only UCDP and the Census API require credentials. Every other source is open.

UCDP is reachable by two routes and defaults to the one needing no
credential. The bulk CSV archive at https://ucdp.uu.se/downloads/ is free and
open, and the API maintainer confirms it carries the full data the API draws
on. The API route is available for anyone holding a token, requested by email
to mertcan.yilmaz@pcr.uu.se; academic replies take about a week and the quota
is 5,000 requests per day with errors counted, which is why credential
failures here fail on the first attempt rather than retrying.

Switch routes with `active_route` in `config/sources.yml`. Both routes pin the
same dataset version and pass through identical validation and checksumming,
so running both and comparing row counts is a genuine reproducibility check.
The bulk archive is global, so the Nigeria restriction is applied after
download rather than in the request.

## Build stages

Each stage is verified before the next begins.

| Stage | Contents | Status |
|---|---|---|
| 1 | Repo skeleton, config, run ledger | complete |
| 2 | Source adapters with retry, schema validation, quarantine | complete |
| 3 | Health metrics and outcome diagnostics | complete |
| 4 | Geographic crosswalks | complete |
| 5 | Locality-week panel builder | complete |
| 6 | Feature construction | complete |
| 7 | Baseline models | complete |
| 8 | Main model and H1 ablation | complete |
| 9 | Alert budget allocator | complete |
| 7 | Baseline models | complete |
| 8 | Main model and H1 ablation | complete |
| 9 | Alert budget allocator | complete |
| 7 | Main model and calibration | pending |
| 9 | Validation regimes and source-failure experiments | pending |
| 10 | Results export | pending |

## Verification

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Stage 1 checks that configuration parses and validates, that the analytical
decisions in `config/thresholds.yml` are internally consistent, that the
benchmark source cannot enter the feature matrix, and that the run ledger
records and reads back health entries.

Stage 2 checks adapter retry and backoff, that an empty payload is treated as
a failure rather than as an absence of events, that raw files are immutable,
that schema drift raises while individual bad records are quarantined with
reasons attached, and that a thin source is flagged without failing.

## Running ingestion

```bash
PYTHONPATH=src python -m hsre.ingest.run --list
PYTHONPATH=src python -m hsre.ingest.run --source ucdp_ged --start 2016-01-01 --end 2024-12-31
PYTHONPATH=src python -m hsre.ingest.run --setting nigeria
```

Sources on the `export` route have no supported API. Place the downloaded file
in `data/raw/_inbox/<source_name>/` and the adapter registers it with the same
checksum, immutable storage and ledger treatment as an automated pull.

A required source that fails exits non-zero, since a forecast built on a
missing outcome source is invalid rather than degraded. An optional source
that fails is reported and the run continues, with affected outputs marked
degraded downstream.

## Preregistration

`config/thresholds.yml` holds the analytical decisions frozen before test-set
analysis: study period and holdout boundaries, escalation definitions, alert
budget values, completeness rules and evaluation metrics. Changes after
freezing are logged in `docs/deviations.md` with a date and reason.

## Licence

MIT for the code. Data sources retain their own licences; see
`docs/data_dictionary.md`.


## Outcome source hierarchy for Nigeria

ACLED is the primary Nigerian outcome source. UCDP GED and Nigeria Watch are
secondary, retained for the coverage comparison. This ordering follows from
the data rather than from preference.

UCDP codes organised political violence, and its Nigerian coverage is
dominated by the Borno insurgency. Measured against ACLED over 2016 to 2024:

| State | UCDP | ACLED | Ratio |
|---|---|---|---|
| Katsina | 13 | 1,504 | 116x |
| Zamfara | 27 | 1,684 | 62x |
| FCT | 26 | 1,599 | 62x |
| Niger | 31 | 1,202 | 39x |
| Ekiti | 10 | 324 | 32x |
| Borno | 4,061 | 4,568 | 1.1x |

Borno's share of all Nigerian events falls from 56.5% under UCDP to 14.4%
under ACLED. Borno itself barely moves, because UCDP already captures the
insurgency well. The entire gap is non-insurgency violence, which is what this
study is about.

ACLED's sub-event types carry the research question directly: mob violence
(1,698 events), abduction and forced disappearance (2,979), riots (2,609) and
protests (6,168) over the same window.

Nigeria Watch extends further into interpersonal and criminal killing, with
roughly 169,000 violent deaths recorded from 2006 to 2021 and more attributed
to crime than to insurgency. Access is by request. It serves as a breadth
check on ACLED once obtained.

Run `python -m hsre.validate.source_coverage` to reproduce the comparison.

## LGA resolution and event narratives

ACLED offers two routes and the choice matters:

- **api** returns full disaggregated events including `admin2` (LGA),
  `admin3`, actor names and event narratives. Requires free credentials.
- **bulk** returns the weekly aggregated regional file, already at
  week x admin1 x event type. No credentials, but no LGA, actors or
  narratives.

The API route is active by default because LGA robustness analysis and
actor-based youth identification both depend on fields the aggregated file
does not carry. Where the bulk route is active, LGA analysis is reported as
unavailable rather than silently degraded.

## Diagnostics

```bash
python -m hsre.ingest.inspect ucdp_ged                  # columns and coverage
python -m hsre.ingest.inspect ucdp_ged --column adm_1   # value distribution
python -m hsre.validate.outcome_check ucdp_ged          # outcome viability
python -m hsre.validate.source_coverage                 # what each source misses
```

The outcome check exists because sparse panels break the escalation
definition. Where the trailing baseline is zero, any single event counts as
escalation and the outcome measures presence of violence rather than change in
level. `min_future_events` in `config/thresholds.yml` sets the floor that
prevents this.


## Features

Four families, all reading only information available by the end of week t.
Escalation is labelled from weeks t+1 and t+2, so any feature that reads
forward is leakage.

- **lag** the locality's own event history: counts, fatalities, rolling sums
  and means over 1, 4, 12 and 26 weeks, volatility, weeks since last event,
  and short-run trend against the medium-run level.
- **spatial** violence in the four nearest localities by centroid distance,
  lagged identically. Centroids come from ACLED, so no external boundary file
  is required.
- **signal** protest and demonstration activity as leading indicators.
  Protests are excluded from the outcome because they are overwhelmingly
  peaceful, but their relationship to subsequent violence is what makes H1
  testable.
- **health** data-quality measures as predictors: reporting volume against the
  locality's own norm, share of recent weeks with no event, and explicit
  missingness flags. These exist because H4 claims forecast errors track
  measurable data conditions, which is only testable if those conditions are
  in the panel.

Leakage prevention is structural rather than procedural. Every rolling window
shifts by one before aggregating, and `assert_no_leakage` verifies the result
empirically: it perturbs the outcome window and confirms no earlier feature
value moves.

### Undefined ratios

Ratio features are undefined where the denominator is zero, which is common in
quiet localities. Leaving them as NaN would cause complete-case analysis to
delete exactly the localities conventional sources already omit. Each ratio is
therefore filled with a stated value and paired with an indicator column, so
the condition is learnable rather than the row disappearing. On the 2016-2024
Nigeria panel this keeps 100% of labelled rows against 16% before the fix.


## Baselines

```bash
PYTHONPATH=src python -m hsre.models.run_baselines --data path/to/acled.xlsx
```

Writes `reports/tables/baseline_comparison.csv`. Results and interpretation
are in `docs/baseline_results.md`.

Splits are temporal, never random: neighbouring weeks in the same locality are
strongly dependent, so random splitting leaks the future and inflates
performance. Accuracy is not reported, because at a 27% base rate a model
predicting no escalation everywhere scores 73%. Average precision is the
headline metric, with lift against the base rate to expose no-skill models,
and precision and recall at a fixed alert budget because that is what an
institution can act on.


## Main model

```bash
PYTHONPATH=src python -m hsre.models.run_model --data path/to/acled.xlsx
```

Writes `reports/tables/ablation.csv` and `model_comparison.csv`. Results and
interpretation are in `docs/h1_results.md`.

Gradient boosting uses scikit-learn's `HistGradientBoostingClassifier`.
Capacity is selected on the validation period rather than defaulted, because
the outcome is non-stationary: escalation rises from roughly 5% of
locality-weeks in 2016 to 39% in 2024, and an unconstrained model memorises
the training regime and transfers poorly.

**H1 is not supported.** Multi-source features change average precision by
−0.006 on the primary outcome and +0.005 on youth, with overlapping
confidence intervals. Regularised logistic regression on event history alone
outperforms every gradient boosting configuration on both outcomes.


## Alert budgets

```bash
PYTHONPATH=src python -m hsre.alerts.run_budget --data path/to/acled.xlsx --capacity 4
```

Writes `reports/tables/alert_regimes.csv`, `alert_budget_curve.csv` and
`reports/figures/alert_budget_curve.png`. Results are in
`docs/alert_budget_results.md`.

Three regimes are compared: accuracy-optimised (the field default, which
ignores capacity), recall-optimised (which produces alert fatigue), and
alert-budgeted (where the threshold is a function of institutional capacity
rather than model output).

**No regime satisfies the error budget.** The burden ceiling and the
severe-recall floor are jointly infeasible on this panel: capacity-constrained
alerting catches under 25% of escalations, while reaching the recall floor
requires roughly four times the permitted alert volume. That infeasibility is
the finding.
