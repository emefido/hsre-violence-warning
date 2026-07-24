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
cp .env.example .env    # then fill in UCDP and Census credentials
```

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
| 4 | Geographic crosswalks | pending |
| 5 | Locality-week panel builder | pending |
| 6 | Baseline models | pending |
| 7 | Main model and calibration | pending |
| 8 | Alert-budget allocator and threshold regimes | pending |
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

Nigeria Watch is the primary Nigerian outcome source and UCDP GED is the
organised-violence subset. This is a data-driven decision rather than a
preference.

UCDP codes organised political violence. In the ingested extract, Borno holds
roughly half of all Nigerian UCDP events, and Kano, Katsina, Kebbi, Bauchi and
the FCT produce no escalation weeks at all across 2016 to 2024. Zamfara, the
centre of the banditry crisis, records fewer than 20 events over nine years.
Those absences are false rather than informative: the violence is present but
falls outside UCDP's inclusion criteria.

Nigeria Watch records roughly 169,000 violent deaths from 2006 to 2021 and
attributes more of them to crime than to insurgency, covering the cultism,
communal, criminal and interpersonal killing that constitutes youth and
community violence. It is press-derived, so coverage is uneven by region, and
that bias is reported as a finding rather than corrected away.

Run `python -m hsre.validate.source_coverage` to reproduce the comparison.

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
