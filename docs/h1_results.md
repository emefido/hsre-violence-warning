# H1: does multi-source data improve short-horizon forecasts?

**H1 is not supported on either outcome.** Adding public-signal, spatial and
data-health features to event history produces changes of −0.006 AP on the
primary outcome and +0.005 AP on the youth outcome, with overlapping 95%
bootstrap intervals in both cases.

## Ablation

Identical algorithm, hyperparameter search and temporal split across every
row. Only the feature set varies, so differences are attributable to the data
sources rather than to model capacity.

### Primary outcome, test base rate 34.8%

| feature set | n | AP | 95% CI | lift |
|---|---|---|---|---|
| lag_only | 19 | 0.663 | [0.636, 0.690] | 1.90 |
| lag_seasonal | 23 | 0.662 | [0.635, 0.687] | 1.90 |
| lag_spatial | 26 | 0.659 | [0.631, 0.686] | 1.89 |
| lag_signal | 29 | 0.660 | [0.633, 0.688] | 1.90 |
| multi_source | 32 | 0.657 | [0.629, 0.684] | 1.89 |
| multi_source_health | 36 | 0.665 | [0.638, 0.692] | 1.91 |

### Youth outcome, test base rate 26.7%

| feature set | n | AP | 95% CI | lift |
|---|---|---|---|---|
| lag_only | 19 | 0.450 | [0.425, 0.479] | 1.68 |
| lag_seasonal | 23 | 0.448 | [0.425, 0.479] | 1.68 |
| lag_spatial | 26 | 0.453 | [0.430, 0.484] | 1.69 |
| lag_signal | 29 | 0.455 | [0.430, 0.485] | 1.70 |
| multi_source | 32 | 0.455 | [0.429, 0.488] | 1.70 |
| multi_source_health | 36 | 0.447 | [0.423, 0.477] | 1.67 |

## The simplest model wins

| model | primary AP | youth AP |
|---|---|---|
| persistence | 0.594 | 0.360 |
| trailing_rate | 0.673 | 0.468 |
| count_nb | 0.679 | 0.484 |
| **logistic_lag** | **0.682** | **0.501** |
| gradient boosting, best set | 0.665 | 0.455 |

Regularised logistic regression on event history alone outperforms gradient
boosting on every feature set, on both outcomes. This is not a tuning failure:
capacity was selected on the validation period across a four-point grid, and
the selected models are already heavily constrained.

## Why: the outcome is non-stationary

Escalation rates by year:

| year | primary | youth |
|---|---|---|
| 2016 | 5.2% | 4.9% |
| 2018 | 9.6% | 5.2% |
| 2020 | 23.7% | 18.0% |
| 2022 | 33.3% | 29.2% |
| 2024 | 38.6% | 28.2% |

The training period base rate is 11.4% against 26.7% in test. Train and test
are different regimes.

Before capacity selection, gradient boosting scored 0.677 on the training set
and 0.359 on test, a collapse of 0.32. Logistic regression scored 0.337 on
train and 0.501 on test: it performed *better* out of sample than in. The
flexible model memorised a regime that no longer obtained; the constrained
model, unable to memorise, transferred. Constraining capacity recovered most
of the gap (youth 0.359 to 0.450) without closing it.

## What this means for the study

**The null is a finding, not a failure.** Three points follow.

First, it replicates a well-established result in conflict forecasting:
conflict-history baselines are difficult to beat, and added data sources
frequently fail to improve on them. Reporting a null here is more useful than
the alternative, which would be searching specifications until something
crossed a threshold.

Second, it strengthens rather than weakens the study's central argument. If
predictive gains from additional data are marginal, then the case for
evaluating early-warning systems on operational properties rather than
predictive accuracy alone is stronger. A field that keeps adding sources for
0.005 AP while alert burden, data freshness and response capacity go
unmeasured is optimising the wrong quantity.

Third, non-stationarity is itself a reliability finding of the kind H4
concerns. A model trained on 2016-2021 and deployed in 2023 faces a different
world, and nothing in a conventional accuracy report would reveal that. A
service specified with data-health monitoring would detect the drift; one
specified by AP alone would not.

## Threats to this interpretation

The rise in escalation rates has two plausible and non-separable causes:
Nigerian violence genuinely intensified, and ACLED expanded its Nigerian
sourcing over the period. If the second dominates, part of what the models
fail to transfer is a change in measurement rather than in the world. That
possibility cannot be resolved with these data and is reported rather than
assumed away.

The signal family is also thinner than intended. Protest counts enter as
aggregate weekly totals per state, whereas the manuscript describes richer
text-derived features. A stronger signal representation might yet support H1,
and that is a limitation of this implementation rather than a refutation of
the hypothesis.
