# Baseline results

Fitted on the 2016-2024 ACLED Nigeria panel with a temporal split: train to
2021, validate 2022, test from 2023. Baselines see only `lag_` features
(19 of 36), so any improvement from the main model is attributable to the
signal, spatial and health families rather than to a richer algorithm.

## Primary outcome

Test set: 3,811 locality-weeks, 1,326 escalation weeks, base rate 34.8%.

| model | AP | lift | P@2% | R@2% |
|---|---|---|---|---|
| persistence | 0.594 | 1.71 | 0.883 | 0.051 |
| trailing_rate | 0.673 | 1.93 | 0.857 | 0.050 |
| count_nb | 0.679 | 1.95 | 0.883 | 0.051 |
| logistic_lag | 0.682 | 1.96 | 0.870 | 0.051 |

## Youth outcome

Test set: 3,811 locality-weeks, 1,019 escalation weeks, base rate 26.7%.

| model | AP | lift | P@2% | R@2% |
|---|---|---|---|---|
| persistence | 0.360 | 1.35 | 0.636 | 0.048 |
| trailing_rate | 0.468 | 1.75 | 0.688 | 0.052 |
| count_nb | 0.484 | 1.81 | 0.662 | 0.050 |
| logistic_lag | 0.501 | 1.87 | 0.610 | 0.046 |

## Reading these

**Lift matters more than AP.** A model with no skill scores average precision
equal to the base rate, so lift of 1.0 is the null. The strongest baseline
reaches 1.96 on the primary outcome and 1.87 on youth: real but modest skill,
consistent with the literature's finding that conflict-history baselines are
hard to beat and hard to improve upon.

**The youth outcome is harder.** Every baseline scores lower on youth than on
primary. This is expected: mob violence, abduction and demonstration policing
are less autocorrelated than armed conflict, so event history carries less
information about what follows. It also means the youth outcome has more room
for the additional sources to contribute, which is where H1 will be decided.

**Precision at 2% is high, recall is low.** Roughly 87% of the top 2% of
localities do escalate, but that captures only 5% of all escalation weeks.
This is the volume-detection trade-off the alert-budget analysis examines
directly in stage 8, and the shape of that curve is the central empirical
exhibit of the study.

## A trend in the outcome

Escalation rates rise sharply across the study period:

| year | primary | youth |
|---|---|---|
| 2016 | 5.2% | 4.9% |
| 2018 | 9.6% | 5.2% |
| 2020 | 23.7% | 18.0% |
| 2022 | 33.3% | 29.2% |
| 2024 | 38.6% | 28.2% |

The test period is therefore substantially harder than the training period,
and the test base rate (34.8%) exceeds the full-panel rate (21.8%).

Two mechanisms are plausible and they are not separable with these data alone:
Nigerian violence genuinely intensified over the period, and ACLED expanded
its Nigerian sourcing. The second is itself a data-health condition of exactly
the kind H4 concerns, and it is reported rather than adjusted away. Any claim
about deteriorating security in Nigeria drawn from this panel must carry that
caveat.


## Reproducing

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m hsre.models.run_baselines --data path/to/acled.xlsx
```

The negative binomial dispersion parameter is estimated from the training data
by method of moments rather than taking the library default of 1.0, since the
degree of over-dispersion differs between the two outcomes. The effect on
reported metrics is below the third decimal place, but the estimate is used
because a fixed value misstates the variance.
