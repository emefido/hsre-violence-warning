# Alert budgets and the volume-detection trade-off

The central operational result. A model produces a risk score for every
locality-week, but an institution can investigate only a few. What does
constraining alert volume to real capacity actually cost?

Model: regularised logistic regression on event history, the strongest
performer in stage 8. Test period 2023-2024, 3,811 state-weeks over 103 weeks.

## Detection against volume

### Primary outcome, 1,326 escalation weeks

| alerts/week | recall | precision | false alerts/week |
|---|---|---|---|
| 1 | 0.067 | 0.864 | 0.1 |
| 2 | 0.126 | 0.811 | 0.4 |
| 4 | 0.242 | 0.779 | 0.9 |
| 8 | 0.450 | 0.725 | 2.2 |
| 16 | 0.748 | 0.602 | 6.4 |
| 37 | 1.000 | 0.348 | 24.1 |

### Youth outcome, 1,019 escalation weeks

| alerts/week | recall | precision | false alerts/week |
|---|---|---|---|
| 1 | 0.062 | 0.612 | 0.4 |
| 2 | 0.127 | 0.626 | 0.7 |
| 4 | 0.248 | 0.614 | 1.5 |
| 8 | 0.420 | 0.519 | 3.8 |
| 16 | 0.696 | 0.430 | 9.1 |
| 37 | 1.000 | 0.267 | 27.1 |

## Regime comparison at a capacity of four localities per week

### Primary

| regime | alerts/week | precision | recall | false/week | error budget |
|---|---|---|---|---|---|
| accuracy_optimised | 17.9 | 0.575 | 0.801 | 7.6 | fail |
| recall_optimised | 17.9 | 0.575 | 0.800 | 7.6 | fail |
| alert_budgeted | 4.0 | 0.779 | 0.242 | 0.9 | fail |

### Youth

| regime | alerts/week | precision | recall | false/week | error budget |
|---|---|---|---|---|---|
| accuracy_optimised | 16.1 | 0.430 | 0.699 | 9.2 | fail |
| recall_optimised | 20.9 | 0.380 | 0.801 | 12.9 | fail |
| alert_budgeted | 4.0 | 0.614 | 0.248 | 1.5 | fail |

## Every regime fails the error budget

This is the finding, not a defect.

The two constraints are a review burden ceiling of ten alerts per week and a
severe-event recall floor of 70%. No configuration satisfies both.

The accuracy-optimised and recall-optimised regimes fail on burden: they issue
16 to 21 alerts per week against a ceiling of ten. The alert-budgeted regime
satisfies the burden comfortably at four per week, then fails the recall
floor, catching under 25% of escalations against a 70% requirement.

There is no threshold that satisfies both, because the two constraints are
jointly infeasible for this model on this panel. That is a statement about the
service, not about the threshold: **a capacity of four localities per week is
too small for the detection standard the objectives demand.**

## What the trade-off costs

The volume-detection exchange rate is roughly linear rather than favourable.
On the primary outcome, quadrupling alerts from 4 to 16 per week takes recall
from 24% to 75%, while precision falls from 0.78 to 0.60. Reaching 95%
detection requires 28 alerts per week, seven times the stated capacity, at
which point 24 of every 37 alerts is false.

This matters because the literature's implicit preference for sensitivity
assumes detection is cheap at the margin. It is not. Each additional
escalation caught costs progressively more analyst time, and past roughly 16
alerts per week the majority of alerts are false.

## The implication for the manuscript

Combined with the H1 null, the picture is coherent and uncomfortable.

Prediction works: at four alerts per week, roughly 78% of flagged localities
do escalate. That is a usable signal, produced from one open dataset on a
laptop.

Additional data sources do not improve it materially.

And the binding constraint is neither. It is that four localities per week
catches a quarter of escalations, and catching most of them would require
capacity a Nigerian state security committee does not have.

The gap between warning and response is therefore not primarily a modelling
problem or a data problem. It is a resourcing problem, and it is visible only
when alert volume is treated as a scarce quantity governed by response
capacity rather than as a free output of a classifier. Conventional evaluation
reporting average precision alone would show a competent model and reveal none
of this.

## Setting capacity honestly

The capacity of four is illustrative, drawn from `config/thresholds.yml`, not
measured from a real institution. Any deployment should set it from the
responding body's actual review capability, and the objectives should be
renegotiated with that body rather than stipulated by the researcher. An
objective no institution has agreed to is not an objective.

Reproduce with:

```bash
PYTHONPATH=src python -m hsre.alerts.run_budget --data path/to/acled.xlsx --capacity 4
```
