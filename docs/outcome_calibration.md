# Outcome calibration

Thresholds in `config/thresholds.yml` were set from observed base rates on the
2016-2024 ACLED Nigeria panel, before any modelling. This file records the
calibration so the choices are auditable.

## Why two outcomes

`primary` covers serious violence: violence against civilians, battles and
riots. It is comparable to conventional conflict-forecasting targets and lets
results be read against the existing literature.

`youth` covers the subset this study is named for: mob violence, violent
demonstration, abduction, force against protesters and sexual violence. It
excludes most insurgency and counter-insurgency.

The difference is substantive rather than cosmetic. Under `primary`, Borno
holds 9.0% of escalation weeks. Under `youth`, the most affected locality is
the Federal Capital Territory at 7.5%. Two different pictures of the same
country in the same weeks, which is the source-dependency the study examines.

## Calibration results

| | primary | youth |
|---|---|---|
| events 2016-2024 | 22,377 | 6,200 |
| fatalities | 59,477 | 1,786 |
| labelled locality-weeks | 16,909 | 16,909 |
| escalation weeks | 3,687 | 2,867 |
| base rate | 21.80% | 16.96% |
| zero trailing baseline | 50.0% | 83.3% |
| top locality share | Borno 9.0% | FCT 7.5% |
| states with no escalation week | 1 of 37 | 0 of 37 |

## Threshold selection

Candidates tested against the real panel:

**primary**, lethality required:

| min events | percentile | base rate | top locality share |
|---|---|---|---|
| 2 | 0.75 | 37.95% | 6.1% |
| 3 | 0.80 | 29.44% | 7.6% |
| 3 | 0.90 | 28.03% | 7.0% |
| **4** | **0.90** | **21.80%** | **9.0%** |

**youth**, lethality not required:

| min events | percentile | base rate | top locality share |
|---|---|---|---|
| 1 | 0.90 | 35.91% | 4.5% |
| **2** | **0.90** | **16.96%** | **7.5%** |
| 3 | 0.90 | 7.73% | 9.1% |

Selected values balance a plausible escalation rate against concentration. The
rejected permissive settings label roughly a third of all locality-weeks as
escalation, which describes ordinary conditions rather than deterioration.

## Why the youth outcome does not require a death

Mob violence, abduction and the policing of demonstrations are frequently
non-fatal: 6,200 youth-category events carry 1,786 fatalities, against 22,377
primary events carrying 59,477. Requiring a death would discard most of the
outcome and reintroduce the insurgency bias the outcome exists to avoid.

## Why explosions and remote violence are excluded

Air strikes, IEDs and suicide bombing are almost entirely insurgency and
counter-insurgency. Including them would restore Borno's dominance without
adding anything about youth and community violence.

## Why protests are excluded from the outcome but kept as a predictor

The great majority are peaceful and are not violence. They enter the panel as
a leading feature, following the protest-to-conflict literature.

## A note on week alignment

ACLED weeks begin on Saturday. A Monday-anchored grid produces zero overlap on
merge and therefore an empty panel with no error raised. `week_anchor` is
configured, asserted at build time, and covered by a regression test.
