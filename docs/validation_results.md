# Degradation and transfer

Two validation regimes conventional forecasting evaluation omits. Model:
regularised logistic regression, test period 2023-2024.

## H4: forecast errors track data conditions

**H4 is supported, with an important qualification about which conditions
matter.**

### Primary outcome, clean baseline AP 0.6809

| failure | severity | AP | change | recall at 4/wk |
|---|---|---|---|---|
| signal delayed | 1 week | 0.6809 | −0.0% | 0.244 |
| signal delayed | 2 weeks | 0.6808 | −0.0% | 0.244 |
| signal removed | total | 0.6803 | −0.0% | 0.244 |
| spatial removed | total | 0.6781 | −0.3% | 0.247 |
| volume truncated | 25% | 0.6783 | −0.3% | 0.245 |
| volume truncated | 50% | 0.6735 | −1.0% | 0.244 |
| localities masked | 10% | 0.6483 | −4.7% | 0.232 |
| localities masked | 25% | 0.6236 | −8.3% | 0.225 |
| **localities masked** | **50%** | **0.5769** | **−15.2%** | **0.195** |

### Youth outcome, clean baseline AP 0.4966

| failure | severity | AP | change | recall at 4/wk |
|---|---|---|---|---|
| signal delayed | 1 week | 0.4958 | −0.2% | 0.247 |
| signal removed | total | 0.4950 | −0.3% | 0.245 |
| spatial removed | total | 0.4906 | −1.2% | 0.247 |
| volume truncated | 50% | 0.4885 | −1.6% | 0.244 |
| localities masked | 10% | 0.4851 | −2.3% | 0.249 |
| localities masked | 25% | 0.4200 | −15.4% | 0.209 |
| **localities masked** | **50%** | **0.4172** | **−16.0%** | **0.209** |

### Reading these

**Not all data failures are equal, and the difference is stark.** Losing the
protest signal entirely costs under 0.5% AP. Losing spatial features costs
around 1%. Halving reported volume costs 1 to 2%. But localities that stop
reporting cost 15 to 16%.

This is coherent with the H1 null. The model relies almost entirely on event
history, so failures affecting auxiliary sources are nearly free while
failures affecting the outcome source itself are expensive.

**Degradation is gradual rather than abrupt.** No failure mode produces a
collapse; the worst case is a 16% relative loss. Operationally this means a
service can continue running in a flag-raised state rather than being obliged
to abstain, which is the more useful of the two failure behaviours.

**The dangerous failure is the silent one.** A locality that stops reporting
looks identical to a locality that has become peaceful. Masking 25% of
localities costs 15% AP on the youth outcome, and nothing in a conventional
accuracy report would indicate that reporting had stopped rather than violence
having ceased. This is precisely why source completeness belongs among the
service-level indicators rather than in a footnote.

## Geographic transfer: near zero skill

| held out | primary lift | youth lift |
|---|---|---|
| Borno | 1.10 | 1.11 |
| FCT | 1.19 | 1.01 |
| Lagos | 1.12 | 1.00 |
| Zamfara | 0.91 | 0.98 |
| Katsina | 1.04 | 0.97 |

Lift of 1.00 means average precision equal to the base rate, which is no skill
at all.

**On states the model has never seen, it performs at or barely above chance.**
Two of ten holdouts score below 1.00, meaning worse than simply predicting the
base rate. The best result is 1.19 on the FCT for the primary outcome.

Compare this with in-sample temporal performance, where lift reaches 1.96.
Nearly all of the model's apparent skill comes from learning each locality's
own level, not from any generalisable relationship between predictors and
escalation.

### What this means

The model is effectively a well-calibrated memory of which places are violent.
That is genuinely useful for allocating scarce mediators among known
localities, which is the actual deployment scenario. It is close to useless
for the harder and arguably more valuable task: identifying a quiet locality
about to deteriorate.

It also bounds any claim about transferability. A model trained on Nigerian
states would not transfer to new administrative units, to a neighbouring
country, or to the United States. What the study argues transfers is the
architecture and the reliability objectives, not the fitted model, and this
result is the evidence for that distinction rather than an assumption behind
it.

Reviewers will ask about this. Reporting it directly is better than having it
inferred from a temporal-only evaluation.

## Reproducing

```bash
PYTHONPATH=src python -m hsre.models.run_validation --data path/to/acled.xlsx
```

Writes `reports/tables/degradation.csv` and `geographic_transfer.csv`.
