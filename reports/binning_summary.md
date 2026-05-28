# Phase 3 WoE Binning And IV

## Method

- Fitted `optbinning.OptimalBinning` on the training sample only.
- Applied the fitted bins to both train and test.
- Missing values use empirical WoE, preserving missingness as an informative bin.
- Minimum bin size target: 5% of the training sample.

## Outputs

- `reports/binning_tables.csv`
- `reports/iv_ranking.csv`
- `data/processed/train_woe.csv`
- `data/processed/test_woe.csv`
- `models/binning_artifacts.joblib`

## IV Screening

- Candidate predictors with IV >= 0.02: 10
- High-IV leakage review threshold: IV > 0.50

- `RevolvingUtilizationOfUnsecuredLines` IV=1.120: suspiciously strong - review for leakage
- `NumberOfTimes90DaysLate` IV=0.830: suspiciously strong - review for leakage
- `NumberOfTime30-59DaysPastDueNotWorse` IV=0.736: suspiciously strong - review for leakage

High IV does not automatically mean leakage here because delinquency-history
and utilization variables are expected to be strong in this dataset. These
variables still need sign, stability, and business-sense checks in modelling.
