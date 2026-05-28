# Phase 2 Cleaning Decisions

## Target Definition

- Target: `SeriousDlqin2yrs`
- Bad event: `SeriousDlqin2yrs = 1`
- Good event: `SeriousDlqin2yrs = 0`

## Deterministic Cleaning Before Split

- Removed 1 row where `age <= 0`.
- Replaced 96/98 delinquency-count sentinel values with missing values:
- `NumberOfTime30-59DaysPastDueNotWorse`: 269 values replaced with missing
- `NumberOfTimes90DaysLate`: 269 values replaced with missing
- `NumberOfTime60-89DaysPastDueNotWorse`: 269 values replaced with missing

Missing values are retained for later WoE binning rather than imputed in Phase 2.

## Train/Test Split

- Split type: stratified random split on `SeriousDlqin2yrs`
- Random state: 42
- Test size: 25%
- Train rows: 112,499; bad rate: 6.68%
- Test rows: 37,500; bad rate: 6.69%

## Train-Fitted Outlier Caps

Caps were fit on the training sample only and then applied to train and test.

- `RevolvingUtilizationOfUnsecuredLines` upper cap: 1.35955
- `DebtRatio` upper cap: 6,184.02

## Saved Outputs

- `data/processed/train_clean.csv`
- `data/processed/test_clean.csv`
- `reports/cleaning_metadata.json`
