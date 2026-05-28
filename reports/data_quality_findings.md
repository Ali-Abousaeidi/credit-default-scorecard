# Phase 1 Data Quality Findings

## Dataset Snapshot

- Source file: `data/raw/cs-training.csv`
- Shape: 150,000 rows x 11 columns
- Target bad rate: 6.68%
- Target classes:
- `SeriousDlqin2yrs=0`: 139,974 rows (93.32%).
- `SeriousDlqin2yrs=1`: 10,026 rows (6.68%).

## Key Findings

- The dataset is strongly imbalanced: only 6.68% of rows are bad accounts.
- `MonthlyIncome` is missing in 29,731 rows (19.82%).
- `NumberOfDependents` is missing in 3,924 rows (2.62%).
- `age` has 1 invalid zero-age rows.
- `RevolvingUtilizationOfUnsecuredLines` is greater than 1 in 3,321 rows (2.21%), so utilization has extreme values that need deliberate treatment.
- `DebtRatio` has a p99 of 4,979.04 and a max of 329,664.00, indicating a long right tail.

## Sentinel-Like Delinquency Values

- `NumberOfTime30-59DaysPastDueNotWorse` has 5 rows with value 96 (0.00%).
- `NumberOfTime30-59DaysPastDueNotWorse` has 264 rows with value 98 (0.18%).
- `NumberOfTimes90DaysLate` has 5 rows with value 96 (0.00%).
- `NumberOfTimes90DaysLate` has 264 rows with value 98 (0.18%).
- `NumberOfTime60-89DaysPastDueNotWorse` has 5 rows with value 96 (0.00%).
- `NumberOfTime60-89DaysPastDueNotWorse` has 264 rows with value 98 (0.18%).

## Phase 2 Cleaning Decisions To Implement

- Keep missing values available for WoE binning instead of blindly imputing them.
- Treat or cap the 96/98 delinquency values explicitly before binning.
- Correct or remove the invalid age-zero record.
- Cap or bin extreme utilization and debt-ratio values based on train-only logic.
- Split train/test before fitting binning, feature selection, or any model.

## Generated Outputs

- `reports/data_dictionary.csv`
- `reports/data_quality_summary.csv`
- `reports/sentinel_value_counts.csv`
- `reports/target_summary.csv`
- `reports/figures/target_distribution.png`
- `reports/figures/missingness.png`
- `reports/figures/revolving_utilization_distribution.png`
