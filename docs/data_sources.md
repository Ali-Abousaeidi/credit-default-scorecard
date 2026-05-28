# Data Sources

## Primary Dataset

This project uses the Give Me Some Credit credit scoring dataset.

The reproducible project source is OpenML dataset `46929`, `GiveMeSomeCredit`,
which is marked public and cites the original Kaggle competition:

```text
Credit Fusion and Will Cukierski. Give Me Some Credit.
https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.
```

OpenML's curated file renames the original Kaggle target to
`FinancialDistressNextTwoYears` with values `No` and `Yes`. The project fetch
script maps it back into the scorecard convention used in the plan:

- `No` -> `SeriousDlqin2yrs = 0`
- `Yes` -> `SeriousDlqin2yrs = 1`

The normalized raw project file is written to:

```text
data/raw/cs-training.csv
```

Raw data is ignored by git.
