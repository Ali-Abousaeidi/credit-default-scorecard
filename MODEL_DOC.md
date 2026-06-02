# Model Documentation

## 1. Model Overview

Model name: Credit Default Prediction & Scorecard

Purpose: estimate probability of serious delinquency within two years and convert the fitted model into a points-based credit scorecard.

Champion model: logistic regression on Weight-of-Evidence transformed predictors.

Challenger model: XGBoost classifier trained on cleaned raw predictors.

## 2. Data

Primary source: OpenML dataset `46929`, `GiveMeSomeCredit`, citing the original Kaggle Give Me Some Credit competition.

Raw normalized file:

```text
data/raw/cs-training.csv
```

Target:

- `SeriousDlqin2yrs = 1`: bad account
- `SeriousDlqin2yrs = 0`: good account

Sample after deterministic cleaning:

- Train rows: 112,499
- Test rows: 37,500
- Train bad rate: 6.68%
- Test bad rate: 6.69%

## 3. Data Quality And Cleaning

Phase 1 EDA found:

- 19.82% missing `MonthlyIncome`
- 2.62% missing `NumberOfDependents`
- 1 invalid `age = 0` record
- 2.21% of records with `RevolvingUtilizationOfUnsecuredLines > 1`
- 96/98 sentinel-like values in delinquency count fields

Cleaning decisions:

- Remove the one invalid age-zero row.
- Replace 96/98 delinquency count values with missing.
- Keep missing values for WoE binning rather than imputing them.
- Fit utilization and debt-ratio outlier caps on train only, then apply to train and test.
- Split train/test before binning, feature selection, model fitting, score scaling, or validation.

## 4. Binning And Feature Screening

Method: `optbinning.OptimalBinning`, fitted on the training sample only.

Information Value outputs:

```text
reports/iv_ranking.csv
reports/binning_tables.csv
```

High-IV review flags:

- `RevolvingUtilizationOfUnsecuredLines`
- `NumberOfTimes90DaysLate`
- `NumberOfTime30-59DaysPastDueNotWorse`

These variables are plausible high-signal credit risk predictors, but they remain subject to sign checks, validation, and business review.

## 5. Model Development

Candidate features: all predictors with IV >= 0.02.

Feature selection removed:

- `NumberOfOpenCreditLinesAndLoans_woe`: p-value above 0.05
- `NumberOfTime60-89DaysPastDueNotWorse_woe`: wrong coefficient sign after controlling for other delinquency fields

Final scorecard characteristics:

- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfDependents`

All selected coefficients have the expected negative sign under the WoE orientation: higher WoE means safer, so coefficients should reduce default log-odds.

Model fit:

- Pseudo R-squared: 0.2414
- AIC: 41,899.15
- Max VIF: 1.41

## 6. Scorecard Scaling

Scaling convention:

- Base score: 600
- Base odds good:bad: 50:1
- PDO: 20
- Factor: 28.853901
- Offset: 487.122876

The scorecard table is saved at:

```text
reports/scorecard_points.csv
```

Higher score means lower estimated default risk.

## 7. Validation

Validation sample: held-out test set only.

| Metric | Value |
|--------|-------|
| AUC | 0.8481 |
| Gini | 0.6962 |
| KS | 0.5434 |
| Brier score | 0.0510 |
| Score PSI | 0.0003 |

Bootstrap 95% confidence intervals:

| Metric | 95% CI |
|--------|--------|
| AUC | [0.8408, 0.8555] |
| Gini | [0.6817, 0.7110] |
| KS | [0.5287, 0.5593] |

Rank ordering:

- Eight quantile score bands
- Ordered from lowest score / riskiest to highest score / safest
- Observed bad rate declines monotonically across bands

Stability:

- Score and characteristic PSI are computed train vs test on a random stratified
  split. Because both samples are drawn from the same pool with no time dimension,
  these PSI values are ~0 by construction and are **not** evidence of out-of-time
  stability. They are reported only as a sanity check that the split did not
  distort the score distribution. True population-stability monitoring would
  require out-of-time samples (see Limitations).
- Selected raw-characteristic PSI values are all below 0.10, as expected for a
  random split.

Calibration:

- Raw scorecard PD mean: 6.73%
- Observed test bad rate: 6.69%
- Platt scaling benchmark has effectively unchanged AUC, Brier score, and log loss.
- Calibration outputs are saved in `reports/calibration_summary.md`.

Binning diagnostics:

- Final scorecard characteristics have bin-level population share, bad-rate,
  and WoE plots in `reports/figures/bin_diagnostic_*.png`.

## 8. Explainability

SHAP values are calculated with `shap.LinearExplainer` from the fitted scorecard coefficients.

Top global drivers by mean absolute SHAP:

- `RevolvingUtilizationOfUnsecuredLines`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `age`
- `NumberOfTimes90DaysLate`

Individual reason-code examples are saved at:

```text
reports/reason_codes_examples.csv
```

## 9. Challenger Model

XGBoost challenger results:

| Model | AUC | Gini | KS |
|-------|-----|------|----|
| Logistic scorecard | 0.8481 | 0.6962 | 0.5434 |
| XGBoost challenger | 0.8667 | 0.7335 | 0.5752 |

Interpretation: XGBoost improves discrimination, but the logistic scorecard remains the champion because it is transparent, points-based, easier to validate, and better aligned with scorecard governance expectations.

## 10. Limitations

- The dataset is a public competition dataset, not a bank-owned production portfolio.
- No time field is available, so validation is a stratified holdout rather than true out-of-time validation. Consequently the reported train-vs-test PSI is ~0 by construction and does not demonstrate population stability over time.
- `DebtRatio` has mixed semantics: when `MonthlyIncome` is missing (~20% of rows) the field holds a raw dollar amount rather than a ratio (median ~1159 vs ~0.30 when income is present). It therefore partly proxies the income-missing population and carries the largest model coefficient despite the lowest IV. A production redevelopment should add an explicit income-missing indicator and regime-aware binning.
- No reject inference is included.
- No fairness testing is included.
- No production monitoring pipeline is included.
- Regulatory notes are contextual and not legal advice.
- `age` is used for this public-data demonstration but would require legal,
  compliance, and fair-lending review before any production use.

## 11. Monitoring Plan

Recommended monitoring if this were productionized:

- Monthly score distribution PSI
- Monthly PSI on key characteristics
- Bad-rate rank-ordering by score band
- Calibration drift: predicted PD vs observed default rate
- Override and exception monitoring
- Periodic coefficient/sign review after redevelopment samples accumulate

## 12. Key Files

```text
reports/data_quality_findings.md
reports/cleaning_decisions.md
reports/binning_summary.md
reports/model_summary.md
reports/scorecard_summary.md
reports/validation_summary.md
reports/calibration_summary.md
reports/bin_diagnostics_summary.md
reports/explainability_summary.md
reports/challenger_summary.md
docs/regulatory_context.md
docs/streamlit_demo.md
docs/lending_club_extension.md
```
