# Phase 4 Logistic Regression Model

## Model Choice

Champion model: logistic regression on WoE-transformed predictors.

Expected sign convention: because higher WoE means safer / lower default risk,
selected coefficients should be negative when modelling `SeriousDlqin2yrs = 1`.

## Feature Selection

Started from all IV-screened WoE predictors and removed features that were not
statistically significant at p <= 0.05 or had the wrong sign.

- Removed `NumberOfOpenCreditLinesAndLoans_woe`: p-value above 0.05 (coef=-0.0643, p=0.183).
- Removed `NumberOfTime60-89DaysPastDueNotWorse_woe`: wrong coefficient sign for WoE orientation (coef=0.5536, p=6.7e-25).

## Final Features

- `RevolvingUtilizationOfUnsecuredLines_woe`
- `age_woe`
- `NumberOfTime30-59DaysPastDueNotWorse_woe`
- `DebtRatio_woe`
- `MonthlyIncome_woe`
- `NumberOfTimes90DaysLate_woe`
- `NumberRealEstateLoansOrLines_woe`
- `NumberOfDependents_woe`

## Fit Summary

- Final feature count: 8
- Pseudo R-squared: 0.2414
- AIC: 41,899.15
- Max VIF: 1.41
- Train AUC sanity check: 0.8491
- Test AUC early holdout check: 0.8481

Full validation is handled in Phase 6; the test AUC here is only a quick
smoke check that the serialized model scores the holdout sample.

## Outputs

- `reports/model_coefficients.csv`
- `reports/feature_selection_decisions.csv`
- `reports/vif_report.csv`
- `models/logistic_scorecard_model.joblib`
