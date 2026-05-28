# Phase 7 Explainability

## SHAP Method

SHAP values are calculated with `shap.LinearExplainer` using the fitted
logistic scorecard coefficients. Contributions are on the log-odds scale of
default. Positive values raise predicted default risk; negative values lower it.

## Global Importance

- `RevolvingUtilizationOfUnsecuredLines`: mean |SHAP| 0.5703
- `NumberOfTime30-59DaysPastDueNotWorse`: mean |SHAP| 0.2747
- `DebtRatio`: mean |SHAP| 0.2141
- `age`: mean |SHAP| 0.2014
- `NumberOfTimes90DaysLate`: mean |SHAP| 0.1703

## Individual Reason-Code Examples

- `highest_risk`: score 426.63, PD 89.06%; largest contribution is `NumberOfTime30-59DaysPastDueNotWorse` (raises PD).
- `lowest_risk`: score 642.86, PD 0.45%; largest contribution is `RevolvingUtilizationOfUnsecuredLines` (lowers PD).
- `median_risk`: score 591.86, PD 2.58%; largest contribution is `NumberOfTime30-59DaysPastDueNotWorse` (lowers PD).

## Outputs

- `reports/shap_global_importance.csv`
- `reports/reason_codes_examples.csv`
- `reports/figures/shap_global_importance.png`
- `reports/figures/reason_codes_highest_risk.png`
- `reports/figures/reason_codes_median_risk.png`
- `reports/figures/reason_codes_lowest_risk.png`
- `docs/regulatory_context.md`
