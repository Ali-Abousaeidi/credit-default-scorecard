# Phase 7 Explainability

## SHAP Method

SHAP values are calculated with `shap.LinearExplainer` using the fitted
logistic scorecard coefficients. Contributions are on the log-odds scale of
default. Positive values raise predicted default risk; negative values lower it.

## Global Importance

- `Revolving utilization`: mean |SHAP| 0.5703
- `30-59 DPD count`: mean |SHAP| 0.2747
- `Debt ratio`: mean |SHAP| 0.2141
- `Age`: mean |SHAP| 0.2014
- `90+ DPD count`: mean |SHAP| 0.1703

## Individual Reason-Code Examples

- `highest_risk`: score 426.63, PD 89.06%; largest contribution is `30-59 DPD count` (raises PD).
- `lowest_risk`: score 642.86, PD 0.45%; largest contribution is `Revolving utilization` (lowers PD).
- `median_risk`: score 591.86, PD 2.58%; largest contribution is `30-59 DPD count` (lowers PD).

## Outputs

- `reports/shap_global_importance.csv`
- `reports/reason_codes_examples.csv`
- `reports/figures/shap_global_importance.png`
- `reports/figures/reason_codes_highest_risk.png`
- `reports/figures/reason_codes_median_risk.png`
- `reports/figures/reason_codes_lowest_risk.png`
- `docs/regulatory_context.md`
