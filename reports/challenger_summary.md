# Phase 8 Challenger Model

## Champion vs Challenger

| Model | AUC | Gini | KS |
|-------|-----|------|----|
| Logistic scorecard | 0.8481 | 0.6962 | 0.5434 |
| XGBoost challenger | 0.8667 | 0.7335 | 0.5752 |

## Interpretation

The challenger AUC delta is +0.0186; the KS delta is +0.0319.

The XGBoost challenger is useful as a performance benchmark, but the logistic
scorecard remains the champion because it is transparent, directly convertible
to points, easier to validate, and easier to explain in a regulated credit-risk
setting.

## Outputs

- `reports/challenger_comparison.csv`
- `reports/figures/champion_challenger_roc.png`
- `models/xgboost_challenger.joblib`
