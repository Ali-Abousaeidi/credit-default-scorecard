# Phase 6 Validation

## Held-Out Test Metrics

- AUC: 0.8481
- Gini: 0.6962
- KS: 0.5434
- Brier score: 0.0510

## Rank Ordering

- Score bands are eight quantile bands ordered from lowest score / riskiest to highest score / safest.
- Monotonic bad-rate decrease across score bands: True

## Stability

- Score PSI train vs test: 0.0003
- Score PSI interpretation: no significant shift

## Outputs

- `reports/validation_metrics.csv`
- `reports/rank_ordering_table.csv`
- `reports/calibration_table.csv`
- `reports/psi_summary.csv`
- `reports/figures/roc_curve.png`
- `reports/figures/ks_curve.png`
- `reports/figures/rank_ordering_bad_rate.png`
- `reports/figures/calibration_curve.png`
- `reports/figures/calibration_by_decile.png`
