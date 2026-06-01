# Calibration Comparison

## Method

Platt scaling is fit on the champion model's training predictions and evaluated
on the held-out test predictions. This is a calibration check, not a replacement
for the deployed scorecard points.

## Results

| Probability | AUC | Brier | Log loss | Mean PD | Observed bad rate |
|-------------|-----|-------|----------|---------|-------------------|
| Raw scorecard PD | 0.8481 | 0.0510 | 0.1861 | 0.0673 | 0.0669 |
| Platt calibrated PD | 0.8481 | 0.0510 | 0.1861 | 0.0673 | 0.0669 |

## Interpretation

The raw scorecard is already close to the observed central tendency. Platt
scaling is included to demonstrate a validation workflow for probability
calibration and to create a benchmark for future redevelopment.

## Outputs

- `reports/calibration_comparison.csv`
- `data/processed/test_scores_calibrated.csv`
- `reports/figures/calibration_comparison.png`
