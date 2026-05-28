# Phase 5 Scorecard

## Scaling

- Base score: 600
- Base odds good:bad: 50:1
- PDO: 20
- Factor: 28.853901
- Offset: 487.122876

Score formula:

```text
score = offset - factor * logit(PD)
```

Higher scores indicate lower estimated default risk.

## Score Distribution

- Train mean score: 582.97
- Train min/max score: 421.74 / 642.86
- Test mean score: 582.96
- Test min/max score: 426.63 / 642.86
- Best test profile: score 642.86, PD 0.45%
- Worst test profile: score 426.63, PD 89.06%

## Outputs

- `reports/scorecard_points.csv` (46 attribute rows)
- `data/processed/train_scores.csv`
- `data/processed/test_scores.csv`
- `reports/figures/score_distribution.png`
