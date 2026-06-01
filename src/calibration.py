"""Post-model PD calibration checks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SCORES_FILE,
    TRAIN_SCORES_FILE,
)


def _logit(probability: pd.Series) -> np.ndarray:
    """Convert probabilities to logits with numerical clipping."""
    clipped = probability.clip(1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).to_numpy().reshape(-1, 1)


def fit_platt_scaler(train_scores: pd.DataFrame) -> LogisticRegression:
    """Fit Platt scaling on champion train predictions."""
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(_logit(train_scores["pd"]), train_scores[TARGET_COLUMN])
    return calibrator


def build_calibration_comparison(
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw and Platt-calibrated probabilities."""
    calibrator = fit_platt_scaler(train_scores)
    calibrated_pd = calibrator.predict_proba(_logit(test_scores["pd"]))[:, 1]

    scored = test_scores.copy()
    scored["pd_platt"] = calibrated_pd

    rows = []
    for label, column in [("raw_scorecard_pd", "pd"), ("platt_calibrated_pd", "pd_platt")]:
        rows.append(
            {
                "probability": label,
                "auc": roc_auc_score(scored[TARGET_COLUMN], scored[column]),
                "brier": brier_score_loss(scored[TARGET_COLUMN], scored[column]),
                "log_loss": log_loss(scored[TARGET_COLUMN], scored[column]),
                "mean_pd": scored[column].mean(),
                "observed_bad_rate": scored[TARGET_COLUMN].mean(),
            }
        )

    return scored, pd.DataFrame(rows)


def save_calibration_comparison_plot(scored: pd.DataFrame) -> None:
    """Save raw-vs-calibrated PD decile plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_rows = []
    for label, column in [("Raw scorecard PD", "pd"), ("Platt calibrated PD", "pd_platt")]:
        temp = scored.copy()
        temp["band"] = pd.qcut(temp[column], q=10, labels=False, duplicates="drop") + 1
        grouped = (
            temp.groupby("band", observed=True)
            .agg(mean_pd=(column, "mean"), observed_bad_rate=(TARGET_COLUMN, "mean"))
            .reset_index()
        )
        grouped["probability"] = label
        plot_rows.append(grouped)

    plot_data = pd.concat(plot_rows, ignore_index=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 5.8))
    sns.lineplot(
        data=plot_data,
        x="mean_pd",
        y="observed_bad_rate",
        hue="probability",
        marker="o",
    )
    max_value = max(plot_data["mean_pd"].max(), plot_data["observed_bad_rate"].max())
    plt.plot([0, max_value], [0, max_value], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed bad rate")
    plt.title("Raw vs Platt-calibrated PD")
    plt.tight_layout(pad=1.4)
    plt.savefig(
        FIGURES_DIR / "calibration_comparison.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def write_calibration_report(comparison: pd.DataFrame) -> None:
    """Write calibration comparison report."""
    raw = comparison[comparison["probability"] == "raw_scorecard_pd"].iloc[0]
    calibrated = comparison[comparison["probability"] == "platt_calibrated_pd"].iloc[0]
    report = f"""# Calibration Comparison

## Method

Platt scaling is fit on the champion model's training predictions and evaluated
on the held-out test predictions. This is a calibration check, not a replacement
for the deployed scorecard points.

## Results

| Probability | AUC | Brier | Log loss | Mean PD | Observed bad rate |
|-------------|-----|-------|----------|---------|-------------------|
| Raw scorecard PD | {raw["auc"]:.4f} | {raw["brier"]:.4f} | {raw["log_loss"]:.4f} | {raw["mean_pd"]:.4f} | {raw["observed_bad_rate"]:.4f} |
| Platt calibrated PD | {calibrated["auc"]:.4f} | {calibrated["brier"]:.4f} | {calibrated["log_loss"]:.4f} | {calibrated["mean_pd"]:.4f} | {calibrated["observed_bad_rate"]:.4f} |

## Interpretation

The raw scorecard is already close to the observed central tendency. Platt
scaling is included to demonstrate a validation workflow for probability
calibration and to create a benchmark for future redevelopment.

## Outputs

- `reports/calibration_comparison.csv`
- `data/processed/test_scores_calibrated.csv`
- `reports/figures/calibration_comparison.png`
"""
    (REPORTS_DIR / "calibration_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run calibration comparison."""
    if not TRAIN_SCORES_FILE.exists() or not TEST_SCORES_FILE.exists():
        raise FileNotFoundError("Run `python -m src.scorecard` before calibration.")

    train_scores = pd.read_csv(TRAIN_SCORES_FILE)
    test_scores = pd.read_csv(TEST_SCORES_FILE)
    scored, comparison = build_calibration_comparison(train_scores, test_scores)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "calibration_comparison.csv", index=False)
    scored.to_csv(TEST_SCORES_FILE.parent / "test_scores_calibrated.csv", index=False)
    save_calibration_comparison_plot(scored)
    write_calibration_report(comparison)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
