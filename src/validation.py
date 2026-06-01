"""Held-out model validation metrics and plots."""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from src.config import (
    FIGURES_DIR,
    LOGISTIC_MODEL_FILE,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_CLEAN_FILE,
    TEST_SCORES_FILE,
    TRAIN_CLEAN_FILE,
    TRAIN_SCORES_FILE,
)

BOOTSTRAP_ITERATIONS = 300


def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test score outputs from Phase 5."""
    if not TRAIN_SCORES_FILE.exists() or not TEST_SCORES_FILE.exists():
        raise FileNotFoundError("Run `python -m src.scorecard` before validation.")
    return pd.read_csv(TRAIN_SCORES_FILE), pd.read_csv(TEST_SCORES_FILE)


def calculate_discrimination(scores: pd.DataFrame) -> dict[str, float]:
    """Calculate AUC, Gini, KS, and Brier score on the holdout sample."""
    y_true = scores[TARGET_COLUMN]
    pd_pred = scores["pd"]
    auc = roc_auc_score(y_true, pd_pred)
    fpr, tpr, _ = roc_curve(y_true, pd_pred)
    ks = float(np.max(tpr - fpr))
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks,
        "brier": float(brier_score_loss(y_true, pd_pred)),
    }


def bootstrap_metric_confidence_intervals(
    scores: pd.DataFrame,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for AUC, Gini, and KS on the holdout sample."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    y_true = scores[TARGET_COLUMN].to_numpy()
    pd_pred = scores["pd"].to_numpy()
    n_rows = len(scores)

    for _ in range(n_iterations):
        sample_index = rng.integers(0, n_rows, size=n_rows)
        if np.unique(y_true[sample_index]).size < 2:
            continue

        sample_scores = pd.DataFrame(
            {
                TARGET_COLUMN: y_true[sample_index],
                "pd": pd_pred[sample_index],
            }
        )
        rows.append(calculate_discrimination(sample_scores))

    boot = pd.DataFrame(rows)
    summary_rows = []
    for metric in ["auc", "gini", "ks"]:
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(boot[metric].mean()),
                "lower_95": float(boot[metric].quantile(0.025)),
                "upper_95": float(boot[metric].quantile(0.975)),
                "bootstrap_iterations": int(len(boot)),
            }
        )
    return pd.DataFrame(summary_rows)


def make_rank_ordering_table(scores: pd.DataFrame, n_bands: int = 8) -> pd.DataFrame:
    """Create score-band rank ordering table for the holdout sample."""
    ranked = scores.copy()
    ranked["score_band"] = pd.qcut(
        ranked["score"],
        q=n_bands,
        labels=False,
        duplicates="drop",
    )
    ranked["score_band"] = ranked["score_band"] + 1

    table = (
        ranked.groupby("score_band", observed=True)
        .agg(
            count=(TARGET_COLUMN, "size"),
            bads=(TARGET_COLUMN, "sum"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            mean_score=("score", "mean"),
            mean_pd=("pd", "mean"),
            observed_bad_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
        .sort_values("score_band")
    )
    table["goods"] = table["count"] - table["bads"]
    return table


def make_calibration_table(scores: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Create predicted-vs-observed calibration table."""
    calibrated = scores.copy()
    calibrated["pd_band"] = pd.qcut(
        calibrated["pd"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    calibrated["pd_band"] = calibrated["pd_band"] + 1

    return (
        calibrated.groupby("pd_band", observed=True)
        .agg(
            count=(TARGET_COLUMN, "size"),
            mean_pd=("pd", "mean"),
            observed_bad_rate=(TARGET_COLUMN, "mean"),
            min_pd=("pd", "min"),
            max_pd=("pd", "max"),
        )
        .reset_index()
        .sort_values("pd_band")
    )


def _psi_for_aligned_distributions(expected_counts: pd.Series, actual_counts: pd.Series) -> float:
    """Calculate PSI from aligned bin counts."""
    bins = expected_counts.index.union(actual_counts.index)
    expected = expected_counts.reindex(bins, fill_value=0).astype(float)
    actual = actual_counts.reindex(bins, fill_value=0).astype(float)

    expected_pct = expected / expected.sum()
    actual_pct = actual / actual.sum()
    epsilon = 1e-6
    expected_pct = expected_pct.clip(lower=epsilon)
    actual_pct = actual_pct.clip(lower=epsilon)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def psi_numeric(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """Calculate PSI for a numeric variable using train quantile bins."""
    expected_non_missing = expected.dropna()
    if expected_non_missing.nunique() <= 1:
        expected_counts = expected.astype("string").fillna("Missing").value_counts()
        actual_counts = actual.astype("string").fillna("Missing").value_counts()
        return _psi_for_aligned_distributions(expected_counts, actual_counts)

    _, bins = pd.qcut(
        expected_non_missing,
        q=n_bins,
        retbins=True,
        duplicates="drop",
    )
    bins = np.unique(bins)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_binned = pd.cut(expected, bins=bins, include_lowest=True).astype("string")
    actual_binned = pd.cut(actual, bins=bins, include_lowest=True).astype("string")
    expected_binned = expected_binned.fillna("Missing")
    actual_binned = actual_binned.fillna("Missing")
    return _psi_for_aligned_distributions(
        expected_binned.value_counts(),
        actual_binned.value_counts(),
    )


def psi_interpretation(psi: float) -> str:
    """Interpret PSI using common credit-risk thresholds."""
    if psi < 0.10:
        return "no significant shift"
    if psi < 0.25:
        return "moderate shift - monitor"
    return "significant shift"


def calculate_psi_summary(train_scores: pd.DataFrame, test_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate PSI on score and selected raw characteristics."""
    model_artifact = joblib.load(LOGISTIC_MODEL_FILE)
    selected_features = model_artifact["selected_features"]
    selected_raw_features = [feature[: -len("_woe")] for feature in selected_features]

    train_clean = pd.read_csv(TRAIN_CLEAN_FILE)
    test_clean = pd.read_csv(TEST_CLEAN_FILE)

    rows = [
        {
            "variable": "score",
            "psi": psi_numeric(train_scores["score"], test_scores["score"]),
        }
    ]
    for feature in selected_raw_features:
        rows.append(
            {
                "variable": feature,
                "psi": psi_numeric(train_clean[feature], test_clean[feature]),
            }
        )

    summary = pd.DataFrame(rows)
    summary["interpretation"] = summary["psi"].map(psi_interpretation)
    return summary.sort_values("psi", ascending=False)


def save_validation_plots(
    scores: pd.DataFrame,
    rank_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> None:
    """Save ROC, KS, rank-ordering, and calibration plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    y_true = scores[TARGET_COLUMN]
    pd_pred = scores["pd"]
    fpr, tpr, thresholds = roc_curve(y_true, pd_pred)

    plt.figure(figsize=(7.2, 5.8))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, pd_pred):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout(pad=1.4)
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    ks_values = tpr - fpr
    plt.figure(figsize=(8.2, 5.8))
    plt.plot(thresholds, tpr, label="Cumulative bads")
    plt.plot(thresholds, fpr, label="Cumulative goods")
    plt.plot(thresholds, ks_values, label="KS gap")
    plt.xlabel("PD threshold")
    plt.ylabel("Rate")
    plt.title("KS curve")
    plt.legend()
    plt.tight_layout(pad=1.4)
    plt.savefig(FIGURES_DIR / "ks_curve.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    plt.figure(figsize=(8.2, 5.8))
    sns.lineplot(
        data=rank_table,
        x="score_band",
        y="observed_bad_rate",
        marker="o",
        color="#E45756",
    )
    plt.xlabel("Score band, low score to high score")
    plt.ylabel("Observed bad rate")
    plt.title("Rank ordering by score band")
    plt.tight_layout(pad=1.4)
    plt.savefig(
        FIGURES_DIR / "rank_ordering_bad_rate.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, pd_pred, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7.2, 5.8))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed bad rate")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout(pad=1.4)
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    plt.figure(figsize=(8.2, 5.8))
    sns.lineplot(
        data=calibration_table,
        x="mean_pd",
        y="observed_bad_rate",
        marker="o",
        color="#4C78A8",
    )
    max_pd = calibration_table["mean_pd"].max()
    plt.plot([0, max_pd], [0, max_pd], "--", color="gray")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed bad rate")
    plt.title("Calibration by PD decile")
    plt.tight_layout(pad=1.4)
    plt.savefig(
        FIGURES_DIR / "calibration_by_decile.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def write_validation_report(
    metrics: dict[str, float],
    confidence_intervals: pd.DataFrame,
    rank_table: pd.DataFrame,
    psi_summary: pd.DataFrame,
) -> None:
    """Write validation summary report."""
    monotonic_rank_order = rank_table["observed_bad_rate"].is_monotonic_decreasing
    score_psi = psi_summary.loc[psi_summary["variable"] == "score", "psi"].iloc[0]
    ci_lookup = confidence_intervals.set_index("metric")
    report = f"""# Phase 6 Validation

## Held-Out Test Metrics

- AUC: {metrics["auc"]:.4f}
- Gini: {metrics["gini"]:.4f}
- KS: {metrics["ks"]:.4f}
- Brier score: {metrics["brier"]:.4f}

## Bootstrap 95% Confidence Intervals

- AUC: [{ci_lookup.loc["auc", "lower_95"]:.4f}, {ci_lookup.loc["auc", "upper_95"]:.4f}]
- Gini: [{ci_lookup.loc["gini", "lower_95"]:.4f}, {ci_lookup.loc["gini", "upper_95"]:.4f}]
- KS: [{ci_lookup.loc["ks", "lower_95"]:.4f}, {ci_lookup.loc["ks", "upper_95"]:.4f}]

## Rank Ordering

- Score bands are eight quantile bands ordered from lowest score / riskiest to highest score / safest.
- Monotonic bad-rate decrease across score bands: {monotonic_rank_order}

## Stability

- Score PSI train vs test: {score_psi:.4f}
- Score PSI interpretation: {psi_interpretation(score_psi)}

## Outputs

- `reports/validation_metrics.csv`
- `reports/validation_confidence_intervals.csv`
- `reports/rank_ordering_table.csv`
- `reports/calibration_table.csv`
- `reports/psi_summary.csv`
- `reports/figures/roc_curve.png`
- `reports/figures/ks_curve.png`
- `reports/figures/rank_ordering_bad_rate.png`
- `reports/figures/calibration_curve.png`
- `reports/figures/calibration_by_decile.png`
"""
    (REPORTS_DIR / "validation_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 6 held-out validation."""
    train_scores, test_scores = load_scores()
    metrics = calculate_discrimination(test_scores)
    confidence_intervals = bootstrap_metric_confidence_intervals(test_scores)
    rank_table = make_rank_ordering_table(test_scores)
    calibration_table = make_calibration_table(test_scores)
    psi_summary = calculate_psi_summary(train_scores, test_scores)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "validation_metrics.csv", index=False)
    confidence_intervals.to_csv(REPORTS_DIR / "validation_confidence_intervals.csv", index=False)
    rank_table.to_csv(REPORTS_DIR / "rank_ordering_table.csv", index=False)
    calibration_table.to_csv(REPORTS_DIR / "calibration_table.csv", index=False)
    psi_summary.to_csv(REPORTS_DIR / "psi_summary.csv", index=False)

    save_validation_plots(test_scores, rank_table, calibration_table)
    write_validation_report(metrics, confidence_intervals, rank_table, psi_summary)

    score_psi = psi_summary.loc[psi_summary["variable"] == "score", "psi"].iloc[0]
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test Gini: {metrics['gini']:.4f}")
    print(f"Test KS: {metrics['ks']:.4f}")
    print(f"Score PSI: {score_psi:.4f}")


if __name__ == "__main__":
    main()
