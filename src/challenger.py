"""XGBoost challenger model benchmark."""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_CLEAN_FILE,
    TEST_SCORES_FILE,
    TRAIN_CLEAN_FILE,
)

CHALLENGER_MODEL_FILE = MODELS_DIR / "xgboost_challenger.joblib"


def ks_statistic(y_true: pd.Series, pd_pred: pd.Series) -> float:
    """Calculate KS from ROC curve coordinates."""
    fpr, tpr, _ = roc_curve(y_true, pd_pred)
    return float((tpr - fpr).max())


def train_xgboost_challenger(train: pd.DataFrame) -> XGBClassifier:
    """Train an XGBoost challenger on cleaned raw predictors."""
    x_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model


def challenger_metrics(model: XGBClassifier, test: pd.DataFrame) -> dict[str, float]:
    """Calculate challenger holdout metrics."""
    x_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    pd_pred = pd.Series(model.predict_proba(x_test)[:, 1], index=test.index)
    auc = roc_auc_score(y_test, pd_pred)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_test, pd_pred),
    }


def champion_metrics(test_scores: pd.DataFrame) -> dict[str, float]:
    """Calculate champion holdout metrics from saved scorecard outputs."""
    y_test = test_scores[TARGET_COLUMN]
    pd_pred = test_scores["pd"]
    auc = roc_auc_score(y_test, pd_pred)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_test, pd_pred),
    }


def save_roc_comparison(
    test: pd.DataFrame,
    test_scores: pd.DataFrame,
    challenger_model: XGBClassifier,
) -> None:
    """Save champion-vs-challenger ROC comparison."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_test = test[TARGET_COLUMN]
    champion_pd = test_scores["pd"]
    challenger_pd = challenger_model.predict_proba(test.drop(columns=[TARGET_COLUMN]))[:, 1]

    champ_fpr, champ_tpr, _ = roc_curve(y_test, champion_pd)
    chall_fpr, chall_tpr, _ = roc_curve(y_test, challenger_pd)
    champ_auc = roc_auc_score(y_test, champion_pd)
    chall_auc = roc_auc_score(y_test, challenger_pd)

    plt.figure(figsize=(7.2, 5.8))
    plt.plot(champ_fpr, champ_tpr, label=f"Scorecard AUC={champ_auc:.3f}")
    plt.plot(chall_fpr, chall_tpr, label=f"XGBoost AUC={chall_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Champion vs challenger ROC")
    plt.legend(loc="lower right")
    plt.tight_layout(pad=1.4)
    plt.savefig(
        FIGURES_DIR / "champion_challenger_roc.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def write_challenger_report(comparison: pd.DataFrame) -> None:
    """Write challenger comparison summary."""
    champion = comparison[comparison["model"] == "logistic_scorecard"].iloc[0]
    challenger = comparison[comparison["model"] == "xgboost_challenger"].iloc[0]
    auc_delta = challenger["auc"] - champion["auc"]
    ks_delta = challenger["ks"] - champion["ks"]

    report = f"""# Phase 8 Challenger Model

## Champion vs Challenger

| Model | AUC | Gini | KS |
|-------|-----|------|----|
| Logistic scorecard | {champion["auc"]:.4f} | {champion["gini"]:.4f} | {champion["ks"]:.4f} |
| XGBoost challenger | {challenger["auc"]:.4f} | {challenger["gini"]:.4f} | {challenger["ks"]:.4f} |

## Interpretation

The challenger AUC delta is {auc_delta:+.4f}; the KS delta is {ks_delta:+.4f}.

The XGBoost challenger is useful as a performance benchmark, but the logistic
scorecard remains the champion because it is transparent, directly convertible
to points, easier to validate, and easier to explain in a regulated credit-risk
setting.

## Outputs

- `reports/challenger_comparison.csv`
- `reports/figures/champion_challenger_roc.png`
- `models/xgboost_challenger.joblib`
"""
    (REPORTS_DIR / "challenger_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 8 challenger benchmark."""
    if not TRAIN_CLEAN_FILE.exists() or not TEST_CLEAN_FILE.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` before challenger training.")
    if not TEST_SCORES_FILE.exists():
        raise FileNotFoundError("Run `python -m src.scorecard` before challenger comparison.")

    train = pd.read_csv(TRAIN_CLEAN_FILE)
    test = pd.read_csv(TEST_CLEAN_FILE)
    test_scores = pd.read_csv(TEST_SCORES_FILE)

    challenger_model = train_xgboost_challenger(train)
    champ = champion_metrics(test_scores)
    chall = challenger_metrics(challenger_model, test)

    comparison = pd.DataFrame(
        [
            {"model": "logistic_scorecard", **champ},
            {"model": "xgboost_challenger", **chall},
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHALLENGER_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "challenger_comparison.csv", index=False)
    joblib.dump(challenger_model, CHALLENGER_MODEL_FILE)
    save_roc_comparison(test, test_scores, challenger_model)
    write_challenger_report(comparison)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
