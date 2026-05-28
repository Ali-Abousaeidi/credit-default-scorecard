"""Scorecard scaling and points table generation."""

from __future__ import annotations

import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    BASE_ODDS,
    BASE_SCORE,
    FIGURES_DIR,
    LOGISTIC_MODEL_FILE,
    PDO,
    REPORTS_DIR,
    SCORECARD_FILE,
    TARGET_COLUMN,
    TEST_SCORES_FILE,
    TEST_WOE_FILE,
    TRAIN_SCORES_FILE,
    TRAIN_WOE_FILE,
)


def scorecard_factor() -> float:
    """Return points-to-double-odds factor."""
    return PDO / math.log(2)


def scorecard_offset() -> float:
    """Return scorecard offset for the configured base score and odds."""
    return BASE_SCORE - scorecard_factor() * math.log(BASE_ODDS)


def original_feature_name(woe_feature: str) -> str:
    """Strip the `_woe` suffix from a transformed feature name."""
    if not woe_feature.endswith("_woe"):
        raise ValueError(f"Expected WoE feature name ending in `_woe`: {woe_feature}")
    return woe_feature[: -len("_woe")]


def load_model_artifact() -> dict:
    """Load the trained logistic model artifact."""
    if not LOGISTIC_MODEL_FILE.exists():
        raise FileNotFoundError("Run `python -m src.model` before scorecard generation.")
    return joblib.load(LOGISTIC_MODEL_FILE)


def build_scorecard_table(model_artifact: dict) -> pd.DataFrame:
    """Convert model coefficients and bin WoE values into scorecard points."""
    binning_tables = pd.read_csv(REPORTS_DIR / "binning_tables.csv")
    model = model_artifact["model"]
    selected_features = model_artifact["selected_features"]

    factor = scorecard_factor()
    offset = scorecard_offset()
    intercept = float(model.params["const"])
    n_features = len(selected_features)

    rows = []
    for woe_feature in selected_features:
        feature = original_feature_name(woe_feature)
        beta = float(model.params[woe_feature])
        table = binning_tables[binning_tables["variable"] == feature].copy()
        table = table[table["WoE"].notna() & table["Bin"].notna()]
        table = table[(table["Count"] > 0) | (~table["Bin"].isin(["Special", "Missing"]))]

        for bin_order, row in table.reset_index(drop=True).iterrows():
            woe = float(row["WoE"])
            attribute_logit = beta * woe + intercept / n_features
            points = offset / n_features - factor * attribute_logit
            rows.append(
                {
                    "characteristic": feature,
                    "woe_feature": woe_feature,
                    "attribute": row["Bin"],
                    "bin_order": int(bin_order),
                    "count": int(row["Count"]),
                    "count_pct": float(row["Count (%)"]),
                    "event_rate": float(row["Event rate"]),
                    "woe": woe,
                    "coefficient": beta,
                    "points": float(points),
                    "points_rounded": int(round(points)),
                }
            )

    return pd.DataFrame(rows)


def score_data(model_artifact: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Score a WoE-transformed dataset."""
    model = model_artifact["model"]
    selected_features = model_artifact["selected_features"]
    params = model.params

    linear_predictor = np.full(len(data), float(params["const"]))
    for feature in selected_features:
        linear_predictor += data[feature].to_numpy() * float(params[feature])

    pd_values = 1 / (1 + np.exp(-linear_predictor))
    scores = scorecard_offset() - scorecard_factor() * linear_predictor

    return pd.DataFrame(
        {
            TARGET_COLUMN: data[TARGET_COLUMN].astype(int),
            "pd": pd_values,
            "score": scores,
        }
    )


def save_score_distribution_plot(train_scores: pd.DataFrame, test_scores: pd.DataFrame) -> None:
    """Save score distribution plot for train and test."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = pd.concat(
        [
            train_scores.assign(sample="train"),
            test_scores.assign(sample="test"),
        ],
        ignore_index=True,
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5.8))
    sns.histplot(
        data=plot_data,
        x="score",
        hue="sample",
        bins=50,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("Score distribution")
    plt.xlabel("Score, higher is safer")
    plt.ylabel("Density")
    plt.tight_layout(pad=1.4)
    plt.savefig(FIGURES_DIR / "score_distribution.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def write_scorecard_report(
    scorecard: pd.DataFrame,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
) -> None:
    """Write Phase 5 scorecard summary."""
    factor = scorecard_factor()
    offset = scorecard_offset()
    best_test = test_scores.loc[test_scores["score"].idxmax()]
    worst_test = test_scores.loc[test_scores["score"].idxmin()]

    report = f"""# Phase 5 Scorecard

## Scaling

- Base score: {BASE_SCORE}
- Base odds good:bad: {BASE_ODDS:.0f}:1
- PDO: {PDO}
- Factor: {factor:.6f}
- Offset: {offset:.6f}

Score formula:

```text
score = offset - factor * logit(PD)
```

Higher scores indicate lower estimated default risk.

## Score Distribution

- Train mean score: {train_scores["score"].mean():.2f}
- Train min/max score: {train_scores["score"].min():.2f} / {train_scores["score"].max():.2f}
- Test mean score: {test_scores["score"].mean():.2f}
- Test min/max score: {test_scores["score"].min():.2f} / {test_scores["score"].max():.2f}
- Best test profile: score {best_test["score"]:.2f}, PD {best_test["pd"]:.2%}
- Worst test profile: score {worst_test["score"]:.2f}, PD {worst_test["pd"]:.2%}

## Outputs

- `reports/scorecard_points.csv` ({len(scorecard):,} attribute rows)
- `data/processed/train_scores.csv`
- `data/processed/test_scores.csv`
- `reports/figures/score_distribution.png`
"""
    (REPORTS_DIR / "scorecard_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 5 scorecard point scaling."""
    model_artifact = load_model_artifact()
    train_woe = pd.read_csv(TRAIN_WOE_FILE)
    test_woe = pd.read_csv(TEST_WOE_FILE)

    scorecard = build_scorecard_table(model_artifact)
    train_scores = score_data(model_artifact, train_woe)
    test_scores = score_data(model_artifact, test_woe)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(SCORECARD_FILE, index=False)
    train_scores.to_csv(TRAIN_SCORES_FILE, index=False)
    test_scores.to_csv(TEST_SCORES_FILE, index=False)
    save_score_distribution_plot(train_scores, test_scores)
    write_scorecard_report(scorecard, train_scores, test_scores)

    print(f"Wrote scorecard: {SCORECARD_FILE} ({len(scorecard):,} rows)")
    print(f"Train score mean: {train_scores['score'].mean():.2f}")
    print(f"Test score mean: {test_scores['score'].mean():.2f}")


if __name__ == "__main__":
    main()
