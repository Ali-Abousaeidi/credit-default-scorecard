"""SHAP explainability and reason-code outputs for the scorecard model."""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.config import (
    FIGURES_DIR,
    LOGISTIC_MODEL_FILE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SCORES_FILE,
    TEST_WOE_FILE,
    TRAIN_WOE_FILE,
)


def display_name(woe_feature: str) -> str:
    """Return a readable feature name from a WoE feature."""
    return woe_feature[: -len("_woe")] if woe_feature.endswith("_woe") else woe_feature


def load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load model artifact and WoE datasets."""
    if not LOGISTIC_MODEL_FILE.exists():
        raise FileNotFoundError("Run `python -m src.model` before explainability.")
    return (
        joblib.load(LOGISTIC_MODEL_FILE),
        pd.read_csv(TRAIN_WOE_FILE),
        pd.read_csv(TEST_WOE_FILE),
        pd.read_csv(TEST_SCORES_FILE),
    )


def calculate_shap_values(
    model_artifact: dict,
    train_woe: pd.DataFrame,
    test_woe: pd.DataFrame,
) -> shap.Explanation:
    """Calculate exact linear SHAP values on the log-odds scale."""
    model = model_artifact["model"]
    selected_features = model_artifact["selected_features"]
    coef = model.params[selected_features].to_numpy()
    intercept = float(model.params["const"])

    background = train_woe[selected_features].sample(
        n=min(1000, len(train_woe)),
        random_state=42,
    )
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer((coef, intercept), masker)
    return explainer(test_woe[selected_features])


def make_global_importance(
    shap_values: shap.Explanation,
    selected_features: list[str],
) -> pd.DataFrame:
    """Create global mean absolute SHAP importance table."""
    importance = pd.DataFrame(
        {
            "feature": selected_features,
            "characteristic": [display_name(feature) for feature in selected_features],
            "mean_abs_shap_log_odds": np.abs(shap_values.values).mean(axis=0),
            "mean_shap_log_odds": shap_values.values.mean(axis=0),
        }
    )
    return importance.sort_values("mean_abs_shap_log_odds", ascending=False)


def save_global_importance_plot(importance: pd.DataFrame) -> None:
    """Save a SHAP global importance plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_data = importance.sort_values("mean_abs_shap_log_odds", ascending=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_data,
        x="mean_abs_shap_log_odds",
        y="characteristic",
        color="#4C78A8",
    )
    plt.xlabel("Mean absolute SHAP value, log-odds scale")
    plt.ylabel("")
    plt.title("Global SHAP importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=160)
    plt.close()


def make_reason_code_examples(
    shap_values: shap.Explanation,
    selected_features: list[str],
    test_scores: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Create individual reason-code examples from SHAP contributions."""
    examples = {
        "highest_risk": int(test_scores["pd"].idxmax()),
        "median_risk": int((test_scores["pd"] - test_scores["pd"].median()).abs().idxmin()),
        "lowest_risk": int(test_scores["pd"].idxmin()),
    }

    rows = []
    for example_name, row_index in examples.items():
        values = shap_values.values[row_index]
        order = np.argsort(np.abs(values))[::-1][:top_n]
        for rank, feature_position in enumerate(order, start=1):
            shap_value = float(values[feature_position])
            rows.append(
                {
                    "example": example_name,
                    "row_index": row_index,
                    "score": float(test_scores.loc[row_index, "score"]),
                    "pd": float(test_scores.loc[row_index, "pd"]),
                    "actual_target": int(test_scores.loc[row_index, TARGET_COLUMN]),
                    "rank": rank,
                    "feature": selected_features[feature_position],
                    "characteristic": display_name(selected_features[feature_position]),
                    "shap_value_log_odds": shap_value,
                    "direction": "raises PD" if shap_value > 0 else "lowers PD",
                }
            )
    return pd.DataFrame(rows)


def save_reason_code_plots(reason_codes: pd.DataFrame) -> None:
    """Save bar plots for individual reason-code examples."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for example, data in reason_codes.groupby("example"):
        plot_data = data.sort_values("shap_value_log_odds", ascending=True)
        colors = ["#E45756" if value > 0 else "#4C78A8" for value in plot_data["shap_value_log_odds"]]
        plt.figure(figsize=(8, 4.5))
        plt.barh(plot_data["characteristic"], plot_data["shap_value_log_odds"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("SHAP contribution to log-odds of default")
        plt.ylabel("")
        plt.title(f"Reason-code example: {example.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"reason_codes_{example}.png", dpi=160)
        plt.close()


def write_explainability_report(
    importance: pd.DataFrame,
    reason_codes: pd.DataFrame,
) -> None:
    """Write Phase 7 explainability report."""
    top_features = [
        f"- `{row.characteristic}`: mean |SHAP| {row.mean_abs_shap_log_odds:.4f}"
        for row in importance.head(5).itertuples(index=False)
    ]
    example_lines = []
    for example, data in reason_codes.groupby("example"):
        pd_value = data["pd"].iloc[0]
        score = data["score"].iloc[0]
        leading = data.sort_values("rank").iloc[0]
        example_lines.append(
            f"- `{example}`: score {score:.2f}, PD {pd_value:.2%}; "
            f"largest contribution is `{leading.characteristic}` ({leading.direction})."
        )

    report = f"""# Phase 7 Explainability

## SHAP Method

SHAP values are calculated with `shap.LinearExplainer` using the fitted
logistic scorecard coefficients. Contributions are on the log-odds scale of
default. Positive values raise predicted default risk; negative values lower it.

## Global Importance

{chr(10).join(top_features)}

## Individual Reason-Code Examples

{chr(10).join(example_lines)}

## Outputs

- `reports/shap_global_importance.csv`
- `reports/reason_codes_examples.csv`
- `reports/figures/shap_global_importance.png`
- `reports/figures/reason_codes_highest_risk.png`
- `reports/figures/reason_codes_median_risk.png`
- `reports/figures/reason_codes_lowest_risk.png`
- `docs/regulatory_context.md`
"""
    (REPORTS_DIR / "explainability_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 7 SHAP explainability outputs."""
    model_artifact, train_woe, test_woe, test_scores = load_inputs()
    selected_features = model_artifact["selected_features"]
    shap_values = calculate_shap_values(model_artifact, train_woe, test_woe)
    importance = make_global_importance(shap_values, selected_features)
    reason_codes = make_reason_code_examples(shap_values, selected_features, test_scores)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    reason_codes.to_csv(REPORTS_DIR / "reason_codes_examples.csv", index=False)
    save_global_importance_plot(importance)
    save_reason_code_plots(reason_codes)
    write_explainability_report(importance, reason_codes)

    print(f"Wrote SHAP global importance: {REPORTS_DIR / 'shap_global_importance.csv'}")
    print(f"Wrote reason-code examples: {REPORTS_DIR / 'reason_codes_examples.csv'}")


if __name__ == "__main__":
    main()
