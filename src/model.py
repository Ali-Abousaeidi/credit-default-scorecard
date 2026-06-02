"""Logistic scorecard model training pipeline step."""

from __future__ import annotations

import joblib
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import (
    LOGISTIC_MODEL_FILE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_WOE_FILE,
    TRAIN_WOE_FILE,
)

MAX_PVALUE = 0.05
MAX_VIF = 5.0


def load_woe_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 3 WoE-transformed train/test files."""
    if not TRAIN_WOE_FILE.exists() or not TEST_WOE_FILE.exists():
        raise FileNotFoundError("Run `python -m src.binning` before model training.")
    return pd.read_csv(TRAIN_WOE_FILE), pd.read_csv(TEST_WOE_FILE)


def fit_logit(train: pd.DataFrame, features: list[str]) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    """Fit a statsmodels logistic regression."""
    y = train[TARGET_COLUMN]
    x = sm.add_constant(train[features], has_constant="add")
    return sm.Logit(y, x).fit(disp=False, maxiter=200)


def select_features(
    train: pd.DataFrame,
    candidate_features: list[str],
) -> tuple[list[str], sm.discrete.discrete_model.BinaryResultsWrapper, pd.DataFrame]:
    """Iteratively remove non-significant or wrong-signed WoE predictors."""
    selected = candidate_features.copy()
    decisions = []

    while True:
        model = fit_logit(train, selected)
        params = model.params.drop("const")
        pvalues = model.pvalues.drop("const")

        non_significant = pvalues[pvalues > MAX_PVALUE]
        if not non_significant.empty:
            feature_to_remove = non_significant.sort_values(ascending=False).index[0]
            decisions.append(
                {
                    "feature": feature_to_remove,
                    "reason": f"p-value above {MAX_PVALUE}",
                    "coefficient": float(params[feature_to_remove]),
                    "pvalue": float(pvalues[feature_to_remove]),
                }
            )
            selected.remove(feature_to_remove)
            continue

        wrong_signed = params[params >= 0]
        if not wrong_signed.empty:
            feature_to_remove = wrong_signed.sort_values(ascending=False).index[0]
            decisions.append(
                {
                    "feature": feature_to_remove,
                    "reason": "wrong coefficient sign for WoE orientation",
                    "coefficient": float(params[feature_to_remove]),
                    "pvalue": float(pvalues[feature_to_remove]),
                }
            )
            selected.remove(feature_to_remove)
            continue

        return selected, model, pd.DataFrame(decisions)


def calculate_vif(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate VIF for selected model features."""
    x = data[features]
    rows = []
    for i, feature in enumerate(features):
        vif_value = float(variance_inflation_factor(x.values, i))
        rows.append(
            {
                "feature": feature,
                "vif": vif_value,
                "passes_vif_check": bool(vif_value <= MAX_VIF),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def coefficient_table(model: sm.discrete.discrete_model.BinaryResultsWrapper) -> pd.DataFrame:
    """Return coefficient table with sign checks."""
    params = model.params
    table = pd.DataFrame(
        {
            "feature": params.index,
            "coefficient": params.values,
            "std_error": model.bse.values,
            "z_value": model.tvalues.values,
            "pvalue": model.pvalues.values,
        }
    )
    table["expected_sign"] = table["feature"].map(
        lambda feature: "negative" if feature != "const" else "n/a"
    )
    table["passes_sign_check"] = table.apply(
        lambda row: True if row["feature"] == "const" else row["coefficient"] < 0,
        axis=1,
    )
    table["passes_significance_check"] = table.apply(
        lambda row: True if row["feature"] == "const" else row["pvalue"] <= MAX_PVALUE,
        axis=1,
    )
    return table


def predict_pd(
    model: sm.discrete.discrete_model.BinaryResultsWrapper,
    data: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    """Predict probability of default."""
    x = sm.add_constant(data[features], has_constant="add")
    return pd.Series(model.predict(x), index=data.index, name="pd")


def write_model_report(
    selected_features: list[str],
    model: sm.discrete.discrete_model.BinaryResultsWrapper,
    decisions: pd.DataFrame,
    vif: pd.DataFrame,
    train_auc: float,
    test_auc: float,
) -> None:
    """Write a concise Phase 4 model report."""
    if decisions.empty:
        decision_lines = ["- No candidate features were removed."]
    else:
        decision_lines = [
            f"- Removed `{row.feature}`: {row.reason} "
            f"(coef={row.coefficient:.4f}, p={row.pvalue:.3g})."
            for row in decisions.itertuples(index=False)
        ]

    feature_lines = [f"- `{feature}`" for feature in selected_features]
    report = f"""# Phase 4 Logistic Regression Model

## Model Choice

Champion model: logistic regression on WoE-transformed predictors.

Expected sign convention: because higher WoE means safer / lower default risk,
selected coefficients should be negative when modelling `SeriousDlqin2yrs = 1`.

## Feature Selection

Started from all IV-screened WoE predictors and removed features that were not
statistically significant at p <= {MAX_PVALUE} or had the wrong sign.

{chr(10).join(decision_lines)}

## Final Features

{chr(10).join(feature_lines)}

## Fit Summary

- Final feature count: {len(selected_features)}
- Pseudo R-squared: {model.prsquared:.4f}
- AIC: {model.aic:,.2f}
- Max VIF: {vif["vif"].max():.2f}
- Train AUC sanity check: {train_auc:.4f}
- Test AUC early holdout check: {test_auc:.4f}

Full validation is handled in Phase 6; the test AUC here is only a quick
smoke check that the serialized model scores the holdout sample.

## Outputs

- `reports/model_coefficients.csv`
- `reports/feature_selection_decisions.csv`
- `reports/vif_report.csv`
- `models/logistic_scorecard_model.joblib`
"""
    (REPORTS_DIR / "model_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 4 logistic model training."""
    train, test = load_woe_data()
    candidate_features = [column for column in train.columns if column != TARGET_COLUMN]
    selected_features, model, decisions = select_features(train, candidate_features)

    coefficients = coefficient_table(model)
    vif = calculate_vif(train, selected_features)
    train_pd = predict_pd(model, train, selected_features)
    test_pd = predict_pd(model, test, selected_features)
    train_auc = roc_auc_score(train[TARGET_COLUMN], train_pd)
    test_auc = roc_auc_score(test[TARGET_COLUMN], test_pd)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGISTIC_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    coefficients.to_csv(REPORTS_DIR / "model_coefficients.csv", index=False)
    decisions.to_csv(REPORTS_DIR / "feature_selection_decisions.csv", index=False)
    vif.to_csv(REPORTS_DIR / "vif_report.csv", index=False)
    joblib.dump(
        {
            "model": model,
            "selected_features": selected_features,
            "target": TARGET_COLUMN,
            "train_auc": train_auc,
            "test_auc": test_auc,
        },
        LOGISTIC_MODEL_FILE,
    )
    write_model_report(selected_features, model, decisions, vif, train_auc, test_auc)

    print(f"Selected {len(selected_features)} features")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Wrote model artifact: {LOGISTIC_MODEL_FILE}")


if __name__ == "__main__":
    main()
