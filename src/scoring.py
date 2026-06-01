"""Reusable applicant scoring helpers for the Streamlit app and tests."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BINNING_ARTIFACT_FILE,
    CLEANING_METADATA_FILE,
    LOGISTIC_MODEL_FILE,
    TARGET_COLUMN,
)
from src.scorecard import score_data

FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

LATE_PAYMENT_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

DISPLAY_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 DPD count",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines/loans",
    "NumberOfTimes90DaysLate": "90+ DPD count",
    "NumberRealEstateLoansOrLines": "Real estate loans/lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 DPD count",
    "NumberOfDependents": "Dependents",
}


@dataclass(frozen=True)
class ScoringArtifacts:
    """Loaded scoring artifacts."""

    model_artifact: dict
    binning_artifact: dict
    cleaning_metadata: dict


def load_scoring_artifacts() -> ScoringArtifacts:
    """Load fitted artifacts required for applicant scoring."""
    missing = [
        str(path)
        for path in [LOGISTIC_MODEL_FILE, BINNING_ARTIFACT_FILE, CLEANING_METADATA_FILE]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing fitted artifacts. Run `python -m src.run_pipeline` first. "
            f"Missing: {missing}"
        )

    return ScoringArtifacts(
        model_artifact=joblib.load(LOGISTIC_MODEL_FILE),
        binning_artifact=joblib.load(BINNING_ARTIFACT_FILE),
        cleaning_metadata=json.loads(CLEANING_METADATA_FILE.read_text(encoding="utf-8")),
    )


def normalize_applicant_input(applicant: dict | pd.DataFrame) -> pd.DataFrame:
    """Convert applicant input into a raw feature dataframe."""
    if isinstance(applicant, pd.DataFrame):
        data = applicant.copy()
    else:
        data = pd.DataFrame([applicant])

    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Applicant input missing required columns: {missing}")

    data = data[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def apply_scoring_cleaning(data: pd.DataFrame, cleaning_metadata: dict) -> pd.DataFrame:
    """Apply the same deterministic cleaning and train-fitted caps used by the model."""
    cleaned = data.copy()
    cleaned.loc[cleaned["age"] <= 0, "age"] = np.nan

    for column in LATE_PAYMENT_COLUMNS:
        cleaned.loc[cleaned[column].isin([96, 98]), column] = np.nan

    for column, cap in cleaning_metadata["fitted_upper_caps"].items():
        cleaned[column] = cleaned[column].clip(upper=float(cap))

    return cleaned


def transform_raw_to_woe(data: pd.DataFrame, binners: dict) -> pd.DataFrame:
    """Transform raw applicant features to WoE values."""
    transformed = pd.DataFrame({TARGET_COLUMN: np.zeros(len(data), dtype=int)})
    for feature, optb in binners.items():
        transformed[f"{feature}_woe"] = optb.transform(
            data[feature].values,
            metric="woe",
            metric_missing="empirical",
            metric_special="empirical",
        )
    return transformed


def score_applicants(applicant: dict | pd.DataFrame, artifacts: ScoringArtifacts | None = None) -> pd.DataFrame:
    """Score one or more applicants and return PD, score, and raw inputs."""
    artifacts = artifacts or load_scoring_artifacts()
    raw = normalize_applicant_input(applicant)
    cleaned = apply_scoring_cleaning(raw, artifacts.cleaning_metadata)
    woe = transform_raw_to_woe(cleaned, artifacts.binning_artifact["binners"])
    scores = score_data(artifacts.model_artifact, woe)
    return pd.concat([cleaned.reset_index(drop=True), scores[["pd", "score"]]], axis=1)


def reason_codes_for_applicant(
    applicant: dict | pd.DataFrame,
    artifacts: ScoringArtifacts | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return model-contribution reason codes for a single applicant."""
    artifacts = artifacts or load_scoring_artifacts()
    raw = normalize_applicant_input(applicant).head(1)
    cleaned = apply_scoring_cleaning(raw, artifacts.cleaning_metadata)
    woe = transform_raw_to_woe(cleaned, artifacts.binning_artifact["binners"])
    model = artifacts.model_artifact["model"]
    selected_features = artifacts.model_artifact["selected_features"]

    rows = []
    for feature in selected_features:
        raw_feature = feature[: -len("_woe")]
        contribution = float(model.params[feature]) * float(woe.loc[0, feature])
        rows.append(
            {
                "characteristic": DISPLAY_LABELS.get(raw_feature, raw_feature),
                "woe": float(woe.loc[0, feature]),
                "coefficient": float(model.params[feature]),
                "log_odds_contribution": contribution,
                "direction": "raises PD" if contribution > 0 else "lowers PD",
            }
        )

    return (
        pd.DataFrame(rows)
        .assign(abs_contribution=lambda frame: frame["log_odds_contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .drop(columns=["abs_contribution"])
        .head(top_n)
        .reset_index(drop=True)
    )
