"""Streamlit demo for the credit scorecard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import SCORECARD_FILE
from src.scoring import (
    DISPLAY_LABELS,
    FEATURE_COLUMNS,
    load_scoring_artifacts,
    reason_codes_for_applicant,
    score_applicants,
)

DEFAULT_APPLICANT = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 5500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def number_input_for_feature(feature: str) -> float:
    """Render a Streamlit numeric input for a scorecard feature."""
    label = DISPLAY_LABELS.get(feature, feature)
    default = DEFAULT_APPLICANT[feature]
    if feature in {
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    }:
        return float(st.number_input(label, min_value=0, value=int(default), step=1))

    if feature == "MonthlyIncome":
        return float(st.number_input(label, min_value=0.0, value=float(default), step=250.0))

    return float(st.number_input(label, min_value=0.0, value=float(default), step=0.01))


def main() -> None:
    """Run the Streamlit scorecard demo."""
    st.set_page_config(page_title="Credit Scorecard Demo", layout="wide")
    st.title("Credit Default Scorecard")

    try:
        artifacts = load_scoring_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Applicant Inputs")
        applicant = {feature: number_input_for_feature(feature) for feature in FEATURE_COLUMNS}

    scored = score_applicants(applicant, artifacts)
    reason_codes = reason_codes_for_applicant(applicant, artifacts)

    with right:
        st.subheader("Score Result")
        score = scored.loc[0, "score"]
        pd_value = scored.loc[0, "pd"]
        st.metric("Credit score", f"{score:,.0f}")
        st.metric("Predicted default probability", f"{pd_value:.2%}")

        if score >= 620:
            band = "Low risk"
        elif score >= 560:
            band = "Medium risk"
        else:
            band = "High risk"
        st.metric("Risk band", band)

    st.subheader("Top Reason Codes")
    display_reasons = reason_codes.copy()
    display_reasons["log_odds_contribution"] = display_reasons["log_odds_contribution"].round(4)
    display_reasons["woe"] = display_reasons["woe"].round(4)
    st.dataframe(
        display_reasons[
            ["characteristic", "direction", "log_odds_contribution", "woe"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if SCORECARD_FILE.exists():
        st.subheader("Scorecard Points")
        scorecard = pd.read_csv(SCORECARD_FILE)
        st.dataframe(
            scorecard[
                [
                    "characteristic",
                    "attribute",
                    "event_rate",
                    "woe",
                    "points_rounded",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
