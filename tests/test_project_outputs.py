"""Regression tests for the scorecard project outputs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.binning import iv_band
from src.config import (
    BASE_ODDS,
    BASE_SCORE,
    BINNING_ARTIFACT_FILE,
    CLEANING_METADATA_FILE,
    LOGISTIC_MODEL_FILE,
    PDO,
    REPORTS_DIR,
    STATIC_SCORECARD_APP_FILE,
    TARGET_COLUMN,
    TEST_CLEAN_FILE,
    TEST_SCORES_FILE,
    TEST_WOE_FILE,
    TRAIN_CLEAN_FILE,
    TRAIN_WOE_FILE,
)
from src.lending_club_adapter import construct_lending_club_target
from src.scorecard import scorecard_factor, scorecard_offset
from src.scoring import score_applicants


def require_file(path):
    """Skip a test if a generated pipeline artifact is missing."""
    if not path.exists():
        pytest.skip(f"Generated artifact missing: {path}. Run `python -m src.run_pipeline`.")


def test_train_test_split_preserves_target_rate() -> None:
    """The stratified split should keep train/test bad rates close."""
    require_file(TRAIN_CLEAN_FILE)
    require_file(TEST_CLEAN_FILE)
    train = pd.read_csv(TRAIN_CLEAN_FILE)
    test = pd.read_csv(TEST_CLEAN_FILE)

    assert abs(train[TARGET_COLUMN].mean() - test[TARGET_COLUMN].mean()) < 0.002


def test_woe_outputs_have_no_missing_values() -> None:
    """WoE outputs should not contain missing values after empirical missing-bin handling."""
    require_file(TRAIN_WOE_FILE)
    require_file(TEST_WOE_FILE)
    train = pd.read_csv(TRAIN_WOE_FILE)
    test = pd.read_csv(TEST_WOE_FILE)

    assert train.isna().sum().sum() == 0
    assert test.isna().sum().sum() == 0


def test_score_increases_when_risk_decreases() -> None:
    """A safer applicant profile should receive a higher score and lower PD."""
    require_file(LOGISTIC_MODEL_FILE)
    require_file(BINNING_ARTIFACT_FILE)
    require_file(CLEANING_METADATA_FILE)

    low_risk = {
        "RevolvingUtilizationOfUnsecuredLines": 0.02,
        "age": 62,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.20,
        "MonthlyIncome": 9000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0,
    }
    high_risk = {
        "RevolvingUtilizationOfUnsecuredLines": 1.20,
        "age": 28,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 1.20,
        "MonthlyIncome": 1200,
        "NumberOfOpenCreditLinesAndLoans": 2,
        "NumberOfTimes90DaysLate": 1,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 3,
    }

    scored = score_applicants(pd.DataFrame([low_risk, high_risk]))
    assert scored.loc[0, "score"] > scored.loc[1, "score"]
    assert scored.loc[0, "pd"] < scored.loc[1, "pd"]


def test_validation_metrics_are_in_expected_ranges() -> None:
    """Validation metrics should stay in plausible scorecard ranges."""
    metrics_file = REPORTS_DIR / "validation_metrics.csv"
    require_file(metrics_file)
    metrics = pd.read_csv(metrics_file).iloc[0]

    assert 0.80 <= metrics["auc"] <= 0.90
    assert 0.60 <= metrics["gini"] <= 0.80
    assert 0.45 <= metrics["ks"] <= 0.65


def test_static_scorecard_app_contains_embedded_artifacts() -> None:
    """The standalone HTML app should include the generated scorecard payload."""
    require_file(STATIC_SCORECARD_APP_FILE)
    app_html = STATIC_SCORECARD_APP_FILE.read_text(encoding="utf-8")

    assert "Credit Default Scorecard" in app_html
    assert "scorecard-data" in app_html
    assert "RevolvingUtilizationOfUnsecuredLines" in app_html


def test_scorecard_factor_matches_pdo_definition() -> None:
    """Factor must equal PDO / ln(2) by the points-to-double-odds definition."""
    assert scorecard_factor() == pytest.approx(PDO / math.log(2))


def test_scorecard_offset_anchors_base_score_at_base_odds() -> None:
    """At the base good:bad odds, the score must equal the configured base score."""
    logit_bad_at_base = math.log(1.0 / BASE_ODDS)  # log-odds of a BAD account at 50:1 good:bad
    score_at_base = scorecard_offset() - scorecard_factor() * logit_bad_at_base
    assert score_at_base == pytest.approx(BASE_SCORE)


def test_pdo_shifts_score_when_odds_double() -> None:
    """Doubling good:bad odds must move the score by exactly PDO points."""
    assert scorecard_factor() * math.log(2) == pytest.approx(PDO)


def test_iv_band_boundaries() -> None:
    """IV banding must follow the standard credit-risk cutoffs."""
    assert iv_band(0.019) == "not useful"
    assert iv_band(0.05) == "weak"
    assert iv_band(0.20) == "medium"
    assert iv_band(0.40) == "strong"
    assert iv_band(0.60).startswith("suspiciously strong")


def test_score_is_offset_minus_factor_times_logit() -> None:
    """The score column must be the exact affine transform of logit(PD)."""
    require_file(TEST_SCORES_FILE)
    scores = pd.read_csv(TEST_SCORES_FILE)
    logit = np.log(scores["pd"] / (1.0 - scores["pd"]))
    reconstructed = scorecard_offset() - scorecard_factor() * logit
    assert np.allclose(reconstructed, scores["score"], atol=1e-6)


def test_lending_club_target_adapter() -> None:
    """The Lending Club adapter should keep only final good/bad statuses."""
    data = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current", "Default"],
            "amount": [1, 2, 3, 4],
        }
    )
    prepared = construct_lending_club_target(data)

    assert prepared["default_target"].tolist() == [0, 1, 1]
    assert "Current" not in prepared["loan_status"].tolist()
