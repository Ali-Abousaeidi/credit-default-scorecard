"""WoE binning and Information Value pipeline step."""

from __future__ import annotations

import joblib
import pandas as pd
from optbinning import OptimalBinning

from src.config import (
    BINNING_ARTIFACT_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_CLEAN_FILE,
    TEST_WOE_FILE,
    TRAIN_CLEAN_FILE,
    TRAIN_WOE_FILE,
)

MONOTONIC_TRENDS = {
    "RevolvingUtilizationOfUnsecuredLines": "ascending",
    "age": "descending",
    "NumberOfTime30-59DaysPastDueNotWorse": "ascending",
    "DebtRatio": "ascending",
    "MonthlyIncome": "descending",
    "NumberOfOpenCreditLinesAndLoans": "auto",
    "NumberOfTimes90DaysLate": "ascending",
    "NumberRealEstateLoansOrLines": "auto",
    "NumberOfTime60-89DaysPastDueNotWorse": "ascending",
    "NumberOfDependents": "auto",
}


def iv_band(iv: float) -> str:
    """Classify Information Value using the standard credit-risk rule of thumb."""
    if iv < 0.02:
        return "not useful"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspiciously strong - review for leakage"


def load_clean_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 2 train/test files."""
    if not TRAIN_CLEAN_FILE.exists() or not TEST_CLEAN_FILE.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` before binning.")
    return pd.read_csv(TRAIN_CLEAN_FILE), pd.read_csv(TEST_CLEAN_FILE)


def fit_binning_for_feature(
    train: pd.DataFrame,
    feature: str,
) -> tuple[OptimalBinning, pd.DataFrame]:
    """Fit one train-only optimal binning model and return its table."""
    y = train[TARGET_COLUMN].values
    trend = MONOTONIC_TRENDS.get(feature, "auto")

    optb = OptimalBinning(
        name=feature,
        dtype="numerical",
        prebinning_method="cart",
        solver="cp",
        max_n_prebins=20,
        min_prebin_size=0.03,
        min_bin_size=0.05,
        monotonic_trend=trend,
    )
    optb.fit(train[feature].values, y)
    table = optb.binning_table.build().copy()
    table.insert(0, "variable", feature)
    table.insert(1, "monotonic_trend", trend)
    table.insert(2, "status", optb.status)
    return optb, table


def fit_binning(train: pd.DataFrame) -> tuple[dict[str, OptimalBinning], pd.DataFrame, pd.DataFrame]:
    """Fit binning models for every predictor on train only."""
    features = [column for column in train.columns if column != TARGET_COLUMN]
    binners = {}
    tables = []
    iv_rows = []

    for feature in features:
        optb, table = fit_binning_for_feature(train, feature)
        binners[feature] = optb
        tables.append(table)

        data_bins = table[~table["Bin"].astype(str).isin(["Special", "Missing", "Totals"])]
        iv = float(optb.binning_table.iv)
        iv_rows.append(
            {
                "variable": feature,
                "iv": iv,
                "iv_band": iv_band(iv),
                "status": optb.status,
                "monotonic_trend": MONOTONIC_TRENDS.get(feature, "auto"),
                "n_data_bins": int(len(data_bins)),
                "selected_for_modeling": bool(iv >= 0.02),
                "review_flag": bool(iv > 0.50),
            }
        )

    binning_tables = pd.concat(tables, ignore_index=True)
    iv_ranking = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)
    return binners, binning_tables, iv_ranking


def transform_to_woe(data: pd.DataFrame, binners: dict[str, OptimalBinning]) -> pd.DataFrame:
    """Transform predictors to WoE values using fitted train-only binners."""
    transformed = pd.DataFrame({TARGET_COLUMN: data[TARGET_COLUMN].astype(int)})
    for feature, optb in binners.items():
        transformed[f"{feature}_woe"] = optb.transform(
            data[feature].values,
            metric="woe",
            metric_missing="empirical",
            metric_special="empirical",
        )
    return transformed


def write_binning_report(iv_ranking: pd.DataFrame) -> None:
    """Write a concise Phase 3 binning summary."""
    high_iv = iv_ranking[iv_ranking["review_flag"]]
    high_iv_lines = [
        f"- `{row.variable}` IV={row.iv:.3f}: {row.iv_band}"
        for row in high_iv.itertuples(index=False)
    ]
    if not high_iv_lines:
        high_iv_lines = ["- No variables above the high-IV review threshold."]

    selected_count = int(iv_ranking["selected_for_modeling"].sum())
    report = f"""# Phase 3 WoE Binning And IV

## Method

- Fitted `optbinning.OptimalBinning` on the training sample only.
- Applied the fitted bins to both train and test.
- Missing values use empirical WoE, preserving missingness as an informative bin.
- Minimum bin size target: 5% of the training sample.

## Outputs

- `reports/binning_tables.csv`
- `reports/iv_ranking.csv`
- `data/processed/train_woe.csv`
- `data/processed/test_woe.csv`
- `models/binning_artifacts.joblib`

## IV Screening

- Candidate predictors with IV >= 0.02: {selected_count}
- High-IV leakage review threshold: IV > 0.50

{chr(10).join(high_iv_lines)}

High IV does not automatically mean leakage here because delinquency-history
and utilization variables are expected to be strong in this dataset. These
variables still need sign, stability, and business-sense checks in modelling.
"""
    (REPORTS_DIR / "binning_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 3 WoE binning and IV ranking."""
    train, test = load_clean_splits()
    binners, binning_tables, iv_ranking = fit_binning(train)

    train_woe = transform_to_woe(train, binners)
    test_woe = transform_to_woe(test, binners)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BINNING_ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)

    binning_tables.to_csv(REPORTS_DIR / "binning_tables.csv", index=False)
    iv_ranking.to_csv(REPORTS_DIR / "iv_ranking.csv", index=False)
    train_woe.to_csv(TRAIN_WOE_FILE, index=False)
    test_woe.to_csv(TEST_WOE_FILE, index=False)
    joblib.dump(
        {
            "binners": binners,
            "iv_ranking": iv_ranking,
            "target": TARGET_COLUMN,
        },
        BINNING_ARTIFACT_FILE,
    )
    write_binning_report(iv_ranking)

    print(f"Wrote IV ranking: {REPORTS_DIR / 'iv_ranking.csv'}")
    print(f"Wrote train WoE data: {TRAIN_WOE_FILE} ({train_woe.shape[0]:,} rows)")
    print(f"Wrote test WoE data: {TEST_WOE_FILE} ({test_woe.shape[0]:,} rows)")


if __name__ == "__main__":
    main()
