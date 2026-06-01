"""Load, clean, and split the credit default dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEANING_METADATA_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_GMSC_FILE,
    TARGET_COLUMN,
    TEST_CLEAN_FILE,
    TEST_SIZE,
    TRAIN_CLEAN_FILE,
)

LATE_PAYMENT_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

CAP_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
]

UPPER_CAP_QUANTILE = 0.995


def load_raw_data(path: Path = RAW_GMSC_FILE) -> pd.DataFrame:
    """Load the raw Give Me Some Credit training file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Place Kaggle's cs-training.csv file in data/raw/ before running."
        )

    data = pd.read_csv(path)

    # Kaggle's file often contains an unnamed row-id column.
    unnamed_cols = [col for col in data.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        data = data.drop(columns=unnamed_cols)

    return data


def validate_target(data: pd.DataFrame, target: str = TARGET_COLUMN) -> None:
    """Check that the expected binary target exists."""
    if target not in data.columns:
        raise ValueError(f"Expected target column '{target}' was not found.")

    observed = set(data[target].dropna().unique())
    if not observed.issubset({0, 1}):
        raise ValueError(f"Target '{target}' must be binary 0/1. Found: {observed}")


def apply_basic_cleaning(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic, non-fitted cleaning before the train/test split."""
    cleaned = data.copy()
    metadata: dict[str, object] = {
        "raw_rows": int(len(cleaned)),
        "target_column": TARGET_COLUMN,
        "basic_cleaning": [],
    }

    invalid_age_mask = cleaned["age"] <= 0
    invalid_age_rows = int(invalid_age_mask.sum())
    if invalid_age_rows:
        cleaned = cleaned.loc[~invalid_age_mask].copy()
    metadata["basic_cleaning"].append(
        {
            "rule": "drop_age_le_zero",
            "rows_removed": invalid_age_rows,
            "reason": "Age must be positive; one zero-age record is invalid.",
        }
    )

    sentinel_counts = {}
    for column in LATE_PAYMENT_COLUMNS:
        sentinel_mask = cleaned[column].isin([96, 98])
        sentinel_counts[column] = int(sentinel_mask.sum())
        cleaned.loc[sentinel_mask, column] = np.nan
    metadata["basic_cleaning"].append(
        {
            "rule": "replace_96_98_delinquency_counts_with_missing",
            "columns": LATE_PAYMENT_COLUMNS,
            "counts": sentinel_counts,
            "reason": "96/98 are sentinel-like values, not plausible delinquency counts.",
        }
    )

    metadata["post_basic_cleaning_rows"] = int(len(cleaned))
    return cleaned, metadata


def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an honest stratified train/test split before fitted preprocessing."""
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )
    return train.sort_index().reset_index(drop=True), test.sort_index().reset_index(drop=True)


def fit_upper_caps(train: pd.DataFrame) -> dict[str, float]:
    """Fit outlier caps on the training sample only."""
    caps = {}
    for column in CAP_COLUMNS:
        caps[column] = float(train[column].quantile(UPPER_CAP_QUANTILE))
    return caps


def apply_upper_caps(data: pd.DataFrame, caps: dict[str, float]) -> pd.DataFrame:
    """Apply train-fitted upper caps to a dataframe."""
    capped = data.copy()
    for column, cap in caps.items():
        capped[column] = capped[column].clip(upper=cap)
    return capped


def save_processed_data(train: pd.DataFrame, test: pd.DataFrame, metadata: dict) -> None:
    """Persist processed train/test data and cleaning metadata."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLEANING_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_CLEAN_FILE, index=False)
    test.to_csv(TEST_CLEAN_FILE, index=False)
    CLEANING_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_cleaning_decisions(metadata: dict, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Write a human-readable Phase 2 cleaning decision report."""
    cap_lines = [
        f"- `{column}` upper cap: {cap:,.6g}"
        for column, cap in metadata["fitted_upper_caps"].items()
    ]

    sentinel_rule = metadata["basic_cleaning"][1]
    sentinel_lines = [
        f"- `{column}`: {count:,} values replaced with missing"
        for column, count in sentinel_rule["counts"].items()
    ]

    report = f"""# Phase 2 Cleaning Decisions

## Target Definition

- Target: `{TARGET_COLUMN}`
- Bad event: `{TARGET_COLUMN} = 1`
- Good event: `{TARGET_COLUMN} = 0`

## Deterministic Cleaning Before Split

- Removed {metadata["basic_cleaning"][0]["rows_removed"]:,} row where `age <= 0`.
- Replaced 96/98 delinquency-count sentinel values with missing values:
{chr(10).join(sentinel_lines)}

Missing values are retained for later WoE binning rather than imputed in Phase 2.

## Train/Test Split

- Split type: stratified random split on `{TARGET_COLUMN}`
- Random state: {RANDOM_STATE}
- Test size: {TEST_SIZE:.0%}
- Train rows: {len(train):,}; bad rate: {train[TARGET_COLUMN].mean():.2%}
- Test rows: {len(test):,}; bad rate: {test[TARGET_COLUMN].mean():.2%}

## Train-Fitted Outlier Caps

Caps were fit on the training sample only and then applied to train and test.

{chr(10).join(cap_lines)}

## Saved Outputs

- `data/processed/train_clean.csv`
- `data/processed/test_clean.csv`
- `reports/cleaning_metadata.json`
"""
    (CLEANING_METADATA_FILE.parent / "cleaning_decisions.md").write_text(
        report, encoding="utf-8"
    )


def prepare_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full Phase 2 preparation workflow."""
    validate_target(data)
    cleaned, metadata = apply_basic_cleaning(data)
    train, test = split_train_test(cleaned)

    caps = fit_upper_caps(train)
    train = apply_upper_caps(train, caps)
    test = apply_upper_caps(test, caps)

    metadata.update(
        {
            "split": {
                "type": "stratified_random",
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_bad_rate": float(train[TARGET_COLUMN].mean()),
                "test_bad_rate": float(test[TARGET_COLUMN].mean()),
            },
            "fitted_upper_caps": caps,
            "outputs": {
                "train_clean": str(TRAIN_CLEAN_FILE),
                "test_clean": str(TEST_CLEAN_FILE),
                "cleaning_metadata": str(CLEANING_METADATA_FILE),
            },
        }
    )
    return train, test, metadata


def main() -> None:
    """Run Phase 2 cleaning and splitting."""
    data = load_raw_data()
    train, test, metadata = prepare_train_test(data)
    save_processed_data(train, test, metadata)
    write_cleaning_decisions(metadata, train, test)

    print(f"Wrote train data: {TRAIN_CLEAN_FILE} ({len(train):,} rows)")
    print(f"Wrote test data: {TEST_CLEAN_FILE} ({len(test):,} rows)")
    print(f"Train bad rate: {train[TARGET_COLUMN].mean():.2%}")
    print(f"Test bad rate: {test[TARGET_COLUMN].mean():.2%}")


if __name__ == "__main__":
    main()
