"""Optional Lending Club scale-up adapter.

This module is intentionally separate from the Give Me Some Credit pipeline.
It documents and implements the target-construction step needed before the
same scorecard workflow can be reused on a larger, messier loan dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LOAN_STATUS_COLUMN = "loan_status"
TARGET_COLUMN = "default_target"
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}


def construct_lending_club_target(data: pd.DataFrame) -> pd.DataFrame:
    """Construct a binary good/bad target from Lending Club loan status."""
    if LOAN_STATUS_COLUMN not in data.columns:
        raise ValueError(f"Expected `{LOAN_STATUS_COLUMN}` column in Lending Club data.")

    eligible = data[data[LOAN_STATUS_COLUMN].isin(BAD_STATUSES | GOOD_STATUSES)].copy()
    eligible[TARGET_COLUMN] = eligible[LOAN_STATUS_COLUMN].isin(BAD_STATUSES).astype(int)
    return eligible


def load_and_prepare_lending_club(path: Path | str) -> pd.DataFrame:
    """Load a Lending Club CSV and construct the default target."""
    data = pd.read_csv(path, low_memory=False)
    return construct_lending_club_target(data)


def main() -> None:
    """Show usage for the optional Lending Club extension."""
    print(
        "Place a Lending Club accepted loans CSV in data/raw/ and call "
        "`load_and_prepare_lending_club(path)` before adapting the Phase 2+ pipeline."
    )


if __name__ == "__main__":
    main()
