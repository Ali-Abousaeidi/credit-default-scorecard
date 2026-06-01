"""Run the full project pipeline in order."""

from __future__ import annotations

from collections.abc import Callable

from src import (
    binning,
    calibration,
    challenger,
    data_prep,
    diagnostics,
    eda,
    explainability,
    fetch_data,
    model,
    scorecard,
    validation,
)

PIPELINE_STEPS: list[tuple[str, Callable[[], None]]] = [
    ("Fetch data", fetch_data.main),
    ("EDA", eda.main),
    ("Clean and split", data_prep.main),
    ("WoE binning", binning.main),
    ("Logistic model", model.main),
    ("Scorecard", scorecard.main),
    ("Validation", validation.main),
    ("Calibration", calibration.main),
    ("Diagnostics", diagnostics.main),
    ("Explainability", explainability.main),
    ("Challenger", challenger.main),
]


def main() -> None:
    """Run every pipeline step."""
    for step_name, step in PIPELINE_STEPS:
        print(f"\n=== {step_name} ===")
        step()


if __name__ == "__main__":
    main()
