"""Phase 1 exploratory data analysis for the raw credit dataset."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, RAW_GMSC_FILE, REPORTS_DIR, TARGET_COLUMN
from src.data_prep import load_raw_data, validate_target

LATE_PAYMENT_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


@dataclass(frozen=True)
class VariableDefinition:
    """Human-readable variable definition for the project data dictionary."""

    column: str
    description: str
    expected_risk_direction: str
    phase_1_note: str


DATA_DICTIONARY = [
    VariableDefinition(
        TARGET_COLUMN,
        "Person experienced serious delinquency within two years.",
        "Target: 1 is bad, 0 is good.",
        "Direct target field for Give Me Some Credit.",
    ),
    VariableDefinition(
        "RevolvingUtilizationOfUnsecuredLines",
        "Credit card and unsecured line utilization.",
        "Higher utilization should generally mean higher risk.",
        "Values above 1 need review as extreme utilization/outlier behavior.",
    ),
    VariableDefinition(
        "age",
        "Borrower age in years.",
        "Younger borrowers may carry higher risk; relationship can be nonlinear.",
        "Age 0 is invalid and must be handled in cleaning.",
    ),
    VariableDefinition(
        "NumberOfTime30-59DaysPastDueNotWorse",
        "Count of 30-59 day delinquencies in the last two years.",
        "Higher count should mean higher risk.",
        "Values 96 and 98 are sentinel-like outliers to treat deliberately.",
    ),
    VariableDefinition(
        "DebtRatio",
        "Monthly debt payments and living costs divided by monthly gross income.",
        "Higher ratio should generally mean higher risk.",
        "Very large values need review, especially when income is missing.",
    ),
    VariableDefinition(
        "MonthlyIncome",
        "Monthly gross income.",
        "Higher income should generally mean lower risk.",
        "Missing values should likely be retained as a WoE bin, not blindly imputed.",
    ),
    VariableDefinition(
        "NumberOfOpenCreditLinesAndLoans",
        "Open installment loans and credit lines.",
        "Ambiguous/nonlinear; too few or too many lines may be risky.",
        "Review binning manually rather than forcing a simple direction.",
    ),
    VariableDefinition(
        "NumberOfTimes90DaysLate",
        "Count of 90+ day delinquencies.",
        "Higher count should mean higher risk.",
        "Values 96 and 98 are sentinel-like outliers to treat deliberately.",
    ),
    VariableDefinition(
        "NumberRealEstateLoansOrLines",
        "Mortgage and real estate loan count, including home equity lines.",
        "Ambiguous/nonlinear; moderate secured credit can be lower risk.",
        "Review binning manually.",
    ),
    VariableDefinition(
        "NumberOfTime60-89DaysPastDueNotWorse",
        "Count of 60-89 day delinquencies in the last two years.",
        "Higher count should mean higher risk.",
        "Values 96 and 98 are sentinel-like outliers to treat deliberately.",
    ),
    VariableDefinition(
        "NumberOfDependents",
        "Number of dependents in family excluding the borrower.",
        "More dependents may mean higher obligations, but effect is often weak.",
        "Missing values should be reviewed and likely kept as a bin.",
    ),
]


def make_data_dictionary() -> pd.DataFrame:
    """Return the project data dictionary as a dataframe."""
    return pd.DataFrame([definition.__dict__ for definition in DATA_DICTIONARY])


def summarize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact column-level quality summary."""
    rows = []
    for column in data.columns:
        series = data[column]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "min": float(numeric.min()) if numeric.notna().any() else None,
                "p01": float(numeric.quantile(0.01)) if numeric.notna().any() else None,
                "median": float(numeric.median()) if numeric.notna().any() else None,
                "mean": float(numeric.mean()) if numeric.notna().any() else None,
                "p99": float(numeric.quantile(0.99)) if numeric.notna().any() else None,
                "max": float(numeric.max()) if numeric.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_target(data: pd.DataFrame) -> pd.DataFrame:
    """Return count and rate by target class."""
    summary = (
        data[TARGET_COLUMN]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
        .sort_values(TARGET_COLUMN)
    )
    summary["rate"] = summary["count"] / len(data)
    return summary


def summarize_sentinels(data: pd.DataFrame) -> pd.DataFrame:
    """Count sentinel-like 96/98 values in delinquency count columns."""
    rows = []
    for column in LATE_PAYMENT_COLUMNS:
        for value in [96, 98]:
            count = int((data[column] == value).sum())
            rows.append(
                {
                    "column": column,
                    "sentinel_value": value,
                    "count": count,
                    "rate": count / len(data),
                }
            )
    return pd.DataFrame(rows)


def save_figures(data: pd.DataFrame, column_summary: pd.DataFrame) -> None:
    """Save simple Phase 1 EDA plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    target_summary = summarize_target(data)
    plt.figure(figsize=(7.5, 4.8))
    sns.barplot(data=target_summary, x=TARGET_COLUMN, y="count", color="#4C78A8")
    plt.title("Target distribution")
    plt.xlabel("Serious delinquency within two years")
    plt.ylabel("Count")
    plt.tight_layout(pad=1.4)
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close()

    missing = column_summary[column_summary["missing_count"] > 0].copy()
    if not missing.empty:
        missing = missing.sort_values("missing_pct", ascending=False)
        missing["display_column"] = missing["column"].replace(
            {
                "MonthlyIncome": "Monthly income",
                "NumberOfDependents": "Dependents",
            }
        )
        plt.figure(figsize=(8.5, 4.8))
        sns.barplot(data=missing, y="display_column", x="missing_pct", color="#F58518")
        plt.title("Missing value rate")
        plt.xlabel("Missing rate")
        plt.ylabel("")
        plt.tight_layout(pad=1.4)
        plt.savefig(FIGURES_DIR / "missingness.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
        plt.close()

    utilization_clip = data["RevolvingUtilizationOfUnsecuredLines"].clip(
        upper=data["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99)
    )
    plt.figure(figsize=(9, 4.8))
    sns.histplot(utilization_clip, bins=50, color="#54A24B")
    plt.title("Revolving utilization distribution, clipped at p99")
    plt.xlabel("Revolving utilization")
    plt.ylabel("Rows")
    plt.tight_layout(pad=1.4)
    plt.savefig(
        FIGURES_DIR / "revolving_utilization_distribution.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def write_markdown_report(
    data: pd.DataFrame,
    column_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    sentinel_summary: pd.DataFrame,
) -> None:
    """Write a concise markdown data-quality findings report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    bad_rate = data[TARGET_COLUMN].mean()
    monthly_income_missing = int(data["MonthlyIncome"].isna().sum())
    monthly_income_missing_rate = data["MonthlyIncome"].isna().mean()
    dependents_missing = int(data["NumberOfDependents"].isna().sum())
    dependents_missing_rate = data["NumberOfDependents"].isna().mean()
    age_zero = int((data["age"] == 0).sum())
    utilization_gt_one = int((data["RevolvingUtilizationOfUnsecuredLines"] > 1).sum())
    utilization_gt_one_rate = (data["RevolvingUtilizationOfUnsecuredLines"] > 1).mean()
    debt_ratio_p99 = data["DebtRatio"].quantile(0.99)
    debt_ratio_max = data["DebtRatio"].max()

    sentinel_lines = []
    for _, row in sentinel_summary.iterrows():
        if row["count"] > 0:
            sentinel_lines.append(
                f"- `{row['column']}` has {row['count']:,} rows "
                f"with value {int(row['sentinel_value'])} ({row['rate']:.2%})."
            )
    if not sentinel_lines:
        sentinel_lines.append("- No 96/98 sentinel-like values found.")

    target_lines = []
    for _, row in target_summary.iterrows():
        target_lines.append(
            f"- `{TARGET_COLUMN}={int(row[TARGET_COLUMN])}`: "
            f"{int(row['count']):,} rows ({row['rate']:.2%})."
        )

    report = f"""# Phase 1 Data Quality Findings

## Dataset Snapshot

- Source file: `data/raw/cs-training.csv`
- Shape: {data.shape[0]:,} rows x {data.shape[1]:,} columns
- Target bad rate: {bad_rate:.2%}
- Target classes:
{chr(10).join(target_lines)}

## Key Findings

- The dataset is strongly imbalanced: only {bad_rate:.2%} of rows are bad accounts.
- `MonthlyIncome` is missing in {monthly_income_missing:,} rows ({monthly_income_missing_rate:.2%}).
- `NumberOfDependents` is missing in {dependents_missing:,} rows ({dependents_missing_rate:.2%}).
- `age` has {age_zero:,} invalid zero-age rows.
- `RevolvingUtilizationOfUnsecuredLines` is greater than 1 in {utilization_gt_one:,} rows ({utilization_gt_one_rate:.2%}), so utilization has extreme values that need deliberate treatment.
- `DebtRatio` has a p99 of {debt_ratio_p99:,.2f} and a max of {debt_ratio_max:,.2f}, indicating a long right tail.

## Sentinel-Like Delinquency Values

{chr(10).join(sentinel_lines)}

## Phase 2 Cleaning Decisions To Implement

- Keep missing values available for WoE binning instead of blindly imputing them.
- Treat or cap the 96/98 delinquency values explicitly before binning.
- Correct or remove the invalid age-zero record.
- Cap or bin extreme utilization and debt-ratio values based on train-only logic.
- Split train/test before fitting binning, feature selection, or any model.

## Generated Outputs

- `reports/data_dictionary.csv`
- `reports/data_quality_summary.csv`
- `reports/sentinel_value_counts.csv`
- `reports/target_summary.csv`
- `reports/figures/target_distribution.png`
- `reports/figures/missingness.png`
- `reports/figures/revolving_utilization_distribution.png`
"""

    (REPORTS_DIR / "data_quality_findings.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run Phase 1 EDA and save report artifacts."""
    data = load_raw_data(RAW_GMSC_FILE)
    validate_target(data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dictionary = make_data_dictionary()
    column_summary = summarize_columns(data)
    target_summary = summarize_target(data)
    sentinel_summary = summarize_sentinels(data)

    data_dictionary.to_csv(REPORTS_DIR / "data_dictionary.csv", index=False)
    column_summary.to_csv(REPORTS_DIR / "data_quality_summary.csv", index=False)
    target_summary.to_csv(REPORTS_DIR / "target_summary.csv", index=False)
    sentinel_summary.to_csv(REPORTS_DIR / "sentinel_value_counts.csv", index=False)

    save_figures(data, column_summary)
    write_markdown_report(data, column_summary, target_summary, sentinel_summary)

    print(f"Wrote Phase 1 EDA outputs to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
