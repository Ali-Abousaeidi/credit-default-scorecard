"""Binning diagnostic plots for scorecard characteristics."""

from __future__ import annotations

import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, LOGISTIC_MODEL_FILE, REPORTS_DIR
from src.scoring import DISPLAY_LABELS


def safe_filename(value: str) -> str:
    """Create a filesystem-safe lowercase filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_selected_characteristics() -> list[str]:
    """Load selected raw characteristics from the model artifact."""
    artifact = joblib.load(LOGISTIC_MODEL_FILE)
    return [feature[: -len("_woe")] for feature in artifact["selected_features"]]


def save_bin_diagnostic_plot(table: pd.DataFrame, characteristic: str) -> None:
    """Save count-share, bad-rate, and WoE diagnostics for one variable."""
    data = table[
        (table["variable"] == characteristic)
        & table["Bin"].notna()
        & ~table["Bin"].isin(["Special"])
        & table["WoE"].notna()
    ].copy()
    if data.empty:
        return

    data["attribute"] = data["Bin"].astype(str)
    x = range(len(data))
    label = DISPLAY_LABELS.get(characteristic, characteristic)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].bar(x, data["Count (%)"], color="#4C78A8")
    axes[0].set_ylabel("Population share")
    axes[0].set_title(f"Binning diagnostic: {label}")

    axes[1].plot(x, data["Event rate"], marker="o", color="#E45756")
    axes[1].set_ylabel("Bad rate")

    axes[2].bar(x, data["WoE"], color="#54A24B")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("WoE")
    axes[2].set_xlabel("Bin")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(data["attribute"], rotation=35, ha="right")

    fig.tight_layout(pad=1.4)
    fig.savefig(
        FIGURES_DIR / f"bin_diagnostic_{safe_filename(characteristic)}.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)


def write_diagnostic_report(selected_characteristics: list[str]) -> None:
    """Write bin diagnostic output index."""
    lines = [
        f"- `reports/figures/bin_diagnostic_{safe_filename(characteristic)}.png`"
        for characteristic in selected_characteristics
    ]
    report = f"""# Binning Diagnostics

Generated bin-level diagnostic plots for the final scorecard characteristics.
Each plot shows training-sample population share, observed bad rate, and WoE by
bin so monotonicity and business sense can be reviewed visually.

## Outputs

{chr(10).join(lines)}
"""
    (REPORTS_DIR / "bin_diagnostics_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Create bin-level diagnostic plots for final scorecard variables."""
    if not (REPORTS_DIR / "binning_tables.csv").exists():
        raise FileNotFoundError("Run `python -m src.binning` before diagnostics.")
    if not LOGISTIC_MODEL_FILE.exists():
        raise FileNotFoundError("Run `python -m src.model` before diagnostics.")

    table = pd.read_csv(REPORTS_DIR / "binning_tables.csv")
    selected_characteristics = load_selected_characteristics()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for characteristic in selected_characteristics:
        save_bin_diagnostic_plot(table, characteristic)
    write_diagnostic_report(selected_characteristics)
    print(f"Wrote {len(selected_characteristics)} bin diagnostic plots")


if __name__ == "__main__":
    main()
