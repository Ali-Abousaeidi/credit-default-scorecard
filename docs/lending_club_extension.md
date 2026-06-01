# Lending Club Scale-Up Extension

The main repository uses Give Me Some Credit because it is compact, public, and
ideal for a clean scorecard demonstration. A useful future extension is to run
the same scorecard workflow on Lending Club accepted loans.

## Why It Is Separate

Lending Club data is much larger and messier. It also contains leakage traps:
post-origination fields such as recoveries, collection recoveries, last payment
amount, settlement status, and hardship outcomes should not be available at the
application scoring point.

## Target Construction

Use `src/lending_club_adapter.py` to construct a binary target from
`loan_status`:

- bad: `Charged Off`, `Default`
- good: `Fully Paid`
- dropped: loans still in progress, such as `Current`

## Recommended Workflow

1. Place the raw accepted-loans CSV in `data/raw/`.
2. Use `load_and_prepare_lending_club(path)` to create a modelling file.
3. Drop clear leakage fields before EDA.
4. Prefer a time-based split using issue date:
   - train: older vintages
   - validation/test: newer vintages
   - optional out-of-time sample for PSI
5. Reuse the same project phases: EDA, cleaning, WoE/IV, logistic scorecard,
   validation, SHAP, challenger model.

## Status

This is a scaffold, not yet a full second production pipeline. It is included
so the project has a clear scale-up path without bloating the GitHub repo with
large raw data files.
