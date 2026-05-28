# Phase 0 Scope

## Project Objective

Build an end-to-end credit default probability model and convert it into a
traditional points-based scorecard suitable for a credit risk portfolio project.

## Primary Dataset

Primary target dataset: Kaggle "Give Me Some Credit".

Expected raw input:

```text
data/raw/cs-training.csv
```

## Target Definition

For the Give Me Some Credit dataset, the target is `SeriousDlqin2yrs`.

- Bad event: `SeriousDlqin2yrs = 1`
- Good event: `SeriousDlqin2yrs = 0`
- Performance window: two years after the observation point represented in the dataset

## Leakage Controls

The following operations must be fit only on the training sample:

- WoE binning
- IV screening thresholds
- Feature selection
- Logistic regression
- Scorecard scaling
- Calibration logic

The test sample is used only for held-out validation.

## Phase 0 Definition Of Done

- Repository structure exists.
- README stub explains the project and target.
- Dependency file exists.
- Empty data and artifact folders are present but ignored by git.
- Source package stubs exist for the later pipeline phases.
