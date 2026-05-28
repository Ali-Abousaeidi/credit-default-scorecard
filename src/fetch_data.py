"""Download and normalize the Give Me Some Credit raw dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests
from scipy.io import arff

from src.config import (
    OPENML_DOWNLOAD_URL,
    OPENML_MD5,
    RAW_GMSC_FILE,
    RAW_OPENML_ARFF_FILE,
    TARGET_COLUMN,
)


OPENML_TARGET_COLUMN = "FinancialDistressNextTwoYears"


def md5sum(path: Path) -> str:
    """Return the MD5 checksum for a file."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_md5: str) -> None:
    """Download a file and verify its checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with destination.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    observed_md5 = md5sum(destination)
    if observed_md5 != expected_md5:
        raise ValueError(
            f"Checksum mismatch for {destination}. "
            f"Expected {expected_md5}, observed {observed_md5}."
        )


def _decode_nominal_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Decode byte-valued ARFF nominal columns into strings."""
    decoded = data.copy()
    for column in decoded.columns:
        if decoded[column].dtype == object:
            decoded[column] = decoded[column].map(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
    return decoded


def convert_openml_arff_to_training_csv(
    arff_path: Path = RAW_OPENML_ARFF_FILE,
    csv_path: Path = RAW_GMSC_FILE,
) -> pd.DataFrame:
    """Convert OpenML's ARFF file into the Kaggle-style training CSV."""
    rows, _ = arff.loadarff(arff_path)
    data = _decode_nominal_columns(pd.DataFrame(rows))

    if OPENML_TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected OpenML target '{OPENML_TARGET_COLUMN}' was not found.")

    data[TARGET_COLUMN] = data[OPENML_TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    data = data.drop(columns=[OPENML_TARGET_COLUMN])

    ordered_columns = [TARGET_COLUMN] + [col for col in data.columns if col != TARGET_COLUMN]
    data = data[ordered_columns]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    return data


def main() -> None:
    """Download the raw source and create data/raw/cs-training.csv."""
    if not RAW_OPENML_ARFF_FILE.exists():
        print(f"Downloading OpenML source to {RAW_OPENML_ARFF_FILE}")
        download_file(OPENML_DOWNLOAD_URL, RAW_OPENML_ARFF_FILE, OPENML_MD5)
    else:
        observed_md5 = md5sum(RAW_OPENML_ARFF_FILE)
        if observed_md5 != OPENML_MD5:
            raise ValueError(
                f"Existing file checksum mismatch. Expected {OPENML_MD5}, observed {observed_md5}."
            )
        print(f"Using existing verified source file: {RAW_OPENML_ARFF_FILE}")

    data = convert_openml_arff_to_training_csv()
    print(f"Wrote normalized training CSV: {RAW_GMSC_FILE}")
    print(f"Rows: {data.shape[0]:,}; columns: {data.shape[1]:,}")
    print(f"Bad rate ({TARGET_COLUMN}=1): {data[TARGET_COLUMN].mean():.2%}")


if __name__ == "__main__":
    main()
