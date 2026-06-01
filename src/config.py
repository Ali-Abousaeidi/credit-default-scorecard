"""Project-level paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_GMSC_FILE = RAW_DATA_DIR / "cs-training.csv"
RAW_OPENML_ARFF_FILE = RAW_DATA_DIR / "GiveMeSomeCredit.arff"
TRAIN_CLEAN_FILE = PROCESSED_DATA_DIR / "train_clean.csv"
TEST_CLEAN_FILE = PROCESSED_DATA_DIR / "test_clean.csv"
TRAIN_WOE_FILE = PROCESSED_DATA_DIR / "train_woe.csv"
TEST_WOE_FILE = PROCESSED_DATA_DIR / "test_woe.csv"
TRAIN_SCORES_FILE = PROCESSED_DATA_DIR / "train_scores.csv"
TEST_SCORES_FILE = PROCESSED_DATA_DIR / "test_scores.csv"
CLEANING_METADATA_FILE = REPORTS_DIR / "cleaning_metadata.json"
BINNING_ARTIFACT_FILE = MODELS_DIR / "binning_artifacts.joblib"
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_scorecard_model.joblib"
SCORECARD_FILE = REPORTS_DIR / "scorecard_points.csv"
TARGET_COLUMN = "SeriousDlqin2yrs"
RANDOM_STATE = 42
TEST_SIZE = 0.25

BASE_SCORE = 600
BASE_ODDS = 50.0
PDO = 20

OPENML_DATASET_ID = 46929
OPENML_FILE_ID = 22125240
OPENML_DOWNLOAD_URL = (
    "https://openml.org/data/v1/download/22125240/GiveMeSomeCredit.arff"
)
OPENML_MD5 = "6d013a631b97b13d5372f097697e25a1"
