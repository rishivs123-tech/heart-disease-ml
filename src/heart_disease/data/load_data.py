"""Data acquisition, parsing, and persistence module for the 920-record Heart Disease cohort.

Matches PRD Section 6 (Primary dataset, 4 sources, 920 records, binary target conversion).
"""

import os
import urllib.request
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UCI_SOURCES = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Hungary": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "Switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "VA Long Beach": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
}

EXPECTED_COUNTS = {
    "Cleveland": 303,
    "Hungary": 294,
    "Switzerland": 123,
    "VA Long Beach": 200
}

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"
]
ALL_COLUMNS = ["dataset"] + FEATURE_COLUMNS + ["num"]


def fetch_raw_uci_datasets(output_raw_path: str = "data/raw/heart_disease_combined_raw.csv") -> pd.DataFrame:
    """Download raw data files from UCI repository and concatenate them into a single dataframe."""
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
    all_records = []

    for source_name, url in UCI_SOURCES.items():
        logger.info("Fetching %s records from %s...", source_name, url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Network error fetching %s directly: %s. Using fallback mirror.", source_name, e)
            raise RuntimeError(f"Could not download {source_name} data from {url}: {e}")

        source_records = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 14:
                source_records.append([source_name] + parts)
            else:
                logger.warning("Skipping malformed row with %d elements in %s", len(parts), source_name)

        actual_count = len(source_records)
        expected = EXPECTED_COUNTS[source_name]
        logger.info("Source %s: retrieved %d records (expected %d).", source_name, actual_count, expected)
        all_records.extend(source_records)

    df_raw = pd.DataFrame(all_records, columns=ALL_COLUMNS)
    df_raw.to_csv(output_raw_path, index=False)
    logger.info("Raw combined dataset saved to %s (Total records: %d)", output_raw_path, len(df_raw))
    return df_raw


def clean_and_format_dataset(
    df_raw: pd.DataFrame,
    handle_invalid_zeros: bool = True
) -> pd.DataFrame:
    """Clean missing values ('?'), handle physiologically invalid zero measurements,

    convert data types, and create the binary classification target.
    """
    df = df_raw.copy()

    # Replace '?' and empty strings with NaN
    df = df.replace(["?", "", " ", "null", "NULL", "NaN"], np.nan)

    # Cast numeric and categorical columns
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Original target num (0 to 4)
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df["num_original"] = df["num"]

    # Target conversion according to PRD 6.5:
    # 0: Disease absent
    # 1: Disease present (> 0)
    df["target"] = (df["num"] > 0).astype(int)

    # Physiologically invalid zeros:
    # In clinical reality, resting blood pressure (trestbps) and cholesterol (chol) cannot be 0 in living humans.
    # In the Switzerland and VA Long Beach cohorts, missing cholesterol was often recorded as 0.
    if handle_invalid_zeros:
        invalid_trestbps = (df["trestbps"] == 0)
        invalid_chol = (df["chol"] == 0)
        logger.info(
            "Flagging physiologically invalid zeros as missing: trestbps=0 (%d cases), chol=0 (%d cases)",
            invalid_trestbps.sum(),
            invalid_chol.sum()
        )
        df.loc[invalid_trestbps, "trestbps"] = np.nan
        df.loc[invalid_chol, "chol"] = np.nan

    return df


def prepare_datasets(
    raw_path: str = "data/raw/heart_disease_combined_raw.csv",
    processed_path: str = "data/processed/heart_disease_cleaned.csv",
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete data preparation: download/load, clean, create binary target,

    and generate reproducible 80/20 train/test splits.
    """
    if os.path.exists(raw_path):
        logger.info("Loading existing raw dataset from %s", raw_path)
        df_raw = pd.read_csv(raw_path, dtype=str)
    else:
        logger.info("Raw dataset not found at %s. Fetching from UCI...", raw_path)
        df_raw = fetch_raw_uci_datasets(raw_path)

    df_cleaned = clean_and_format_dataset(df_raw, handle_invalid_zeros=True)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_cleaned.to_csv(processed_path, index=False)
    logger.info("Saved cleaned dataset to %s (%d records)", processed_path, len(df_cleaned))

    # Stratified 80/20 train/test split matching PRD 11.1 (736 train / 184 test)
    train_df, test_df = train_test_split(
        df_cleaned,
        test_size=test_size,
        stratify=df_cleaned["target"],
        random_state=random_state
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(
        "Generated stratified split: Train = %d records, Test = %d records (saved to %s and %s)",
        len(train_df),
        len(test_df),
        train_path,
        test_path
    )

    return df_cleaned, train_df, test_df


if __name__ == "__main__":
    prepare_datasets()
