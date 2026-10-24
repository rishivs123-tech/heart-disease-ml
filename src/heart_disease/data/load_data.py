"""Data loader and preprocessing utilities for UCI Heart Disease dataset."""

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
    """Download raw processed data files from UCI repository and combine them."""
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
    all_records = []

    for source_name, url in UCI_SOURCES.items():
        logger.info("Fetching %s data from %s", source_name, url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            logger.error("Failed to download %s: %s", source_name, e)
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
                logger.warning("Skipping invalid row in %s: %s", source_name, parts)

        actual_count = len(source_records)
        expected = EXPECTED_COUNTS[source_name]
        logger.info("Source %s: loaded %d rows (expected %d)", source_name, actual_count, expected)
        all_records.extend(source_records)

    df_raw = pd.DataFrame(all_records, columns=ALL_COLUMNS)
    df_raw.to_csv(output_raw_path, index=False)
    logger.info("Saved raw combined data to %s (%d records)", output_raw_path, len(df_raw))
    return df_raw


def clean_and_format_dataset(
    df_raw: pd.DataFrame,
    handle_invalid_zeros: bool = True
) -> pd.DataFrame:
    """Parse numeric columns, clean missing placeholders, and encode binary target."""
    df = df_raw.copy()

    # replace missing value markers with NaN
    df = df.replace(["?", "", " ", "null", "NULL", "NaN"], np.nan)

    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # preserve original num rating (0-4)
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df["num_original"] = df["num"]

    # binary classification: 0 = absent, 1 = present
    df["target"] = (df["num"] > 0).astype(int)

    # 0 values in trestbps and chol are physiologically impossible (missing data entry)
    if handle_invalid_zeros:
        invalid_trestbps = (df["trestbps"] == 0)
        invalid_chol = (df["chol"] == 0)
        if invalid_trestbps.any() or invalid_chol.any():
            logger.info(
                "Replacing zero values with NaN: trestbps=0 (%d), chol=0 (%d)",
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
    """Load or fetch data, clean features, and create train/test splits."""
    if os.path.exists(raw_path):
        logger.info("Reading raw dataset from %s", raw_path)
        df_raw = pd.read_csv(raw_path, dtype=str)
    else:
        logger.info("Fetching raw files from UCI repository...")
        df_raw = fetch_raw_uci_datasets(raw_path)

    df_cleaned = clean_and_format_dataset(df_raw, handle_invalid_zeros=True)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_cleaned.to_csv(processed_path, index=False)
    logger.info("Saved cleaned data to %s (%d records)", processed_path, len(df_cleaned))

    # 80/20 train/test split stratified by target class
    train_df, test_df = train_test_split(
        df_cleaned,
        test_size=test_size,
        stratify=df_cleaned["target"],
        random_state=random_state
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(
        "Train/test split saved: train=%d, test=%d (%s, %s)",
        len(train_df),
        len(test_df),
        train_path,
        test_path
    )

    return df_cleaned, train_df, test_df


if __name__ == "__main__":
    prepare_datasets()
