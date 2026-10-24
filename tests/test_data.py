"""Automated unit and integration tests for dataset acquisition and cleaning."""

import os
import pandas as pd
import numpy as np
import pytest
from src.heart_disease.data.load_data import clean_and_format_dataset
from src.heart_disease.data.validate_data import validate_dataset, EXPECTED_ROW_COUNT, EXPECTED_CLASSES


def test_clean_and_format_dataset():
    """Test data cleaning, invalid zeros flagging, and binary target conversion."""
    raw_dummy = pd.DataFrame({
        "dataset": ["Cleveland", "Hungary"],
        "age": ["63", "55"],
        "sex": ["1", "0"],
        "cp": ["4", "2"],
        "trestbps": ["140", "0"],  # 0 should become NaN
        "chol": ["0", "250"],      # 0 should become NaN
        "fbs": ["1", "?"],         # ? should become NaN
        "restecg": ["0", "1"],
        "thalch": ["150", "130"],
        "exang": ["0", "1"],
        "oldpeak": ["2.3", "0.0"],
        "slope": ["2", "1"],
        "ca": ["0", "?"],
        "thal": ["7", "?"],
        "num": ["2", "0"]          # 2 -> target 1, 0 -> target 0
    })

    cleaned = clean_and_format_dataset(raw_dummy, handle_invalid_zeros=True)

    assert cleaned["target"].iloc[0] == 1
    assert cleaned["target"].iloc[1] == 0
    assert np.isnan(cleaned["trestbps"].iloc[1])
    assert np.isnan(cleaned["chol"].iloc[0])
    assert np.isnan(cleaned["fbs"].iloc[1])


def test_processed_dataset_acceptance():
    """Verify that the actual saved processed dataset satisfies PRD acceptance criteria."""
    data_path = "data/processed/heart_disease_cleaned.csv"
    if not os.path.exists(data_path):
        pytest.skip("Processed dataset not yet generated.")

    df = pd.read_csv(data_path)
    report = validate_dataset(df)

    assert report["passed"] is True
    assert report["record_count"] == EXPECTED_ROW_COUNT
    assert report["target_distribution"][0] == EXPECTED_CLASSES[0]
    assert report["target_distribution"][1] == EXPECTED_CLASSES[1]
    assert report["physiologically_invalid_zeros"]["trestbps"] == 0
    assert report["physiologically_invalid_zeros"]["chol"] == 0
