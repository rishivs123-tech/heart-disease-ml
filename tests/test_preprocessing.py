"""Unit tests for leakage-safe preprocessing pipelines and M1-M4 imputation."""

import numpy as np
import pandas as pd
import pytest
from src.heart_disease.data.preprocess import (
    ALL_FEATURES,
    create_pipeline,
    create_preprocessor,
    get_feature_names_out
)
from src.heart_disease.models.build_models import get_model


@pytest.fixture
def sample_data():
    """Create a sample dataframe with realistic missing values for testing transformers."""
    return pd.DataFrame({
        "age": [54.0, 60.0, np.nan, 45.0],
        "trestbps": [130.0, np.nan, 140.0, 120.0],
        "chol": [240.0, 290.0, np.nan, 180.0],
        "thalch": [150.0, 140.0, 160.0, np.nan],
        "oldpeak": [1.2, np.nan, 0.0, 2.4],
        "sex": [1.0, 0.0, 1.0, 0.0],
        "cp": [4.0, 3.0, np.nan, 1.0],
        "fbs": [0.0, 1.0, np.nan, 0.0],
        "restecg": [0.0, 1.0, 2.0, 0.0],
        "exang": [1.0, 0.0, np.nan, 1.0],
        "slope": [2.0, np.nan, 1.0, 2.0],
        "ca": [0.0, np.nan, 1.0, 2.0],
        "thal": [3.0, 7.0, np.nan, 6.0],
        "target": [1, 0, 1, 0]
    })


@pytest.mark.parametrize("strategy", ["M1", "M2", "M3", "M4"])
def test_imputation_strategies_no_nans(sample_data, strategy):
    """Ensure that all imputation strategies transform data with zero remaining NaNs."""
    X = sample_data[ALL_FEATURES]
    y = sample_data["target"]

    preprocessor = create_preprocessor(strategy)
    X_trans = preprocessor.fit_transform(X, y)

    assert not np.isnan(X_trans).any(), f"Strategy {strategy} left NaNs in transformed output!"
    feature_names = get_feature_names_out(preprocessor)
    assert len(feature_names) == X_trans.shape[1]


def test_full_pipeline_fit_predict(sample_data):
    """Verify end-to-end pipeline training and prediction."""
    X = sample_data[ALL_FEATURES]
    y = sample_data["target"]

    model = get_model("Logistic Regression")
    pipeline = create_pipeline("M1", model)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
