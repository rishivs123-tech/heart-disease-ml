"""Leakage-safe preprocessing and imputation pipelines for Heart Disease classification.

Implements the four missing-data strategies defined in PRD FR-04:
- M1: Simple median/mode imputation
- M2: Median/mode imputation with missingness indicators
- M3: KNN imputation for continuous variables + categorical mode imputation
- M4: Iterative imputation (MICE) for continuous variables + categorical mode imputation
"""

import logging
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor(strategy: Literal["M1", "M2", "M3", "M4"] = "M1") -> ColumnTransformer:
    """Construct a leakage-safe ColumnTransformer for the specified imputation strategy.

    Parameters
    ----------
    strategy : {'M1', 'M2', 'M3', 'M4'}
        M1: Median for numeric, Mode for categorical
        M2: Median/Mode with missingness indicator features
        M3: KNNImputer (k=5) for numeric, Mode for categorical
        M4: IterativeImputer (MICE) for numeric, Mode for categorical

    Returns
    -------
    ColumnTransformer
        Configured transformer pipeline ready to fit on training data.
    """
    if strategy == "M1":
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=False)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

    elif strategy == "M2":
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

    elif strategy == "M3":
        num_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=False)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

    elif strategy == "M4":
        num_pipeline = Pipeline([
            ("imputer", IterativeImputer(max_iter=10, random_state=42)),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=False)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}. Choose from 'M1', 'M2', 'M3', 'M4'.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def create_pipeline(strategy: str, model: BaseEstimator) -> Pipeline:
    """Build a complete scikit-learn Pipeline chaining preprocessor and classifier."""
    preprocessor = create_preprocessor(strategy)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve interpretable feature names after ColumnTransformer fitting."""
    try:
        names = preprocessor.get_feature_names_out()
        cleaned_names = []
        for n in names:
            # Clean scikit-learn prefixes for display clarity
            clean = n.replace("num__", "").replace("cat__", "")
            cleaned_names.append(clean)
        return cleaned_names
    except Exception as e:
        logger.warning("Could not auto-extract feature names: %s. Generating manual fallback.", e)
        # Fallback names
        cat_names = []
        for c in CATEGORICAL_FEATURES:
            cat_names.append(f"{c}_cat")
        return NUMERIC_FEATURES + cat_names
