"""Model training and artifact serialization module.

Handles:
- Fitting preprocessor + classifier pipelines
- Evaluation on the test set
- Saving fitted pipelines to artifacts/models/
- Saving run metadata to artifacts/metadata/
"""

import datetime
import json
import logging
import os
import joblib
import pandas as pd
from src.heart_disease.data.preprocess import ALL_FEATURES, create_pipeline, get_feature_names_out
from src.heart_disease.evaluation.metrics import calculate_metrics
from src.heart_disease.models.build_models import get_all_baseline_models, get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_persist_model(
    model_name: str,
    strategy: str = "M1",
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    models_dir: str = "artifacts/models",
    metadata_dir: str = "artifacts/metadata",
    random_state: int = 42
) -> dict:
    """Train a single model pipeline on the training set, evaluate on test set, and persist artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = df_train[ALL_FEATURES], df_train["target"]
    X_test, y_test = df_test[ALL_FEATURES], df_test["target"]

    estimator = get_model(model_name, random_state=random_state)
    pipeline = create_pipeline(strategy, estimator)

    logger.info("Fitting %s pipeline with %s imputation on %d training records...", model_name, strategy, len(X_train))
    pipeline.fit(X_train, y_train)

    # Test set predictions
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        import numpy as np
        decision = pipeline.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-decision))
    else:
        y_prob = None

    test_metrics = calculate_metrics(y_test, y_pred, y_prob)
    logger.info("Test results for %s (%s): Accuracy=%.4f, ROC-AUC=%s", model_name, strategy, test_metrics["accuracy"], test_metrics["roc_auc"])

    # Persist model
    clean_name = model_name.lower().replace(" ", "_")
    model_filename = f"{clean_name}_{strategy.lower()}.joblib"
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(pipeline, model_path)

    # Extract feature names
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_out(preprocessor)

    # Save metadata
    meta = {
        "model_name": model_name,
        "imputation_strategy": strategy,
        "model_path": model_path,
        "training_date": datetime.datetime.now().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": test_metrics,
        "feature_names": feature_names,
        "hyperparameters": estimator.get_params()
    }
    # Convert non-serializable objects in hyperparameters
    serializable_params = {}
    for k, v in meta["hyperparameters"].items():
        try:
            json.dumps(v)
            serializable_params[k] = v
        except Exception:
            serializable_params[k] = str(v)
    meta["hyperparameters"] = serializable_params

    meta_filename = f"{clean_name}_{strategy.lower()}_metadata.json"
    meta_path = os.path.join(metadata_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def train_all_baseline_pipelines(
    strategy: str = "M1",
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv"
) -> list[dict]:
    """Train and persist all five baseline models using a specified strategy."""
    models = ["Logistic Regression", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Gradient Boosting"]
    results = []
    for m in models:
        meta = train_and_persist_model(m, strategy=strategy, train_path=train_path, test_path=test_path)
        results.append(meta)
    return results


if __name__ == "__main__":
    train_all_baseline_pipelines(strategy="M1")
