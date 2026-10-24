"""Unit tests for machine learning model building and evaluation metrics."""

import pytest
from src.heart_disease.models.build_models import get_model, get_all_baseline_models
from src.heart_disease.evaluation.metrics import calculate_metrics


def test_get_all_baseline_models():
    """Verify factory returns all five required models with paper initial configurations."""
    models = get_all_baseline_models()
    expected_names = [
        "Logistic Regression", "Random Forest", "Support Vector Machine",
        "K-Nearest Neighbors", "Gradient Boosting"
    ]
    for name in expected_names:
        assert name in models
        assert hasattr(models[name], "fit")
        assert hasattr(models[name], "predict")


def test_calculate_metrics():
    """Test performance metric computations (accuracy, specificity, recall, mcc, roc_auc)."""
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    y_prob = [0.9, 0.4, 0.1, 0.2, 0.8]

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.8
    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["specificity"] == 1.0
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["roc_auc"] is not None
    assert metrics["mcc"] > 0
