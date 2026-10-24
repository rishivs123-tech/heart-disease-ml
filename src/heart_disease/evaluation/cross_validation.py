"""Leakage-safe repeated Stratified K-Fold Cross-Validation runner.

Matches PRD Section 11.1 and 11.2 experimental protocol.
Ensures that all imputation, scaling, and encoding transformations are fitted
strictly on the training partition of each fold.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.heart_disease.data.preprocess import create_pipeline, ALL_FEATURES
from src.heart_disease.evaluation.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_cv(
    model_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "M1",
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """Run stratified K-fold cross-validation for a given model and imputation strategy.

    Returns fold-level metrics and aggregate mean/std.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]

        # Build pipeline with preprocessor + model (strictly leakage safe)
        pipeline = create_pipeline(strategy, estimator)
        pipeline.fit(X_train_f, y_train_f)

        y_pred = pipeline.predict(X_val_f)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_val_f)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X_val_f)
            y_prob = 1 / (1 + np.exp(-decision))
        else:
            y_prob = None

        m = calculate_metrics(y_val_f, y_pred, y_prob)
        m["fold"] = fold_idx + 1
        fold_metrics.append(m)

        y_true_all.extend(y_val_f.tolist())
        y_pred_all.extend(y_pred.tolist())
        if y_prob is not None:
            y_prob_all.extend(y_prob.tolist())

    # Aggregate overall metrics across folds
    df_folds = pd.DataFrame(fold_metrics)
    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1_score", "mcc", "roc_auc"]

    summary = {
        "model": model_name,
        "strategy": strategy,
        "random_state": random_state,
        "n_splits": n_splits,
        "folds": fold_metrics
    }

    for col in metric_cols:
        valid_vals = df_folds[col].dropna()
        summary[f"{col}_mean"] = round(float(valid_vals.mean()), 4) if len(valid_vals) > 0 else None
        summary[f"{col}_std"] = round(float(valid_vals.std()), 4) if len(valid_vals) > 0 else None

    # Overall pooled metrics
    pooled = calculate_metrics(y_true_all, y_pred_all, y_prob_all if y_prob_all else None)
    summary["pooled_metrics"] = pooled

    return summary
