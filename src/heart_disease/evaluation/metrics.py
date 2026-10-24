"""Comprehensive evaluation metrics for Heart Disease classification.

Matches PRD FR-05:
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity (True Negative Rate)
- F1-Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- Confusion Matrix parameters (TN, FP, FN, TP)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score
)


def calculate_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_prob: np.ndarray | list | None = None
) -> dict[str, float]:
    """Compute all required classification performance metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,), optional
        Predicted probabilities for positive class (1).

    Returns
    -------
    dict[str, float]
        Dictionary with all evaluation metrics formatted to 4 decimal places.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    roc_auc = np.nan
    if y_prob is not None:
        try:
            y_prob_arr = np.asarray(y_prob, dtype=float)
            if not np.isnan(y_prob_arr).any():
                roc_auc = float(roc_auc_score(y_true, y_prob_arr))
        except Exception:
            roc_auc = np.nan

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "mcc": round(mcc, 4),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
