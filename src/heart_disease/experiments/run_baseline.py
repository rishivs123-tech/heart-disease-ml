"""Baseline Reproduction Experiment runner.

Matches PRD Phase 2 and Section 11.1:
- Evaluates the five baseline classifiers on the 80/20 split and 5-fold CV
- Computes Accuracy, Precision, Recall, Specificity, F1, MCC, ROC-AUC
- Compares empirical results against reference study benchmarks
- Saves metrics to results/metrics/baseline_reproduction.csv
- Generates publication-ready figures in results/figures/
"""

import json
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

from src.heart_disease.data.preprocess import ALL_FEATURES, create_pipeline
from src.heart_disease.evaluation.cross_validation import evaluate_model_cv
from src.heart_disease.evaluation.metrics import calculate_metrics
from src.heart_disease.models.build_models import get_all_baseline_models, get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Benchmark values from reference paper (Table 2 & comparative sections)
REFERENCE_PAPER_BENCHMARKS = {
    "Logistic Regression": {"accuracy": 0.8315, "precision": 0.8400, "recall": 0.8500, "f1_score": 0.8450, "roc_auc": 0.8950},
    "Random Forest": {"accuracy": 0.8478, "precision": 0.8600, "recall": 0.8600, "f1_score": 0.8600, "roc_auc": 0.9120},
    "Support Vector Machine": {"accuracy": 0.8370, "precision": 0.8500, "recall": 0.8500, "f1_score": 0.8500, "roc_auc": 0.8980},
    "K-Nearest Neighbors": {"accuracy": 0.8207, "precision": 0.8300, "recall": 0.8400, "f1_score": 0.8350, "roc_auc": 0.8720},
    "Gradient Boosting": {"accuracy": 0.8533, "precision": 0.8700, "recall": 0.8650, "f1_score": 0.8675, "roc_auc": 0.9180}
}


def run_baseline_reproduction(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    output_dir: str = "results",
    random_state: int = 42
) -> pd.DataFrame:
    """Execute baseline reproduction experiment across all five baseline classifiers."""
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = df_train[ALL_FEATURES], df_train["target"]
    X_test, y_test = df_test[ALL_FEATURES], df_test["target"]

    baseline_models = get_all_baseline_models(random_state=random_state)
    results = []
    roc_data = {}
    cm_data = {}

    plt.figure(figsize=(9, 7))

    for name, estimator in baseline_models.items():
        logger.info("Evaluating baseline model: %s...", name)

        # 1. 5-Fold Cross-Validation on Development Split (736 records)
        cv_res = evaluate_model_cv(
            model_name=name,
            estimator=estimator,
            X=X_train,
            y=y_train,
            strategy="M1",
            n_splits=5,
            random_state=random_state
        )

        # 2. Fit on full training set and evaluate on held-out test split (184 records)
        pipeline = create_pipeline("M1", estimator)
        pipeline.fit(X_train, y_train)

        y_test_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            dec = pipeline.decision_function(X_test)
            y_test_prob = 1 / (1 + np.exp(-dec))
        else:
            y_test_prob = None

        test_m = calculate_metrics(y_test, y_test_pred, y_test_prob)

        # Compute ROC Curve for test set
        if y_test_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, roc_auc)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        cm_data[name] = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

        # Benchmark comparison
        paper_ref = REFERENCE_PAPER_BENCHMARKS.get(name, {})

        results.append({
            "model": name,
            "cv_accuracy_mean": cv_res["accuracy_mean"],
            "cv_accuracy_std": cv_res["accuracy_std"],
            "cv_f1_mean": cv_res["f1_score_mean"],
            "cv_f1_std": cv_res["f1_score_std"],
            "cv_roc_auc_mean": cv_res["roc_auc_mean"],
            "test_accuracy": test_m["accuracy"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_specificity": test_m["specificity"],
            "test_f1_score": test_m["f1_score"],
            "test_mcc": test_m["mcc"],
            "test_roc_auc": test_m["roc_auc"],
            "paper_accuracy_ref": paper_ref.get("accuracy"),
            "paper_f1_ref": paper_ref.get("f1_score"),
            "paper_roc_auc_ref": paper_ref.get("roc_auc"),
            "acc_delta_vs_paper": round(test_m["accuracy"] - paper_ref.get("accuracy", 0), 4)
        })

    # Save ROC curves figure
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curves - Baseline Supervised Classifiers (Held-Out Test Cohort)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_fig_path = os.path.join(output_dir, "figures", "baseline_roc_curves.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()

    # Save Confusion Matrices Figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for idx, (name, cm) in enumerate(cm_data.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
    plt.tight_layout()
    cm_fig_path = os.path.join(output_dir, "figures", "baseline_confusion_matrices.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()

    # Save Metrics CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "metrics", "baseline_reproduction.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info("Baseline reproduction results saved to %s", csv_path)

    # Save Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_results))
    width = 0.35
    plt.bar(x - width/2, df_results["test_accuracy"], width, label="Our Reproduced Accuracy", color="#3b82f6")
    plt.bar(x + width/2, df_results["paper_accuracy_ref"], width, label="Reference Paper Benchmark", color="#94a3b8")
    plt.xticks(x, df_results["model"], rotation=15)
    plt.ylabel("Accuracy Score")
    plt.ylim([0.7, 0.95])
    plt.title("Reproduced Model Accuracy vs Reference Study Benchmark")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, "figures", "baseline_paper_comparison.png")
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()

    return df_results


if __name__ == "__main__":
    run_baseline_reproduction()
