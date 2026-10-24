"""Missing-Data Imputation Experiment runner.

Matches PRD FR-04, Section 11.2, and Research Question RQ2:
- Evaluates 5 classifiers across 4 missing-data strategies (M1, M2, M3, M4)
- Repeated across 3 random seeds (42, 101, 2024) and 5 folds = 300 evaluations
- Conducts paired t-tests and Wilcoxon signed-rank tests
- Generates comparative boxplots, bar charts, and metric exports
"""

import itertools
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.heart_disease.data.preprocess import ALL_FEATURES, create_pipeline
from src.heart_disease.evaluation.cross_validation import evaluate_model_cv
from src.heart_disease.evaluation.statistics import compute_confidence_interval, paired_comparison_test
from src.heart_disease.models.build_models import get_all_baseline_models, get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["Logistic Regression", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Gradient Boosting"]
STRATEGIES = ["M1", "M2", "M3", "M4"]
SEEDS = [42, 101, 2024]
N_SPLITS = 5


def run_imputation_experiments(
    train_path: str = "data/processed/train.csv",
    output_dir: str = "results"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete missing-data evaluation across 5 models, 4 strategies, 3 seeds, 5 folds."""
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    df_train = pd.read_csv(train_path)
    X_train, y_train = df_train[ALL_FEATURES], df_train["target"]

    all_fold_evaluations = []
    summary_records = []

    logger.info("Executing 300 evaluations: 5 models x 4 strategies x 3 seeds x 5 folds...")

    for model_name in MODELS:
        for strategy in STRATEGIES:
            logger.info("Evaluating: Model='%s' | Imputation='%s'", model_name, strategy)

            strategy_accuracies = []
            strategy_f1s = []
            strategy_aucs = []

            for seed in SEEDS:
                estimator = get_model(model_name, random_state=seed)
                cv_res = evaluate_model_cv(
                    model_name=model_name,
                    estimator=estimator,
                    X=X_train,
                    y=y_train,
                    strategy=strategy,
                    n_splits=N_SPLITS,
                    random_state=seed
                )

                for f in cv_res["folds"]:
                    fold_rec = {
                        "model": model_name,
                        "strategy": strategy,
                        "seed": seed,
                        "fold": f["fold"],
                        "accuracy": f["accuracy"],
                        "precision": f["precision"],
                        "recall": f["recall"],
                        "specificity": f["specificity"],
                        "f1_score": f["f1_score"],
                        "mcc": f["mcc"],
                        "roc_auc": f["roc_auc"]
                    }
                    all_fold_evaluations.append(fold_rec)
                    strategy_accuracies.append(f["accuracy"])
                    strategy_f1s.append(f["f1_score"])
                    if f["roc_auc"] is not None:
                        strategy_aucs.append(f["roc_auc"])

            # Compute 95% Confidence Intervals
            acc_mean, acc_low, acc_high = compute_confidence_interval(strategy_accuracies)
            f1_mean, f1_low, f1_high = compute_confidence_interval(strategy_f1s)
            auc_mean, auc_low, auc_high = compute_confidence_interval(strategy_aucs) if strategy_aucs else (None, None, None)

            summary_records.append({
                "model": model_name,
                "strategy": strategy,
                "n_evaluations": len(strategy_accuracies),
                "accuracy_mean": acc_mean,
                "accuracy_ci_lower": acc_low,
                "accuracy_ci_upper": acc_high,
                "f1_mean": f1_mean,
                "f1_ci_lower": f1_low,
                "f1_ci_upper": f1_high,
                "roc_auc_mean": auc_mean,
                "roc_auc_ci_lower": auc_low,
                "roc_auc_ci_upper": auc_high
            })

    df_folds = pd.DataFrame(all_fold_evaluations)
    df_summary = pd.DataFrame(summary_records)

    # Save CSVs
    folds_csv = os.path.join(output_dir, "metrics", "imputation_experiments.csv")
    summary_csv = os.path.join(output_dir, "metrics", "imputation_summary.csv")
    df_folds.to_csv(folds_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    logger.info("Saved fold evaluations to %s and summary to %s", folds_csv, summary_csv)

    # Statistical hypothesis comparisons (M1 vs M2, M1 vs M3, M1 vs M4 for each model)
    stat_records = []
    for model_name in MODELS:
        df_m = df_folds[df_folds["model"] == model_name]
        m1_scores = df_m[df_m["strategy"] == "M1"]["accuracy"].values

        for strat in ["M2", "M3", "M4"]:
            comp_scores = df_m[df_m["strategy"] == strat]["accuracy"].values
            res = paired_comparison_test(comp_scores, m1_scores, metric_name="accuracy")
            res["model"] = model_name
            res["comparison"] = f"{strat} vs M1 (Baseline Median)"
            stat_records.append(res)

    df_stats = pd.DataFrame(stat_records)
    stats_csv = os.path.join(output_dir, "metrics", "statistical_comparisons.csv")
    df_stats.to_csv(stats_csv, index=False)
    logger.info("Saved statistical comparisons to %s", stats_csv)

    # Generate comparative boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_folds, x="model", y="accuracy", hue="strategy", palette="Set2")
    plt.title("Classification Accuracy by Imputation Strategy Across 3 Repeated 5-Fold CV Runs")
    plt.ylabel("Accuracy")
    plt.xlabel("Classifier")
    plt.ylim([0.65, 0.95])
    plt.legend(title="Imputation Strategy", loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "figures", "imputation_comparison_boxplot.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()

    # Generate summary heatmaps of mean accuracy
    pivot_acc = df_summary.pivot(index="model", columns="strategy", values="accuracy_mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_acc, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Mean Accuracy'})
    plt.title("Mean CV Accuracy: Models vs Imputation Strategies")
    plt.ylabel("Classifier")
    plt.xlabel("Imputation Strategy")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "figures", "imputation_strategy_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    return df_folds, df_summary, df_stats


if __name__ == "__main__":
    run_imputation_experiments()
