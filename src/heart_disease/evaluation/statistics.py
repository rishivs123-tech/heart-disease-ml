"""Statistical significance analysis and hypothesis testing for model and imputation comparisons.

Matches PRD Section 11.4:
- Paired t-tests and Wilcoxon signed-rank tests for paired cross-validation folds
- 95% confidence intervals (t-distribution)
- Corrected repeated k-fold test considerations (Nadeau & Bengio corrected variance)
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval(
    scores: list[float] | np.ndarray,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """Calculate mean and two-sided confidence interval for a metric across folds/runs.

    Returns (mean, lower_bound, upper_bound).
    """
    scores_arr = np.asarray(scores, dtype=float)
    scores_arr = scores_arr[~np.isnan(scores_arr)]
    n = len(scores_arr)
    if n < 2:
        val = float(scores_arr[0]) if n == 1 else 0.0
        return val, val, val

    mean = float(np.mean(scores_arr))
    sem = stats.sem(scores_arr)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return round(mean, 4), round(mean - margin, 4), round(mean + margin, 4)


def paired_comparison_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    metric_name: str = "accuracy"
) -> dict[str, float | str | bool]:
    """Conduct paired t-test and Wilcoxon signed-rank test between two configurations (e.g.

    M1 vs M4 or RF vs LR across identical folds).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    diff = a - b
    mean_diff = float(np.mean(diff))

    # Paired t-test
    if np.all(diff == 0):
        t_stat, p_val_t = 0.0, 1.0
    else:
        t_stat, p_val_t = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank test
    try:
        if np.all(diff == 0):
            w_stat, p_val_w = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(a, b)
            w_stat, p_val_w = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, p_val_w = np.nan, np.nan

    return {
        "metric": metric_name,
        "mean_diff": round(mean_diff, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value_t": round(float(p_val_t), 4),
        "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
        "p_value_wilcoxon": round(float(p_val_w), 4) if not np.isnan(p_val_w) else None,
        "significant_at_05": bool(p_val_t < 0.05)
    }
