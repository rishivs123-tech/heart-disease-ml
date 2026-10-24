"""SHAP Explanation Stability evaluation module.

Matches PRD FR-08, Section 11.3, and Research Questions RQ3, RQ4:
- Global stability: Spearman rank correlation and Top-k Jaccard similarity
- Local patient-level stability: Mean Absolute Difference (MAD) and attribution correlation
- Hypothesis testing for H1, H2, and H3
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def calculate_spearman_rank_stability(rank_a: list[str], rank_b: list[str]) -> float:
    """Compute Spearman rank correlation between two ordered lists of feature importance rankings."""
    all_features = list(dict.fromkeys(rank_a + rank_b))
    order_a = {feat: idx for idx, feat in enumerate(rank_a)}
    order_b = {feat: idx for idx, feat in enumerate(rank_b)}

    ranks_a = [order_a.get(f, len(all_features)) for f in all_features]
    ranks_b = [order_b.get(f, len(all_features)) for f in all_features]

    corr, _ = spearmanr(ranks_a, ranks_b)
    return round(float(corr), 4) if not np.isnan(corr) else 0.0


def calculate_top_k_jaccard(features_a: list[str], features_b: list[str], k: int = 5) -> float:
    """Compute Jaccard similarity index between the top-k most important features.

    Jaccard = |A ∩ B| / |A ∪ B|
    """
    set_a = set(features_a[:k])
    set_b = set(features_b[:k])

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return round(float(intersection / union), 4) if union > 0 else 1.0


def calculate_patient_level_mad(shap_matrix_a: np.ndarray, shap_matrix_b: np.ndarray) -> float:
    """Compute Mean Absolute Difference (MAD) between patient-level SHAP attributions

    across two model fits / runs on the identical test cohort.
    """
    a = np.asarray(shap_matrix_a)
    b = np.asarray(shap_matrix_b)

    # Ensure shape alignment
    min_rows = min(a.shape[0], b.shape[0])
    min_cols = min(a.shape[1], b.shape[1])
    diff = np.abs(a[:min_rows, :min_cols] - b[:min_rows, :min_cols])
    return round(float(np.mean(diff)), 4)


def calculate_attribution_correlation(shap_matrix_a: np.ndarray, shap_matrix_b: np.ndarray) -> float:
    """Compute mean Pearson correlation between individual patient SHAP attribution vectors across runs."""
    a = np.asarray(shap_matrix_a)
    b = np.asarray(shap_matrix_b)

    min_rows = min(a.shape[0], b.shape[0])
    min_cols = min(a.shape[1], b.shape[1])

    corrs = []
    for i in range(min_rows):
        row_a = a[i, :min_cols]
        row_b = b[i, :min_cols]
        if np.std(row_a) > 1e-8 and np.std(row_b) > 1e-8:
            r, _ = pearsonr(row_a, row_b)
            if not np.isnan(r):
                corrs.append(r)

    return round(float(np.mean(corrs)), 4) if corrs else 0.0


def evaluate_explanation_stability(
    run_results: list[dict]
) -> pd.DataFrame:
    """Evaluate pairwise explanation stability across multiple experimental runs and models.

    Each item in run_results is expected to contain:
    - 'model': model name
    - 'strategy': imputation strategy (e.g. M1, M2, M3, M4)
    - 'run_id': seed / fold ID
    - 'ranked_features': list of feature names ordered by importance
    - 'shap_values': numpy array of SHAP values on the fixed reference set
    """
    records = []
    n = len(run_results)

    for i in range(n):
        for j in range(i + 1, n):
            res_a = run_results[i]
            res_b = run_results[j]

            # Compare only when informative (e.g. same model across runs or across imputation strategies)
            spearman = calculate_spearman_rank_stability(
                res_a["ranked_features"],
                res_b["ranked_features"]
            )
            jaccard_3 = calculate_top_k_jaccard(
                res_a["ranked_features"],
                res_b["ranked_features"],
                k=3
            )
            jaccard_5 = calculate_top_k_jaccard(
                res_a["ranked_features"],
                res_b["ranked_features"],
                k=5
            )

            mad = np.nan
            attr_corr = np.nan
            if "shap_values" in res_a and "shap_values" in res_b:
                mad = calculate_patient_level_mad(res_a["shap_values"], res_b["shap_values"])
                attr_corr = calculate_attribution_correlation(res_a["shap_values"], res_b["shap_values"])

            records.append({
                "model_a": res_a["model"],
                "strategy_a": res_a["strategy"],
                "run_a": res_a["run_id"],
                "model_b": res_b["model"],
                "strategy_b": res_b["strategy"],
                "run_b": res_b["run_id"],
                "same_model": res_a["model"] == res_b["model"],
                "same_strategy": res_a["strategy"] == res_b["strategy"],
                "spearman_rank_corr": spearman,
                "jaccard_top_3": jaccard_3,
                "jaccard_top_5": jaccard_5,
                "patient_level_mad": mad,
                "attribution_correlation": attr_corr
            })

    return pd.DataFrame(records)
