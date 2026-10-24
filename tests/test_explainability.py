"""Unit tests for SHAP explainer and stability metric calculations."""

import numpy as np
import pytest
from src.heart_disease.explainability.stability import (
    calculate_spearman_rank_stability,
    calculate_top_k_jaccard,
    calculate_patient_level_mad,
    calculate_attribution_correlation
)


def test_spearman_rank_stability():
    """Verify Spearman rank calculation for identical and reversed rankings."""
    rank_a = ["oldpeak", "cp", "thalch", "age", "trestbps"]
    rank_b = ["oldpeak", "cp", "thalch", "age", "trestbps"]
    assert calculate_spearman_rank_stability(rank_a, rank_b) == 1.0

    rank_rev = list(reversed(rank_a))
    assert calculate_spearman_rank_stability(rank_a, rank_rev) == -1.0


def test_top_k_jaccard():
    """Verify Jaccard similarity between top-k feature lists."""
    list_a = ["a", "b", "c", "d", "e"]
    list_b = ["a", "b", "c", "x", "y"]
    # Top 3: {a, b, c} vs {a, b, c} -> 3/3 = 1.0
    assert calculate_top_k_jaccard(list_a, list_b, k=3) == 1.0
    # Top 5: {a, b, c, d, e} vs {a, b, c, x, y} -> 3 / 7 = 0.4286
    assert calculate_top_k_jaccard(list_a, list_b, k=5) == round(3 / 7, 4)


def test_patient_level_mad():
    """Verify patient-level MAD calculation."""
    m_a = np.array([[0.5, 0.2], [0.1, -0.4]])
    m_b = np.array([[0.4, 0.2], [0.1, -0.2]])
    # Diff: [[0.1, 0.0], [0.0, 0.2]] -> mean = 0.3 / 4 = 0.075
    assert calculate_patient_level_mad(m_a, m_b) == 0.075
