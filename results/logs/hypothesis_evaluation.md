# Empirical Research Findings & Hypothesis Testing

## Research Hypotheses Evaluation

### Hypothesis H1: Missing-Data Strategies & Predictive Performance
- **Statement**: Missing-data handling strategies may produce measurable differences in predictive performance.
- **Empirical Status**: ✅ SUPPORTED
- **Evidence**: Observed cross-validation accuracy differences across M1 (median), M2 (indicator), M3 (KNN), and M4 (MICE). Iterative and KNN imputation yielded minor improvements over simple median, particularly for models sensitive to covariance structure such as Logistic Regression and SVM.

---

### Hypothesis H2: Explanation Sensitivity
- **Statement**: SHAP explanations may vary across models, data splits, and imputation strategies.
- **Empirical Status**: ✅ SUPPORTED
- **Evidence**: Patient-level and global SHAP attributions demonstrated measurable variance. Random Forest maintained high rank correlation (0.987), but top-5 feature overlap was not 100% (75.6%), indicating attribution sensitivity to training subsamples and feature representations.

---

### Hypothesis H3: Predictive Accuracy vs Explanation Stability Trade-off
- **Statement**: The model with the best predictive performance may not have the most stable explanations.
- **Empirical Status**: ✅ SUPPORTED
- **Evidence**: While ensemble methods (Gradient Boosting and Random Forest) achieved the highest predictive accuracy and ROC-AUC, linear models exhibited competitive or higher attribution stability due to the strict monotonic nature of coefficients, confirming that predictive optimality does not guarantee explanation invariance.

---

## Explanation Stability Summary Table

| Model Family | Spearman Rank Correlation | Top-5 Jaccard Similarity | Patient-Level Attribution Correlation | Patient-Level MAD |
|---|---|---|---|---|
| **Gradient Boosting** | 0.8883 (±0.0632) | 0.7556 | 0.8946 | 0.0552 |
| **Logistic Regression** | 0.9592 (±0.0257) | 0.8222 | 0.9553 | 0.0334 |
| **Random Forest** | 0.9868 (±0.0056) | 0.7556 | 0.9723 | 0.0041 |
