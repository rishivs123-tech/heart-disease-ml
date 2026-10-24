# Experimental Protocol & Methodology

## 1. Experimental Objectives
This document establishes the controlled, leakage-safe experimental protocol designed to evaluate:
1. **RQ1**: Baseline reproduction against the reference study on the 920-record combined heart disease cohort.
2. **RQ2**: Comparative impact of four missing-data imputation strategies (M1–M4).
3. **RQ3**: Stability of global feature rankings and patient-level SHAP attributions across repeated splits.
4. **RQ4**: The empirical trade-off between predictive accuracy and explanation stability.

---

## 2. Dataset Protocol
- **Cohort**: 920 patient records from Cleveland (303), Hungary (294), Switzerland (123), VA Long Beach (200).
- **Features**: 13 clinical attributes (demographics, symptoms, hemodynamics, ECG, exercise stress, scintigraphy).
- **Target**: Binary classification ($0 = \text{Absent}$, $1 = \text{Present}$, corresponding to $\ge 50\%$ diameter stenosis).
- **Invalid Measurement Policy**: Records with `trestbps = 0` (1 record) and `chol = 0` (172 records) are flagged as missing (`NaN`) and imputed within training folds.

---

## 3. Imputation Strategies (M1–M4)
- **M1**: Simple Median for numeric features + Most Frequent for categorical features.
- **M2**: Median/Mode augmented with `MissingIndicator` features.
- **M3**: Distance-Weighted K-Nearest Neighbors Imputation ($k=5$) for continuous features, mode for categorical.
- **M4**: Multivariate Imputation by Chained Equations (Iterative / MICE, 10 cycles) for continuous features, mode for categorical.

---

## 4. Evaluation Scale (300 Evaluations)
- **Models**: Logistic Regression, Random Forest, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting.
- **Imputation Strategies**: M1, M2, M3, M4.
- **Repeated Seeds**: 42, 101, 2024.
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the 736 development records.
- **Total Evaluations**: $5 \times 4 \times 3 \times 5 = 300$ model-fit evaluations.
- **Held-Out Test Set**: 184 records (20% stratified holdout) preserved strictly for final generalization evaluation.

---

## 5. Statistical Significance Framework
- Paired comparisons between imputation strategies (M2 vs M1, M3 vs M1, M4 vs M1) and classifiers across matched folds.
- **Tests**: Two-tailed paired $t$-test and Wilcoxon signed-rank test.
- **Threshold**: Statistical significance established at $\alpha = 0.05$.
- **Intervals**: 95% Confidence Intervals computed using the Student's $t$-distribution.

---

## 6. SHAP Stability Measurement
- **Reference Cohort**: Fixed sample of 100 held-out test patients.
- **Global Metric**: Spearman rank correlation ($r_s$) and Top-5 Jaccard similarity ($J_5$).
- **Local Metric**: Patient-level Mean Absolute Difference (MAD) and attribution vector Pearson correlation.
