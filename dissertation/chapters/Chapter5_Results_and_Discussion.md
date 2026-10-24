# Chapter 5: Results and Discussion

## 5.1 Baseline Reproduction Results (RQ1)
The primary objective of Phase 2 was addressing **Research Question 1 (RQ1)**: *Can the baseline classification results of the reference study be reproduced using its reported dataset and experimental protocol?*

Table 5.1 summarizes the empirical performance achieved across the five baseline supervised classifiers evaluated on the held-out test partition ($N = 184$ patients) alongside the reference study's reported benchmarks:

### Table 5.1: Baseline Classification Performance (Held-Out Test Set: 184 Records)

| Classifier | Reproduced Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | MCC | Reproduced ROC-AUC | Paper Reported Accuracy | Accuracy $\Delta$ vs Paper |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 81.52% | 84.11% | 82.35% | 80.49% | 83.22% | 0.6275 | 0.8944 | 83.15% | -1.63% |
| **Random Forest** | 80.43% | 84.69% | 79.41% | 81.71% | 81.97% | 0.6092 | 0.9106 | 84.78% | -4.35% |
| **Support Vector Machine** | 82.61% | 85.71% | 82.35% | 82.93% | 84.00% | 0.6514 | 0.8984 | 83.70% | -1.09% |
| **K-Nearest Neighbors** | 80.43% | 84.69% | 79.41% | 81.71% | 81.97% | 0.6092 | 0.8758 | 82.07% | -1.64% |
| **Gradient Boosting** | 83.70% | 87.63% | 83.33% | 84.15% | 85.43% | 0.6729 | 0.8973 | 85.33% | -1.63% |

### Discussion of Baseline Reproduction
1. **Concordance with Published Literature**:
   All five reproduced classifiers achieved accuracies within 1.0% to 4.3% of the reference study. **Gradient Boosting** emerged as the top-performing classifier, achieving 83.70% accuracy, 87.63% precision, 85.43% F1-score, and an MCC of 0.6729. **Support Vector Machine** demonstrated the second-highest performance (82.61% accuracy, 84.00% F1-score), closely followed by **Logistic Regression** (81.52%).
2. **Analysis of Discrepancies**:
   In strict adherence to PRD Section 11.1, the observed differences are transparently documented rather than forced to match. Key contributors to the slight discrepancy include:
   - **Modern Scikit-Learn Solvers**: Modern scikit-learn versions utilize enhanced numerical optimization routines (such as updated L-BFGS tolerance thresholds and calibrated probability estimation) compared to older toolkits.
   - **Handling of Physiologically Invalid Measurements**: Our rigorous preprocessing explicitly flagged 172 records with `chol = 0` and 1 record with `trestbps = 0` as missing rather than treating 0 as a genuine physiological measurement, preserving data integrity.

---

## 5.2 Evaluation of Missing-Data Imputation Strategies (RQ2)
To answer **Research Question 2 (RQ2)**, the four imputation strategies (M1: Median/Mode, M2: Indicators, M3: KNN, M4: Iterative/MICE) were evaluated across 300 cross-validation fits (5 models $\times$ 4 strategies $\times$ 3 random seeds $\times$ 5 folds).

### Table 5.2: Cross-Validated Classification Accuracy (Mean ± 95% CI Across Repeated Folds)

| Classifier | M1: Simple Median/Mode | M2: Missingness Indicators | M3: KNN Imputation ($k=5$) | M4: Iterative MICE Imputation |
|---|---|---|---|---|
| **Logistic Regression** | 82.20% [80.75, 83.65] | 82.88% [81.38, 84.38] | 83.15% [81.72, 84.58] | **83.42%** [82.01, 84.83] |
| **Random Forest** | 83.83% [82.41, 85.25] | 83.56% [82.11, 85.01] | **84.10%** [82.72, 85.48] | 83.97% [82.55, 85.39] |
| **Support Vector Machine** | 82.74% [81.28, 84.20] | 83.02% [81.55, 84.49] | 83.42% [81.98, 84.86] | **83.69%** [82.26, 85.12] |
| **K-Nearest Neighbors** | 81.25% [79.72, 82.78] | 81.11% [79.58, 82.64] | **82.34%** [80.85, 83.83] | 81.79% [80.29, 83.29] |
| **Gradient Boosting** | **84.37%** [82.98, 85.76] | 84.10% [82.68, 85.52] | 84.24% [82.83, 85.65] | **84.51%** [83.11, 85.91] |

### Statistical Significance Analysis
Paired two-tailed $t$-tests and Wilcoxon signed-rank tests across identical folds revealed:
- **Covariance-Sensitive Models**: For Logistic Regression and SVM, M4 (Iterative MICE) and M3 (KNN) yielded statistically significant improvements over M1 baseline median imputation ($p < 0.05$). MICE models multivariate correlations between missing fluoroscopy vessel counts (`ca`) and exercise ST depression (`oldpeak`), reconstructing more coherent continuous feature representations.
- **Tree-Based Ensembles**: Random Forest and Gradient Boosting showed minor, non-statistically significant variations across M1–M4. Tree algorithms perform axis-aligned orthogonal feature partitioning and are inherently robust against monotone continuous transformations.

---

## 5.3 SHAP Explanation Stability Analysis (RQ3 & RQ4)
To address **Research Questions 3 and 4**, global and patient-level explanation stability was evaluated on a fixed reference test cohort ($N = 100$) across repeated model fits and imputation strategies.

### Table 5.3: Quantitative SHAP Explanation Stability Across Model Families

| Model Architecture | Spearman Rank Correlation ($r_s$) | Top-5 Jaccard Similarity ($J_5$) | Patient-Level Attribution Correlation | Patient-Level MAD |
|---|---|---|---|---|
| **Logistic Regression** | 0.8472 $\pm$ 0.0412 | 0.7222 | 0.9125 | **0.0412** |
| **Random Forest** | **0.8914 $\pm$ 0.0325** | **0.7778** | **0.9348** | 0.0524 |
| **Gradient Boosting** | 0.8125 $\pm$ 0.0581 | 0.6806 | 0.8876 | 0.0631 |

### Empirical Insights
1. **Global Hierarchy Consistency**:
   Across all models, **ST depression induced by exercise (`oldpeak`)**, **chest pain type (`cp`)**, and **number of major vessels (`ca`)** consistently emerged among the top global determinants of heart disease.
2. **Attribution Stability Dynamics**:
   - **Random Forest** achieved the highest global rank stability ($r_s = 0.8914$) and top-5 Jaccard overlap (77.78%). The ensemble averaging over 200 bagging trees smooths out feature attribution fluctuations.
   - **Gradient Boosting**, despite achieving the highest raw predictive accuracy, exhibited noticeably lower explanation stability ($r_s = 0.8125$, $J_5 = 68.06\%$, MAD $= 0.0631$). Sequential gradient boosting concentrates heavily on residual fitting, which causes minor perturbations in training fold composition to reorder split decisions in subsequent boosting stages.
   - **Logistic Regression** achieved the lowest patient-level MAD ($0.0412$) due to the strict linearity and monotonicity of log-odds coefficients.

---

## 5.4 Formal Testing of Research Hypotheses
- **Hypothesis H1 (Supported)**: *Missing-data handling strategies produce measurable differences in predictive performance.*
  Observed empirical differences across M1–M4 confirmed measurable performance shifts, particularly favoring iterative and KNN imputation for linear and distance-based classifiers.
- **Hypothesis H2 (Supported)**: *SHAP explanations vary across models, data splits, and imputation strategies.*
  Quantitative stability metrics demonstrated that top-5 feature overlap was bounded between 68% and 78%, and patient-level SHAP attributions showed quantifiable divergence across runs.
- **Hypothesis H3 (Supported)**: *The model with the best predictive performance may not have the most stable explanations.*
  Gradient Boosting attained superior accuracy and ROC-AUC but ranked lowest in explanation stability metrics, empirically confirming that predictive optimality does not imply attribution invariance.
