# Chapter 3: Research Methodology

## 3.1 Patient Cohort and Data Curation
The empirical investigation utilizes the complete combined 920-record Heart Disease dataset sourced from the official University of California, Irvine (UCI) Machine Learning Repository. The cohort incorporates records from four clinical cardiology institutions:
1. **Cleveland Clinic Foundation, Ohio, USA**: 303 patient records.
2. **Hungarian Institute of Cardiology, Budapest, Hungary**: 294 patient records.
3. **University Hospital, Zurich, Switzerland**: 123 patient records.
4. **Veterans Administration Medical Center, Long Beach, California, USA**: 200 patient records.

Total records: $303 + 294 + 123 + 200 = 920$ patients.

### Target Encoding and Ground Truth
The original angiographic target variable (`num`) encodes disease severity on an ordinal scale from 0 to 4:
- `num = 0`: Absence of significant coronary artery disease (< 50% stenosis in any major vessel). Total: 411 patients (44.67%).
- `num = 1, 2, 3, 4`: Presence of coronary artery disease ($\ge 50\%$ diameter stenosis). Total: 509 patients (55.33%).

Following the reference study and clinical diagnostic conventions, the problem is formulated as binary classification:
$$y = \begin{cases} 0, & \text{if } \text{num} = 0 \text{ (Disease Absent)} \\ 1, & \text{if } \text{num} > 0 \text{ (Disease Present)} \end{cases}$$

## 3.2 Leakage-Safe Data Preprocessing
A paramount methodological requirement is the complete prevention of data leakage between training, validation, and testing partitions. All transformation statistics (medians, means, scaling parameters, imputation estimators, and one-hot encodings) were strictly computed exclusively on the designated training folds.

### Handling Physiologically Invalid Zero Measurements
In clinical medicine, resting arterial blood pressure (`trestbps`) and total serum cholesterol (`chol`) cannot physically equal zero in a viable human subject. In the raw multi-center data:
- 1 patient record contained `trestbps = 0`.
- 172 patient records contained `chol = 0` (primarily in the Switzerland and VA Long Beach cohorts, where unrecorded measurements were coded as 0).
These physiologically invalid entries were explicitly converted to missing values (`NaN`) and processed through the formal experimental imputation pipelines.

## 3.3 Experimental Missing-Data Strategies
Four imputation strategies (M1–M4) were systematically evaluated:
- **M1 (Median / Mode)**: Continuous attributes (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`) are imputed using the training fold median. Categorical attributes (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`) are imputed using the training fold mode (most frequent category). Continuous features are subsequently standardized ($z$-score), and categorical features are one-hot encoded.
- **M2 (Median / Mode + Indicators)**: Extends M1 by generating auxiliary binary indicator columns ($I_j \in \{0, 1\}$) denoting whether feature $j$ was initially missing, explicitly enabling classifiers to exploit informative missingness.
- **M3 (K-Nearest Neighbors Imputation)**: Continuous attributes are imputed using distance-weighted K-Nearest Neighbors ($k=5$) fitted on training fold records. Categorical variables are mode-imputed.
- **M4 (Iterative Imputation / MICE)**: Continuous attributes are modeled using Multivariate Imputation by Chained Equations (`IterativeImputer`), iteratively estimating missing values via Bayesian ridge regression over 10 cycles.

## 3.4 Baseline Classifier Architectures
The five supervised classifiers were instantiated with the reference study's configurations:
- **Logistic Regression**: $L_2$ regularization, $C = 1.0$, L-BFGS optimization solver, maximum iterations $= 1000$.
- **Random Forest**: $N = 200$ estimators, maximum tree depth $= 10$, minimum samples split $= 2$.
- **Support Vector Machine**: Radial Basis Function (RBF) kernel, penalty parameter $C = 1.0$, calibrated probability outputs.
- **K-Nearest Neighbors**: $k = 7$ nearest neighbors, uniform weighting, Euclidean distance metric.
- **Gradient Boosting**: $N = 200$ boosting stages, learning rate $\eta = 0.1$, maximum tree depth $= 2$.

## 3.5 Cross-Validation and Experimental Design
To avoid over-optimistic evaluation on small samples, the experimental protocol combines:
1. **Stratified 80/20 Train/Test Partition**: 736 development records and 184 held-out test records.
2. **Repeated 5-Fold Stratified Cross-Validation**: Conducted across 3 distinct random seeds ($S \in \{42, 101, 2024\}$).
Total experimental scale:
$$5 \text{ models} \times 4 \text{ imputation strategies} \times 3 \text{ random seeds} \times 5 \text{ folds} = 300 \text{ model evaluations}$$

## 3.6 Performance Metrics and Statistical Significance
Classification performance is evaluated across seven complementary metrics:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Sensitivity (Recall)} = \frac{TP}{TP + FN}, \quad \text{Specificity} = \frac{TN}{TN + FP}$$
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

Statistical comparisons between imputation strategies and baseline models were performed using **two-tailed paired $t$-tests** and non-parametric **Wilcoxon signed-rank tests** across identical cross-validation folds, accompanied by 95% confidence intervals.

## 3.7 Quantitative Explanation Stability Metrics
Explanation stability was assessed on a fixed reference test cohort ($N = 100$ records) across repeated splits:
1. **Spearman Rank Correlation ($r_s$)**: Measures the monotonic rank consistency of global feature hierarchies:
   $$r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
2. **Top-$k$ Jaccard Similarity Index ($J_k$)**: Measures the proportion of shared critical features in the top-$k$ most important rankings:
   $$J_k(A, B) = \frac{|A_k \cap B_k|}{|A_k \cup B_k|}, \quad k \in \{3, 5\}$$
3. **Patient-Level Mean Absolute Difference (MAD)**: Quantifies the average divergence in individual feature attributions:
   $$\text{MAD} = \frac{1}{N \cdot M} \sum_{i=1}^N \sum_{j=1}^M |\phi_{i,j}^{(A)} - \phi_{i,j}^{(B)}|$$
4. **Attribution Correlation**: Pearson correlation between patient attribution vectors across model runs.
