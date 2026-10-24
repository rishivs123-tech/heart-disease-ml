# Chapter 1: Introduction

## 1.1 Research Background
Cardiovascular diseases (CVDs) represent the leading cause of global mortality, accounting for an estimated 17.9 million deaths annually according to the World Health Organization. Early and accurate detection of ischemic heart disease can dramatically improve patient survival rates and enable preventive therapeutic interventions before irreversible myocardial infarction occurs. Over recent decades, clinical medicine has accumulated substantial structured electronic health records, encompassing routine diagnostic measurements such as electrocardiography (ECG), exercise stress testing, cardiac fluoroscopy, and serum lipid profiling.

Machine learning (ML) algorithms offer tremendous promise in extracting predictive patterns from multi-attribute clinical datasets. By learning complex non-linear interactions across demographic and physiological features, supervised learning classifiers can estimate coronary artery disease risk with high statistical precision.

## 1.2 Problem Statement
Despite extensive published research applying machine learning algorithms to cardiology datasets, three critical challenges persist in the academic literature:

1. **The Pervasiveness of Incomplete and Corrupted Clinical Data**:
   Real-world patient cohorts frequently contain substantial missing values. In multi-center datasets, certain specialized diagnostic procedures (such as radioactive thallium scintigraphy or invasive coronary fluoroscopy) are ordered selectively based on patient symptom severity, resulting in informative missingness patterns. Furthermore, historical recording practices often introduced physiologically impossible placeholder values (such as zero blood pressure or zero serum cholesterol). Standard research implementations frequently resort to naive listwise deletion or simple mean imputation, which can bias parameter estimates and cause information leakage.

2. **The "Black Box" Interpretability Barrier in Clinical Decision Support**:
   Advanced non-linear algorithms, including ensemble decision trees (Random Forest, Gradient Boosting) and non-linear Support Vector Machines (SVM), operate as opaque predictive systems. Medical practitioners cannot ethically or legally adopt algorithmic recommendations without understanding the patient-specific physiological factors driving a given risk assessment. Explainable Artificial Intelligence (XAI) frameworks, particularly SHAP (SHapley Additive exPlanations) grounded in cooperative game theory, have emerged to address this need.

3. **The Unquantified Instability of Feature Attributions**:
   A fundamental yet frequently overlooked research gap is **explanation stability**. While extensive studies evaluate predictive metric consistency (such as cross-validated accuracy), few examine whether model explanations remain invariant across repeated data splits, hyperparameter runs, and alternative missing-data handling pipelines. An algorithm that yields high diagnostic accuracy but produces contradictory clinical explanations when retrained on a slightly perturbed data subset poses severe reliability and trust hazards in healthcare.

## 1.3 Research Objectives
This dissertation aims to systematically reproduce a benchmark comparative study on heart disease prediction and formulate an original scientific research extension centered on missing-data strategies and explanation stability. The primary objectives are:

1. **Rigorous Baseline Reproduction**: Faithfully reproduce the experimental protocol of the reference research paper using the combined 920-record UCI Heart Disease cohort and five supervised classifiers (Logistic Regression, Random Forest, Support Vector Machine, K-Nearest Neighbors, and Gradient Boosting) across a stratified 80/20 train/test split and 5-fold cross-validation.
2. **Systematic Evaluation of Missing-Data Strategies**: Implement and compare four distinct missing-data handling paradigms:
   - **M1**: Simple Median / Mode imputation.
   - **M2**: Median / Mode imputation augmented with Missingness Indicators.
   - **M3**: Distance-weighted K-Nearest Neighbors (KNN, $k=5$) imputation.
   - **M4**: Multivariate Imputation by Chained Equations (Iterative / MICE).
3. **Quantification of SHAP Explanation Stability**: Implement model-specific SHAP explainers (TreeSHAP, LinearSHAP, KernelSHAP) and quantitatively measure attribution stability across repeated runs and data splits using:
   - Spearman rank correlation of global feature importance hierarchies.
   - Top-$k$ Jaccard similarity indices ($k=3, 5$).
   - Patient-level Mean Absolute Difference (MAD) and attribution vector correlation.
4. **Empirical Hypothesis Testing**: Formulate and test three core research hypotheses (H1, H2, H3) concerning the interplay between predictive accuracy, imputation method, and explanation stability.
5. **Interactive Research Software Prototype**: Develop a modern, accessible Streamlit web application providing real-time diagnostic prediction, dynamic local SHAP waterfall visualizations, comparative model benchmarks, and comprehensive research summaries.

## 1.4 Research Scope and Delimitations
- **Included in Scope**: Multi-center raw data validation, leakage-safe preprocessing pipelines, 5 baseline classifiers, 4 imputation strategies, repeated 5-fold cross-validation (300 model fits), global and local SHAP interpretability, statistical hypothesis testing, and local web demonstration.
- **Excluded from Scope**: Autonomous real-time hospital integration, electronic health record (EHR) database synchronization, prospective clinical trials, or prescriptive treatment generation. This project is strictly designated as an academic research prototype.
