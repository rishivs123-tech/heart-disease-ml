# 🫀 Heart Disease Prediction Using Machine Learning
### **Proposed Research Extension**: Explainable and Robust Heart Disease Prediction: Evaluating Missing-Data Strategies and SHAP Explanation Stability

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io)
[![XAI: SHAP](https://img.shields.io/badge/XAI-SHAP%20Explainability-orange.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Pytest-12%20Passed-brightgreen.svg)](tests/)

---

## 📌 1. Project Overview & Research Contribution
This repository contains the complete Master of Engineering (ME) dissertation research software prototype and scientific reproducibility framework. 

The project accomplishes two primary research goals:
1. **Faithful Baseline Reproduction**: Reproduces the comparative study on heart disease prediction using the benchmark **920-record combined UCI Heart Disease cohort** (Cleveland, Hungary, Switzerland, VA Long Beach) across five supervised learning classifiers:
   - Logistic Regression (LR)
   - Random Forest (RF)
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting (GB)
2. **Explainable AI & Robustness Extension**:
   - Compares **four missing-data imputation strategies**:
     - **M1**: Simple Median / Mode Imputation
     - **M2**: Median / Mode with Binary Missingness Indicators
     - **M3**: Distance-Weighted K-Nearest Neighbors Imputation (KNN, $k=5$)
     - **M4**: Multivariate Imputation by Chained Equations (Iterative / MICE)
   - Investigates **SHAP (SHapley Additive exPlanations) Explanation Stability** across repeated stratified data splits and model runs.
   - Quantifies whether predictive optimality correlates with attribution stability across model families.
3. **Interactive Local Demonstrator**: Provides a full-featured, dark/light theme modern **Streamlit Web Application** featuring real-time clinical prediction, dynamic patient-level SHAP waterfall charts, model benchmark comparisons, and dissertation findings.

---

## 📦 2. What to Do with the Provided ZIP File (`heart-disease-ml.zip`)

If you downloaded the project as a `.zip` archive:

### Step 1: Extract the Archive
Extract the contents of `heart-disease-ml.zip` into your preferred project directory on your computer:
- **Windows**: Right-click `heart-disease-ml.zip` ➔ **Extract All...** ➔ Choose destination (e.g., `C:\Projects\heart-disease-ml`)
- **Mac/Linux**:
  ```bash
  unzip heart-disease-ml.zip -d heart-disease-ml
  cd heart-disease-ml
  ```

### Step 2: Open in VS Code (or your preferred IDE)
Launch Visual Studio Code and open the extracted folder:
```bash
code .
```
*(Or in VS Code: **File** ➔ **Open Folder...** ➔ Select the extracted folder).*

### Step 3: Set Up Python Virtual Environment
Open a terminal in the project root:
```bash
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
*(All exact package versions are also recorded in `requirements-lock.txt` for absolute reproducibility).*

---

## 🚀 3. Quick Start: Launching the Streamlit Web Application

To launch the interactive multi-page web application:
```bash
streamlit run app/Home.py
```
Your default browser will automatically open to `http://localhost:8501`.

### 🖥️ Application Pages Overview:
1. **Home (`app/Home.py`)**: Executive summary, research questions, system architecture, and 920-record dataset inspection.
2. **1. Clinical Prediction (`app/pages/1_Prediction.py`)**: Interactive diagnostic prediction form with clinical defaults, input validation, risk gauge charts, and preset patient case studies.
3. **2. Prediction Explanation (`app/pages/2_Explanation.py`)**: Local patient-level SHAP waterfall attributions, color-coded risk factors (red = increases risk, green = protective), and clinical narrative generation.
4. **3. Model Performance (`app/pages/3_Model_Performance.py`)**: Empirical reproduction benchmark table, confusion matrices, ROC curves, cross-validation metrics, and statistical significance tests.
5. **4. Research Findings (`app/pages/4_Research_Findings.py`)**: Quantitative SHAP stability metrics (Spearman rank, Top-5 Jaccard, MAD), global feature rankings, empirical hypothesis evaluation (H1–H3), and clinical limitations.

---

## 🧪 4. Running the Automated Test Suite

Run the full automated test suite with pytest:
```bash
# Windows PowerShell:
$env:PYTHONPATH="."
pytest

# Linux / macOS:
PYTHONPATH=. pytest
```
**Test Results**: 12/12 unit and integration tests passing covering data validation, preprocessing, M1–M4 imputation, model factories, metrics calculation, and SHAP stability algorithms.

---

## 🔬 5. Reproducing the Experiments from Scratch

All baseline and experimental metrics, figures, logs, and trained model pipelines are **precomputed and saved** in the repository. However, you can regenerate everything from scratch at any time:

### 1. Ingest and Validate Dataset (920 Records)
```bash
python src/heart_disease/data/load_data.py
python src/heart_disease/data/validate_data.py
```
- Fetches raw records from the 4 official UCI cardiology subsets (Cleveland: 303, Hungary: 294, Switzerland: 123, VA Long Beach: 200).
- Flags physiologically impossible measurements (`trestbps=0`, `chol=0`).
- Generates `results/logs/data_quality_report.json` and `.md`.

### 2. Train All Baseline Models
```bash
python -m src.heart_disease.models.train
```
- Trains all 5 classifiers on 736 development records and serializes fitted pipelines to `artifacts/models/`.

### 3. Run Baseline Reproduction Experiment
```bash
python -m src.heart_disease.experiments.run_baseline
```
- Evaluates 80/20 train/test split and 5-fold cross-validation.
- Exports `results/metrics/baseline_reproduction.csv` and publication figures in `results/figures/`.

### 4. Run Missing-Data Imputation Experiments (300 Evaluations)
```bash
python -m src.heart_disease.experiments.run_imputation
```
- Evaluates 5 models $\times$ 4 strategies $\times$ 3 random seeds $\times$ 5 folds = 300 evaluations.
- Performs paired $t$-tests and Wilcoxon signed-rank tests.
- Exports `results/metrics/imputation_summary.csv` and `results/metrics/statistical_comparisons.csv`.

### 5. Run SHAP Explanation Stability Analysis
```bash
python -m src.heart_disease.experiments.run_stability
```
- Computes Spearman rank correlation, top-5 Jaccard similarity, and patient-level MAD.
- Formulates formal empirical evaluation of Hypotheses H1, H2, and H3.
- Exports `results/logs/hypothesis_evaluation.json` and `.md`.

---

## 📁 6. Repository File and Folder Structure

```
PROJECT-X/
├── data/
│   ├── raw/
│   │   └── heart_disease_combined_raw.csv    # Raw concatenated 920-record dataset
│   ├── processed/
│   │   ├── heart_disease_cleaned.csv         # Cleaned dataset (920 rows)
│   │   ├── train.csv                         # Stratified 80% training split (736 rows)
│   │   └── test.csv                          # Stratified 20% held-out test split (184 rows)
│   └── README.md                             # Dataset documentation and clinical dictionary
├── configs/
│   ├── baseline.yaml                         # 80/20 split and baseline parameters
│   ├── models.yaml                           # Model configurations and hyperparameter grids
│   └── experiments.yaml                      # Imputation strategies (M1-M4) and stability settings
├── notebooks/
│   ├── 01_dataset_audit.ipynb                # Dataset verification & data quality audit
│   ├── 02_exploratory_analysis.ipynb         # Clinical EDA and biomarker correlations
│   ├── 03_baseline_reproduction.ipynb        # 5 baseline models on 80/20 split & 5-fold CV
│   ├── 04_imputation_experiments.ipynb       # M1-M4 missing data comparison across 300 fits
│   └── 05_shap_stability.ipynb               # SHAP explainability and stability analysis
├── src/
│   └── heart_disease/
│       ├── data/
│       │   ├── load_data.py                  # Multi-center dataset ingestion and zero-cleaning
│       │   ├── validate_data.py              # Schema assertions and quality report generator
│       │   └── preprocess.py                 # Leakage-safe ColumnTransformers (M1-M4)
│       ├── models/
│       │   ├── build_models.py               # Classifier factory (LR, RF, SVM, KNN, GB)
│       │   ├── train.py                      # Pipeline fitting, test evaluation, persistence
│       │   └── tune_models.py                # GridSearchCV hyperparameter tuning
│       ├── evaluation/
│       │   ├── metrics.py                    # Accuracy, Precision, Recall, Specificity, F1, MCC, AUC
│       │   ├── cross_validation.py           # Leakage-safe repeated Stratified K-Fold CV
│       │   └── statistics.py                 # Paired t-tests, Wilcoxon signed-rank, 95% CIs
│       ├── explainability/
│       │   ├── shap_explainer.py             # Model-specific SHAP explainers (Tree, Linear, Kernel)
│       │   └── stability.py                  # Spearman rank, Top-k Jaccard, Patient-level MAD
│       └── experiments/
│           ├── run_baseline.py               # Baseline reproduction script
│           ├── run_imputation.py             # Missing-data 300 evaluations runner
│           └── run_stability.py              # SHAP stability and hypothesis testing runner
├── artifacts/
│   ├── models/                               # Serialized Joblib pipelines (*.joblib)
│   └── metadata/                             # Training parameters and test metrics (*_metadata.json)
├── results/
│   ├── metrics/                              # CSV experiment tables (baseline, imputation, stability)
│   ├── figures/                              # High-resolution plots (ROC, confusion matrix, heatmaps)
│   ├── shap/                                 # SHAP explanation exports
│   └── logs/                                 # Data quality report & hypothesis evaluation
├── app/
│   ├── Home.py                               # Streamlit application main page
│   ├── utils.py                              # Styling, cached loaders, and clinical presets
│   └── pages/
│       ├── 1_Prediction.py                   # Patient prediction interface
│       ├── 2_Explanation.py                  # Interactive SHAP waterfall studio
│       ├── 3_Model_Performance.py            # Comparative benchmark dashboard
│       └── 4_Research_Findings.py            # Stability findings and hypothesis synthesis
├── dissertation/
│   ├── chapters/
│   │   ├── Chapter1_Introduction.md          # Background, problem statement, objectives, scope
│   │   ├── Chapter2_Literature_Review.md     # Tabular ML, missing data, XAI, stability
│   │   ├── Chapter3_Research_Methodology.md  # Experimental design, M1-M4, SHAP stability
│   │   ├── Chapter4_System_Design_and_Implementation.md # 5-layer architecture & testing
│   │   ├── Chapter5_Results_and_Discussion.md # Baseline reproduction, imputation, stability
│   │   ├── Chapter6_Conclusion_and_Future_Work.md # Contributions, limitations, future work
│   │   └── Appendices.md                     # Data dictionary, setup & reproduction instructions
│   └── Full_Dissertation.md                  # Unified comprehensive academic dissertation draft
├── tests/
│   ├── test_data.py                          # Tests for data loading and acceptance criteria
│   ├── test_preprocessing.py                 # Tests for M1-M4 imputation and feature names
│   ├── test_models.py                        # Tests for classifiers and metric computations
│   └── test_explainability.py                # Tests for SHAP stability mathematical functions
├── requirements.txt                          # Dependencies with version ranges
├── requirements-lock.txt                     # Frozen exact package versions
├── .gitignore                                # Git ignore file
└── README.md                                 # This documentation file
```

---

## 📑 7. Master Dissertation Documentation

The complete academic dissertation is provided in both modular chapter format and as a unified master manuscript:
- **Unified Master Dissertation**: [`dissertation/Full_Dissertation.md`](dissertation/Full_Dissertation.md)
- **Individual Chapters**:
  - [Chapter 1: Introduction](dissertation/chapters/Chapter1_Introduction.md)
  - [Chapter 2: Literature Review](dissertation/chapters/Chapter2_Literature_Review.md)
  - [Chapter 3: Research Methodology](dissertation/chapters/Chapter3_Research_Methodology.md)
  - [Chapter 4: System Design and Implementation](dissertation/chapters/Chapter4_System_Design_and_Implementation.md)
  - [Chapter 5: Results and Discussion](dissertation/chapters/Chapter5_Results_and_Discussion.md)
  - [Chapter 6: Conclusion and Future Work](dissertation/chapters/Chapter6_Conclusion_and_Future_Work.md)
  - [Appendices](dissertation/chapters/Appendices.md)

---

## ⚖️ 8. Research Prototype and Medical Disclaimer

> **IMPORTANT CLINICAL & ETHICAL NOTICE**:  
> This software system is an academic research prototype developed as part of a Master of Engineering dissertation. It is intended solely for scientific investigation, machine learning benchmarking, and explainability evaluation.  
> - It is **NOT** a certified medical diagnostic device.  
> - It has not undergone prospective clinical validation or regulatory clearance.  
> - It must **NEVER** be used to provide real-world clinical diagnoses, medical recommendations, or autonomous treatment planning.
