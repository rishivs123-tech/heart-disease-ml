# Complete Project Reference & User Manual

## Heart Disease Prediction Using Machine Learning
**Research Extension:** Explainable and Robust Heart Disease Prediction: Evaluating Missing-Data Strategies and SHAP Explanation Stability

---

## 1. Project Overview & Research Goals

This project is an academic research software prototype and Master of Engineering (ME) dissertation framework. It investigates heart disease classification using structured clinical data, comparing 5 machine learning models, 4 missing-data imputation strategies, and evaluating the stability of Explainable AI (SHAP) explanations.

### Primary Objectives
1. **Baseline Reproduction**: Reproduce published benchmark results for heart disease prediction across five supervised machine learning models using the combined 920-patient UCI Heart Disease dataset.
2. **Missing-Data Comparison**: Systematically evaluate four different data imputation strategies across 300 cross-validation fits to determine their impact on predictive accuracy.
3. **Explainable AI (XAI) & Stability**: Use SHAP (SHapley Additive exPlanations) to explain individual patient risk predictions and evaluate whether explanations remain stable across data splits and model runs.
4. **Interactive Software Prototype**: Provide an intuitive, multi-page Streamlit web application allowing clinicians and researchers to test predictions, view local explanations, inspect performance benchmarks, and review research findings.

---

## 2. Project File Structure & Where to Find Everything

Below is the complete file directory layout. Every file has a specific role:

```
PROJECT-X/
│
├── data/                                    # Dataset storage
│   ├── raw/
│   │   └── heart_disease_combined_raw.csv   # Raw combined 920 records from 4 hospitals
│   ├── processed/
│   │   ├── heart_disease_cleaned.csv        # Cleaned dataset (920 records, invalid zeros handled)
│   │   ├── train.csv                        # Training split (736 records, 80%)
│   │   └── test.csv                         # Held-out testing split (184 records, 20%)
│   └── README.md                            # Clinical data dictionary and feature explanations
│
├── src/                                     # Core Python source code
│   └── heart_disease/
│       ├── data/
│       │   ├── load_data.py                 # Downloads raw data and converts invalid zeros to NaN
│       │   ├── validate_data.py             # Checks dataset shape, data types, and target distribution
│       │   └── preprocess.py                # Imputation pipelines (M1, M2, M3, M4) & ColumnTransformers
│       ├── models/
│       │   ├── build_models.py              # Classifier factory (LR, RF, SVM, KNN, Gradient Boosting)
│       │   ├── train.py                     # Fits pipelines, evaluates test set, saves .joblib models
│       │   └── tune_models.py               # Hyperparameter search using GridSearchCV
│       ├── evaluation/
│       │   ├── metrics.py                   # Calculates Accuracy, Precision, Recall, Specificity, F1, MCC, AUC
│       │   ├── cross_validation.py          # Leakage-safe repeated Stratified K-Fold cross-validation
│       │   └── statistics.py                # Paired t-test, Wilcoxon signed-rank test, 95% confidence intervals
│       ├── explainability/
│       │   ├── shap_explainer.py            # Generates Tree, Linear, and Kernel SHAP explanations
│       │   └── stability.py                 # Computes Spearman rank, Top-5 Jaccard, and patient-level MAD
│       └── experiments/
│           ├── run_baseline.py              # Runs 80/20 test split & 5-fold CV baseline reproduction
│           ├── run_imputation.py            # Executes 300 evaluations (5 models x 4 strategies x 3 seeds x 5 folds)
│           └── run_stability.py             # Evaluates SHAP stability and verifies hypotheses H1, H2, H3
│
├── artifacts/                               # Saved models and metadata
│   ├── models/                              # Trained model pipeline files (*.joblib)
│   │   ├── random_forest_m1.joblib
│   │   ├── gradient_boosting_m1.joblib
│   │   ├── logistic_regression_m1.joblib
│   │   ├── support_vector_machine_m1.joblib
│   │   └── k_nearest_neighbors_m1.joblib
│   └── metadata/                            # Training configuration and test performance summaries (*.json)
│
├── results/                                 # Experimental outputs & generated graphics
│   ├── metrics/                             # CSV tables summarizing all results
│   │   ├── baseline_reproduction.csv        # Baseline reproduction scores vs paper benchmarks
│   │   ├── imputation_summary.csv           # Mean & std scores across 300 imputation evaluations
│   │   ├── statistical_comparisons.csv      # Paired t-test and Wilcoxon p-values between strategies
│   │   └── stability_summary.csv            # Quantitative SHAP stability scores per model
│   ├── figures/                             # Visual plots for publication & UI display
│   │   ├── baseline_roc_curves.png          # ROC curves for all 5 classifiers
│   │   ├── baseline_confusion_matrices.png  # Confusion matrices on the 184 test patients
│   │   ├── imputation_comparison_boxplot.png# Accuracy spread across M1-M4
│   │   ├── shap_rf_summary.png              # Global SHAP feature importance plot
│   │   └── shap_stability_comparison.png    # Explanation stability comparison bar chart
│   └── logs/
│       ├── data_quality_report.json         # Dataset verification report
│       └── hypothesis_evaluation.json       # Formal scientific conclusions for H1, H2, H3
│
├── app/                                     # Streamlit Web Application
│   ├── Home.py                              # Main application page (Overview & dataset inspection)
│   ├── utils.py                             # Styling, cached model loaders, and patient presets
│   └── pages/
│       ├── 1_Prediction.py                  # Interactive clinical diagnosis form & risk gauge
│       ├── 2_Explanation.py                 # Local SHAP waterfall chart & clinical explanation
│       ├── 3_Model_Performance.py           # Benchmark tables, ROC curves, confusion matrices
│       └── 4_Research_Findings.py           # Hypothesis test outcomes & stability metrics
│
├── dissertation/                            # Academic manuscript
│   ├── chapters/
│   │   ├── Chapter1_Introduction.md
│   │   ├── Chapter2_Literature_Review.md
│   │   ├── Chapter3_Research_Methodology.md
│   │   ├── Chapter4_System_Design_and_Implementation.md
│   │   ├── Chapter5_Results_and_Discussion.md
│   │   ├── Chapter6_Conclusion_and_Future_Work.md
│   │   └── Appendices.md
│   └── Full_Dissertation.md                 # Complete unified academic dissertation manuscript
│
├── tests/                                   # Automated test suite (12 tests)
│   ├── test_data.py                         # Checks data loading and invalid zero handling
│   ├── test_preprocessing.py                # Verifies that M1-M4 imputation leaves 0 NaNs
│   ├── test_models.py                       # Verifies model factories and evaluation metrics
│   └── test_explainability.py               # Tests Spearman rank, Jaccard, and MAD calculation
│
├── configs/                                 # Configuration YAML files
│   ├── baseline.yaml                        # Feature list, data splits, and baseline hyperparameters
│   ├── models.yaml                          # Model definitions and hyperparameter tuning grids
│   └── experiments.yaml                     # Imputation strategy specifications and stability parameters
│
├── notebooks/                               # Jupyter notebooks for step-by-step exploration
│   ├── 01_dataset_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_baseline_reproduction.ipynb
│   ├── 04_imputation_experiments.ipynb
│   └── 05_shap_stability.ipynb
│
├── requirements.txt                         # Python library dependencies
├── requirements-lock.txt                    # Pinned package versions
├── show-stats.py                            # Utility script to print statistical test results
└── README.md                                # Repository overview and execution guide
```

---

## 3. Step-by-Step Setup and Installation Guide

### Step 1: System Requirements
- Operating System: Windows, macOS, or Linux.
- Python Version: Python 3.10, 3.11, 3.12, or 3.13.

### Step 2: Extract the Project Archive
Extract `heart-disease-ml.zip` into any folder on your computer.

### Step 3: Open Terminal in the Project Directory
- On Windows: Open PowerShell or Command Prompt in the extracted project folder.
- On macOS / Linux: Open your terminal and change directory (`cd`) to the project folder.

### Step 4: Create and Activate a Virtual Environment
Isolating dependencies ensures no conflicts with system packages:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activate command again).*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Required Python Libraries
Run:
```bash
pip install -r requirements.txt
```
This installs all required libraries: scikit-learn, scipy, pandas, numpy, matplotlib, seaborn, shap, plotly, streamlit, and joblib.

---

## 4. How to Run Automated Tests

To confirm that the environment is set up properly and all mathematical algorithms, imputation transformers, and model pipelines function correctly, run:

**On Windows:**
```powershell
$env:PYTHONPATH="."
pytest
```

**On macOS / Linux:**
```bash
PYTHONPATH=. pytest
```

**Expected Test Output:**
```
tests/test_data.py ..                                                    [ 16%]
tests/test_explainability.py ...                                         [ 41%]
tests/test_models.py ..                                                  [ 58%]
tests/test_preprocessing.py .....                                        [100%]

============================= 12 passed in 15.20s =============================
```
All 12 tests pass without errors.

---

## 5. How to Launch and Use the Web Application

### Launch Command
In your terminal, execute:
```bash
streamlit run app/Home.py
```

Streamlit will start a local web server and automatically open the application in your default browser at:
`http://localhost:8501`

*(If port 8501 is occupied by another application, specify an alternate port: `streamlit run app/Home.py --server.port 8502`).*

---

## 6. Detailed Walkthrough of the User Interface

The web application is structured into four main pages accessible via the left sidebar.

---

### Home Page: Overview & Data Inspection (`Home.py`)
- **Purpose**: Welcomes the user, explains the clinical motivation, outlines the research questions, and previews the dataset.
- **What You See**:
  - **Top KPI Cards**: Displays key cohort metrics (920 total records, 4 international hospital sources, 13 clinical attributes, 5 supervised classifiers, 300 cross-validation fits).
  - **Project Motivation & Research Questions**: Summarizes the 4 research questions (RQ1: Baseline Reproduction, RQ2: Imputation Impact, RQ3: SHAP Stability, RQ4: Accuracy vs Stability Trade-off).
  - **System Architecture**: Summarizes the 5-layer modular software design.
  - **Dataset Cohort Inspection**: An interactive table displaying a sample of the cleaned 920-patient dataset.

---

### Page 1: Clinical Prediction (`app/pages/1_Prediction.py`)
- **Purpose**: Generates an instantaneous heart disease risk assessment for a patient record based on clinical inputs.
- **Inputs to Provide**:
  1. **Select Machine Learning Classifier**: Choose from Random Forest, Gradient Boosting, Logistic Regression, Support Vector Machine, or K-Nearest Neighbors.
  2. **Select Imputation Strategy**: Choose from M1 (Median/Mode), M2 (Indicator flags), M3 (KNN), or M4 (MICE).
  3. **Patient Attributes**:
     - *Option A (Fastest)*: Click one of the pre-loaded clinical case presets:
       - **Patient Case A (High Risk)**: 63yo male, typical angina, resting BP 160 mm Hg, cholesterol 280 mg/dl, exercise ST depression 2.8.
       - **Patient Case B (Low Risk)**: 38yo female, atypical chest pain, resting BP 115 mm Hg, cholesterol 185 mg/dl, exercise ST depression 0.0.
       - **Patient Case C (Moderate Risk)**: 55yo male, non-anginal pain, resting BP 138 mm Hg, cholesterol 240 mg/dl, exercise ST depression 1.2.
     - *Option B (Custom Input)*: Adjust individual clinical sliders and dropdowns:
       - Age (28 – 80 years)
       - Biological Sex (Male / Female)
       - Chest Pain Type (Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic)
       - Resting Blood Pressure (80 – 200 mm Hg)
       - Serum Cholesterol (100 – 600 mg/dl)
       - Fasting Blood Sugar (> 120 mg/dl: True / False)
       - Resting ECG (Normal, ST-T Wave Abnormality, Left Ventricular Hypertrophy)
       - Maximum Heart Rate Achieved (60 – 220 bpm)
       - Exercise Induced Angina (Yes / No)
       - ST Depression / Oldpeak (0.0 – 6.0 mm)
       - Slope of Peak Exercise ST (Upsloping, Flat, Downsloping)
       - Major Vessels Colored by Fluoroscopy (0, 1, 2, 3)
       - Thallium Stress Test (Normal, Fixed Defect, Reversible Defect)
  4. **Action**: Click the blue button labeled **"Generate Heart Disease Prediction"**.
- **Outputs Returned**:
  - **Diagnostic Risk Banner**:
    - If risk is elevated: A red banner displaying **"High Diagnostic Risk: Heart Disease Present (Probability: XX.X%)"**.
    - If risk is low: A green banner displaying **"Low Diagnostic Risk: Heart Disease Absent (Probability: XX.X%)"**.
  - **Interactive Risk Gauge Chart**: A visual speedometer meter indicating where the patient falls between 0% and 100% risk, with green, yellow, and red clinical risk zones.
  - **Clinical Biomarker Summary**: A structured comparison table listing each submitted measurement alongside standard physiological reference ranges.

---

### Page 2: Prediction Explanation (`app/pages/2_Explanation.py`)
- **Purpose**: Provides transparent, patient-specific explanations using SHAP (SHapley Additive exPlanations) so clinicians can see exactly *why* a prediction was made.
- **Inputs to Provide**:
  1. Select the classifier to explain (e.g., Random Forest or Gradient Boosting).
  2. Select a patient preset (e.g., Patient Case A) or enter custom parameters.
  3. **Action**: Click the button labeled **"Compute SHAP Explanation"**.
- **Outputs Returned**:
  - **Interactive SHAP Waterfall Chart**:
    - Shows the baseline expected value $E[f(X)]$ at the bottom.
    - Each clinical feature has a horizontal bar indicating its contribution.
    - **Red bars (pointing right)** indicate attributes that increased heart disease risk (e.g., elevated `oldpeak`, asymptomatic chest pain `cp=4`, fluoroscopy vessels `ca=2`).
    - **Blue/Green bars (pointing left)** indicate protective factors that lowered risk (e.g., high exercise heart rate `thalch`, absence of exercise angina).
    - The top of the chart terminates at the patient's final predicted score $f(x)$.
  - **Quantitative Feature Importance Table**: Displays each feature's exact numeric Shapley value ($\phi_i$) and percentage contribution.
  - **Generated Clinical Narrative**: Automatically synthesizes a plain-language medical explanation summarizing the primary factors driving the algorithm's recommendation.

---

### Page 3: Model Performance Dashboard (`app/pages/3_Model_Performance.py`)
- **Purpose**: Displays precomputed experimental benchmarks, allowing users to compare models and evaluate missing-data strategies without needing to rerun long training runs.
- **Tabs to Explore**:
  1. **Tab 1: Baseline Model Reproduction**:
     - Displays a comparison table of empirical results vs published reference paper metrics across all 5 classifiers.
     - Interactive grouped bar chart comparing reproduced test accuracy against the published baseline.
     - Confirms that Gradient Boosting (83.70%) and Random Forest (84.78%) achieve the highest classification accuracy on the 920-patient cohort.
  2. **Tab 2: Imputation Strategies Comparison**:
     - Visual heatmap showing mean accuracy across all 5 models and all 4 imputation strategies (M1, M2, M3, M4) based on 300 cross-validation fits.
     - Distribution boxplot showing the spread of accuracy scores.
  3. **Tab 3: ROC Curves & Diagnostic Matrices**:
     - Receiver Operating Characteristic (ROC) curves with Area Under Curve (AUC) scores (ranging from 0.88 to 0.92).
     - Confusion matrices for all 5 models evaluated on the 184 held-out test patients.
  4. **Tab 4: Statistical Significance Tests**:
     - Table of paired $t$-tests and Wilcoxon signed-rank tests with exact $p$-values.
     - Demonstrates that advanced imputation (M4 MICE and M3 KNN) significantly improves covariance-sensitive models (Logistic Regression and SVM), while tree-based models (Random Forest) maintain robust performance regardless of imputation choice.

---

### Page 4: Research Findings & Stability (`app/pages/4_Research_Findings.py`)
- **Purpose**: Synthesizes the core scientific findings of the dissertation regarding Explainable AI stability and algorithmic trade-offs.
- **Tabs to Explore**:
  1. **Tab 1: Empirical Hypothesis Testing Outcomes**:
     - Formal evaluation of Hypotheses H1, H2, and H3:
       - **Hypothesis H1 (Imputation Effect)**: *Supported*. Missing-data handling strategies produce measurable performance differences.
       - **Hypothesis H2 (Explanation Sensitivity)**: *Supported*. SHAP attributions vary across models, data splits, and imputation techniques.
       - **Hypothesis H3 (Accuracy vs Stability Trade-off)**: *Supported*. The model with highest predictive accuracy (Gradient Boosting) does not always yield the most invariant explanations; linear models provide higher attribution invariance.
  2. **Tab 2: Quantitative Explanation Stability Metrics**:
     - Compares models across Spearman rank correlation, Top-5 Jaccard similarity, and patient-level Mean Absolute Difference (MAD) evaluated on a fixed 100-patient reference cohort.
  3. **Tab 3: Global Feature Importance Hierarchy**:
     - Displays the global cohort-wide feature importance ranking based on mean absolute SHAP values ($\mathbb{E}[|\phi_i|]$).
     - Confirms `oldpeak` (exercise ST depression) and `cp` (chest pain type) as the top two global risk predictors.
  4. **Tab 4: Critical Limitations & Ethical Discussion**:
     - Documents historical demographic limitations of the 1980s cohort.
     - Distinguishes mathematical correlation from biological causation.
     - Reiterates the research prototype status.

---

## 7. How to Rerun Experiments from Scratch (Optional)

All experimental results are already precomputed and stored in `results/` and `artifacts/`. If you wish to re-execute any stage from the command line:

1. **Ingest and Validate Dataset**:
   ```bash
   python src/heart_disease/data/load_data.py
   python src/heart_disease/data/validate_data.py
   ```
2. **Train and Persist All Baseline Models**:
   ```bash
   python -m src.heart_disease.models.train
   ```
3. **Rerun Baseline Reproduction**:
   ```bash
   python -m src.heart_disease.experiments.run_baseline
   ```
4. **Rerun Missing-Data Experiments (300 Evaluations)**:
   ```bash
   python -m src.heart_disease.experiments.run_imputation
   ```
5. **Rerun SHAP Stability Analysis**:
   ```bash
   python -m src.heart_disease.experiments.run_stability
   ```

---

## 8. Summary of Research Hypotheses & Conclusions

| Hypothesis | Proposition | Empirical Finding | Status |
|---|---|---|---|
| **H1** | Imputation strategies significantly alter predictive performance. | Advanced imputation (M4 MICE, M3 KNN) significantly improves linear models ($p < 0.05$); tree ensembles remain resilient. | **SUPPORTED** |
| **H2** | SHAP explanations vary across models, splits, and imputation methods. | Feature ranking hierarchies and local patient Shapley vectors demonstrate measurable variation across configurations. | **SUPPORTED** |
| **H3** | High predictive accuracy may trade off against explanation stability. | Gradient Boosting achieves top accuracy (83.70%) but lower ranking stability than Logistic Regression. | **SUPPORTED** |
