# Appendices

## Appendix A: Clinical Data Dictionary & Permissible Encodings

| Feature Name | Clinical Description | Data Type | Permissible Range / Encodings | Missing Count (before imputation) |
|---|---|---|---|---|
| `age` | Patient Age in years | Continuous (Numeric) | 28 – 77 | 0 (0.0%) |
| `sex` | Biological Sex | Categorical (Binary) | 0 = Female, 1 = Male | 0 (0.0%) |
| `cp` | Chest Pain Type | Categorical (Nominal) | 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal Pain, 4 = Asymptomatic | 0 (0.0%) |
| `trestbps` | Resting Blood Pressure (in mm Hg upon admission) | Continuous (Numeric) | 80 – 200 (0 flagged as invalid) | 59 (6.4%) |
| `chol` | Serum Cholesterol in mg/dl | Continuous (Numeric) | 85 – 603 (0 flagged as invalid) | 202 (22.0%) |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | Categorical (Binary) | 0 = False, 1 = True | 90 (9.8%) |
| `restecg` | Resting Electrocardiographic Results | Categorical (Nominal) | 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy | 2 (0.2%) |
| `thalch` | Maximum Heart Rate Achieved during exercise | Continuous (Numeric) | 60 – 202 | 55 (6.0%) |
| `exang` | Exercise Induced Angina | Categorical (Binary) | 0 = No, 1 = Yes | 55 (6.0%) |
| `oldpeak` | ST depression induced by exercise relative to rest | Continuous (Numeric) | -2.6 – 6.2 | 62 (6.7%) |
| `slope` | The slope of the peak exercise ST segment | Categorical (Ordinal) | 1 = Upsloping, 2 = Flat, 3 = Downsloping | 309 (33.6%) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy | Discrete (Ordinal) | 0, 1, 2, 3 | 611 (66.4%) |
| `thal` | Thallium Heart Stress Test Status | Categorical (Nominal) | 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect | 486 (52.8%) |
| `target` | Binary Diagnostic Label | Target (Binary) | 0 = Disease Absent, 1 = Disease Present | 0 (0.0%) |

---

## Appendix B: Reproduction and Environment Execution Guide

### B.1 Python Virtual Environment Configuration
To reproduce the complete experimental findings and launch the demonstration interface:

```bash
# 1. Create a dedicated virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 3. Install tested dependencies
pip install -r requirements.txt
```

### B.2 Executing Experiments
```bash
# Ingest and validate 920-record dataset
python src/heart_disease/data/load_data.py
python src/heart_disease/data/validate_data.py

# Run baseline reproduction (80/20 split and 5-fold CV)
python -m src.heart_disease.experiments.run_baseline

# Run missing-data experiments (300 evaluations across M1-M4)
python -m src.heart_disease.experiments.run_imputation

# Run SHAP stability analysis and hypothesis evaluation
python -m src.heart_disease.experiments.run_stability

# Run automated test suite
pytest
```

### B.3 Launching the Streamlit Demonstration Application
```bash
streamlit run app/Home.py
```
