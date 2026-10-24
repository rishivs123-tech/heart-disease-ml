# Heart Disease Dataset Documentation

## 1. Dataset Overview
- **Dataset Title**: Combined 920-Record UCI Heart Disease Cohort
- **Official UCI Repository**: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Primary Reference Study**: *Heart Disease Prediction Using Artificial Intelligence Algorithms: A Comparative Study*
- **Total Records**: 920
- **Total Features**: 13 clinical attributes + 1 source origin column + 1 target
- **Target Distribution**:
  - `0` (Absence of Heart Disease, < 50% diameter narrowing): **411 patients (44.67%)**
  - `1` (Presence of Heart Disease, >= 50% diameter narrowing): **509 patients (55.33%)**
- **Train / Test Partition**:
  - Stratified 80% Training Set: **736 patients**
  - Stratified 20% Held-Out Testing Set: **184 patients**

---

## 2. Source Subsets
The combined 920-record dataset originates from four clinical cardiology institutions:
1. **Cleveland Clinic Foundation, Ohio, USA**: 303 records (Principal investigator: Robert Detrano, M.D., Ph.D.)
2. **Hungarian Institute of Cardiology, Budapest, Hungary**: 294 records (Principal investigator: Andras Janosi, M.D.)
3. **University Hospital, Zurich, Switzerland**: 123 records (Principal investigator: William Steinbrunn, M.D.)
4. **V.A. Medical Center, Long Beach, California, USA**: 200 records (Principal investigator: E. Pfisterer, M.D.)

---

## 3. Clinical Data Dictionary

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
| `num` | Original angiographic disease status | Target (Ordinal) | 0 = No disease; 1, 2, 3, 4 = Severity degrees | 0 (0.0%) |
| `target` | Binary Diagnostic Label | Target (Binary) | 0 = Disease Absent, 1 = Disease Present | 0 (0.0%) |

---

## 4. Physiologically Invalid Measurements Handling
In clinical medicine, resting blood pressure (`trestbps`) and serum cholesterol (`chol`) cannot physically be zero in a living patient. In the original raw UCI files, certain recording centers (notably Switzerland and VA Long Beach) recorded missing measurements as 0. 
Following clinical data preprocessing protocols:
- 1 occurrence of `trestbps = 0` is converted to `NaN`.
- 172 occurrences of `chol = 0` are converted to `NaN`.
These missing values are subsequently handled rigorously using the four experimental imputation strategies (M1, M2, M3, M4) within cross-validation folds to prevent information leakage.
