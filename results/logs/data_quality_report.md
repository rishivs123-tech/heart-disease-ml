# Data Quality and Acceptance Report
**Generated for**: 920-Record UCI Heart Disease Cohort  
**Validation Status**: ✅ PASSED  
**Total Records**: 920 (Expected: 920)  
**Total Columns**: 17  

---

## 1. Class Distribution
- **Class 0 (Disease Absent)**: 411 (44.67%)
- **Class 1 (Disease Present)**: 509 (55.33%)

## 2. Institutional Cohort Breakdown
- **Cleveland**: 303 records (32.93%)
- **Hungary**: 294 records (31.96%)
- **VA Long Beach**: 200 records (21.74%)
- **Switzerland**: 123 records (13.37%)

## 3. Missingness Summary per Feature

| Feature | Missing Values (Count) | Missing Percentage (%) |
|---|---|---|
| `dataset` | 0 | 0.0% |
| `age` | 0 | 0.0% |
| `sex` | 0 | 0.0% |
| `cp` | 0 | 0.0% |
| `trestbps` | 60 | 6.52% |
| `chol` | 202 | 21.96% |
| `fbs` | 90 | 9.78% |
| `restecg` | 2 | 0.22% |
| `thalch` | 55 | 5.98% |
| `exang` | 55 | 5.98% |
| `oldpeak` | 62 | 6.74% |
| `slope` | 309 | 33.59% |
| `ca` | 611 | 66.41% |
| `thal` | 486 | 52.83% |
| `num` | 0 | 0.0% |
| `num_original` | 0 | 0.0% |
| `target` | 0 | 0.0% |

## 4. Validation Checks & Warnings
- All fundamental schema and record count assertions passed successfully.
