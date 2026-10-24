# Python API & Module Reference

This document provides technical documentation for all modular components in the `src/heart_disease` package.

---

## 1. `heart_disease.data`

### `load_data.py`
- `fetch_raw_uci_datasets(output_raw_path: str) -> pd.DataFrame`:
  Downloads raw `.data` files from the 4 official UCI heart disease repository endpoints and combines them into 920 rows.
- `clean_and_format_dataset(df_raw: pd.DataFrame, handle_invalid_zeros: bool = True) -> pd.DataFrame`:
  Replaces missing representations (`?`, empty strings) with `NaN`, flags physiologically impossible zeros in `trestbps` and `chol` as `NaN`, and converts `num` to binary `target`.
- `prepare_datasets(raw_path, processed_path, train_path, test_path, test_size=0.20, random_state=42)`:
  Coordinates end-to-end data preparation and exports `train.csv` (736 records) and `test.csv` (184 records).

### `validate_data.py`
- `validate_dataset(df: pd.DataFrame) -> dict`:
  Validates row count (920), required columns (14 clinical + dataset + target), target balance, and missingness statistics.
- `generate_data_quality_report(data_path, output_dir) -> dict`:
  Executes validation assertions and generates both JSON and Markdown data quality reports.

### `preprocess.py`
- `create_preprocessor(strategy: Literal['M1', 'M2', 'M3', 'M4']) -> ColumnTransformer`:
  Constructs a leakage-safe `ColumnTransformer` encapsulating the specified imputation strategy.
- `create_pipeline(strategy: str, model: BaseEstimator) -> Pipeline`:
  Chains preprocessor and classifier into a single scikit-learn `Pipeline`.
- `get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]`:
  Extracts clean, human-readable feature names post-transformation for SHAP interpretability.

---

## 2. `heart_disease.models`

### `build_models.py`
- `get_model(model_name: str, random_state: int = 42, **kwargs) -> BaseEstimator`:
  Instantiates any of the five baseline classifiers with the reference study's configurations.
- `get_all_baseline_models(random_state: int = 42) -> dict[str, BaseEstimator]`:
  Returns a dictionary containing all five baseline models.

### `train.py`
- `train_and_persist_model(model_name, strategy, train_path, test_path, models_dir, metadata_dir)`:
  Fits a model pipeline on the training split, evaluates on test split, and serializes artifacts (`.joblib` and `_metadata.json`).
- `train_all_baseline_pipelines(strategy='M1') -> list[dict]`:
  Fits and persists all five baseline models.

### `tune_models.py`
- `tune_model(model_name, strategy, train_path, cv_splits=5, random_state=42) -> dict`:
  Executes `GridSearchCV` optimization over defined hyperparameter spaces in `configs/models.yaml`.

---

## 3. `heart_disease.evaluation`

### `metrics.py`
- `calculate_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]`:
  Computes Accuracy, Precision, Recall, Specificity, F1, MCC, ROC-AUC, and Confusion Matrix metrics.

### `cross_validation.py`
- `evaluate_model_cv(model_name, estimator, X, y, strategy='M1', n_splits=5, random_state=42) -> dict`:
  Executes leakage-safe Stratified K-Fold CV, returning fold-level and aggregate mean $\pm$ std metrics.

### `statistics.py`
- `compute_confidence_interval(scores, confidence=0.95) -> tuple[float, float, float]`:
  Calculates mean and two-sided 95% confidence intervals via Student's $t$-distribution.
- `paired_comparison_test(scores_a, scores_b, metric_name='accuracy') -> dict`:
  Conducts two-tailed paired $t$-tests and Wilcoxon signed-rank tests across matched folds.

---

## 4. `heart_disease.explainability`

### `shap_explainer.py`
- `HeartDiseaseSHAPExplainer(fitted_pipeline, model_name, background_sample=None)`:
  Instantiates model-appropriate explainers (`TreeExplainer`, `LinearExplainer`, `KernelExplainer`).
- `explain_dataset(X: pd.DataFrame) -> tuple[np.ndarray, float]`:
  Computes matrix of SHAP values and base expected value.
- `get_global_feature_importance(X: pd.DataFrame) -> pd.DataFrame`:
  Returns features ordered by mean absolute SHAP value ($\mathbb{E}[|\phi_i|]$).
- `explain_patient(patient_df: pd.DataFrame) -> dict`:
  Returns patient-level feature attributions sorted by magnitude.

### `stability.py`
- `calculate_spearman_rank_stability(rank_a, rank_b) -> float`:
  Computes Spearman rank correlation between feature hierarchies.
- `calculate_top_k_jaccard(features_a, features_b, k=5) -> float`:
  Computes Jaccard similarity index across the top-$k$ decisive features.
- `calculate_patient_level_mad(shap_matrix_a, shap_matrix_b) -> float`:
  Calculates Mean Absolute Difference (MAD) between patient Shapley attributions across runs.
- `evaluate_explanation_stability(run_results: list[dict]) -> pd.DataFrame`:
  Coordinates pairwise stability analysis across model families and imputation strategies.
