"""SHAP explanation utilities for local and global interpretability.

Provides wrappers around TreeExplainer, LinearExplainer, and KernelExplainer to generate:
- Patient-level attribution waterfall plots
- Global feature importance rankings
"""

import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.heart_disease.data.preprocess import get_feature_names_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HeartDiseaseSHAPExplainer:
    """Wrapper for computing and visualizing SHAP values across multiple model types."""

    def __init__(self, fitted_pipeline, model_name: str, background_sample: pd.DataFrame | None = None):
        self.pipeline = fitted_pipeline
        self.model_name = model_name
        self.preprocessor = fitted_pipeline.named_steps["preprocessor"]
        self.model = fitted_pipeline.named_steps["model"]

        # Extract transformed feature names
        self.feature_names = get_feature_names_out(self.preprocessor)

        # Build appropriate SHAP explainer
        self.explainer = self._build_explainer(background_sample)

    def _transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw input dataframe through the fitted preprocessor."""
        return self.preprocessor.transform(X)

    def _build_explainer(self, background_sample: pd.DataFrame | None):
        """Construct model-appropriate SHAP explainer (Tree, Linear, or Kernel)."""
        model_type = type(self.model).__name__.lower()

        if "randomforest" in model_type or "gradientboosting" in model_type:
            logger.info("Using shap.TreeExplainer for %s", self.model_name)
            return shap.TreeExplainer(self.model)

        elif "logisticregression" in model_type:
            logger.info("Using shap.LinearExplainer for %s", self.model_name)
            if background_sample is not None:
                bg_trans = self._transform_data(background_sample)
                # Use a small background summary (max 50 points)
                if len(bg_trans) > 50:
                    bg_trans = shap.kmeans(bg_trans, 50).data
                return shap.LinearExplainer(self.model, bg_trans)
            return shap.LinearExplainer(self.model)

        else:
            # For SVM and KNN: KernelExplainer with background kmeans
            logger.info("Using shap.KernelExplainer for %s", self.model_name)
            if background_sample is not None:
                bg_trans = self._transform_data(background_sample)
                if len(bg_trans) > 40:
                    bg_summary = shap.kmeans(bg_trans, 40).data
                else:
                    bg_summary = bg_trans
            else:
                bg_summary = np.zeros((1, len(self.feature_names)))

            predict_fn = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
            return shap.KernelExplainer(predict_fn, bg_summary)

    def explain_dataset(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values and base value for a dataset.

        Returns (shap_values, base_value).
        """
        X_trans = self._transform_data(X)
        shap_raw = self.explainer.shap_values(X_trans)

        # Handle different SHAP output formats (binary class output)
        if isinstance(shap_raw, list):
            # Binary classification list: [class_0_shap, class_1_shap] -> use positive class (1)
            shap_vals = shap_raw[1]
        elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
            shap_vals = shap_raw[:, :, 1]
        else:
            shap_vals = shap_raw

        # Base value
        base_val = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(base_val, list | np.ndarray) and len(base_val) > 1:
            base_val = float(base_val[1])
        else:
            base_val = float(base_val) if isinstance(base_val, float | int | np.number) else 0.0

        return shap_vals, base_val

    def get_global_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute global feature importance as mean(|SHAP|) across records."""
        shap_vals, _ = self.explain_dataset(X)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return df_imp

    def plot_summary(self, X: pd.DataFrame, output_path: str = "results/figures/shap_summary.png"):
        """Generate and save global SHAP summary bar chart."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_imp = self.get_global_feature_importance(X)

        plt.figure(figsize=(10, 6))
        plt.barh(df_imp["feature"][:12][::-1], df_imp["mean_abs_shap"][:12][::-1], color="#2563eb")
        plt.xlabel("Mean |SHAP value| (Average Impact on Model Output)")
        plt.title(f"Global Feature Importance ({self.model_name}) - SHAP Summary")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("Saved SHAP summary figure to %s", output_path)

    def explain_patient(self, patient_df: pd.DataFrame) -> dict:
        """Compute individual SHAP explanation for a single patient record.

        Returns feature contributions sorted by absolute magnitude.
        """
        X_trans = self._transform_data(patient_df)
        shap_raw = self.explainer.shap_values(X_trans)

        if isinstance(shap_raw, list):
            patient_shap = shap_raw[1][0]
        elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
            patient_shap = shap_raw[0, :, 1]
        else:
            patient_shap = shap_raw[0]

        base_val = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(base_val, list | np.ndarray) and len(base_val) > 1:
            base_val = float(base_val[1])
        else:
            base_val = float(base_val) if isinstance(base_val, float | int | np.number) else 0.0

        contributions = []
        for feat_name, shap_v in zip(self.feature_names, patient_shap):
            contributions.append({
                "feature": feat_name,
                "shap_value": round(float(shap_v), 4),
                "abs_impact": round(float(abs(shap_v)), 4),
                "direction": "Increases Risk" if shap_v > 0 else "Decreases Risk"
            })

        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        return {
            "model": self.model_name,
            "base_value": round(base_val, 4),
            "output_sum": round(base_val + float(np.sum(patient_shap)), 4),
            "contributions": contributions
        }
