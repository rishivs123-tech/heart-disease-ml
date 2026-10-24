"""Shared UI utilities, styling, and model loaders for the Streamlit application.

Provides rich visual theming, cached artifact loaders, and clinical presets.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .disclaimer-card {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        color: #92400e;
        margin-bottom: 1.8rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    .metric-card {
        background: white;
        padding: 1.4rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .result-badge-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.3rem;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    
    .result-badge-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.3rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
</style>
"""


def apply_custom_theme():
    """Inject modern CSS styles into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def display_medical_disclaimer():
    """Render mandatory academic research disclaimer matching PRD FR-12."""
    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>⚠️ Research Prototype & Medical Disclaimer:</strong><br>
            This web application is an academic dissertation research prototype designed exclusively for scientific demonstration,
            reproducibility evaluation, and Explainable AI (XAI) analysis. It is <strong>NOT</strong> a certified medical device,
            has not undergone prospective clinical trials, and must <strong>NEVER</strong> be utilized for clinical diagnosis,
            triage, or real-world treatment planning.
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def load_fitted_pipeline(model_name: str, strategy: str = "M1"):
    """Load cached fitted pipeline from artifacts/models/."""
    clean_name = model_name.lower().replace(" ", "_")
    path = f"artifacts/models/{clean_name}_{strategy.lower()}.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    # Fallback to M1 if specific strategy not saved
    fallback_path = f"artifacts/models/{clean_name}_m1.joblib"
    if os.path.exists(fallback_path):
        return joblib.load(fallback_path)
    return None


@st.cache_data
def load_benchmark_metrics():
    """Load baseline reproduction metrics from results/metrics/baseline_reproduction.csv."""
    path = "results/metrics/baseline_reproduction.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_imputation_summary():
    """Load imputation strategy summary from results/metrics/imputation_summary.csv."""
    path = "results/metrics/imputation_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_statistical_comparisons():
    """Load statistical test comparisons from results/metrics/statistical_comparisons.csv."""
    path = "results/metrics/statistical_comparisons.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_stability_summary():
    """Load SHAP stability metrics from results/metrics/stability_summary.csv."""
    path = "results/metrics/stability_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_sample(n: int = 5):
    """Load cleaned dataset head for interactive exploration."""
    path = "data/processed/heart_disease_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path).head(n)
    return pd.DataFrame()


PRELOADED_CASES = {
    "Patient Case A: High Risk (63yo, Typical Angina, Elevated Oldpeak)": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 160.0, "chol": 280.0,
        "fbs": 1, "restecg": 2, "thalch": 108.0, "exang": 1, "oldpeak": 2.8,
        "slope": 2, "ca": 2, "thal": 7
    },
    "Patient Case B: Low Risk (38yo, Asymptomatic, Excellent Exercise ECG)": {
        "age": 38, "sex": 0, "cp": 2, "trestbps": 115.0, "chol": 185.0,
        "fbs": 0, "restecg": 0, "thalch": 178.0, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 0, "thal": 3
    },
    "Patient Case C: Moderate Risk (55yo, Atypical Chest Pain)": {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 138.0, "chol": 240.0,
        "fbs": 0, "restecg": 1, "thalch": 145.0, "exang": 0, "oldpeak": 1.2,
        "slope": 2, "ca": 1, "thal": 6
    }
}
