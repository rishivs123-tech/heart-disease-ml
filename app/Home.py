"""Heart Disease Prediction & Explainability System - Main Entry Point.

Matches PRD Page 1 (Home) User Interface Requirements.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

try:
    from app.utils import apply_custom_theme, display_medical_disclaimer, load_dataset_sample
except ImportError:
    from utils import apply_custom_theme, display_medical_disclaimer, load_dataset_sample

st.set_page_config(
    page_title="Heart Disease ML & Explainability",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

# Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.4rem; font-weight: 800;">🫀 Heart Disease Prediction Using Machine Learning</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.15rem; opacity: 0.95;">
            <strong>Research Extension:</strong> Explainable & Robust Heart Disease Prediction: Evaluating Missing-Data Strategies and SHAP Explanation Stability
        </p>
        <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.85;">
            Master of Engineering (ME) Dissertation Research Prototype & Scientific Reproducibility Framework
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

display_medical_disclaimer()

# Overview & Metrics KPI
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Total Cohort Records", value="920", help="Aggregated from 4 international cardiology centers")
with col2:
    st.metric(label="Cardiology Centers", value="4 Sites", help="Cleveland, Hungary, Switzerland, VA Long Beach")
with col3:
    st.metric(label="Clinical Attributes", value="13 Features", help="Hemodynamic, electrocardiographic, and demographic parameters")
with col4:
    st.metric(label="Supervised Classifiers", value="5 Models", help="LR, RF, SVM, KNN, Gradient Boosting")
with col5:
    st.metric(label="Evaluations Conducted", value="300 Fits", help="5 models x 4 imputations x 3 seeds x 5 folds")

st.markdown("---")

# Main Content Grid
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📌 Project Overview & Research Motivation")
    st.markdown(
        """
        Cardiovascular diseases remain the leading cause of global mortality. While machine learning algorithms demonstrate 
        high predictive capability on structured patient records, real-world clinical datasets consistently present two critical hurdles:
        1. **Complex Missing-Data Patterns**: Real clinical records frequently suffer from missing measurements (e.g., fluoroscopy vessel counts or stress test thallium) or physiologically invalid encodings (such as zero cholesterol).
        2. **Attribution Instability in Explainable AI (XAI)**: Models that achieve high classification accuracy may generate dramatically different explanations for similar patients across data splits, imputation choices, or training runs.
        
        This dissertation project accomplishes a **controlled, leakage-safe empirical reproduction** of a benchmark 920-record study and extends it by systematically quantifying:
        - The predictive impact of **four distinct missing-data imputation strategies (M1–M4)**.
        - The stability and consistency of **SHAP (SHapley Additive exPlanations)** global rankings and patient-level attributions.
        """
    )
    
    st.subheader("🎯 Primary Research Questions")
    st.markdown(
        """
        - **RQ1 (Baseline Reproduction)**: Can the published classification metrics be reproduced using the reported 80/20 train/test protocol?
        - **RQ2 (Imputation Impact)**: How do Simple Median/Mode, Missingness Indicators, KNN, and Iterative (MICE) imputation alter classifier performance?
        - **RQ3 (Explanation Stability)**: How stable are global feature hierarchies and patient-level SHAP attributions across repeated stratified splits?
        - **RQ4 (Accuracy vs Stability Trade-off)**: Does the top-performing predictive classifier also produce the most stable explanations?
        """
    )

with right_col:
    st.subheader("🏗️ System Architecture (5 Layers)")
    st.markdown(
        """
        - **Layer 1: Dataset Layer**: Multi-center raw data ingestion (Cleveland, Hungary, Switzerland, VA Long Beach) and schema verification.
        - **Layer 2: Preprocessing Layer**: Leakage-safe handling of invalid zeros and scikit-learn imputation pipelines (M1–M4).
        - **Layer 3: Machine Learning Layer**: 5 baseline models (LR, RF, SVM, KNN, GB) with stratified 5-fold cross-validation.
        - **Layer 4: Explainability & Stability**: Model-specific SHAP explainers (Tree, Linear, Kernel), Spearman rank correlation, and top-$k$ Jaccard similarity.
        - **Layer 5: Application Layer**: Interactive Streamlit demonstration interface with real-time inference and local feature attribution.
        """
    )
    
    st.info("💡 **Navigation Guide**: Use the sidebar to explore Patient Prediction, Patient SHAP Explanations, Model Performance Benchmarks, and Research Findings.")

st.markdown("---")
st.subheader("🔍 Dataset Cohort Inspection (Cleaned 920-Record Sample)")
st.dataframe(load_dataset_sample(8), use_container_width=True)

st.caption("Heart Disease Machine Learning System • Version 1.0 • Academic Research Software Prototype")
