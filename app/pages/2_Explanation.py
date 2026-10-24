"""Prediction Explanation Interface - Explainable AI (XAI) Studio.

Matches PRD Page 3 (Prediction Explanation) User Interface Requirements.
"""

import os
import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from app.utils import (
        apply_custom_theme,
        display_medical_disclaimer,
        load_fitted_pipeline,
        PRELOADED_CASES
    )
except ImportError:
    from utils import (
        apply_custom_theme,
        display_medical_disclaimer,
        load_fitted_pipeline,
        PRELOADED_CASES
    )

from src.heart_disease.data.preprocess import ALL_FEATURES
from src.heart_disease.explainability.shap_explainer import HeartDiseaseSHAPExplainer

st.set_page_config(
    page_title="Prediction Explanation (SHAP)",
    page_icon="🔍",
    layout="wide"
)

apply_custom_theme()

st.title("🔍 Patient-Level SHAP Interpretability Studio")
st.markdown(r"Deconstruct the model's diagnostic output into additive Shapley attribution values ($\phi_i$) for individual patient clinical features.")

display_medical_disclaimer()

# Patient Selection Source
st.markdown("### 👤 Select Patient Record to Explain")

tabs = st.tabs(["Patient from Prediction Page", "Explore Representative Case Studies"])

with tabs[0]:
    if "current_patient_df" in st.session_state:
        patient_df = st.session_state["current_patient_df"]
        model_name = st.session_state.get("current_model_name", "Random Forest")
        strat_code = st.session_state.get("current_strategy", "M1")
        pred_label = st.session_state.get("current_prediction", 1)
        prob_val = st.session_state.get("current_probability", 0.5)

        st.success(f"Loaded patient currently evaluated in Prediction Form (Model: **{model_name}**, Imputation: **{strat_code}**). Predicted Probability: **{prob_val*100:.1f}%**")
    else:
        st.info("No active patient evaluated yet on the Prediction page. Preloading default clinical case below.")
        patient_df = pd.DataFrame([PRELOADED_CASES["Patient Case A: High Risk (63yo, Typical Angina, Elevated Oldpeak)"]])[ALL_FEATURES]
        model_name = "Random Forest"
        strat_code = "M1"
        pred_label = 1
        prob_val = 0.85

with tabs[1]:
    case_select = st.selectbox("Select Benchmark Case Study:", list(PRELOADED_CASES.keys()))
    if st.button("Load Selected Case Study"):
        patient_df = pd.DataFrame([PRELOADED_CASES[case_select]])[ALL_FEATURES]
        model_name = "Random Forest"
        strat_code = "M1"
        st.session_state["current_patient_df"] = patient_df
        st.rerun()

st.markdown("---")

# Feature Values Summary Table
with st.expander("🔎 View Patient Feature Values", expanded=False):
    st.dataframe(patient_df, use_container_width=True)

# Generate SHAP Explanation
pipeline = load_fitted_pipeline(model_name, strat_code)

if pipeline is None:
    st.error(f"Fitted pipeline for {model_name} ({strat_code}) not found.")
else:
    with st.spinner("Calculating SHAP Shapley values for this patient..."):
        # Load small background cohort for reference
        train_ref_path = "data/processed/train.csv"
        df_train_bg = pd.read_csv(train_ref_path)[ALL_FEATURES].iloc[:40] if os.path.exists(train_ref_path) else None

        explainer = HeartDiseaseSHAPExplainer(pipeline, model_name, background_sample=df_train_bg)
        explanation = explainer.explain_patient(patient_df)

    base_val = explanation["base_value"]
    output_sum = explanation["output_sum"]
    contribs = explanation["contributions"]

    # Metric Badges
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(r"Base Expected Value ($\mathbb{E}[f(x)])$", f"{base_val:.4f}", help="Population baseline log-odds or expected output before observing patient features.")
    with m_col2:
        total_shap = sum(c["shap_value"] for c in contribs)
        st.metric(r"Total Feature Attribution ($\sum \phi_i$)", f"{total_shap:+.4f}", help="Sum of individual feature contributions pushing the prediction toward or away from disease presence.")
    with m_col3:
        st.metric("Final Model Output Value", f"{output_sum:.4f}", help="Base value + sum of SHAP attributions.")

    st.markdown("### 📊 Interactive SHAP Attribution Waterfall")
    st.caption("🔴 **Red bars** indicate features increasing heart disease risk • 🟢 **Green bars** indicate features protective against heart disease.")

    # Build Interactive Horizontal Bar Chart with Plotly
    top_contribs = contribs[:12][::-1]
    features = [c["feature"] for c in top_contribs]
    shap_vals = [c["shap_value"] for c in top_contribs]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Local SHAP Feature Contributions ({model_name})",
        xaxis_title="SHAP Value (Impact on Prediction Output)",
        yaxis_title="Clinical Feature",
        height=480,
        margin=dict(l=20, r=40, t=50, b=40),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain Language Clinical Interpretation Table
    st.markdown("### 🩺 Clinical Narrative Interpretation")
    st.markdown(
        r"""
        How to read these attributions:
        - **Positive SHAP ($\phi > 0$)**: These attributes contributed evidence toward the diagnosis of **Heart Disease Present**.
        - **Negative SHAP ($\phi < 0$)**: These attributes served as protective factors, reducing the predicted likelihood of heart disease.
        """
    )

    narrative_rows = []
    for c in contribs[:8]:
        direction_badge = "🔴 Increases Disease Risk" if c["shap_value"] > 0 else "🟢 Protective Factor"
        narrative_rows.append({
            "Feature": c["feature"],
            "SHAP Contribution": f"{c['shap_value']:+.4f}",
            "Clinical Impact": direction_badge,
            "Absolute Magnitude": f"{c['abs_impact']:.4f}"
        })

    st.table(pd.DataFrame(narrative_rows))

    st.markdown(
        """
        > **Academic Note on Causality vs Attribution**: SHAP values measure empirical model attribution—how much 
        > each feature altered the algorithm's specific mathematical output. They do **not** imply direct biomedical causation.
        """
    )
