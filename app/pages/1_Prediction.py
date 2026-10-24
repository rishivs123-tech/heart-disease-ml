"""Heart Disease Prediction Interface."""

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
import streamlit as st
import plotly.graph_objects as go

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

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

apply_custom_theme()

st.title("🩺 Clinical Patient Heart Disease Prediction")
st.markdown("Enter patient demographic, hemodynamic, and electrocardiographic parameters to generate diagnostic risk estimates.")

display_medical_disclaimer()

# Model and Imputation Selector
col_m1, col_m2 = st.columns(2)
with col_m1:
    selected_model = st.selectbox(
        "Select Machine Learning Classifier:",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"],
        index=0,
        help="Choose the trained baseline algorithm to execute inference."
    )
with col_m2:
    selected_strategy = st.selectbox(
        "Select Missing-Data Imputation Pipeline:",
        ["M1 (Median/Mode)", "M2 (Indicators)", "M3 (KNN Imputer)", "M4 (Iterative MICE)"],
        index=0,
        help="Choose which imputation strategy pipeline to apply to clinical inputs."
    )

strat_code = selected_strategy[:2]

# Preloaded Clinical Vignettes Selector
st.markdown("#### ⚡ Quick Preset Vignettes")
preset_choice = st.selectbox(
    "Or load a preconfigured clinical case study:",
    ["-- Manual Patient Entry --"] + list(PRELOADED_CASES.keys())
)

default_vals = {}
if preset_choice != "-- Manual Patient Entry --":
    default_vals = PRELOADED_CASES[preset_choice]

st.markdown("---")
st.subheader("📋 Patient Clinical Parameters Form")

with st.form("patient_prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics & Symptoms**")
        age = st.slider("Patient Age (years):", 20, 85, int(default_vals.get("age", 54)))
        sex_label = st.radio("Biological Sex:", ["Female", "Male"], index=int(default_vals.get("sex", 1)))
        sex = 1 if sex_label == "Male" else 0

        cp_options = {
            1: "1: Typical Angina",
            2: "2: Atypical Angina",
            3: "3: Non-Anginal Chest Pain",
            4: "4: Asymptomatic"
        }
        cp_def = int(default_vals.get("cp", 4))
        cp = st.selectbox("Chest Pain Type (cp):", options=list(cp_options.keys()), format_func=lambda x: cp_options[x], index=list(cp_options.keys()).index(cp_def))

        trestbps = st.number_input(
            "Resting Blood Pressure (trestbps in mm Hg):",
            min_value=80.0, max_value=220.0,
            value=float(default_vals.get("trestbps", 130.0)),
            step=1.0,
            help="Resting arterial blood pressure upon hospital admission."
        )

    with c2:
        st.markdown("**Biomarkers & Electrocardiography**")
        chol = st.number_input(
            "Serum Cholesterol (chol in mg/dl):",
            min_value=80.0, max_value=600.0,
            value=float(default_vals.get("chol", 230.0)),
            step=1.0,
            help="Total fasting serum cholesterol."
        )

        fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs):", ["No (<= 120)", "Yes (> 120)"], index=int(default_vals.get("fbs", 0)))
        fbs = 1 if "Yes" in fbs_label else 0

        restecg_options = {
            0: "0: Normal",
            1: "1: ST-T Wave Abnormality (T inversion / ST depression)",
            2: "2: Left Ventricular Hypertrophy (Estes' criteria)"
        }
        restecg_def = int(default_vals.get("restecg", 0))
        restecg = st.selectbox("Resting ECG (restecg):", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x], index=list(restecg_options.keys()).index(restecg_def))

        thalch = st.slider("Max Heart Rate Achieved (thalch in bpm):", 60.0, 220.0, float(default_vals.get("thalch", 150.0)), step=1.0)

    with c3:
        st.markdown("**Exercise Stress & Angiography**")
        exang_label = st.radio("Exercise-Induced Angina (exang):", ["No", "Yes"], index=int(default_vals.get("exang", 0)))
        exang = 1 if "Yes" in exang_label else 0

        oldpeak = st.number_input(
            "ST Depression Induced by Exercise (oldpeak):",
            min_value=-2.6, max_value=6.5,
            value=float(default_vals.get("oldpeak", 1.0)),
            step=0.1,
            help="ST segment depression relative to resting baseline."
        )

        slope_options = {
            1: "1: Upsloping",
            2: "2: Flat",
            3: "3: Downsloping"
        }
        slope_def = int(default_vals.get("slope", 2))
        slope = st.selectbox("Slope of Peak Exercise ST (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x], index=list(slope_options.keys()).index(slope_def))

        ca = st.slider(
            "Number of Major Vessels Colored (ca: 0-3):",
            0, 3, int(default_vals.get("ca", 0)),
            help="Fluoroscopy visualization of coronary arteries."
        )

        thal_options = {
            3: "3: Normal Blood Flow",
            6: "6: Fixed Defect",
            7: "7: Reversible Defect"
        }
        thal_def = int(default_vals.get("thal", 3))
        thal = st.selectbox("Thallium Heart Scintigraphy (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x], index=list(thal_options.keys()).index(thal_def))

    submit_button = st.form_submit_button("🚀 Run Patient Risk Prediction", use_container_width=True)

if submit_button:
    # Build dataframe for inference
    patient_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalch": thalch,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal
    }
    df_patient = pd.DataFrame([patient_dict])[ALL_FEATURES]

    # Store in session state for explanation page
    st.session_state["current_patient_df"] = df_patient
    st.session_state["current_model_name"] = selected_model
    st.session_state["current_strategy"] = strat_code

    pipeline = load_fitted_pipeline(selected_model, strat_code)

    if pipeline is None:
        st.error(f"Fitted pipeline for {selected_model} ({strat_code}) could not be loaded. Please ensure training artifacts exist.")
    else:
        y_pred = int(pipeline.predict(df_patient)[0])

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df_patient)[0]
            prob_disease = float(probs[1])
        elif hasattr(pipeline, "decision_function"):
            dec = pipeline.decision_function(df_patient)[0]
            prob_disease = float(1 / (1 + np.exp(-dec)))
        else:
            prob_disease = float(y_pred)

        st.session_state["current_prediction"] = y_pred
        st.session_state["current_probability"] = prob_disease

        st.markdown("---")
        st.subheader("🎯 Diagnostic Risk Evaluation Result")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if y_pred == 1:
                st.markdown(
                    f"""
                    <div class="result-badge-danger">
                        🚨 HEART DISEASE PRESENT (RISK DETECTED)<br>
                        <span style="font-size: 1rem; font-weight: 400;">Model Output: Class 1 (Probability: {prob_disease*100:.1f}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-badge-success">
                        ✅ NO HEART DISEASE DETECTED (NORMAL)<br>
                        <span style="font-size: 1rem; font-weight: 400;">Model Output: Class 0 (Probability: {prob_disease*100:.1f}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Risk tier
            if prob_disease < 0.30:
                tier = "🟢 Low Clinical Risk (< 30%)"
                desc = "Hemodynamic and stress biomarkers are within expected non-pathological ranges."
            elif prob_disease < 0.60:
                tier = "🟡 Moderate Clinical Risk (30% – 60%)"
                desc = "Borderline findings warranting secondary clinical evaluation and lifestyle intervention."
            elif prob_disease < 0.80:
                tier = "🟠 High Clinical Risk (60% – 80%)"
                desc = "Significant coronary risk markers detected; recommended for diagnostic angiography confirmation."
            else:
                tier = "🔴 Critical Clinical Risk (> 80%)"
                desc = "Severe multi-attribute cardiac abnormalities present."

            st.markdown(f"**Assessed Risk Category:** `{tier}`")
            st.caption(desc)

        with res_col2:
            # Interactive Gauge Chart with Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob_disease * 100, 1),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cardiac Disease Probability (%)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#ef4444" if prob_disease >= 0.5 else "#10b981"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d1fae5'},
                        {'range': [30, 60], 'color': '#fef3c7'},
                        {'range': [60, 100], 'color': '#fee2e2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.info("👉 **Next Step**: Click on **'2 Explanation'** in the left sidebar to inspect the detailed SHAP waterfall attribution breakdown for this specific patient!")
