import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parent / "model_deployments"
FEATURE_ORDER = [
    "TyG_Index",
    "Waist Measurement",
    "newhdl",
    "Jenis_Kelamin",
    "Systolic",
    "Weight",
    "newtg",
    "Diastolic",
    "Usia",
    "newua",
    "Height",
]


@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    lbm_model = joblib.load(MODEL_DIR / "lbm_model.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    stacking_model = joblib.load(MODEL_DIR / "stacking_meta_model.pkl")
    return scaler, lbm_model, xgb_model, stacking_model


def build_feature_frame(values):
    gender_value = 1 if values["gender"] == "Male" else 0
    record = {
        "TyG_Index": values["tyg_index"],
        "Jenis_Kelamin": gender_value,
        "newhdl": values["hdl"],
        "Waist Measurement": values["waist_measurement"],
        "Systolic": values["systolic"],
        "Weight": values["weight"],
        "newtg": values["triglycerols"],
        "Diastolic": values["diastolic"],
        "Usia": values["age"],
        "newua": values["uric_acid"],
        "Height": values["height"],
    }
    return pd.DataFrame([record], columns=FEATURE_ORDER)


def classify_probability(probability):
    if probability >= 0.7:
        return "High risk"
    if probability >= 0.4:
        return "Moderate risk"
    return "Low risk"


def recommendation_for_user(prediction, probability):
    if prediction == 1:
        return (
            "The model indicates a likely MetS risk. Please arrange a clinical review with your healthcare provider, "
            "reduce saturated fats and refined sugar, walk at least 30–45 minutes most days, improve sleep quality, "
            "and monitor blood pressure, waist size, and weight over the next few weeks."
        )
    return (
        "The model indicates a low MetS risk at the moment. Keep a healthy routine by staying active, eating balanced meals, "
        "maintaining a healthy weight, limiting sugary drinks and processed foods, and continuing regular health check-ups."
    )


def predict_with_models(scaler, lbm_model, xgb_model, stacking_model, values):
    feature_frame = build_feature_frame(values)
    scaled_features = scaler.transform(feature_frame)

    lbm_prob = float(lbm_model.predict_proba(scaled_features)[0, 1])
    xgb_prob = float(xgb_model.predict_proba(scaled_features)[0, 1])
    stack_input = np.array([[xgb_prob, lbm_prob]], dtype=float)
    stack_prob = float(stacking_model.predict_proba(stack_input)[0, 1])
    stack_pred = int(stacking_model.predict(stack_input)[0])

    results = {
        "LBM": {
            "prediction": int(lbm_model.predict(scaled_features)[0]),
            "probability": lbm_prob,
        },
        "XGB": {
            "prediction": int(xgb_model.predict(scaled_features)[0]),
            "probability": xgb_prob,
        },
        "Stacking": {
            "prediction": stack_pred,
            "probability": stack_prob,
        },
    }
    return results


st.set_page_config(page_title="MetS Prediction System", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f5fbff 0%, #edf6ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a, #1e3a8a 60%, #2563eb);
            border-radius: 20px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 10px 25px rgba(30, 58, 138, 0.15);
            margin-bottom: 1.5rem;
        }
        .hero-title {
            color: #f8fbff;
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
        }
        .hero-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            margin-top: 0.4rem;
        }
        .section-card {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            margin-bottom: 1.2rem;
        }
        .model-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 8px 16px rgba(37, 99, 235, 0.08);
            height: 100%;
        }
        .model-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }
        .model-probability {
            font-size: 2rem;
            font-weight: 800;
            color: #1d4ed8;
            margin: 0.2rem 0;
        }
        .model-label {
            color: #475569;
            font-size: 0.92rem;
        }
        .stForm {
            background: transparent;
        }
        div[data-testid="stNumberInput"] > div {
            background: #fff;
            border-radius: 12px;
        }
        div[data-testid="stSelectbox"] > div {
            background: #fff;
            border-radius: 12px;
        }
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            height: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <p class="hero-title">MetS Prediction System</p>
        <p class="hero-subtitle">Clinical assessment for metabolic syndrome risk using LBM, XGB, and Stacking models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("mets_form"):
    st.markdown('<div class="section-card"><h3 style="margin:0 0 0.8rem 0; color:#0f172a;">Patient information</h3></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=120, value=42)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=72.0)
        waist_measurement = st.number_input("Waist measurement (cm)", min_value=40.0, max_value=200.0, value=91.0)

    with col2:
        systolic = st.number_input("Systolic (mmHg)", min_value=60.0, max_value=220.0, value=120.0)
        diastolic = st.number_input("Diastolic (mmHg)", min_value=40.0, max_value=140.0, value=80.0)
        tyg_index = st.number_input("TyG index", min_value=3.0, max_value=15.0, value=8.5, step=0.1)
        hdl = st.number_input("HDL (mg/dL)", min_value=10.0, max_value=120.0, value=45.0, step=0.5)

    with col3:
        triglycerols = st.number_input("Triglycerols (mg/dL)", min_value=20.0, max_value=600.0, value=140.0, step=1.0)
        uric_acid = st.number_input("Uric acid level (mg/dL)", min_value=1.0, max_value=15.0, value=5.6, step=0.1)

    submitted = st.form_submit_button("Run prediction")

if submitted:
    values = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "waist_measurement": waist_measurement,
        "systolic": systolic,
        "diastolic": diastolic,
        "tyg_index": tyg_index,
        "hdl": hdl,
        "triglycerols": triglycerols,
        "uric_acid": uric_acid,
    }

    scaler, lbm_model, xgb_model, stacking_model = load_models()
    results = predict_with_models(scaler, lbm_model, xgb_model, stacking_model, values)

    st.markdown('<div class="section-card"><h3 style="margin:0 0 0.8rem 0; color:#0f172a;">Prediction summary</h3></div>', unsafe_allow_html=True)
    summary_cols = st.columns(3)
    for idx, (model_name, result) in enumerate(results.items()):
        probability = result["probability"]
        label = "MetS risk" if result["prediction"] == 1 else "No MetS risk"
        with summary_cols[idx]:
            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-title">{model_name}</div>
                    <div class="model-probability">{probability * 100:.1f}%</div>
                    <div class="model-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    tabs = st.tabs(["LBM Model", "XGB Model", "Stacking Model"])

    for tab, (_, result) in zip(tabs, results.items()):
        with tab:
            probability = result["probability"]
            prediction = result["prediction"]
            label = "MetS risk detected" if prediction == 1 else "No MetS risk detected"
            risk_level = classify_probability(probability)

            st.markdown(
                f"""
                <div class="section-card">
                    <div style="font-size:1.1rem; font-weight:700; color:#0f172a;">{label}</div>
                    <div style="font-size:0.95rem; color:#475569; margin-top:0.3rem;">Risk level: {risk_level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Probability of MetS", f"{probability * 100:.1f}%")
            with col_b:
                st.info(recommendation_for_user(prediction, probability))

    st.markdown("---")
    avg_probability = np.mean([result["probability"] for result in results.values()])
    consensus_prediction = 1 if avg_probability >= 0.5 else 0
    st.markdown('<div class="section-card"><h3 style="margin:0 0 0.8rem 0; color:#0f172a;">Overall interpretation</h3></div>', unsafe_allow_html=True)
    st.write(f"Average probability across the three models: {avg_probability * 100:.1f}%")
    if consensus_prediction == 1:
        st.warning("Overall assessment: High likelihood of metabolic syndrome. Follow-up with a clinician is strongly recommended.")
    else:
        st.success("Overall assessment: Low likelihood of metabolic syndrome based on the current input values.")
    st.info(
        "General recommendation: maintain a healthy weight, exercise regularly, reduce sugar and saturated fat intake, "
        "keep blood pressure and waist circumference in check, and schedule routine medical follow-up."
    )
else:
    st.markdown(
        """
        <div class="section-card">
            <div style="font-size:1rem; color:#475569;">Enter the patient data and click the prediction button to generate MetS risk results.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
