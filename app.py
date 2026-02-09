import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🩺 Diabetes Risk Prediction System</h1>
    <p style='text-align: center; color: gray;'>
    AI-powered health risk assessment using Logistic Regression
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Input Section
# -----------------------------
st.subheader("👤 Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        help="Number of times pregnant"
    )

    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=50,
        max_value=300,
        value=120,
        help="Blood sugar concentration after 2 hours"
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=40,
        max_value=200,
        value=100,
        help="Diastolic blood pressure"
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=5,
        max_value=100,
        value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (µU/ml)",
        min_value=0,
        max_value=900,
        value=80,
        help="2-hour serum insulin level"
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=30.0,
        help="Body Mass Index"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        help="Genetic likelihood of diabetes"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.markdown("---")

# -----------------------------
# Prediction Button
# -----------------------------
center_col = st.columns([1, 2, 1])[1]
with center_col:
    predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:
    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if probability >= 0.4:
        st.error("⚠️ **High Risk of Diabetes**")
    else:
        st.success("✅ **Low Risk of Diabetes**")

    st.write(f"**Predicted Probability:** `{probability * 100:.2f}`")
    st.progress(int(probability*100))

    # Medical disclaimer
    st.info(
        "⚠️ This tool is for educational purposes only. "
        "Please consult a medical professional for diagnosis."
    )
