"""Streamlit UI for teen stress prediction."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "final_pipeline.joblib"
BADGES = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Teen Stress Level Prediction")
st.write("Enter daily habits to estimate self-reported stress (1-5). Informational tool only, not medical advice.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 13, 18, 16)
    sleep = st.slider("Sleep hours", 4.0, 10.0, 7.0, 0.1)
    screen = st.slider("Screen time hours", 2.0, 12.0, 7.0, 0.1)
    social = st.slider("Social media hours", 0.0, 10.0, 5.0, 0.1)
with col2:
    exercise = st.slider("Exercise hours", 0.0, 3.0, 1.5, 0.1)
    gender = st.selectbox("Gender", ["F", "M"])
    support = st.selectbox("Support system", ["Low", "Moderate", "High"])
    academic = st.selectbox("Academic performance", ["Poor", "Average", "Good", "Excellent"])

if st.button("Predict Stress Level"):
    pipe = load_model()
    row = pd.DataFrame([{"age": age, "social_media_hours": social,
                         "exercise_hours": exercise, "sleep_hours": sleep,
                         "screen_time_hours": screen, "gender": gender,
                         "support_system": support, "academic_performance": academic}])
    proba = pipe.predict_proba(row)[0]
    level = int(pipe.predict(row)[0]) + 1
    st.subheader(f"Predicted level: {level} ({BADGES[level]})")
    st.bar_chart(pd.DataFrame({"proba": proba}, index=[1, 2, 3, 4, 5]))

st.caption("Model: LogisticRegression on 8 lifestyle features. Test macro F1 about 0.21 (weak signal).")
