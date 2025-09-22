import streamlit as st
from model_training import predict_student

st.title("⚠️ Risk Analysis")

age = st.slider("Age", 15, 25, 18)
study = st.slider("Study Hours/Week", 0, 40, 10)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep = st.slider("Sleep Hours", 4, 10, 6)
gender = st.radio("Gender", ["Male", "Female"])
medium = st.radio("Medium of Instruction", ["English", "Regional"])
extra = st.radio("Extra Curricular", ["Yes", "No"])

if st.button("Analyze Risk"):
    perf, risk = predict_student(age, study, attendance, sleep, gender, medium, extra)
    st.info(f"Risk Status: {risk}")

import pandas as pd
from model_training import perf_model, risk_model

st.subheader("Or Upload a CSV file")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    perf_preds = perf_model.predict(df)
    risk_preds = risk_model.predict(df)

    df["Predicted Performance"] = perf_preds
    df["Predicted Risk"] = risk_preds

    st.write("Results:", df.head())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results as CSV", csv, "results.csv", "text/csv")
