import streamlit as st
from model_training import predict_student

st.title("🎯 Student Performance Prediction")

age = st.number_input("Age", min_value=15, max_value=25, value=18)
study = st.number_input("Study Hours/Week", min_value=0, max_value=40, value=10)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
sleep = st.number_input("Sleep Hours", min_value=4, max_value=10, value=6)
gender = st.radio("Gender", ["Male", "Female"])
medium = st.radio("Medium of Instruction", ["English", "Regional"])
extra = st.radio("Extra Curricular", ["Yes", "No"])

if st.button("Predict Performance"):
    perf, risk = predict_student(age, study, attendance, sleep, gender, medium, extra)
    st.success(f"Predicted Performance: {perf}")

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
