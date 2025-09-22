import streamlit as st

st.title("ℹ️ About Project")

st.write("""
This project predicts student performance and identifies academic risks.  

### Features Used:
- Age
- Study Hours/Week
- Attendance (%)
- Sleep Hours
- Gender
- Medium of Instruction
- Extra Curricular Activities

**Models Used:** Random Forest Classifiers  
**Dataset:** Custom dataset with 1000 student records  
""")

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
