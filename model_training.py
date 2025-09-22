import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("studentpredictionandriskanalysis.csv")

# Create synthetic labels
def assign_performance(row):
    if row["Study Hours/Week"] >= 15 and row["Attendance (%)"] >= 80 and row["Sleep Hours"] >= 6:
        return "High Performer"
    elif row["Study Hours/Week"] >= 7 and row["Attendance (%)"] >= 60:
        return "Medium Performer"
    else:
        return "Low Performer"

def assign_risk(row):
    if row["Attendance (%)"] < 60 or row["Study Hours/Week"] < 5:
        return "At Risk"
    else:
        return "Not At Risk"

df["Performance"] = df.apply(assign_performance, axis=1)
df["Risk"] = df.apply(assign_risk, axis=1)

# Features and targets
X = df.drop(["Performance", "Risk"], axis=1)
y_perf = df["Performance"]
y_risk = df["Risk"]

categorical = ["Gender", "Medium of Instruction", "Extra Curricular"]
numerical = ["Age", "Study Hours/Week", "Attendance (%)", "Sleep Hours"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical),
    ("num", "passthrough", numerical)
])

# Train Performance model
perf_model = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])
perf_model.fit(X, y_perf)

# Train Risk model
risk_model = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])
risk_model.fit(X, y_risk)

# Function for predictions
def predict_student(age, study_hours, attendance, sleep, gender, medium, extra):
    input_df = pd.DataFrame([{
        "Age": age,
        "Study Hours/Week": study_hours,
        "Attendance (%)": attendance,
        "Sleep Hours": sleep,
        "Gender": gender,
        "Medium of Instruction": medium,
        "Extra Curricular": extra
    }])
    perf = perf_model.predict(input_df)[0]
    risk = risk_model.predict(input_df)[0]
    return perf, risk
