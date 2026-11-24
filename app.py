
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/build_model.pkl")

st.set_page_config(
    page_title="Build Failure Prediction",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Build Failure Prediction System")

st.write(
    "Predict whether a software build will succeed "
    "or fail using Machine Learning."
)

st.divider()

st.subheader("Enter Build Information")

code_changes = st.number_input(
    "Number of Code Changes", min_value=0, value=10
)

files_modified = st.number_input(
    "Files Modified", min_value=0, value=5
)

errors = st.number_input(
    "Number of Errors", min_value=0, value=0
)

warnings = st.number_input(
    "Number of Warnings", min_value=0, value=2
)

build_duration = st.number_input(
    "Build Duration (seconds)", min_value=0, value=60
)

previous_failures = st.number_input(
    "Previous Build Failures", min_value=0, value=0
)

if st.button("Predict Build Status", type="primary"):

    input_data = pd.DataFrame([{
        "code_changes": code_changes,
        "files_modified": files_modified,
        "errors": errors,
        "warnings": warnings,
        "build_duration": build_duration,
        "previous_failures": previous_failures
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ BUILD FAILURE PREDICTED")
    else:
        st.success("✅ BUILD SUCCESS PREDICTED")

    st.metric(
        "Estimated Failure Probability",
        f"{probability * 100:.2f}%"
    )
