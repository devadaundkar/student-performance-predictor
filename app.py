"""
app.py
-------
Streamlit web app for real-time score prediction.

Run locally:   streamlit run app.py
Deploy (free): push this folder to GitHub, then create an app on
               Streamlit Community Cloud pointing at app.py.

If model.joblib doesn't exist yet (e.g. first run on a fresh server), the app
generates the data and trains the model automatically.
"""

import os

import joblib
import pandas as pd
import streamlit as st

from preprocessing import FEATURE_COLUMNS, engineer_features

MODEL_PATH = "model.joblib"
REPORT_DIR = "reports"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")


@st.cache_resource
def load_bundle():
    """Load the saved model once and cache it; train it first if missing."""
    if not os.path.exists(MODEL_PATH):
        from generate_data import generate_student_data
        from train_model import train_and_save_model
        os.makedirs("data", exist_ok=True)
        if not os.path.exists("data/students.csv"):
            generate_student_data().to_csv("data/students.csv", index=False)
        train_and_save_model()
    return joblib.load(MODEL_PATH)


def risk_band(score: float):
    if score >= 75:
        return "🟢", "Strong predicted performance."
    if score >= 50:
        return "🟡", "Moderate predicted performance - room to improve."
    return "🔴", "At risk - consider more study time and support."


def main():
    st.title("🎓 Student Performance Predictor")
    bundle = load_bundle()
    model = bundle["model"]

    tab_predict, tab_perf = st.tabs(["Predict", "Model performance"])

    with tab_predict:
        st.write("Enter a student's details to predict the likely final exam score.")
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("Study hours per day", 0.0, 10.0, 4.0, 0.5)
            attendance = st.slider("Attendance (%)", 50.0, 100.0, 80.0, 1.0)
            previous_score = st.slider("Previous exam score", 35.0, 100.0, 65.0, 1.0)
        with col2:
            sleep_hours = st.slider("Sleep hours per day", 4.0, 9.0, 7.0, 0.5)
            extra_classes = st.selectbox("Taking extra classes?", ["No", "Yes"])
            parental_education_years = st.slider(
                "Parental education (years of schooling)", 8, 20, 14, 1)

        raw_input = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "sleep_hours": sleep_hours,
            "extra_classes": 1 if extra_classes == "Yes" else 0,
            "parental_education_years": parental_education_years,
        }])
        # Same feature engineering as training -> identical model inputs
        model_input = engineer_features(raw_input)[FEATURE_COLUMNS]

        if st.button("Predict Final Score"):
            prediction = float(max(0, min(100, model.predict(model_input)[0])))
            icon, message = risk_band(prediction)
            st.metric("Predicted final score", f"{prediction:.1f} / 100")
            st.progress(int(prediction))
            st.write(f"{icon} {message}")
            st.caption(f"Typical error of this model: about ±{bundle['test_metrics']['mae']:.1f} points (MAE).")

    with tab_perf:
        m = bundle["test_metrics"]
        st.subheader(f"Selected model: {bundle['model_name']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (points)", f"{m['mae']:.2f}")
        c2.metric("RMSE (points)", f"{m['rmse']:.2f}")
        c3.metric("R²", f"{m['r2']:.3f}")
        st.caption(f"Evaluated on {bundle['n_test']} held-out students "
                   f"(trained on {bundle['n_train']}).")
        st.subheader("All models compared")
        st.dataframe(pd.DataFrame(bundle["comparison"]), hide_index=True)
        for img, title in [("feature_importance.png", "Feature importance"),
                           ("residuals.png", "Residual analysis")]:
            path = os.path.join(REPORT_DIR, img)
            if os.path.exists(path):
                st.subheader(title)
                st.image(path)
        st.info("Note: this model was trained on a synthetic dataset, so it "
                "demonstrates the pipeline rather than real-world accuracy.")


if __name__ == "__main__":
    main()
