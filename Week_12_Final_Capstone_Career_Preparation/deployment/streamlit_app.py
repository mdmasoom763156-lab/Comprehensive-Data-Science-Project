"""Interactive Streamlit frontend for churn-risk prediction."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.predict import predict_customer


st.set_page_config(page_title="Churn Risk Studio", page_icon="📊", layout="wide")
st.title("Customer Churn Risk Studio")
st.caption("Enter account details to estimate churn risk and receive a retention recommendation.")

with st.form("prediction_form"):
    left, middle, right = st.columns(3)
    with left:
        tenure = st.number_input("Tenure (months)", 0, 120, 8)
        monthly = st.number_input("Monthly charges", 0.0, 1000.0, 180.0)
        total = st.number_input("Total charges", 0.0, 100000.0, 1400.0)
    with middle:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", ["Electronic Check", "Credit Card", "Bank Transfer"])
    with right:
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    result = predict_customer(
        {
            "Tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "PaymentMethod": payment,
            "PaperlessBilling": paperless,
            "SeniorCitizen": senior,
        }
    )
    st.subheader("Prediction result")
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{result['churn_probability']:.1%}")
    b.metric("Risk level", result["risk_level"])
    c.metric("Model decision", "Likely churn" if result["prediction"] else "Likely stay")
    if result["risk_level"] == "High":
        st.error(result["recommended_action"])
    elif result["risk_level"] == "Medium":
        st.warning(result["recommended_action"])
    else:
        st.success(result["recommended_action"])

with st.expander("Responsible-use note"):
    st.write("Use this educational model to prioritize human review, not to deny service or make fully automated decisions.")
