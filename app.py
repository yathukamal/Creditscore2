
#Create streamlit app
import streamlit as st
import pandas as pd
import joblib

#--------------------
# Load trained model
#---------------------
model = joblib.load("logistic_model_4.pkl")

#--------------------
#Page title
#--------------------
st.title("Credit Default Predictor")

st.write(
    "Adjust the financial indicators below to estimate"
    "the customer's likelihood of default."
)

#--------------------
#Input sliders
#--------------------
r_tax_debt = st.slider(
    "Tax / Debt Ratio",
    min_value=0.00,
    max_value=0.40,
    value=0.0052,
    step=0.0010
)

r_debt_income = st.slider(
    "Debt / Income Ratio",
    min_value=0.0000,
    max_value=37.00,
    value=4.9116,
    step=0.1
)

r_expenditure_debt = st.slider(
    "Expenditure / Debt Ratio",
    min_value=0.0000,
    max_value=10.01,
    value=0.1786,
    step=0.01
)

r_travel_debt = st.slider(
    "Travel / Debt Ratio",
    min_value=0.0000,
    max_value=7.34,
    value=0.0483,
    step=0.01
)

r_health_debt = st.slider(
    "Health / Debt Ratio",
    min_value=0.0000,
    max_value=2.03,
    value=0.0068,
    step=0.01
)

r_groceries_savings = st.slider(
    "Groceries / Savings Ratio",
    min_value=0.0000,
    max_value=3.56,
    value=0.0478,
    step=0.01
)

r_groceries_income = st.slider(
    "Groceries / Income Ratio",
    min_value=0.0000,
    max_value=0.56,
    value=0.1330,
    step=0.001
)

r_health = st.slider(
    "Health Ratio",
    min_value=0.0000,
    max_value=1.00,
    value=0.4787,
    step=0.01
)

r_entertainment = st.slider(
    "Entertainment Ratio",
    min_value=0.40,
    max_value=0.79,
    value=0.5119,
    step=0.001
)

#----------------------
#Create input DataFrame
#----------------------

user_input = pd.DataFrame([{
    "R_TAX_DEBT": r_tax_debt,
    "R_DEBT_INCOME": r_debt_income,
    "R_EXPENDITURE_DEBT": r_expenditure_debt,
    "R_TRAVEL_DEBT": r_travel_debt,
    "R_HEALTH_DEBT": r_health_debt,
    "R_GROCERIES_SAVINGS": r_groceries_savings,
    "R_GROCERIES_INCOME": r_groceries_income,
    "R_HEALTH": r_health,
    "R_ENTERTAINMENT": r_entertainment
}])


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Default Risk"):

    probabilities = model.predict_proba(user_input)[0]


    non_default_probability = probabilities[0]
    default_probability = probabilities[1]

    prediction = model.predict(user_input)[0]


    # -------------------------
    # Display probabilities
    # -------------------------

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Non-default probability",
            f"{non_default_probability:.1%}"
)

    with col2:
        st.metric(
            "Default probability",
            f"{default_probability:.1%}"
        )


    # -------------------------
    # Display prediction
    # -------------------------

    if prediction == 1:
        st.error("Prediction: Likely to default")
    else:
        st.success("Prediction: Likely not to default")