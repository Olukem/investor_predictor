import streamlit as st
import pandas as pd
import mlflow.sklearn

# ====== CUSTOM PAGE CONFIG ======
st.set_page_config(
    page_title="Long-Term Investor Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

# ====== CUSTOM DARK THEME STYLE ======
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .stApp {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .stButton>button {
        background-color: #444444;
        color: white;
        border-radius: 5px;
    }
    .stSelectbox label, .stNumberInput label {
        color: #f0f0f0 !important;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 13px;
        color: #999999;
        padding: 10px;
    }
    .footer a {
        color: #bbbbbb;
        margin: 0 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ====== LOAD MODEL ======
run_id = "7e7edbfec7c6489d9965aa803cddd396"
model_uri = f"runs:/{run_id}/catboost_model"
model = mlflow.sklearn.load_model(model_uri)

# ====== APP HEADER ======
st.title("Long-Term Investor Prediction")
st.markdown("Enter Client Information.")

# ====== CATEGORICAL INPUTS ======
job = st.selectbox("Job", [
    "admin.", "technician", "services", "management", "retired", 
    "blue-collar", "unemployed", "entrepreneur", "housemaid", 
    "self-employed", "student", "unknown"
])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default Credit?", ["yes", "no"])
housing = st.selectbox("Housing Loan?", ["yes", "no"])
loan = st.selectbox("Personal Loan?", ["yes", "no"])
contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
month = st.selectbox("Last Contact Month", [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", 
    "sep", "oct", "nov", "dec"
])
poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "other", "unknown"])

# ====== NUMERIC INPUTS ======
age = st.number_input("Age", min_value=18, max_value=100, value=35)
balance = st.number_input("Account Balance (€)", value=1000)
day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
duration = st.number_input("Call Duration (sec)", min_value=0, max_value=5000, value=250)
campaign = st.number_input("Contacts During Campaign", min_value=1, max_value=50, value=2)
pdays = st.number_input("Days Since Last Contact", value=999)
previous = st.number_input("Previous Campaign Contacts", value=0)

# ====== PREPARE DATAFRAME ======
input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

# ====== PREDICT ======
if st.button("Predict Now"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.success("✅ YES — Likely to deposit" if prediction == 1 else "❌ NO — Unlikely to deposit")
    st.write(f"**Confidence Score:** {probability:.2%}")

# ====== FOOTER ======
# st.markdown("""
#     <style>
#     .footer {
#         position: fixed;
#         bottom: 0;
#         left: 0;
#         right: 0;
#         background: #1e1e1e;
#         text-align: center;
#         padding: 12px 0;
#         font-size: 14px;
#         font-weight: bold;
#         color: #d4bfff;
#     }
#     .footer a {
#         color: #f0a500;
#         text-decoration: none;
#         margin: 0 10px;
#     }
#     .footer a:hover {
#         text-decoration: underline;
#     }
#     </style>

#     <div class="footer">
#         Developed by Oluwakemi – ©2025<br>
#         <a href="https://www.linkedin.com/in/oluwakemi-sorinmade/" target="_blank">LinkedIn</a>
#         <a href="https://github.com/Olukem" target="_blank">GitHub</a>
#     </div>
# """, unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1e1e1e;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        font-weight: bold;
        color: #d4bfff;
        z-index: 100;
    }
    .footer a {
        color: #d4bfff; /* Match text color */
        text-decoration: none;
        margin: 0 15px;
        font-size: 20px;
    }
    .footer a:hover {
        color: white;
    }
    </style>

    <div class="footer">
        Developed by Oluwakemi – ©2025<br>
        <a href="https://www.linkedin.com/in/oluwakemi-sorinmade/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/Olukem" target="_blank">
            <i class="fab fa-github"></i>
        </a>
    </div>
""", unsafe_allow_html=True)
