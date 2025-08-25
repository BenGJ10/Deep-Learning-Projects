import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Loading all the saved models

model = tf.keras.models.load_model("models/ann_model.h5")

with open("models/gender_label_encoder.pkl", 'rb') as file:
    gender_label_encoder = pickle.load(file)

with open('models/geography_ohe.pkl', 'rb') as file:
    geography_ohe = pickle.load(file)

with open('models/standard_scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

# Streamlit App

# Set page configuration
st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon = "📉",
    layout = "centered",
)

# Custom CSS styling
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 {
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 30px;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html = True)

# App Title
st.title("📉 Customer Churn Prediction")
st.markdown("Use the form below to input customer details and predict the likelihood of churn.")

# Input layout
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', geography_ohe.categories_[0])
        gender = st.selectbox('👤 Gender', gender_label_encoder.classes_)
        age = st.slider('🎂 Age', 18, 92)
        tenure = st.slider('📆 Tenure (Years)', 0, 10)
        num_of_products = st.slider('🛒 Number of Products', 1, 4)

    with col2:
        credit_score = st.number_input('💳 Credit Score', min_value = 0)
        balance = st.number_input('🏦 Balance', min_value = 0.0)
        estimated_salary = st.number_input('💰 Estimated Salary', min_value = 0.0)
        has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
        is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

    submitted = st.form_submit_button("🔍 Predict Churn")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One Hot Encoding 'Geography'
geography_encoded = geography_ohe.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns = geography_ohe.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop = True), geography_encoded_df], axis = 1)

# Scale the input data
input_data_scaled = standard_scaler.transform(input_data)

# Model Prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]


st.markdown("---")
st.subheader("🔎 Prediction Result")
st.progress(int(prediction_probability * 100))

st.metric(label="📊 Churn Probability", value = f"{prediction_probability:.2%}")

if prediction_probability > 0.5:
    st.markdown(
        f"<div style='color:red; font-size:20px; font-weight:bold;'>⚠️ The customer is likely to churn.</div>",
        unsafe_allow_html = True
    )
else:
    st.markdown(
        f"<div style='color:green; font-size:20px; font-weight:bold;'>✅ The customer is not likely to churn.</div>",
        unsafe_allow_html = True
    )