import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and column structure
model = joblib.load('salary_model.pkl')
model_columns = joblib.load('model_columns.pkl')


st.title(" AI Job Salary Predictor")
st.markdown("""
This app predicts the **Average Annual Salary** for AI roles based on industry, experience, and company details.
""")

st.sidebar.header("User Input Features")

# Create input widgets for the user
industry = st.sidebar.selectbox("Industry", ['Healthcare', 'Tech', 'Finance', 'E-commerce', 'Automotive', 'Education', 'Retail'])
experience = st.sidebar.selectbox("Experience Level", ['Entry', 'Mid', 'Senior'])
comp_size = st.sidebar.selectbox("Company Size", ['Startup', 'Mid', 'Large'])
emp_type = st.sidebar.selectbox("Employment Type", ['Full-time', 'Contract', 'Remote', 'Internship'])

# 3. Process Input to match Training Format
def predict_salary():
    # Create a dataframe with all zeros, matching our training columns
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Set the selected categories to 1 (One-Hot Encoding)
    # We use "C(column)[T.value]" format because of how pd.get_dummies works
    if f"experience_level_{experience}" in model_columns:
        input_data[f"experience_level_{experience}"] = 1
    if f"industry_{industry}" in model_columns:
        input_data[f"industry_{industry}"] = 1
    if f"company_size_{comp_size}" in model_columns:
        input_data[f"company_size_{comp_size}"] = 1
    if f"employment_type_{emp_type}" in model_columns:
        input_data[f"employment_type_{emp_type}"] = 1

    prediction = model.predict(input_data)
    return prediction[0]

if st.button("Predict Salary"):
    result = predict_salary()
    st.success(f"### The estimated annual salary is: ${result:,.2f}")
    
    # Add some context for the project report
    st.info("Note: This prediction is based on the Linear Regression model trained on ourstreamlit run app.py AI Job Market dataset.")