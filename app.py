# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
# Load data, train model, and encode
@st.cache_data
def train_and_save_model():
    df = pd.read_csv("D:\\TechNest\heart.csv")  # Place heart.csv in same folder

    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    le_dict = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model and encoders
    joblib.dump(model, "D:\\TechNest\heart_model.pkl")
    joblib.dump(le_dict, "D:\\TechNest\encoders.pkl")

    return model, le_dict, list(X.columns)

# Load or train
model, le_dict, feature_order = train_and_save_model()

# Streamlit UI
# st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title(" Heart Disease Prediction App")
st.markdown("Enter the patient details below:")

# Input Fields
age = st.slider("Age", 1, 100, 45)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
rbp = st.number_input("Resting Blood Pressure", min_value=0, value=120)
chol = st.number_input("Cholesterol", min_value=0, value=200)
fbs = st.selectbox("FastingBS (1 = True, 0 = False)", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "T"])
maxhr = st.number_input("Max Heart Rate", min_value=0, value=150)
angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, value=1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# On click Predict
if st.button("Predict"):
    try:
        data = {
            'Age': age,
            'Sex': le_dict['Sex'].transform([sex])[0],
            'ChestPainType': le_dict['ChestPainType'].transform([cp])[0],
            'RestingBP': rbp,
            'Cholesterol': chol,
            'FastingBS': fbs,
            'RestingECG': le_dict['RestingECG'].transform([ecg])[0],
            'MaxHR': maxhr,
            'ExerciseAngina': le_dict['ExerciseAngina'].transform([angina])[0],
            'Oldpeak': oldpeak,
            'ST_Slope': le_dict['ST_Slope'].transform([slope])[0]
        }

        input_df = pd.DataFrame([data])
        input_df = input_df[feature_order]  # Ensure correct column order
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error(" High Risk of Heart Disease")
        else:
            st.success(" Low Risk of Heart Disease")

    except Exception as e:
        st.warning(f" Error in prediction: {str(e)}")
