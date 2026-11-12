import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("heart_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("❤️ Heart Disease Risk Predictor")
st.write("Patient details daalo — model risk batayega. Ye sirf learning ke liye hai, medical advice nahi hai.")

# Inputs
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 120, 570, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict Risk"):
    data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "ca": ca, "thal": thal
    }])
    data_scaled = scaler.transform(data)
    risk = model.predict_proba(data_scaled)[0][1] * 100

    st.write(f"Risk: **{risk:.1f}%**")
    st.progress(int(risk))

    if risk < 30:
        st.success("🟢 Low Risk — healthy lifestyle continue rakho.")
    elif risk < 70:
        st.warning("🟡 Moderate Risk — doctor se consult karo.")
    else:
        st.error("🔴 High Risk — urgent cardiac evaluation karwao.")
