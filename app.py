import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model and scaler
model = joblib.load("heart_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("❤️ Heart Disease Risk Predictor")
st.write("Patient details daalo — model risk batayega. Ye sirf learning ke liye hai, medical advice nahi hai.")

# 1. Time show karo — sidebar mein
now = datetime.now()
st.sidebar.write("📅 Date:", now.strftime("%d %b %Y"))
st.sidebar.write("🕐 Time:", now.strftime("%I:%M %p"))

# History list — pehli baar khali banao
if "history" not in st.session_state:
    st.session_state.history = []

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
        band = "Low"
        st.success("🟢 Low Risk — healthy lifestyle continue rakho.")
    elif risk < 70:
        band = "Moderate"
        st.warning("🟡 Moderate Risk — doctor se consult karo.")
    else:
        band = "High"
        st.error("🔴 High Risk — urgent cardiac evaluation karwao.")

    # 2. History mein save karo
    st.session_state.history.append({
        "Time": datetime.now().strftime("%I:%M %p"),
        "Age": age,
        "Sex": "Male" if sex == 1 else "Female",
        "BP": trestbps,
        "Cholesterol": chol,
        "Risk %": round(risk, 1),
        "Band": band
    })

# 2. History show karo
st.subheader("📋 Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.write("Abhi koi prediction nahi hai — pehla prediction karo.")
