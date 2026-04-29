import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.save")

st.set_page_config(page_title="EV Runtime Predictor", layout="wide")

st.title("🔋 EV Runtime Predictor (LSTM)")
st.markdown("Predict EV runtime based on battery condition")

# Sidebar
st.sidebar.header("Input Parameters")

battery_pct = st.sidebar.slider("Battery %", 0, 100, 50)
voltage = st.sidebar.number_input("Voltage (V)", value=48.0)
current = st.sidebar.number_input("Current (A)", value=10.0)
temperature = st.sidebar.number_input("Temperature (°C)", value=30.0)
soh = st.sidebar.slider("SOH (%)", 50, 100, 90)

# Prediction function
def predict_runtime():
    effective_soh = (battery_pct / 100.0) * (soh / 100.0) * 100
    
    input_data = np.array([[voltage, current, temperature, effective_soh]])
    input_scaled = scaler.transform(input_data)

    input_seq = np.repeat(input_scaled, 10, axis=0).reshape(1, 10, 4)
    
    pred = model.predict(input_seq)[0][0]
    return pred / 3600

# Button
if st.button("Predict Runtime"):
    runtime = predict_runtime()

    st.subheader("Result")
    st.metric("Estimated Runtime (hours)", f"{runtime:.2f}")

    if runtime > 3:
        st.success("Good battery performance")
    elif runtime > 1:
        st.warning("Moderate runtime")
    else:
        st.error("Low runtime — charge soon")
