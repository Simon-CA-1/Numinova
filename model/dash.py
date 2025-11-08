import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
from pathlib import Path


# -------------------------------
# 1. Load Model and Encoder
# -------------------------------
model_path = Path(__file__).resolve().parent / "equipment_failure_model.pkl"
encoder_path = Path(__file__).resolve().parent / "equipment_label_encoder.pkl"

try:
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
except FileNotFoundError:
    st.error("❌ Model or encoder not found. Please run `train.py` first.")
    st.stop()

# Extract model features
model_features = getattr(model, "feature_names_in_", [])
equipment_types = getattr(le, "classes_", [])


# -------------------------------
# 2. Helper Function
# -------------------------------
def simulate_equipment_reading():
    """Generate a simulated real-time reading."""
    return {
        "equipment_type": random.choice(list(equipment_types)),
        "temperature": np.random.normal(60, 15),
        "pressure": np.random.normal(100, 30),
        "corrosion_index": np.random.randint(20, 90),
        "usage_hours": np.random.randint(100, 8000),
        "last_maintenance_days": np.random.randint(1, 365),
        "vibration_level": np.random.randint(10, 100),
        "gas_leak_detected": np.random.choice([0, 1], p=[0.9, 0.1]),
    }


# -------------------------------
# 3. Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Equipment Monitoring Dashboard", layout="centered")

st.title("🏭 Real-Time Industrial Equipment Monitoring")
st.markdown("### AI-Powered Failure Detection System")

placeholder = st.empty()
chart_data = pd.DataFrame(columns=["temperature", "pressure", "vibration_level", "gas_leak_detected"])

# -------------------------------
# 4. Live Simulation Loop
# -------------------------------
for _ in range(200):  # Adjust duration if needed
    data = simulate_equipment_reading()
    df = pd.DataFrame([data])

    # Encode and align with model
    df["equipment_type"] = le.transform(df["equipment_type"])
    for f in model_features:
        if f not in df.columns:
            df[f] = 0
    df = df[model_features]

    prediction = model.predict(df)[0]

    gas_leak = int(data["gas_leak_detected"])
    is_critical = prediction == 1

    # Status logic
    if gas_leak == 1:
        status = "🚨 GAS LEAK DETECTED — IMMEDIATE ATTENTION REQUIRED"
        color = "red"
    elif is_critical:
        status = "⚠️ CRITICAL FAILURE RISK"
        color = "orange"
    else:
        status = "✅ NORMAL OPERATION"
        color = "green"

    # Append to history for chart
    chart_data.loc[len(chart_data)] = [
        data["temperature"],
        data["pressure"],
        data["vibration_level"],
        gas_leak,
    ]

    # Display Live Dashboard
    with placeholder.container():
        st.markdown(f"### Equipment Type: `{data['equipment_type']}`")
        st.markdown(f"#### Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

        st.write("##### Sensor Readings:")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": ["Temperature (°C)", "Pressure", "Corrosion Index", "Usage Hours",
                               "Days Since Maintenance", "Vibration", "Gas Leak Detected"],
                    "Value": [f"{data['temperature']:.2f}", f"{data['pressure']:.2f}",
                              data['corrosion_index'], data['usage_hours'],
                              data['last_maintenance_days'], data['vibration_level'],
                              "Yes" if gas_leak else "No"]
                }
            ),
            hide_index=True,
            use_container_width=True
        )

        # Charts
        st.line_chart(chart_data[["temperature", "pressure", "vibration_level"]])
        st.bar_chart(chart_data[["gas_leak_detected"]])

    time.sleep(1)
