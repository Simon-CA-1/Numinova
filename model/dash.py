import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# Helper: Simulate equipment readings
# ---------------------------------------------------------
def simulate_equipment_reading(equipment_types):
    equipment_type = random.choice(equipment_types)
    temperature = np.random.normal(60, 15)
    pressure = np.random.normal(100, 30)
    corrosion_index = np.random.randint(20, 90)
    usage_hours = np.random.randint(100, 8000)
    last_maintenance_days = np.random.randint(1, 365)
    vibration_level = np.random.randint(10, 100)
    gas_leak_detected = np.random.choice([0, 1], p=[0.9, 0.1])

    return {
        "equipment_type": equipment_type,
        "temperature": temperature,
        "pressure": pressure,
        "corrosion_index": corrosion_index,
        "usage_hours": usage_hours,
        "last_maintenance_days": last_maintenance_days,
        "vibration_level": vibration_level,
        "gas_leak_detected": gas_leak_detected
    }

# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Industrial Equipment Monitoring", layout="wide")
st.title("🛡️ Industrial Equipment Monitoring Dashboard")
st.markdown("### Proactive Maintenance System — Predict Failures Before They Occur")

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
model_path = Path(__file__).resolve().parent.parent / "model" / "equipment_failure_model.pkl"
encoder_path = Path(__file__).resolve().parent.parent / "model" / "equipment_label_encoder.pkl"

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

equipment_types = list(encoder.classes_)
model_features = getattr(model, "feature_names_in_", [
    "equipment_type", "temperature", "pressure", "corrosion_index",
    "usage_hours", "last_maintenance_days", "vibration_level", "gas_leak_detected"
])

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")
speed = st.sidebar.slider("Simulation Speed (seconds per reading)", 0.5, 5.0, 1.0)
num_samples = st.sidebar.number_input("Number of Readings to Simulate", 5, 100, 20)

st.sidebar.markdown("---")
st.sidebar.info("🟢 <0.68 = Normal\n🟡 0.68–0.8 = Maintenance Soon\n🔴 >0.8 = Critical")

# ---------------------------------------------------------
# Session state for persistent alert log
# ---------------------------------------------------------
if "alerts" not in st.session_state:
    st.session_state.alerts = pd.DataFrame(columns=["Timestamp", "Equipment", "Risk Probability", "Status", "Lead Time"])

# ---------------------------------------------------------
# Display alert log table (persistent)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📋 Proactive Maintenance Alert Log")
alert_container = st.empty()

# Always display the table from session_state
alert_container.dataframe(st.session_state.alerts.astype(str), width="stretch", hide_index=True)

# Download button outside loop (fixed key)
if not st.session_state.alerts.empty:
    st.download_button(
        label="⬇️ Download Alert Log (CSV)",
        data=st.session_state.alerts.to_csv(index=False),
        file_name=f"maintenance_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_alerts_main"
    )

# ---------------------------------------------------------
# Simulation logic
# ---------------------------------------------------------
if st.button("▶ Start Monitoring", key="start_monitoring"):
    live_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i in range(int(num_samples)):
        data = simulate_equipment_reading(equipment_types)
        df = pd.DataFrame([data])
        df["equipment_type"] = encoder.transform(df["equipment_type"])
        df = df[model_features]

        if hasattr(model, "predict_proba"):
            risk_prob = model.predict_proba(df)[0][1]
        else:
            risk_prob = 0.0

        # Risk classification
        if risk_prob > 0.8:
            status = "🔴 CRITICAL FAILURE RISK → Immediate maintenance required!"
            lead_time = "0 days"
        elif 0.68 <= risk_prob <= 0.8:
            status = "🟡 MAINTENANCE DUE SOON → Schedule within 3–5 days."
            lead_time = "3–5 days"
        else:
            status = "🟢 NORMAL OPERATION → No maintenance required."
            lead_time = ">7 days"

        if data["gas_leak_detected"] == 1:
            status = "⚠️ GAS LEAK DETECTED → Immediate attention required!"
            lead_time = "0 days"

        # Live update of current reading
        with live_placeholder.container():
            st.subheader(f"Equipment Type: `{data['equipment_type']}`")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperature (°C)", f"{data['temperature']:.1f}")
            col2.metric("Pressure (bar)", f"{data['pressure']:.1f}")
            col3.metric("Vibration Level", f"{data['vibration_level']}")
            col4.metric("Risk Probability", f"{risk_prob:.2f}")

            readings_df = pd.DataFrame({
                "Metric": [
                    "Temperature (°C)", "Pressure", "Corrosion Index", "Usage Hours",
                    "Days Since Maintenance", "Vibration Level", "Gas Leak Detected"
                ],
                "Value": [
                    f"{data['temperature']:.2f}",
                    f"{data['pressure']:.2f}",
                    str(int(data['corrosion_index'])),
                    str(int(data['usage_hours'])),
                    str(int(data['last_maintenance_days'])),
                    str(int(data['vibration_level'])),
                    "Yes" if data["gas_leak_detected"] == 1 else "No"
                ]
            })

            st.dataframe(readings_df.astype(str), width="stretch", hide_index=True)

            if "CRITICAL" in status or "GAS LEAK" in status:
                st.error(status)
            elif "MAINTENANCE DUE" in status:
                st.warning(status)
            else:
                st.success(status)

            st.caption(f"🕒 Estimated Lead Time: {lead_time}")

        # Append to session log only for risky or alert readings
        if risk_prob >= 0.68 or data["gas_leak_detected"] == 1:
            new_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Equipment": data["equipment_type"],
                "Risk Probability": round(risk_prob, 2),
                "Status": status,
                "Lead Time": lead_time
            }
            st.session_state.alerts.loc[len(st.session_state.alerts)] = new_entry

        # Live update of alert table
        alert_container.dataframe(st.session_state.alerts.astype(str), width="stretch", hide_index=True)

        time.sleep(speed)
        progress_bar.progress((i + 1) / num_samples)

    st.success("✅ Simulation Complete!")
