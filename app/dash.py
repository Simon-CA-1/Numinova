import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time

model = joblib.load("equipment_failure_model.pkl")
le = joblib.load("equipment_label_encoder.pkl")

st.title("Real-Time Equipment Monitoring Dashboard")

placeholder = st.empty()
equipment_types = le.classes_

for _ in range(200):
    data = {
        "equipment_type": random.choice(equipment_types),
        "temperature": np.random.normal(60, 15),
        "pressure": np.random.normal(100, 30),
        "corrosion_index": np.random.randint(20, 90),
        "usage_hours": np.random.randint(100, 8000),
        "last_maintenance_days": np.random.randint(1, 365),
        "vibration_level": np.random.randint(10, 100),
        "gas_leak_detected": np.random.choice([0, 1], p=[0.9, 0.1])
    }
    df = pd.DataFrame([data])
    df['equipment_type'] = le.transform(df['equipment_type'])
    prediction = model.predict(df)[0]
    
    status = "CRITICAL FAILURE RISK" if prediction == 1 else "NORMAL"
    
    placeholder.markdown(f"""
    ### Equipment Type: `{data['equipment_type']}`
    | Metric | Value |
    |:--|:--:|
    | Temperature | {data['temperature']:.2f} °C |
    | Pressure | {data['pressure']:.2f} |
    | Corrosion Index | {data['corrosion_index']} |
    | Usage Hours | {data['usage_hours']} |
    | Days Since Maintenance | {data['last_maintenance_days']} |
    | Vibration | {data['vibration_level']} |
    | Gas Leak Detected | {data['gas_leak_detected']} |
    ---
    ## **Status:** {status}
    """)
    time.sleep(1)