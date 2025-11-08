import pandas as pd
import numpy as np
import joblib
import time
import random

# Load model and encoder
model = joblib.load("equipment_failure_model.pkl")
le = joblib.load("equipment_label_encoder.pkl")

# Example equipment types (same as training)
equipment_types = le.classes_

def simulate_equipment_reading():
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

print("Real-Time Industrial Equipment Monitoring Started...\n")

while True:
    data = simulate_equipment_reading()
    df = pd.DataFrame([data])
    
    # Encode type
    df['equipment_type'] = le.transform(df['equipment_type'])
    
    # Predict
    prediction = model.predict(df)[0]
    status = "CRITICAL FAILURE RISK" if prediction == 1 else "NORMAL"
    
    print(f"Equipment: {data['equipment_type']} | Temp={data['temperature']:.1f}°C | "
          f"Press={data['pressure']:.1f} | Vib={data['vibration_level']} | Gas={data['gas_leak_detected']} "
          f"→ {status}")
    
    time.sleep(1)
