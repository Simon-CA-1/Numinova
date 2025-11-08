import pandas as pd
import numpy as np
import joblib
import time
import random
from pathlib import Path
import sys


def simulate_equipment_reading(equipment_types):
    """Simulate a random equipment reading."""
    equipment_type = random.choice(list(equipment_types))
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


def main():
    # -------------------------------
    # Load Model and Label Encoder
    # -------------------------------
    model_path = Path(__file__).resolve().parent / "equipment_failure_model.pkl"
    encoder_path = Path(__file__).resolve().parent / "equipment_label_encoder.pkl"

    if not model_path.exists() or not encoder_path.exists():
        print(f"❌ Model or encoder not found.\nExpected at:\n  {model_path}\n  {encoder_path}")
        print("Run train.py first to generate them.")
        sys.exit(1)

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    # -------------------------------
    # Identify model features
    # -------------------------------
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        print("⚠️ Warning: Model has no stored feature names. Proceeding with all simulated features.")
        model_features = [
            "equipment_type", "temperature", "pressure", "corrosion_index",
            "usage_hours", "last_maintenance_days", "vibration_level", "gas_leak_detected"
        ]

    equipment_types = getattr(le, "classes_", None)
    if equipment_types is None:
        print("❌ Encoder error: 'classes_' not found.")
        sys.exit(1)

    print("\n✅ Real-Time Industrial Equipment Monitoring Started...\n")

    # -------------------------------
    # Continuous simulation
    # -------------------------------
    while True:
        data = simulate_equipment_reading(equipment_types)
        df = pd.DataFrame([data])

        # Encode categorical field
        df["equipment_type"] = le.transform(df["equipment_type"])

        # Filter to only model-recognized columns
        missing_features = [f for f in model_features if f not in df.columns]
        if missing_features:
            for col in missing_features:
                df[col] = 0  # default neutral value

        df = df[model_features]

        # Predict
        prediction = model.predict(df)[0]
        status = "CRITICAL FAILURE RISK" if prediction == 1 else "NORMAL"

        # Custom display logic for gas leaks
        leak_flag = data.get("gas_leak_detected", 0)
        if leak_flag == 1:
            status = "⚠️ GAS LEAK DETECTED → IMMEDIATE ATTENTION REQUIRED"

        print(
            f"Equipment: {data['equipment_type']} | "
            f"Temp={data['temperature']:.1f}°C | Press={data['pressure']:.1f} | "
            f"Vib={data['vibration_level']} | Gas={data['gas_leak_detected']} → {status}"
        )

        time.sleep(1)


if __name__ == "__main__":
    main()
