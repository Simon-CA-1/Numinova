import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("industrial_equipment_monitoring_dataset_14000_5.csv")

# Encode equipment_type
le = LabelEncoder()
df['equipment_type'] = le.fit_transform(df['equipment_type'])

# Features and target
X = df.drop(columns=['equipment_id', 'critical_failure_risk'])
y = df['critical_failure_risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and label encoder
joblib.dump(model, "equipment_failure_model.pkl")
joblib.dump(le, "equipment_label_encoder.pkl")
print("Model and encoder saved successfully.")
