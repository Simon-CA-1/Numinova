import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import sys


def main():
    # Resolve dataset path relative to repository root (two levels up from this file -> project root)
    dataset_name = "industrial_equipment_monitoring_dataset_14000_5.csv"
    dataset_path = Path(__file__).resolve().parent.parent / 'data' / dataset_name

    if not dataset_path.exists():
        print(f"ERROR: dataset not found at: {dataset_path}")
        print("Make sure you run this script from the repository or that the data file exists under the `data/` folder.")
        sys.exit(1)

    # Load Dataset
    df = pd.read_csv(dataset_path)

    print("\nDataset Loaded Successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2. Encode Categorical Columns (if present)
    le = LabelEncoder()
    if 'equipment_type' in df.columns:
        try:
            df['equipment_type'] = le.fit_transform(df['equipment_type'])
        except Exception as e:
            print(f"Warning: failed to encode 'equipment_type': {e}")
    else:
        print("Warning: 'equipment_type' column not found; skipping LabelEncoder fit.")

    # 3. Automatically Detect Data Leakage
    if 'critical_failure_risk' not in df.columns:
        print("ERROR: target column 'critical_failure_risk' not found in dataset.")
        sys.exit(1)

    correlations = df.corr(numeric_only=True)['critical_failure_risk'].sort_values(ascending=False)
    print("\nFeature Correlations with Target:")
    print(correlations)

    # Remove any feature that is too correlated with the target
    threshold = 0.5
    leaky_features = [c for c, v in correlations.items() if abs(v) > threshold and c != 'critical_failure_risk']

    if leaky_features:
        print(f"\nHighly correlated (leaky) features detected: {leaky_features}")
    else:
        print("\nNo highly correlated features detected.")

    # 4. Feature Selection
    drop_cols = ['equipment_id', 'critical_failure_risk'] + leaky_features
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['critical_failure_risk']

    print(f"\nFinal Features Used for Training: {list(X.columns)}")

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # 7. Evaluate Model
    y_pred = model.predict(X_test)

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Cross-validation for generalization (may take some time)
    try:
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"\nCross-validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    except Exception as e:
        print(f"Warning: cross-validation failed: {e}")

    # 8. Save Model and Encoder (saved into the same folder as this script)
    model_path = Path(__file__).resolve().parent / "equipment_failure_model.pkl"
    encoder_path = Path(__file__).resolve().parent / "equipment_label_encoder.pkl"
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")


if __name__ == '__main__':
    main()
