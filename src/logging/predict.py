import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")


def load_artifacts():
    """Load model, scaler, encoders, and feature names."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load("models/scaler.joblib")
    encoders = joblib.load("models/encoders.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    return model, scaler, encoders, feature_names


def predict(input_data: dict) -> dict:
    """
    Run prediction on a single customer record.
    input_data: dict of raw feature values (same schema as training data)
    Returns: dict with prediction, probability, and risk level
    """
    model, scaler, encoders, feature_names = load_artifacts()

    df = pd.DataFrame([input_data])

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(0, inplace=True)

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0

    # Ensure correct feature order
    df = df[feature_names]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    risk_level = (
        "High" if probability >= 0.7
        else "Medium" if probability >= 0.4
        else "Low"
    )

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "will_churn": bool(prediction == 1),
    }


def get_sample_input() -> dict:
    """Returns a sample customer record for testing."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": "1020.0",
    }


if __name__ == "__main__":
    sample = get_sample_input()
    result = predict(sample)
    print("\n[Predict] Sample prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")