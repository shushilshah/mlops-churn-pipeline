from dotenv import load_dotenv
import pandas as pd
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load("models/scaler.joblib")
    encoders = joblib.load("models/encoders.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    return model, scaler, encoders, feature_names


def predict(input_data: dict) -> dict:
    model, scaler, encoders, feature_names = load_artifacts()

    df = pd.DataFrame([input_data])

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce").fillna(0)

    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0

    df = df[feature_names]
    X_scaled = scaler.transform(df)

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
    result = predict(get_sample_input())
    print("\n[Predict] Sample prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
