import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

PROCESSED_DIR = "data/processed"


def load_raw_data(path: str = "data/churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Preprocess] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"[Preprocess] Cleaned. Churn rate: {df['Churn'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model accuracy."""
    df = df.copy()

    # Charge per month of tenure
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Total services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["TotalServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
        axis=1
    )

    # Long-term contract flag
    df["IsLongTermContract"] = (df["Contract"] != "Month-to-month").astype(int)

    # High monthly charge flag
    df["HighMonthlyCharge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # Tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # Electronic payment + month-to-month = high risk combo
    df["HighRiskCombo"] = (
        (df["Contract"] == "Month-to-month") &
        (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    print(f"[Preprocess] Added 6 engineered features. Total columns: {len(df.columns)}")
    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    print(f"[Preprocess] Encoded {len(categorical_cols)} categorical columns")
    return df, encoders


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(data_path: str = "data/churn.csv"):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = load_raw_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(encoders, "models/encoders.joblib")
    joblib.dump(list(X.columns), "models/feature_names.joblib")

    print(f"[Preprocess] Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(X.columns)}")
    return X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns)


if __name__ == "__main__":
    preprocess()
    print("[Preprocess] Done.")