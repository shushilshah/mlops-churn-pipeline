from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from src.logging.predict import predict, get_sample_input

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps pipeline for predicting customer churn using RandomForest + MLflow tracking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    will_churn: bool


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "endpoints": ["/predict", "/sample", "/health"],
    }


@app.get("/health")
def health():
    model_exists = os.path.exists(
        os.getenv("MODEL_PATH", "models/churn_model.joblib")
    )
    return {
        "status": "healthy" if model_exists else "model_not_found",
        "model_loaded": model_exists,
    }


@app.get("/sample")
def sample():
    """Returns a sample input for testing."""
    return get_sample_input()


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """Predict churn probability for a customer."""
    try:
        result = predict(customer.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))