import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_file_exists():
    assert os.path.exists(
        "data/churn.csv"), "Dataset not found. Run: python data/download_data.py"


def test_data_loads_correctly():
    df = pd.read_csv("data/churn.csv")
    assert len(df) > 0, "Dataset is empty"
    assert "Churn" in df.columns, "Target column 'Churn' missing"
    assert len(df.columns) >= 10, "Too few columns in dataset"


def test_data_has_no_all_null_columns():
    df = pd.read_csv("data/churn.csv")
    null_cols = df.columns[df.isnull().all()].tolist()
    assert len(null_cols) == 0, f"Columns with all nulls: {null_cols}"


def test_preprocess_runs():
    from src.preprocess import preprocess
    X_train, X_test, y_train, y_test, features = preprocess()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(features) > 0
    assert set(y_train.unique()).issubset({0, 1})


def test_preprocess_no_nulls():
    from src.preprocess import preprocess
    X_train, X_test, y_train, y_test, _ = preprocess()
    assert not np.isnan(X_train).any(), "NaN values in training data"
    assert not np.isnan(X_test).any(), "NaN values in test data"


def test_train_test_split_ratio():
    from src.preprocess import preprocess
    X_train, X_test, y_train, y_test, _ = preprocess()
    total = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total
    assert 0.15 <= test_ratio <= 0.25, f"Unexpected test ratio: {test_ratio}"


def test_model_file_exists_after_training():
    """Only runs if training has been done."""
    model_path = os.getenv("MODEL_PATH", "models/churn_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet — skipping inference test")
    assert os.path.exists(model_path)


def test_prediction_output_format():
    """Test prediction returns correct schema."""
    model_path = os.getenv("MODEL_PATH", "models/churn_model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet")

    from src.logging.predict import predict, get_sample_input
    result = predict(get_sample_input())

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "risk_level" in result
    assert "will_churn" in result
    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
