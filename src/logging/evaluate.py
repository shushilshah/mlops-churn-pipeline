import joblib
import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from src.preprocess import preprocess

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
REPORTS_DIR = "reports"


def evaluate():
    """Load saved model, run evaluation, save metrics report."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train.py first.")

    print(f"[Evaluate] Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    _, X_test, _, y_test, feature_names = preprocess()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "test_samples": len(y_test),
        "churn_rate_actual": round(float(y_test.mean()), 4),
        "churn_rate_predicted": round(float(y_pred.mean()), 4),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    report_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Evaluate] Results:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"\n[Evaluate] Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n[Evaluate] Report saved to {report_path}")

    return metrics


if __name__ == "__main__":
    evaluate()