import pandas as pd
import joblib
import gradio as gr
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load("models/scaler.joblib")
    encoders = joblib.load("models/encoders.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    return model, scaler, encoders, feature_names


def predict_churn(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges
):
    input_data = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": str(total_charges),
    }

    try:
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

        risk = "🔴 HIGH RISK" if probability >= 0.7 else "🟡 MEDIUM RISK" if probability >= 0.4 else "🟢 LOW RISK"
        verdict = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"

        return verdict, f"{probability:.1%}", risk

    except Exception as e:
        return f"Error: {str(e)}", "N/A", "N/A"


with gr.Blocks(title="Customer Churn Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Customer Churn Prediction
    **MLOps Pipeline** · Voting Ensemble · MLflow · GitHub Actions CI/CD
    Enter customer details below to predict churn probability.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Customer Profile")
            gender = gr.Dropdown(["Male", "Female"],
                                 label="Gender", value="Male")
            senior = gr.Dropdown(
                [0, 1], label="Senior Citizen (1=Yes)", value=0)
            partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            dependents = gr.Dropdown(
                ["Yes", "No"], label="Dependents", value="No")
            tenure = gr.Slider(0, 72, value=12, step=1,
                               label="Tenure (months)")

        with gr.Column():
            gr.Markdown("### Services")
            phone = gr.Dropdown(
                ["Yes", "No"], label="Phone Service", value="Yes")
            multi_lines = gr.Dropdown(
                ["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
            internet = gr.Dropdown(
                ["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
            security = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Online Security", value="No")
            backup = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Online Backup", value="No")
            device = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Device Protection", value="No")
            support = gr.Dropout(
                ["Yes", "No", "No internet service"], label="Tech Support", value="No")
            tv = gr.Dropdown(["Yes", "No", "No internet service"],
                             label="Streaming TV", value="No")
            movies = gr.Dropdown(
                ["Yes", "No", "No internet service"], label="Streaming Movies", value="No")

        with gr.Column():
            gr.Markdown("### Billing")
            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract", value="Month-to-month"
            )
            paperless = gr.Dropdown(
                ["Yes", "No"], label="Paperless Billing", value="Yes")
            payment = gr.Dropdown(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
                label="Payment Method", value="Electronic check"
            )
            monthly = gr.Slider(18, 120, value=70, step=0.5,
                                label="Monthly Charges ($)")
            total = gr.Number(value=840, label="Total Charges ($)")

    predict_btn = gr.Button("Predict Churn", variant="primary", size="lg")

    with gr.Row():
        verdict_out = gr.Textbox(label="Prediction", interactive=False)
        prob_out = gr.Textbox(label="Churn Probability", interactive=False)
        risk_out = gr.Textbox(label="Risk Level", interactive=False)

    predict_btn.click(
        predict_churn,
        inputs=[gender, senior, partner, dependents, tenure, phone, multi_lines,
                internet, security, backup, device, support, tv, movies,
                contract, paperless, payment, monthly, total],
        outputs=[verdict_out, prob_out, risk_out]
    )

    gr.Markdown(
        "---\n*Built by Shushil Shah · [GitHub](https://github.com/shushilshah) · [LinkedIn](https://linkedin.com/in/shushilshah)*")

if __name__ == "__main__":
    demo.launch()
