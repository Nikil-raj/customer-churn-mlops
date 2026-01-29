from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load model and feature list
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(customer: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([customer])

    # One-hot encode input
    df = pd.get_dummies(df)

    # Add missing columns
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[model_features]

    prediction = model.predict(df)[0]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No"
    }
