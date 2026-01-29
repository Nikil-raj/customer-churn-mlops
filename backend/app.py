from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

class CustomerInput(BaseModel):
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
    TotalCharges: float

    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1025.6
            }
        }


# Load model and feature list
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    df = pd.DataFrame([customer.dict()])

    df = pd.get_dummies(df)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 2)
    }

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
