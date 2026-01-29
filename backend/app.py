from fastapi import FastAPI, UploadFile, File
from io import StringIO
from pydantic import BaseModel
import joblib
import pandas as pd

# ------------------ APP INIT ------------------
app = FastAPI(title="Customer Churn Prediction API")

# ------------------ INPUT SCHEMA ------------------
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

# ------------------ LOAD MODEL ------------------
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")

# ------------------ HEALTH CHECK ------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

# ------------------ SINGLE CUSTOMER PREDICTION ------------------
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

# ------------------ CSV BATCH PREDICTION ------------------
@app.post("/predict-csv")
def predict_churn_csv(file: UploadFile = File(...)):
    contents = file.file.read().decode("utf-8")
    df = pd.read_csv(StringIO(contents))

    customer_ids = df.get("customerID", None)

    df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    df_encoded = pd.get_dummies(df)

    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[model_features]

    predictions = model.predict(df_encoded)
    probabilities = model.predict_proba(df_encoded)[:, 1]

    # 🔢 KPI METRICS
    total_records = len(predictions)
    churned_customers = int((predictions == 1).sum())
    not_churned_customers = total_records - churned_customers
    churn_percentage = round((churned_customers / total_records) * 100, 2)

    results = []
    for i in range(total_records):
        results.append({
            "customerID": customer_ids.iloc[i] if customer_ids is not None else i,
            "churn_prediction": "Yes" if predictions[i] == 1 else "No",
            "churn_probability": round(float(probabilities[i]), 2)
        })

    return {
        "total_records": total_records,
        "churned_customers": churned_customers,
        "not_churned_customers": not_churned_customers,
        "churn_percentage": churn_percentage,
        "predictions": results[:10]
    }

