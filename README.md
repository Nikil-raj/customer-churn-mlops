# Customer Churn Prediction System (ML + FastAPI)

End-to-end customer churn prediction system using Machine Learning, FastAPI, and Docker.

---

## 📌 Business Problem
Customer churn leads to direct revenue loss.  
This project predicts whether a customer is likely to churn so businesses can take proactive retention actions.

---

## 📊 Dataset
- IBM Telco Customer Churn Dataset
- 7,032 customer records
- Customer demographics, service usage, and billing details

---

## 🔍 Key Business Insights
- Month-to-month contract customers have the highest churn risk
- Customers with fewer subscribed services are more likely to churn
- Long-term contracts significantly reduce churn
- Higher customer engagement lowers churn probability

---

## 🧠 Machine Learning Approach
- Data cleaning and preprocessing
- Feature engineering and encoding
- Model training and evaluation

### Model Choice
- **Logistic Regression (class_weight = balanced)**
- Chosen to maximize **recall** and reduce missed churn cases
- Preferred over Random Forest due to business impact

---

## 📈 Model Performance
- Accuracy: ~74%
- Recall (Churn = Yes): ~77%
- Focused on minimizing revenue loss

---

## 🚀 API Deployment
- Built using **FastAPI**
- Provides real-time churn prediction via REST API

---
## 🏗 System Architecture

The churn prediction system follows a modular and production-ready architecture:

1. **Client Layer**
   - Users interact via Swagger UI or REST API requests

2. **API Layer (FastAPI)**
   - Handles incoming requests
   - Validates input using Pydantic schemas
   - Exposes prediction endpoints

3. **Data Processing Layer**
   - Converts input JSON into Pandas DataFrame
   - Applies one-hot encoding
   - Aligns features with trained model schema

4. **Machine Learning Layer**
   - Pre-trained Logistic Regression model
   - Predicts churn and churn probability

5. **Deployment Layer**
   - Docker container ensures consistent runtime
   - Application can be deployed on any cloud or server

This architecture ensures scalability, reliability, and easy deployment.

## 🐳 Docker Usage

### Build Docker Image
```bash
# docker build -t churn-api .


### Run Docker Container
docker run -p 8000:8000 churn-api



### Access API

API Root: http://127.0.0.1:8000

Swagger Docs: http://127.0.0.1:8000/docs

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Logistic Regression

FastAPI

Docker

Git & GitHub

📌 Project Status

Production-grade ML system
Currently enhancing with validation, monitoring, and cloud readiness

👤 Author

Nikil
Aspiring Data Analyst | Machine Learning Enthusiast