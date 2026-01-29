import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode target
y = df["Churn"].map({"Yes": 1, "No": 0})

# Prepare features
X = df.drop(["Churn", "customerID"], axis=1)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Save model and columns
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("Model and feature list saved successfully")

