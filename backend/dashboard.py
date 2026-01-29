import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Customer Churn Dashboard", layout="centered")

st.title("Customer Churn Analysis")
st.write("Upload customer data to identify revenue loss risk.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    files = {"file": uploaded_file}

    response = requests.post(
        "http://localhost:8000/predict-csv",
        files=files
    )

    data = response.json()
    st.write("RAW API RESPONSE:", data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", data["total_records"])
    col2.metric("Churned", data["churned_customers"])
    col3.metric("Churn %", f'{data["churn_percentage"]}%')

    chart_df = pd.DataFrame({
        "Category": ["Churned", "Not Churned"],
        "Customers": [
            data["churned_customers"],
            data["not_churned_customers"]
        ]
    }).set_index("Category")

    st.bar_chart(chart_df)
