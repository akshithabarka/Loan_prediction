# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Prediction App", layout="wide")

st.title("🏦 Loan Prediction System")
st.write("This app trains a Logistic Regression model using the Loan Prediction dataset.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Loan_prediction.csv", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Fill missing values
    cat_cols = [
        'Gender', 'Married', 'Dependents',
        'Self_Employed', 'Loan_Amount_Term',
        'Credit_History'
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    num_cols = ['LoanAmount']

    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Drop Loan_ID if present
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    st.subheader("Processed Dataset")
    st.dataframe(df.head())

    # Features and target
    X = df.drop('Credit_History', axis=1)
    y = df['Credit_History']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"### Accuracy: {accuracy:.2f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplo
