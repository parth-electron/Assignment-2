"""
AI-ML Assignment 2
Customer Churn Prediction using Logistic Regression

Dataset: Telco Customer Churn Dataset (Kaggle)
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

NOTE ON DATA:
This environment has no internet access, so the real Kaggle CSV could not
be downloaded directly here. The script below FIRST tries to load a local
file named 'WA_Fn-UseC_-Telco-Customer-Churn.csv' (the actual Kaggle file —
download it and place it in the same folder as this script/notebook and it
will be used automatically). If that file is not found, it falls back to a
synthetic dataset with the same columns and the same well-known
relationships (month-to-month contracts, low tenure, fiber optic + no tech
support, high monthly charges -> higher churn) purely so every task can be
demonstrated end-to-end. Before submitting, download the real CSV from the
Kaggle link above and place it next to this script so the real data is used.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_dataset(path=DATA_PATH, n_synthetic=2000):
    """Load the real Kaggle Telco churn CSV if present, else build a
    synthetic stand-in with a similar schema and realistic relationships."""
    if os.path.exists(path):
        print(f"Loading real dataset from '{path}'")
        return pd.read_csv(path), False

    print(f"'{path}' not found locally — generating a synthetic dataset "
          f"with a similar schema so the pipeline can run end-to-end.\n"
          f"Replace with the real Kaggle file before final submission.")

    gender = np.random.choice(["Male", "Female"], n_synthetic)
    senior_citizen = np.random.choice([0, 1], n_synthetic, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n_synthetic)
    dependents = np.random.choice(["Yes", "No"], n_synthetic, p=[0.3, 0.7])
    tenure = np.random.randint(0, 73, n_synthetic)
    phone_service = np.random.choice(["Yes", "No"], n_synthetic, p=[0.9, 0.1])
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_synthetic, p=[0.35, 0.44, 0.21]
    )
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n_synthetic, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = np.random.choice(["Yes", "No"], n_synthetic, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"], n_synthetic
    )
    tech_support = np.random.choice(["Yes", "No", "No internet service"], n_synthetic)
    monthly_charges = np.round(np.random.normal(64.8, 30.1, n_synthetic).clip(18, 120), 2)
    total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.9, 1.0, n_synthetic), 2)

    # Churn probability driven by known real-world patterns in this dataset
    churn_score = (
        -2.0
        + 1.6 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.6 * (tech_support == "No")
        - 0.03 * tenure
        + 0.01 * (monthly_charges - 65)
        + 0.4 * (paperless_billing == "Yes")
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = np.where(np.random.rand(n_synthetic) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"C{1000+i}" for i in range(n_synthetic)],
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })
    return df, True


df, is_synthetic = load_dataset()

print("\n=== First five records ===")
print(df.head())

# --------------------------------------------------------------------------
# Task 1 (continued): identify feature types
# --------------------------------------------------------------------------

target_variable = "Churn"

# customerID is an identifier, not a feature -> drop it before modelling
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# TotalCharges sometimes arrives as a string with blanks in the real
# Kaggle file (a handful of new customers have " " instead of 0) -> coerce
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

numerical_features = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c != target_variable]
categorical_features = [c for c in df.columns
                         if c not in numerical_features + [target_variable]]

print("\n=== Feature types ===")
print("Numerical features   :", numerical_features)
print("Categorical features :", categorical_features)
print("Target variable      :", target_variable)

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------

print("\n=== Missing values per column (before handling) ===")
print(df.isnull().sum())

# Handle missing values: numeric -> median, categorical -> mode
for col in numerical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\n=== Missing values per column (after handling) ===")
print(df.isnull().sum())

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}
for col in categorical_features + [target_variable]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\n=== Target encoding map ===")
print(label_encoders[target_variable])

feature_columns = numerical_features + categorical_features
X = df_encoded[feature_columns]
y = df_encoded[target_variable]

# Scale numeric features for a better-conditioned Logistic Regression fit
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")

# --------------------------------------------------------------------------
# Task 3: Model Development
# --------------------------------------------------------------------------

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Model coefficients ===")
for feature, coef in zip(feature_columns, model.coef_[0]):
    print(f"{feature:20s}: {coef:,.4f}")
print(f"{'intercept':20s}: {model.intercept_[0]:,.4f}")

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Evaluation metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Churn", "Churn"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Customer Churn Prediction")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved plot to confusion_matrix.png")

with open("metrics.txt", "w") as f:
    f.write(f"accuracy,{acc:.4f}\n")
    f.write(f"precision,{prec:.4f}\n")
    f.write(f"recall,{rec:.4f}\n")
    f.write(f"f1,{f1:.4f}\n")
    f.write(f"synthetic,{is_synthetic}\n")

print("\nDone.")
