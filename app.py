import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# ---------------- PAGE ----------------
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 AI Fraud Detection Dashboard")

# ---------------- COLUMN NORMALIZATION ----------------
def normalize_and_map_columns(df):
    df.columns = df.columns.str.strip()

    mapping = {
        "amt": "Amount",
        "transaction_amount": "Amount",
        "amount_usd": "Amount",
        "money": "Amount",

        "transaction_time": "Time",
        "timestamp": "Time",

        "label": "Class",
        "fraud": "Class",
        "is_fraud": "Class",
        "target": "Class"
    }

    df = df.rename(columns=lambda x: mapping.get(x.lower(), x))
    return df

# ---------------- ENCODE CATEGORICAL ----------------
def encode_categorical(df, encoders=None):
    if encoders is None:
        encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    return df, encoders

# ---------------- FIX FEATURES ----------------
def fix_features(df, features):
    df.columns = df.columns.str.strip()

    feature_map = {f.lower(): f for f in features}
    df = df.rename(columns=lambda x: feature_map.get(x.lower(), x))

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

# ---------------- LOAD MODEL ----------------
model, scaler, features, encoders = None, None, None, None

if os.path.exists("models/model.pkl"):
    model = pickle.load(open("models/model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    features = pickle.load(open("models/features.pkl", "rb"))
    # encoders = pickle.load(open("models/encoders.pkl", "rb"))
    st.success("✅ Model Loaded")
else:
    st.info("ℹ️ Upload dataset to train model")

# ---------------- UPLOAD ----------------
file = st.file_uploader("📂 Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df = normalize_and_map_columns(df)

    st.subheader("📊 Preview")
    st.dataframe(df.head())

    if "Class" not in df.columns:
        st.error("❌ Dataset must contain 'Class'")
        st.stop()

    X_input = df.drop("Class", axis=1)

    # ---------------- MODEL ----------------
    if model is not None:
        if set([c.lower() for c in X_input.columns]) == set([f.lower() for f in features]):

            st.success("✅ Using existing model")

            X_fixed = fix_features(X_input.copy(), features)
            X_fixed, _ = encode_categorical(X_fixed, encoders)

            X_scaled = scaler.transform(X_fixed)
            preds = model.predict(X_scaled)

        else:
            st.warning("⚠️ Dataset changed → Retraining")

            X = X_input.copy()
            y = df["Class"]

            X, encoders = encode_categorical(X)

            features = list(X.columns)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestClassifier()
            model.fit(X_scaled, y)

            preds = model.predict(X_scaled)

            os.makedirs("models", exist_ok=True)
            pickle.dump(model, open("models/model.pkl", "wb"))
            pickle.dump(scaler, open("models/scaler.pkl", "wb"))
            pickle.dump(features, open("models/features.pkl", "wb"))
            pickle.dump(encoders, open("models/encoders.pkl", "wb"))

            st.success("✅ New model trained")

    else:
        st.info("🔄 Training model...")

        X = X_input.copy()
        y = df["Class"]

        X, encoders = encode_categorical(X)

        features = list(X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier()
        model.fit(X_scaled, y)

        preds = model.predict(X_scaled)

        os.makedirs("models", exist_ok=True)
        pickle.dump(model, open("models/model.pkl", "wb"))
        pickle.dump(scaler, open("models/scaler.pkl", "wb"))
        pickle.dump(features, open("models/features.pkl", "wb"))
        pickle.dump(encoders, open("models/encoders.pkl", "wb"))

        st.success("✅ Model trained")

    # ---------------- RESULTS ----------------
    fraud_count = int(np.sum(preds))
    total = len(preds)
    safe_count = total - fraud_count

    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Fraud 🚨", fraud_count)
    col3.metric("Safe ✅", safe_count)

    if fraud_count > 0:
        st.error(f"🚨 {fraud_count} Fraud detected!")
    else:
        st.success("✅ No Fraud")

    # ---------------- CHART ----------------
    fig, ax = plt.subplots()
    ax.bar(["Fraud", "Safe"], [fraud_count, safe_count])
    st.pyplot(fig)

    # ---------------- TABLE ----------------
    df["Prediction"] = preds

    with st.expander("🔍 View Full Data"):
        st.dataframe(df)