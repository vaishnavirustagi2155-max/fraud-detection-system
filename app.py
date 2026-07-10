import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection", layout="wide")

# 🎨 UI (TEXT WHITE ONLY)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
}

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* Upload box fix */
section[data-testid="stFileUploader"] {
    background-color: #1e2a38;
    border: 1px solid #00c6ff;
    padding: 10px;
    border-radius: 10px;
}

section[data-testid="stFileUploader"] label,
section[data-testid="stFileUploader"] div {
    color: white !important;
}

/* Metrics */
[data-testid="stMetricValue"], 
[data-testid="stMetricLabel"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🚨 Dynamic Fraud Detection System")

uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ⚡ SPEED CONTROL
    if len(df) > 20000:
        st.warning("⚡ Using 10,000 rows for speed")
        df = df.sample(10000, random_state=42)

    df = df.reset_index(drop=True)

    # 🔄 Preprocess
    X = pd.get_dummies(df)
    X.fillna(0, inplace=True)

    # ✅ FIX dtype error
    X = X.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ 🔥 ONLY CHANGE → DYNAMIC MODEL
    model = IsolationForest(contamination="auto", random_state=42)

    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    preds = np.where(preds == -1, 1, 0)

    df["Prediction"] = preds

    # 📊 Metrics
    st.subheader("📈 Results")
    col1, col2 = st.columns(2)
    col1.metric("🚨 Fraud Detected", int(np.sum(preds)))
    col2.metric("📦 Total Records", len(preds))

    # 📄 Data
    st.subheader("📄 Dataset with Predictions")
    st.write(df)

    # 🚨 Fraud rows
    st.subheader("🚨 Fraud Transactions")
    fraud_df = df[df["Prediction"] == 1]

    if len(fraud_df) > 0:
        st.write(fraud_df)
    else:
        st.warning("No fraud detected")

    # 🔍 SHAP
    st.subheader("🔍 SHAP Explanation")

    if len(fraud_df) > 0:

        selected_index = st.selectbox(
            "Select Fraud Transaction Index",
            fraud_df.index.tolist()
        )

        # ⚡ small background for speed
        background = shap.sample(X, 100)

        explainer = shap.Explainer(model, background)

        shap_values = explainer(X.iloc[[selected_index]])

        st.write("### 🧠 Why this transaction is fraud:")

        shap_row = shap_values.values[0]

        feature_impact = sorted(
            zip(X.columns, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        for f, v in feature_impact:
            st.write(f"🔹 {f}: {round(v, 4)}")

        # 📊 Plot
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    else:
        st.info("No fraud → SHAP not available")
