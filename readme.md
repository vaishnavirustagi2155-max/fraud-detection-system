# 💳 Smart Fraud Detection System using Machine Learning

An end-to-end **Machine Learning-based Fraud Detection System** that identifies fraudulent financial transactions using **Random Forest Classifier**. The application provides a simple and interactive **Streamlit dashboard** where users can upload transaction datasets, analyze them, and receive fraud predictions in real time.

---

##  Live Demo

**Streamlit Application**

https://fraud-detection-system-ychf3fbbmh9ky7jfewy4rq.streamlit.app/

---

##  Project Overview

Financial fraud is one of the biggest challenges faced by banks and digital payment systems. Manual verification of millions of daily transactions is impractical.

This project automates fraud detection using Machine Learning by:

- Detecting fraudulent transactions
- Supporting CSV dataset uploads
- Automatically handling different but similar dataset structures
- Performing preprocessing and feature alignment
- Displaying prediction statistics and visualizations
- Providing an easy-to-use Streamlit interface

---

##  Features

-  Upload transaction datasets (.csv)
-  Machine Learning based fraud prediction
-  Automatic column normalization
-  Interactive dashboard
-  Fraud vs Safe visualization
-  Prediction table
-  Dynamic preprocessing pipeline
-  Streamlit web interface

---

## Machine Learning Workflow

```text
Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Random Forest Classifier
      │
      ▼
Fraud Prediction
      │
      ▼
Visualization & Dashboard
```

---

## Tech Stack

**Programming Language**

- Python

**Libraries**

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pickle

**Machine Learning**

- Random Forest Classifier
- StandardScaler
- LabelEncoder

---

##  Project Structure

```text
Fraud-Detection-System/
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── encoders.pkl
│
├── app.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```

---

##  Installation

Clone the repository

```bash
git clone https://github.com/vaishnavirustagi2155-max/fraud-detection-system
```

Move into the project directory

```bash
cd Fraud-Detection-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

##  Usage

1. Launch the application.
2. Upload a CSV transaction dataset.
3. The system preprocesses the dataset automatically.
4. Fraud predictions are generated.
5. View:
   - Total Transactions
   - Fraud Transactions
   - Safe Transactions
   - Prediction Table
   - Visualization Charts

---

##  Output

The application displays:

- Total number of transactions
- Fraud transaction count
- Safe transaction count
- Prediction table
- Fraud vs Safe graph

---

##  Future Improvements

- SHAP Explainable AI
- XGBoost model comparison
- Fraud probability score
- Real-time transaction prediction
- Power BI dashboard
- User authentication
- Cloud database integration
- REST API support

---

##  Developer

**Vaishnavi Rustagi**

Final Year B.Tech Student (Data Science)


