import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting training...")

df = pd.read_csv("creditcard.csv")

print("✅ Dataset loaded")
print("Shape:", df.shape)

X = df.drop("Class", axis=1)
y = df["Class"]

# SAVE FEATURE NAMES
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = model = RandomForestClassifier(n_estimators=20, max_depth=10)
model.fit(X_train, y_train)

# SAVE EVERYTHING
import os
os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(features, open("models/features.pkl", "wb"))

print("✅ Training complete & saved!")