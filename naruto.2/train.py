"""
train.py — Train Shadow Clone Sign Classifier
=============================================
Reads sign_data.csv and trains a Random Forest model.
Saves the model as sign_model.pkl
"""

import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CSV_FILE   = "sign_data.csv"
MODEL_FILE = "sign_model.pkl"

# ── Load data ──
print("Loading data...")
X = []
y = []

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if row:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(y)}")
print(f"shadow_clone: {sum(y == 'shadow_clone')}")
print(f"other:        {sum(y == 'other')}")

# ── Split into train/test ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train ──
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──
print("\nResults:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ── Save model ──
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_FILE}")
