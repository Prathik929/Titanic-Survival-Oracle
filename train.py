"""Train logistic regression on the Kaggle Titanic dataset and export weights for the web UI.
Usage: python train.py            (expects data/train.csv)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DATA = PROJECT_DIR / "data" / "train.csv"
MODEL_OUTPUT = PROJECT_DIR / "web" / "model.js"

if not TRAIN_DATA.is_file():
    raise FileNotFoundError(f"Training data not found: {TRAIN_DATA}")

df = pd.read_csv(TRAIN_DATA)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Family"] = (df["SibSp"] + df["Parch"]).clip(upper=6)

# Feature order MUST match the frontend: female, 1st, 2nd, age/80, family/6, log(fare+1)/5.7
X = np.column_stack([
    (df["Sex"] == "female").astype(float),
    (df["Pclass"] == 1).astype(float),
    (df["Pclass"] == 2).astype(float),
    df["Age"] / 80,
    df["Family"] / 6,
    np.log1p(df["Fare"]) / 5.7,
])
y = df["Survived"].values

model = LogisticRegression(C=1.0, max_iter=1000)
acc = cross_val_score(model, X, y, cv=5).mean()
model.fit(X, y)
print(f"5-fold CV accuracy: {acc:.3f}")

weights = [float(model.intercept_[0])] + [float(w) for w in model.coef_[0]]
MODEL_OUTPUT.write_text(
    "window.MODEL = " + json.dumps({"weights": weights, "accuracy": float(acc)}) + ";",
    encoding="utf-8",
)
print(f"Saved {MODEL_OUTPUT}")
