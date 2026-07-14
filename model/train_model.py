"""
Customer Churn Prediction - Model Training Pipeline
----------------------------------------------------
Generates a realistic synthetic telecom churn dataset (schema modeled on the
public IBM Telco Customer Churn dataset), trains and compares Logistic
Regression, Decision Tree, and Random Forest classifiers, and persists the
best-performing model + preprocessing artifacts for the Flask app to load.

Run:
    python model/train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
N_SAMPLES = 3000

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(HERE), "data", "sample_churn_data.csv")
MODEL_PATH = os.path.join(HERE, "churn_model.pkl")
SCALER_PATH = os.path.join(HERE, "scaler.pkl")
COLUMNS_PATH = os.path.join(HERE, "feature_columns.json")
METRICS_PATH = os.path.join(HERE, "metrics.json")


def generate_dataset(n=N_SAMPLES, seed=RANDOM_STATE):
    """Create a synthetic but behaviourally realistic churn dataset."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n)
    senior_citizen = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, size=n)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21]
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])

    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    monthly_charges = base_charge + rng.normal(15, 12, size=n)
    monthly_charges = np.clip(monthly_charges, 18, 120)
    total_charges = monthly_charges * tenure + rng.normal(0, 50, size=n)
    total_charges = np.clip(total_charges, 0, None)

    # Latent churn probability driven by realistic business logic
    logit = (
        -1.4
        + 1.35 * (contract == "Month-to-month")
        - 0.55 * (contract == "One year")
        - 1.15 * (contract == "Two year")
        + 0.9 * (internet_service == "Fiber optic")
        - 0.6 * (internet_service == "No")
        + 0.55 * (payment_method == "Electronic check")
        - 0.035 * tenure
        + 0.012 * (monthly_charges - 60)
        + 0.4 * senior_citizen
        - 0.3 * (partner == "Yes")
        - 0.25 * (dependents == "Yes")
        + 0.3 * (paperless_billing == "Yes")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless_billing,
            "MonthlyCharges": monthly_charges.round(2),
            "TotalCharges": total_charges.round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return df


def build_features(df):
    X = df.drop(columns=["Churn"]).copy()
    y = (df["Churn"] == "Yes").astype(int)

    X["gender"] = (X["gender"] == "Male").astype(int)
    X["Partner"] = (X["Partner"] == "Yes").astype(int)
    X["Dependents"] = (X["Dependents"] == "Yes").astype(int)
    X["PaperlessBilling"] = (X["PaperlessBilling"] == "Yes").astype(int)

    X = pd.get_dummies(X, columns=["Contract", "InternetService", "PaymentMethod"], drop_first=False)

    return X, y


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset written to {DATA_PATH} ({len(df)} rows)")

    X, y = build_features(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
        }
        fitted[name] = model
        print(f"{name}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    roc_points = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr[::max(1, len(fpr)//30)], tpr[::max(1, len(tpr)//30)])
    ]

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)

    metrics_out = {
        "best_model": best_name,
        "models": results,
        "roc_curve": roc_points,
        "dataset_size": len(df),
        "feature_count": len(feature_columns),
        "test_size": len(y_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
