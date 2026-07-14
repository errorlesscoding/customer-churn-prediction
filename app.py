"""
Customer Churn Prediction — Flask Application
-----------------------------------------------
Serves the portfolio site (templates/index.html) and exposes a JSON
prediction API backed by the trained scikit-learn model in /model.

Run locally:
    pip install -r requirements.txt
    python model/train_model.py   # only needed once, to generate artifacts
    python app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts at startup
# ---------------------------------------------------------------------------
_model = None
_scaler = None
_feature_columns = None
_metrics = None


def load_artifacts():
    global _model, _scaler, _feature_columns, _metrics
    try:
        _model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
        _scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
            _feature_columns = json.load(f)
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            _metrics = json.load(f)
    except FileNotFoundError:
        # Artifacts not generated yet — the app still boots so the site is
        # browsable; run `python model/train_model.py` to enable /predict.
        _model = None


load_artifacts()


def vectorize_input(payload):
    """Convert a JSON payload from the prediction form into the exact
    feature vector the model was trained on."""
    row = {col: 0 for col in _feature_columns}

    row["gender"] = 1 if payload.get("gender") == "Male" else 0
    row["SeniorCitizen"] = int(payload.get("seniorCitizen", 0))
    row["Partner"] = 1 if payload.get("partner") == "Yes" else 0
    row["Dependents"] = 1 if payload.get("dependents") == "Yes" else 0
    row["PaperlessBilling"] = 1 if payload.get("paperlessBilling") == "Yes" else 0
    row["tenure"] = float(payload.get("tenure", 0))
    row["MonthlyCharges"] = float(payload.get("monthlyCharges", 0))
    row["TotalCharges"] = float(payload.get("monthlyCharges", 0)) * float(payload.get("tenure", 0))

    contract_col = f"Contract_{payload.get('contract', 'Month-to-month')}"
    if contract_col in row:
        row[contract_col] = 1

    internet_col = f"InternetService_{payload.get('internetService', 'Fiber optic')}"
    if internet_col in row:
        row[internet_col] = 1

    payment_col = f"PaymentMethod_{payload.get('paymentMethod', 'Electronic check')}"
    if payment_col in row:
        row[payment_col] = 1

    return pd.DataFrame([[row[c] for c in _feature_columns]], columns=_feature_columns)


@app.route("/")
def home():
    return render_template("index.html", metrics=_metrics)


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify(
            {"error": "Model artifacts not found. Run `python model/train_model.py` first."}
        ), 503

    try:
        payload = request.get_json(force=True)
        X = vectorize_input(payload)
        X_scaled = _scaler.transform(X)
        proba = _model.predict_proba(X_scaled)[0][1]
        prediction = "Churn" if proba >= 0.5 else "Stay"

        return jsonify(
            {
                "prediction": prediction,
                "churn_probability": round(float(proba) * 100, 2),
                "stay_probability": round((1 - float(proba)) * 100, 2),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive for demo API
        return jsonify({"error": str(exc)}), 400


@app.route("/api/metrics")
def api_metrics():
    if _metrics is None:
        return jsonify({"error": "Metrics not available. Train the model first."}), 503
    return jsonify(_metrics)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
