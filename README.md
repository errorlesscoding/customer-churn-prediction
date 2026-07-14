# Customer Churn Prediction

An end-to-end machine learning product that predicts whether a telecom customer is likely to churn, packaged as a portfolio-ready web app with a live prediction demo.

**Live demo:** https://your-project-url.com *(replace after deployment)*
**Repository:** https://github.com/yourusername/customer-churn-prediction

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)

---

## What this is

A classification system that scores the probability of an individual customer cancelling their subscription, based on account, billing, and service attributes. It ships as:

- A trained **scikit-learn** model (Logistic Regression / Decision Tree / Random Forest, best one auto-selected on F1 score)
- A **Flask** REST API (`/predict`) that serves live inference
- A single-page **portfolio site** (dark navy / white / violet, glassmorphism UI) presenting the project, dataset, model comparison, live performance charts, and an interactive prediction form

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | HTML5, CSS3 (custom design system), vanilla JS, Chart.js |
| Backend | Python, Flask |
| Machine learning | Pandas, NumPy, Scikit-learn, Joblib |
| Deployment | Render / Railway (API), Vercel (static assets), GitHub |

## Project structure

```
customer-churn-prediction/
├── app.py                     # Flask app: page routes + /predict API
├── requirements.txt
├── Procfile                    # gunicorn entrypoint for Render/Railway
├── model/
│   ├── train_model.py          # dataset generation + training pipeline
│   ├── churn_model.pkl         # trained model (generated)
│   ├── scaler.pkl              # fitted StandardScaler (generated)
│   ├── feature_columns.json    # exact feature order the model expects
│   └── metrics.json            # accuracy/precision/recall/F1/ROC/confusion matrix
├── data/
│   └── sample_churn_data.csv   # synthetic training data (generated)
├── templates/
│   └── index.html              # the portfolio site
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── img/
├── DEPLOYMENT.md
└── RESUME.md                   # resume-ready STAR project description
```

## Getting started

```bash
# 1. Clone and enter the project
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates model/*.pkl, model/metrics.json, data/*.csv)
python model/train_model.py

# 5. Run the app
python app.py
```

Visit **http://localhost:5000** — the homepage, live prediction form, and performance charts are all served from this one Flask app.

> The dataset is synthetically generated with realistic churn logic modeled on the public IBM Telco Customer Churn schema, so the project is fully reproducible without needing to source external data. Swap `model/train_model.py`'s `generate_dataset()` for a real CSV loader to train on your own data.

## Dataset

11 raw fields per customer account (gender, senior citizen status, partner/dependents, tenure, contract type, internet service, payment method, monthly/total charges, paperless billing), expanded to 18 model features after one-hot encoding. Target: `Churn` (Yes/No).

## Model

Three classifiers are trained on an identical, scaled feature set and compared on accuracy, precision, recall, F1, and ROC-AUC. The model with the best F1 score is automatically selected and persisted — see `model/metrics.json` for the exact numbers from the last training run.

## API reference

**POST** `/predict`

```json
{
  "gender": "Female",
  "seniorCitizen": "0",
  "partner": "No",
  "dependents": "No",
  "tenure": "4",
  "monthlyCharges": "84.20",
  "contract": "Month-to-month",
  "internetService": "Fiber optic",
  "paymentMethod": "Electronic check",
  "paperlessBilling": "Yes"
}
```

Response:

```json
{
  "prediction": "Churn",
  "churn_probability": 88.07,
  "stay_probability": 11.93
}
```

**GET** `/api/metrics` — returns the full training metrics JSON used to render the performance charts.
**GET** `/health` — basic liveness/readiness check.

## Deployment

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for step-by-step guides to Render, Railway, and Vercel.

## Resume description

See [`RESUME.md`](./RESUME.md) for a ready-to-paste, STAR-format project bullet for your CV.

## License

MIT — free to use as a template for your own portfolio.
