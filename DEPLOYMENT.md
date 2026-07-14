# Deployment Guide

This project is a single Flask app (backend + templated frontend in one), so the simplest path is to deploy it as one web service. Vercel is included as an option if you later split the static site out from the API.

## 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Customer Churn Prediction"
git branch -M main
git remote add origin https://github.com/yourusername/customer-churn-prediction.git
git push -u origin main
```

> `model/*.pkl` and `data/*.csv` are generated files. Either commit them (simplest, so the deployed app has a model immediately) or add a build step that runs `python model/train_model.py` before start.

## 2. Deploy the Flask app — Render

1. Go to [render.com](https://render.com) → **New → Web Service** → connect your GitHub repo.
2. Configure:
   - **Build command:** `pip install -r requirements.txt && python model/train_model.py`
   - **Start command:** `gunicorn app:app`
   - **Environment:** Python 3.11+
3. Deploy. Render gives you a URL like `https://customer-churn-prediction.onrender.com` — this is your live demo link.

## 2b. Deploy the Flask app — Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**.
2. Railway auto-detects the `Procfile` (`web: gunicorn app:app`).
3. Add a **Deploy → Custom Start Command** override if needed, or a pre-deploy command: `python model/train_model.py`.
4. Deploy and grab the generated `*.up.railway.app` domain.

## 3. (Optional) Deploy the frontend separately — Vercel

If you want the marketing/portfolio page on Vercel and the API on Render/Railway:

1. Extract `templates/index.html`, `static/`, into a standalone static project (or a small Vercel serverless function that proxies to your API).
2. Point the prediction form's `fetch("/predict")` call at your deployed API's full URL instead of a relative path.
3. `vercel deploy` from the project root, or connect the repo in the Vercel dashboard.

## 4. Update the live links

Once deployed, update:
- The **Live Demo** buttons and deploy cards in `templates/index.html`
- `README.md`'s live demo link
- Your resume / LinkedIn project link

## Environment variables

None are required for the default synthetic-data demo. If you later connect a real database or add secrets (e.g. an API key for a hosted dataset), set them in your platform's dashboard rather than committing them — `.env` is already git-ignored.
