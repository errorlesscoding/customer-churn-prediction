# Resume Project Description

## Short version (for a resume bullet block)

**Customer Churn Prediction** | Python, Flask, Scikit-learn, Pandas, NumPy, HTML/CSS/JavaScript
*[github.com/yourusername/customer-churn-prediction](https://github.com/yourusername/customer-churn-prediction) · [Live Demo](https://your-project-url.com)*

- Built an end-to-end customer churn prediction system covering data preprocessing, feature engineering, model training, and deployment.
- Performed exploratory data analysis and engineered 18 model features from 11 raw customer attributes via label and one-hot encoding.
- Trained and benchmarked Logistic Regression, Decision Tree, and Random Forest classifiers, selecting the best performer by F1 score.
- Achieved ~83% accuracy and 0.85 ROC-AUC on a held-out test set.
- Developed a Flask REST API and interactive web app for real-time churn predictions with live probability output.
- Deployed the application to Render/Railway with a Vercel-hosted frontend and a public GitHub repository.
- Followed clean code, modular pipeline design, and responsive UI principles throughout.

## STAR format (for interviews or a longer write-up)

**Situation**
Subscription-based businesses lose a significant share of revenue to customer churn each year, and by the time a cancellation request arrives, the opportunity to intervene has already passed.

**Task**
Design and ship a complete machine learning product — not just a notebook — that scores each customer's likelihood of churning from data already available in a typical CRM, and makes that score usable through a real interface.

**Action**
- Explored and engineered a customer dataset (demographics, tenure, contract, billing, and service attributes), handling missing values and encoding categorical fields.
- Trained and compared three classification algorithms (Logistic Regression, Decision Tree, Random Forest) under identical preprocessing, using stratified train/test splitting to preserve class balance.
- Evaluated models on accuracy, precision, recall, F1, and ROC-AUC, and automated selection of the best-performing model.
- Built a Flask backend exposing a `/predict` endpoint that vectorizes incoming customer data, scales it, and returns a churn probability.
- Designed and built a responsive, production-quality front end presenting the project narrative, dataset, model comparison, live performance charts, and an interactive prediction form.
- Packaged the project for deployment (Render/Railway for the API, Vercel-ready static assets, GitHub for source control) with a documented setup and deployment process.

**Result**
Delivered a fully working, deployable churn-prediction product — model, API, and UI — that demonstrates the complete ML lifecycle from raw data to a live, recruiter-facing demo, achieving ~83% test accuracy and a 0.85 ROC-AUC with the selected model.
