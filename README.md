# Credit Risk Default Prediction Model

An end-to-end machine learning project that predicts loan default probability and translates it into a FICO-style credit score. Includes data merging, feature engineering, class-imbalance handling, hyperparameter tuning, model validation using banking-industry techniques, and a deployed Streamlit app.

**Live App:** [ml-project-credit-risk-model-system.streamlit.app](https://ml-project-credit-risk-model-system.streamlit.app/)

---

## Problem Statement

Lenders need to estimate the probability that a loan applicant will default, so they can price risk and make approval decisions. This project builds a classification model that predicts default probability from applicant demographics, loan details, and credit bureau history, and converts that probability into an interpretable credit score.

## Dataset

Three tables merged into a single dataset of 50,000 loan records:
- **Customers** — age, gender, marital status, employment, income, residence type, address history
- **Loans** — loan purpose, type, sanctioned/disbursed amount, tenure, fees
- **Bureau data** — open/closed accounts, delinquent months, days past due, credit utilization, enquiry count

Target variable: `default` (binary) — imbalanced, with only **8.6% of loans defaulting**.

## Approach

### 1. Data Cleaning
- Merged customer, loan, and bureau tables on customer ID
- Imputed missing `residence_type` with mode
- Removed records with invalid financials (processing fee > 3% of loan amount, zero sanction amount)
- Fixed inconsistent category labels (e.g., "Personaal" → "Personal")

### 2. Feature Engineering
- `loan_to_income` — loan amount relative to income
- `delinquency_ratio` — % of loan tenure spent delinquent
- `avg_dpd_per_delinquency` — average days past due per delinquent month

### 3. Feature Selection
- **Variance Inflation Factor (VIF):** identified and removed features with severe multicollinearity (`sanction_amount`, `processing_fee`, `gst`, `net_disbursement` — VIF was infinite on several of these)
- **Weight of Evidence / Information Value (WOE/IV):** a credit-scoring-industry technique that ranks features by predictive power; selected the 10 strongest predictors (credit utilization ratio, delinquency ratio, and loan-to-income ranked highest)

### 4. Handling Class Imbalance
Compared three strategies on Logistic Regression and XGBoost:

| Strategy | Defaulter Recall | Defaulter Precision |
|---|---|---|
| Class weighting | 0.82 | 0.77 |
| Random undersampling | 0.98 | 0.54 |
| **SMOTETomek** | **0.94** | **0.57** |

SMOTETomek (oversampling combined with data cleaning) gave the best balance and was used for the final model.

### 5. Hyperparameter Tuning
Used **Optuna** (Bayesian optimization) to tune both Logistic Regression and XGBoost, optimizing for F1-score via cross-validation.

### 6. Model Evaluation
Final model: **Logistic Regression** (selected over XGBoost for interpretability — standard practice in credit scoring, where regulators require explainable decisions).

- **AUC-ROC: 0.98**
- **Gini coefficient: 0.97**
- **Recall on defaulters: 94%**
- Validated using **KS statistic** and **decile rank-ordering** — standard techniques used by banks/NBFCs to check that the model correctly separates high-risk from low-risk applicants across probability bands

### 7. Deployment
Built a Streamlit app that takes applicant details as input and:
- Predicts default probability
- Converts it into a **FICO-style credit score (300–900 scale)**
- Assigns a risk rating (Poor / Average / Good / Excellent) with a visual gauge

## Tech Stack
Python • Pandas • NumPy • Scikit-learn • XGBoost • Optuna • Imbalanced-learn (SMOTETomek) • Statsmodels (VIF) • Plotly • Streamlit

## Project Structure
```
├── credit_risk_model.ipynb       # Full EDA, feature engineering, modeling, evaluation
├── customers.csv                 # Customer demographic data
├── loans.csv                     # Loan details
├── bureau_data.csv               # Credit bureau history
├── artifacts/
│   └── model_data.joblib         # Saved model, scaler, and feature list
├── main.py                       # Streamlit app
├── prediction_helper.py          # Preprocessing + credit score conversion logic
├── requirements.txt
└── README.md
```
## Key Takeaways
- Accuracy is misleading on imbalanced data — comparing recall/precision trade-offs across three imbalance-handling strategies mattered more than any single accuracy number
- WOE/IV and VIF are standard tools in real credit-risk modeling, not just academic exercises — they directly informed which features made it into the final model
- Interpretability can matter more than raw performance: Logistic Regression was chosen over a stronger-scoring XGBoost model because credit decisions need to be explainable
- Translating a raw probability into a familiar score (300–900) makes the model's output usable by a non-technical end user
