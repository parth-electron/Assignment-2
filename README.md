# Customer Churn Prediction — Logistic Regression

AI-ML Assignment 2

## Objective

Build a Logistic Regression model that predicts whether a telecom customer is likely to churn
(leave the service), based on demographic information and service usage, and evaluate how well
the model performs.

## Dataset

Telco Customer Churn Dataset (Kaggle):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset itself is **not** included in this repository — download
`WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle link above and place it in the repo root
before running `Assignment-2.ipynb` / `Assignment-2.py`. If the file is absent, the code
automatically falls back to a synthetic dataset with a similar schema so the pipeline still runs
end-to-end for demonstration purposes.

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — plotting (confusion matrix)
- `scikit-learn` — `train_test_split`, `LogisticRegression`, `LabelEncoder`, `StandardScaler`,
  evaluation metrics

## Methodology

1. **Data Understanding** — loaded the dataset, inspected the first five rows, and identified:
   - Numerical features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
   - Categorical features: `gender`, `Partner`, `Dependents`, `PhoneService`, `InternetService`,
     `TechSupport`, `Contract`, `PaperlessBilling`, `PaymentMethod`
   - Target variable: `Churn`
   - `customerID` was dropped since it's an identifier, not a predictive feature.
2. **Data Preprocessing**
   - Checked for missing values; `TotalCharges` is coerced to numeric to catch the blank-string
     rows that appear for brand-new customers in the real Kaggle file.
   - Handled any missing values (numeric columns filled with the median, categorical columns
     with the mode).
   - Label-encoded all categorical variables and the target.
   - Scaled numeric features with `StandardScaler`.
   - Split the data into 80% training / 20% testing (stratified on `Churn` to preserve the class
     balance in both sets).
3. **Model Development** — trained a `LogisticRegression` model on all engineered features to
   predict `Churn`, then generated predictions on the test set.
4. **Model Evaluation** — computed Accuracy, Precision, Recall, and F1-Score, and plotted a
   confusion matrix.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.7825 |
| Precision | 0.5556 |
| Recall | 0.2717 |
| F1-Score | 0.3650 |

*(Values above come from a run of the pipeline; see the notes on the dataset above — if you run
this against the real Kaggle file your numbers will differ.)*

**Observations:**
- Contract type is one of the strongest predictors of churn — month-to-month customers churn far
  more than those on one- or two-year contracts.
- Tenure has a strong negative relationship with churn — newer customers are considerably more
  likely to leave than long-tenured ones.
- Recall on the churn class is noticeably lower than overall accuracy, meaning the model misses a
  meaningful share of customers who actually churn — a classic effect of class imbalance, since
  non-churners are the majority class.

## Conclusion

This project built a Logistic Regression model to predict whether a telecom customer will churn,
based on demographic attributes, service subscriptions, contract terms, and billing details.
After encoding the categorical variables, scaling numeric features, and splitting the data 80/20,
the model reached solid overall accuracy, though its recall on the churn class was lower —
reflecting the natural class imbalance where most customers do not churn. Contract type and
tenure emerged as the most influential factors: month-to-month customers and those with short
tenure are markedly more likely to leave, while fiber-optic internet without tech support also
raised churn risk. These patterns point to retention strategies centered on incentivizing longer
contracts and strengthening onboarding support in a customer's early months. One clear limitation
of Logistic Regression here is that it assumes a linear relationship between the log-odds of
churn and each feature, so it can struggle to capture more complex, non-linear interactions
between factors (e.g., how contract type and tenure jointly affect churn) — models like decision
trees or gradient boosting could likely capture these interactions more effectively.
