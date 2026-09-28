# AI Customer Support Ticket Triage

## Project overview
This capstone system classifies customer support tickets by **category** and **urgency**, then suggests a support routing queue.

### Main modules
- Dashboard
- Analyze Ticket
- Ticket History
- Model Performance
- Confidence-aware human-review flag
- CSV export
- TF-IDF text representation
- Model comparison: Logistic Regression, Linear SVM and Multinomial Naive Bayes

### Categories
- Account
- Billing
- Product
- Technical

### Urgency levels
- Low
- Medium
- High

## Technology stack
Python, Streamlit, Pandas, scikit-learn, TF-IDF, Logistic Regression, Linear SVM, Multinomial Naive Bayes and Joblib.

## Dataset
The included `data/sample_tickets.csv` contains **1,200 synthetic/demo examples** generated for development and demonstration. It is not real customer data.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Submission note
Do not describe the synthetic dataset as real-world customer data. For a production-style study, use only approved and anonymized data.

## Model results in this package
The values below are from the included synthetic/demo dataset and fixed 80/20 stratified test split.

- Category accuracy: 1.0000
- Category macro-F1: 1.0000
- Urgency accuracy: 1.0000
- Urgency macro-F1: 1.0000
- Best category model: Logistic Regression
- Best urgency model: Logistic Regression

These numbers are demonstration results only and should not be presented as production performance.
