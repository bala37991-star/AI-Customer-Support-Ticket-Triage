from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sample_tickets.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df["ticket_text"].astype(str)
y_category = df["category"].astype(str)
y_urgency = df["urgency"].astype(str)

X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
    X, y_category, test_size=0.20, random_state=42, stratify=y_category
)
X_urg_train, X_urg_test, y_urg_train, y_urg_test = train_test_split(
    X, y_urgency, test_size=0.20, random_state=42, stratify=y_urgency
)

def make_pipeline(estimator):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            min_df=1,
        )),
        ("classifier", estimator),
    ])

candidate_factories = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=3000),
    "Linear SVM": lambda: LinearSVC(),
    "Multinomial Naive Bayes": lambda: MultinomialNB(),
}

comparison = []

def compare_models(X_train, X_test, y_train, y_test, task):
    best_name, best_model, best_f1 = None, None, -1.0
    for name, factory in candidate_factories.items():
        model = make_pipeline(factory())
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        comparison.append({
            "task": task,
            "model": name,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        })
        if macro_f1 > best_f1:
            best_name, best_model, best_f1 = name, model, macro_f1
    return best_name, best_model

best_category_name, best_category_model = compare_models(
    X_cat_train, X_cat_test, y_cat_train, y_cat_test, "Category"
)
best_urgency_name, best_urgency_model = compare_models(
    X_urg_train, X_urg_test, y_urg_train, y_urg_test, "Urgency"
)

if hasattr(best_category_model, "predict_proba"):
    category_model = best_category_model
    deployed_category_name = best_category_name
else:
    category_model = make_pipeline(LogisticRegression(max_iter=3000))
    category_model.fit(X_cat_train, y_cat_train)
    deployed_category_name = "Logistic Regression (confidence-aware deployment)"

if hasattr(best_urgency_model, "predict_proba"):
    urgency_model = best_urgency_model
    deployed_urgency_name = best_urgency_name
else:
    urgency_model = make_pipeline(LogisticRegression(max_iter=3000))
    urgency_model.fit(X_urg_train, y_urg_train)
    deployed_urgency_name = "Logistic Regression (confidence-aware deployment)"

category_pred = category_model.predict(X_cat_test)
urgency_pred = urgency_model.predict(X_urg_test)

category_labels = sorted(y_category.unique().tolist())
urgency_labels = sorted(y_urgency.unique().tolist())

metrics = {
    "dataset_size": int(len(df)),
    "category_test_size": int(len(X_cat_test)),
    "urgency_test_size": int(len(X_urg_test)),
    "best_category_model": best_category_name,
    "best_urgency_model": best_urgency_name,
    "deployed_category_model": deployed_category_name,
    "deployed_urgency_model": deployed_urgency_name,
    "category_accuracy": float(accuracy_score(y_cat_test, category_pred)),
    "category_macro_f1": float(f1_score(y_cat_test, category_pred, average="macro")),
    "urgency_accuracy": float(accuracy_score(y_urg_test, urgency_pred)),
    "urgency_macro_f1": float(f1_score(y_urg_test, urgency_pred, average="macro")),
    "category_labels": category_labels,
    "urgency_labels": urgency_labels,
    "model_comparison": comparison,
    "category_confusion_matrix": confusion_matrix(
        y_cat_test, category_pred, labels=category_labels
    ).tolist(),
    "urgency_confusion_matrix": confusion_matrix(
        y_urg_test, urgency_pred, labels=urgency_labels
    ).tolist(),
    "category_report": classification_report(
        y_cat_test, category_pred, output_dict=True, zero_division=0
    ),
    "urgency_report": classification_report(
        y_urg_test, urgency_pred, output_dict=True, zero_division=0
    ),
}

joblib.dump(category_model, MODELS / "category_model.joblib")
joblib.dump(urgency_model, MODELS / "urgency_model.joblib")

with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("=" * 70)
print("FINAL DEPLOYED MODEL RESULTS")
print("=" * 70)
print(f"Dataset size:            {len(df)}")
print(f"Category test size:      {len(X_cat_test)}")
print(f"Urgency test size:       {len(X_urg_test)}")
print(f"Best category model:     {best_category_name}")
print(f"Best urgency model:      {best_urgency_name}")
print(f"Deployed category model: {deployed_category_name}")
print(f"Deployed urgency model:  {deployed_urgency_name}")
print(f"Category accuracy:       {metrics['category_accuracy']:.4f}")
print(f"Category macro-F1:       {metrics['category_macro_f1']:.4f}")
print(f"Urgency accuracy:        {metrics['urgency_accuracy']:.4f}")
print(f"Urgency macro-F1:        {metrics['urgency_macro_f1']:.4f}")
print("Models and metrics saved in:", MODELS)
