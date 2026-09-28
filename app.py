import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data" / "sample_tickets.csv"
HISTORY = BASE / "data" / "ticket_history.csv"
METRICS = MODELS / "metrics.json"

st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
.main-title {font-size:2.35rem;font-weight:800;margin-bottom:.2rem}
.subtitle {font-size:1.05rem;opacity:.78;margin-bottom:1rem}
.metric-card {padding: .5rem}
</style>
""", unsafe_allow_html=True)

cat_path = MODELS / "category_model.joblib"
urg_path = MODELS / "urgency_model.joblib"

if not cat_path.exists() or not urg_path.exists():
    st.error("Models are missing. Run `python train_model.py` first.")
    st.stop()

category_model = joblib.load(cat_path)
urgency_model = joblib.load(urg_path)

def ensure_history():
    if not HISTORY.exists():
        with open(HISTORY, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "ticket", "category", "urgency",
                "category_confidence", "urgency_confidence",
                "review_required", "routing_queue"
            ])

def load_history():
    ensure_history()
    try:
        return pd.read_csv(HISTORY)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=[
            "timestamp", "ticket", "category", "urgency",
            "category_confidence", "urgency_confidence",
            "review_required", "routing_queue"
        ])

def save_ticket(row):
    ensure_history()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def predict(ticket, threshold=0.60):
    category = category_model.predict([ticket])[0]
    urgency = urgency_model.predict([ticket])[0]
    cat_conf = float(max(category_model.predict_proba([ticket])[0]))
    urg_conf = float(max(urgency_model.predict_proba([ticket])[0]))
    review = min(cat_conf, urg_conf) < threshold
    queue = f"{category.title()} Support — {urgency.title()} priority"
    return category, urgency, cat_conf, urg_conf, review, queue

def load_metrics():
    if METRICS.exists():
        return json.loads(METRICS.read_text(encoding="utf-8"))
    return {}

st.sidebar.title("🤖 AI Support Triage")
st.sidebar.caption("Customer Support Ticket Classification System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Analyze Ticket", "Ticket History", "Model Performance"]
)
threshold = st.sidebar.slider(
    "Human-review threshold",
    min_value=0.50, max_value=0.90, value=0.60, step=0.05
)
st.sidebar.divider()
st.sidebar.info(
    "Training data is synthetic/demo data. Use approved and anonymized real data "
    "for any production-style study."
)

if page == "Dashboard":
    st.markdown('<div class="main-title">AI Customer Support Ticket Triage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-assisted classification, urgency detection and support routing.</div>',
        unsafe_allow_html=True
    )
    df = load_history()
    total = len(df)
    high = int((df["urgency"].astype(str).str.lower() == "high").sum()) if total else 0
    review = int((df["review_required"].astype(str) == "Yes").sum()) if total else 0

    a, b, c, d = st.columns(4)
    a.metric("Tickets Analyzed", total)
    b.metric("High Priority", high)
    c.metric("Human Review", review)
    d.metric("Categories", 4)
    st.divider()

    if total == 0:
        st.info("No tickets analyzed yet. Go to **Analyze Ticket**.")
    else:
        x, y = st.columns(2)
        with x:
            st.subheader("Tickets by Category")
            st.bar_chart(df["category"].value_counts())
        with y:
            st.subheader("Tickets by Urgency")
            st.bar_chart(df["urgency"].value_counts())
        st.subheader("Recent Tickets")
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)

elif page == "Analyze Ticket":
    st.markdown('<div class="main-title">Analyze Customer Ticket</div>', unsafe_allow_html=True)
    st.markdown("Enter a support request to classify it and suggest a routing queue.")
    ticket = st.text_area(
        "Customer ticket",
        placeholder="Example: I cannot access my account and need help immediately.",
        height=170
    )

    examples = {
        "Account": "I cannot log in to my account and need help immediately.",
        "Billing": "I was charged twice and need the billing issue fixed today.",
        "Product": "The product I received is damaged and I need a replacement.",
        "Technical": "The application crashes whenever I open it and I need technical help."
    }
    selected = st.selectbox("Or load an example", ["None"] + list(examples))
    if selected != "None" and st.button("Use Example"):
        ticket = examples[selected]
        st.info(f"Example loaded: {ticket}")

    if st.button("🔍 Analyze & Route Ticket", type="primary", use_container_width=True):
        if not ticket.strip():
            st.warning("Please enter a customer ticket.")
        else:
            category, urgency, cat_conf, urg_conf, review, queue = predict(ticket, threshold)
            save_ticket([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticket, category, urgency,
                round(cat_conf, 4), round(urg_conf, 4),
                "Yes" if review else "No", queue
            ])
            st.success("Ticket analyzed and saved.")
            a, b = st.columns(2)
            a.metric("Predicted Category", category.title())
            b.metric("Predicted Urgency", urgency.title())
            a, b = st.columns(2)
            a.metric("Category Confidence", f"{cat_conf:.1%}")
            b.metric("Urgency Confidence", f"{urg_conf:.1%}")

            if review:
                st.warning("⚠️ Low-confidence prediction: send this ticket to a human reviewer.")
            else:
                st.success("✅ Prediction confidence is above the current review threshold.")
            st.info(f"📌 Suggested routing queue: **{queue}**")

elif page == "Ticket History":
    st.markdown('<div class="main-title">Ticket History</div>', unsafe_allow_html=True)
    df = load_history()
    if df.empty:
        st.info("No ticket history yet.")
    else:
        search = st.text_input("🔎 Search ticket text")
        filtered = (
            df[df["ticket"].astype(str).str.contains(search, case=False, na=False)]
            if search.strip() else df.copy()
        )
        c1, c2 = st.columns(2)
        with c1:
            cf = st.multiselect("Category", sorted(df["category"].dropna().unique()), default=[])
        with c2:
            uf = st.multiselect("Urgency", sorted(df["urgency"].dropna().unique()), default=[])
        if cf:
            filtered = filtered[filtered["category"].isin(cf)]
        if uf:
            filtered = filtered[filtered["urgency"].isin(uf)]
        st.dataframe(filtered.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export filtered history as CSV",
            filtered.to_csv(index=False).encode(),
            "ticket_history_export.csv",
            "text/csv"
        )

else:
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.write("Evaluation on the included synthetic/demo dataset and fixed test split.")
    metrics = load_metrics()

    a, b = st.columns(2)
    a.metric("Category Accuracy", f"{metrics.get('category_accuracy', 0):.1%}")
    b.metric("Urgency Accuracy", f"{metrics.get('urgency_accuracy', 0):.1%}")
    a, b = st.columns(2)
    a.metric("Category Macro-F1", f"{metrics.get('category_macro_f1', 0):.1%}")
    b.metric("Urgency Macro-F1", f"{metrics.get('urgency_macro_f1', 0):.1%}")

    st.warning("These results are for synthetic/demo data and must not be presented as real-world production performance.")

    st.subheader("Model comparison")
    comp = pd.DataFrame(metrics.get("model_comparison", []))
    if not comp.empty:
        st.dataframe(comp.round(4), use_container_width=True, hide_index=True)

    x, y = st.columns(2)
    with x:
        st.subheader("Category Classification Report")
        report = pd.DataFrame(metrics.get("category_report", {})).T
        st.dataframe(report.round(3), use_container_width=True)
    with y:
        st.subheader("Urgency Classification Report")
        report = pd.DataFrame(metrics.get("urgency_report", {})).T
        st.dataframe(report.round(3), use_container_width=True)

    st.subheader("Category Confusion Matrix")
    labels = metrics.get("category_labels", [])
    matrix = metrics.get("category_confusion_matrix", [])
    if labels and matrix:
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), use_container_width=True)

    st.subheader("Urgency Confusion Matrix")
    labels = metrics.get("urgency_labels", [])
    matrix = metrics.get("urgency_confusion_matrix", [])
    if labels and matrix:
        st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), use_container_width=True)

st.divider()
st.caption("AI Customer Support Ticket Triage • Capstone implementation • Human review for uncertain predictions")
