# SUBMISSION GUIDE

## Main files
- `app.py` — Streamlit application
- `train_model.py` — model training and evaluation
- `data/sample_tickets.csv` — 1,200 synthetic/demo examples
- `data/ticket_history.csv` — ticket-history storage
- `models/` — trained Joblib models and metrics
- `docs/AI_Customer_Support_Ticket_Triage_Report.docx` — project report
- `presentation/AI_Customer_Support_Ticket_Triage_Presentation.pptx` — presentation

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Important
The dataset in this package is synthetic/demo data. Do not describe its metrics as production performance.

## Smoke test
All project smoke tests passed.
