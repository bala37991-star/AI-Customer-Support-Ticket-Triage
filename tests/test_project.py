from pathlib import Path
import json
import joblib

ROOT = Path(__file__).resolve().parents[1]
assert (ROOT / "app.py").exists()
assert (ROOT / "train_model.py").exists()
assert (ROOT / "data" / "sample_tickets.csv").exists()
assert (ROOT / "models" / "category_model.joblib").exists()
assert (ROOT / "models" / "urgency_model.joblib").exists()

cat = joblib.load(ROOT / "models" / "category_model.joblib")
urg = joblib.load(ROOT / "models" / "urgency_model.joblib")
sample = "I cannot access my account and need help immediately."
assert cat.predict([sample])[0] in {"account", "billing", "product", "technical"}
assert urg.predict([sample])[0] in {"low", "medium", "high"}

metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
assert metrics["dataset_size"] == 1200
assert 0 <= metrics["category_accuracy"] <= 1
assert 0 <= metrics["urgency_accuracy"] <= 1
print("All project smoke tests passed.")
