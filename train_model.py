"""Reproduces the midterm Decision Tree pipeline (Part IV) and saves it ONCE with joblib.
Run: python train_model.py   (the app never retrains)"""
import json, datetime as dt
import joblib, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from modules.data_processing import clean_df, CAT_FEATURES, NUM_FEATURES, FEATURES, TARGET

df, _ = clean_df(pd.read_csv("data/facility_reservations.csv"))
df.to_csv("data/facility_reservations_cleaned.csv", index=False)
m = df.dropna(subset=[TARGET])
X, y = m[FEATURES], m[TARGET]
pre = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUM_FEATURES),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES)])
model = Pipeline([("preprocessor", pre),
                  ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model.fit(Xtr, ytr)
p = model.predict(Xte)
fi = (pd.DataFrame({"feature": model.named_steps["preprocessor"].get_feature_names_out(),
                    "importance": model.named_steps["classifier"].feature_importances_})
      .sort_values("importance", ascending=False))
metrics = {
    "model_version": "midterm-decision-tree-v1", "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__, "train_rows": len(Xtr), "test_rows": len(Xte),
    "accuracy": accuracy_score(yte, p),
    "precision": precision_score(yte, p, average="weighted", zero_division=0),
    "recall": recall_score(yte, p, average="weighted", zero_division=0),
    "f1": f1_score(yte, p, average="weighted", zero_division=0),
    "majority_baseline": float(yte.value_counts(normalize=True).max()),
    "labels": list(model.classes_), "confusion_matrix": confusion_matrix(yte, p, labels=model.classes_).tolist(),
    "report": classification_report(yte, p, zero_division=0, output_dict=True),
    "feature_importance": fi.head(10).to_dict("records"),
    "categories": {c: sorted(m[c].dropna().unique().tolist()) for c in CAT_FEATURES},
    "participants_range": [float(m.Participants.min()), float(m.Participants.max())],
}
joblib.dump(model, "models/trained_model.joblib")
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
te = Xte.copy(); te[TARGET] = yte
if "Reservation_ID" in m.columns: te.insert(0, "Reservation_ID", m.loc[te.index, "Reservation_ID"])
te.to_csv("data/midterm_test_set.csv", index=False)
print({k: round(metrics[k], 4) for k in ["accuracy", "precision", "recall", "f1", "majority_baseline"]})
