import json, datetime as dt
import joblib, pandas as pd
import streamlit as st
from modules.data_processing import FEATURES, POSITIVE, CAT_FEATURES

@st.cache_resource
def load_model():
    return joblib.load("models/trained_model.joblib")

@st.cache_data
def load_metrics():
    return json.load(open("models/metrics.json"))

def risk_level(p):
    return "High" if p >= 0.60 else "Medium" if p >= 0.40 else "Low"

def score(df: pd.DataFrame) -> pd.DataFrame:
    """Only the four midterm predictors are passed to the saved pipeline."""
    model = load_model()
    X = df[FEATURES]
    out = df.copy()
    out["Predicted_Outcome"] = model.predict(X)
    out["P_Not_Completed"] = model.predict_proba(X)[:, list(model.classes_).index(POSITIVE)].round(3)
    out["Risk_Level"] = out["P_Not_Completed"].apply(risk_level)
    return out

def show():
    m = load_metrics()
    st.header("Prediction: reservation completion risk")
    st.caption(f"Model: {m['model_version']} (Decision Tree, max_depth=5) | trained {m['trained_at']} | scikit-learn {m['sklearn_version']}")
    st.subheader("Model metrics (midterm 20% test set, weighted)")
    c = st.columns(5)
    for col, (lbl, k) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1")]):
        col.metric(lbl, f"{m[k]:.3f}")
    c[4].metric("Majority-class baseline", f"{m['majority_baseline']:.3f}")
    with st.expander("Confusion matrix and most influential features"):
        cm = pd.DataFrame(m["confusion_matrix"], index=[f"Actual {l}" for l in m["labels"]],
                          columns=[f"Pred {l}" for l in m["labels"]])
        st.dataframe(cm)
        st.dataframe(pd.DataFrame(m["feature_importance"]).head(5))
    st.subheader("Predict a new reservation")
    cats = m["categories"]; lo, hi = m["participants_range"]
    with st.form("single"):
        a, b = st.columns(2)
        fac = a.selectbox("Facility type", cats["Facility_Type"])
        usr = a.selectbox("User type", cats["User_Type"])
        pur = b.selectbox("Purpose", cats["Purpose"])
        n = b.number_input("Participants", min_value=1, max_value=2000, value=50, step=1)
        go = st.form_submit_button("Predict")
    if go:
        if n < lo or n > hi:
            st.warning(f"Participants outside the training range ({lo:.0f}-{hi:.0f}); treat the result with extra caution.")
        row = pd.DataFrame([{"Facility_Type": fac, "User_Type": usr, "Purpose": pur, "Participants": float(n)}])[FEATURES]
        r = score(row).iloc[0]
        st.subheader(f"Predicted outcome: {r.Predicted_Outcome}")
        k = st.columns(3)
        k[0].metric("P(Not Completed)", f"{r.P_Not_Completed:.1%}")
        k[1].metric("Risk level", r.Risk_Level)
        k[2].metric("Predicted at", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.warning("Prediction, not an observed fact. The Administrator reviews it and makes the final decision; "
                   "never reject a reservation automatically. Probabilities come from a class-balanced decision tree "
                   "and are not calibrated.")
    st.subheader("Consistency check against the midterm test set")
    t = pd.read_csv("data/midterm_test_set.csv")
    s = score(t)
    acc = (s.Predicted_Outcome == s.Reservation_Outcome).mean()
    st.write(f"Accuracy on the same {len(s)} held-out records: **{acc:.4f}** (midterm notebook: {m['accuracy']:.4f}).")
    st.dataframe(s[["Reservation_ID", *FEATURES, "Reservation_Outcome", "Predicted_Outcome", "P_Not_Completed", "Risk_Level"]].head(5))
    st.subheader("Limitations")
    st.markdown(f"- Accuracy ({m['accuracy']:.3f}) is **below** the majority-class baseline ({m['majority_baseline']:.3f}); "
                "the model is weak and should only prompt a human review.\n"
                "- Uses only 4 submission-time predictors; no leakage fields (Usage_Status, Actual_Attendance, "
                "Cancellation_Days_Before, Attendance_Ratio) are inputs.\n"
                "- Trained on 320 records of synthetic/sample data; may not generalise to new patterns.")
