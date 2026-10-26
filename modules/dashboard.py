import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data_processing import TARGET, validate
from modules.prediction import score

def show(df, role):
    st.header("Facility Administrator Dashboard")
    errs, _ = validate(df)
    if errs:
        st.error(errs); return
    st.subheader("Observed reservation data (historical facts)")
    k = st.columns(3)
    k[0].metric("Total reservations", len(df))
    k[1].metric("Most-used facility", df["Facility_Type"].value_counts().idxmax())
    if TARGET in df.columns:
        d = df.dropna(subset=[TARGET])
        k[2].metric("Observed cancel/no-show rate", f"{(d[TARGET] == 'Not Completed').mean():.1%}")
        u = d.groupby("User_Type")[TARGET].apply(lambda s: (s == "Not Completed").mean() * 100).reset_index(name="%")
        st.plotly_chart(px.bar(u, x="User_Type", y="%", title="Observed cancellation/no-show % by user type"),
                        use_container_width=True)
    s = score(df)
    st.subheader("Predicted risk (model output, NOT observed fact)")
    c1, c2 = st.columns(2)
    order = {"Risk_Level": ["Low", "Medium", "High"]}
    c1.plotly_chart(px.histogram(s, x="Risk_Level", category_orders=order, title="Predicted risk levels"),
                    use_container_width=True)
    fa = s.groupby("Facility_Type")["P_Not_Completed"].mean().reset_index()
    c2.plotly_chart(px.bar(fa, x="Facility_Type", y="P_Not_Completed", title="Average predicted risk by facility"),
                    use_container_width=True)
    st.subheader("Review queue (highest predicted risk first)")
    cols = [c for c in ["Reservation_ID", "Facility_Type", "User_Type", "Purpose", "Participants",
                        "P_Not_Completed", "Risk_Level"] if c in s.columns]
    q = s[s.Risk_Level == "High"].sort_values("P_Not_Completed", ascending=False).head(20).reset_index(drop=True)
    st.dataframe(q[cols])
    st.info("Interpretation: High means similar past reservations were often not completed. Recommended human follow-up: "
            "contact the requester to reconfirm, send a reminder, or prepare a waitlist. Not an automatic rejection.")
    st.subheader("Record final decision")
    if role != "Administrator":
        st.warning("Only the Administrator role can record decisions. You are viewing as a read-only User."); return
    if q.empty:
        st.write("No high-risk reservations in this dataset."); return
    st.session_state.setdefault("log", [])
    row = st.selectbox("Queue row", q.index, format_func=lambda i: str(q.loc[i, cols[0]]))
    act = st.selectbox("Decision", ["Send reminder", "Request reconfirmation", "Offer waitlist", "Keep as is"])
    note = st.text_input("Note (optional)")
    if st.button("Save decision"):
        st.session_state["log"].append({"time": dt.datetime.now().isoformat(timespec="seconds"), "actor": role,
                                        "reservation": str(q.loc[row, cols[0]]), "decision": act, "note": note})
    if st.session_state["log"]:
        st.dataframe(pd.DataFrame(st.session_state["log"]))
