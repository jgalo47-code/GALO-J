import streamlit as st
from modules import data_processing as dp, exploratory_analysis, prediction, dashboard, system_design

st.set_page_config(page_title="Campus Facility Reservation System", layout="wide")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
role = st.sidebar.radio("Role", ["Administrator", "User"])
st.sidebar.caption("Predictions are decision support only; a human decides.")

@st.cache_data
def load_default():
    import pandas as pd
    return pd.read_csv("data/facility_reservations.csv")

@st.cache_data
def clean_cached(raw):
    return dp.clean_df(raw)

raw = st.session_state.get("raw")
if raw is None:
    raw = load_default()
df, n_removed = clean_cached(raw)

if menu == "Home":
    st.title("Campus Facility Reservation and Monitoring")
    st.write("Deploys the midterm Decision Tree that predicts whether a reservation will be **Completed** or "
             "**Not Completed** (cancelled/no-show), to help facility administrators decide where to follow up.")
    st.write("Use the sidebar: upload data, explore it, predict a reservation, and review the risk queue.")
    st.info("Predictions never authorize rejecting a reservation or denying a service. Data is synthetic/sample.")
elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        import pandas as pd
        try:
            new = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}"); st.stop()
        errs, warns = dp.validate(new, prediction.load_metrics()["categories"])
        for e in errs: st.error(e)
        for w in warns: st.warning(w)
        if not errs:
            st.session_state["raw"] = new; raw = new
            df, n_removed = clean_cached(new)
            st.success(f"Accepted. {len(new)} rows loaded; this is now the active dataset.")
    else:
        st.info("No upload: the bundled midterm dataset (facility_reservations.csv) is active.")
    st.dataframe(raw.head(20))
    st.write(f"Rows: {len(raw)} | Columns: {raw.shape[1]}")
    nulls, dup_rows, dup_ids = dp.quality_summary(raw)
    st.write(f"Duplicate rows: {dup_rows} | Duplicate Reservation_IDs: {dup_ids}")
    st.dataframe(nulls)
    st.caption(f"Required columns: {dp.FEATURES}. Optional: Usage_Status, Reservation_ID, Days_Before_Event.")
elif menu == "Data Analysis":
    exploratory_analysis.show(df)
elif menu == "Prediction":
    prediction.show()
elif menu == "Dashboard":
    dashboard.show(df, role)
else:
    system_design.show()
