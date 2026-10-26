import plotly.express as px
import streamlit as st
from modules.data_processing import TARGET

def show(df):
    st.header("Exploratory Data Analysis")
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").T)
    if TARGET not in df.columns:
        st.info("No Usage_Status in this dataset, so outcome-based charts are unavailable.")
    else:
        d = df.dropna(subset=[TARGET])
        st.subheader("Class distribution (Reservation_Outcome)")
        st.plotly_chart(px.pie(d, names=TARGET, hole=0.4), use_container_width=True)
    c1, c2 = st.columns(2)
    if "Facility_Type" in df.columns:
        cnt = df["Facility_Type"].value_counts().rename_axis("Facility_Type").reset_index(name="Reservations")
        c1.plotly_chart(px.bar(cnt, x="Facility_Type", y="Reservations", title="Reservation count by facility type"),
                        use_container_width=True)
    if TARGET in df.columns and "User_Type" in df.columns:
        g = d.groupby("User_Type")[TARGET].agg(Total="size", Failed=lambda s: (s == "Not Completed").sum()).reset_index()
        g["Cancel_NoShow_%"] = (g.Failed / g.Total * 100).round(2)
        c2.plotly_chart(px.bar(g, x="User_Type", y="Cancel_NoShow_%", hover_data=["Total", "Failed"],
                               title="Cancellation/no-show % by user type"), use_container_width=True)
        if "Lead_Time_Category" in d.columns:
            lt = d.dropna(subset=["Lead_Time_Category"])
            order = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
            t = (lt.groupby(["Lead_Time_Category", TARGET]).size().rename("n").reset_index())
            t["%"] = t.n / t.groupby("Lead_Time_Category").n.transform("sum") * 100
            st.plotly_chart(px.bar(t, x="Lead_Time_Category", y="%", color=TARGET, barmode="group",
                                   category_orders={"Lead_Time_Category": order},
                                   title="Outcome % within lead-time category (from Days_Before_Event)"),
                            use_container_width=True)
        st.plotly_chart(px.histogram(d, x="Participants", color=TARGET, barmode="overlay", opacity=0.7,
                                     title="Participants by outcome"), use_container_width=True)
