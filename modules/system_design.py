import streamlit as st

def show():
    st.header("System Design (SAD Integration)")
    st.subheader("Actors")
    st.markdown("- **Administrator**: uploads/validates data, views statistics and predictions, records final decisions.\n"
                "- **User** (student, faculty, staff, organization): submits reservations; read-only view of statistics here.")
    st.subheader("Functional requirements (evidence-based)")
    st.markdown("1. **FR-01** Validate uploaded CSVs (required columns, empty file, nulls, duplicate rows/IDs). "
                "*Evidence: midterm cleaning step found duplicate/inconsistent-text risks.*\n"
                "2. **FR-02** Display reservation counts by facility type. *Evidence: Auditorium is highest with 95 reservations.*\n"
                "3. **FR-03** Display cancellation/no-show % by user type with totals. *Evidence: Organization is highest at 36.79%.*\n"
                "4. **FR-04** Predict Completed/Not Completed with risk level, model version and timestamp for a new reservation.\n"
                "5. **FR-05** Show a risk-ranked review queue and record the Administrator's final decision. "
                "*Evidence: model accuracy 0.412 means every prediction needs human review.*")
    st.subheader("Non-functional requirements")
    st.markdown("1. **NFR-01 Performance**: the saved pipeline is loaded once (`st.cache_resource`), never retrained per refresh.\n"
                "2. **NFR-02 Privacy**: only synthetic/authorized data; no passwords or real student records in the repository.\n"
                "3. **NFR-03 Transparency**: observed facts and predictions labeled separately; model version, timestamp and limitations visible.")
    st.subheader("Updated use case diagram")
    st.image("assets/use_case_diagram.png")
    st.subheader("Decision workflow")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box]; "Upload CSV" -> "Validate + clean" -> "Saved pipeline" -> '
                      '"Risk level (prediction)" -> "Review queue" -> "Administrator reviews" -> "Final decision" -> "Decision log" }')
    st.caption("Only the Administrator can record final decisions; the User role is read-only.")
