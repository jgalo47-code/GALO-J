# Campus Facility Reservation Monitoring (Streamlit)
Deploys the midterm Decision Tree (Completed vs Not Completed) as decision support for the Facility Administrator.

## Run
pip install -r requirements.txt
python train_model.py     # reproduces the midterm pipeline, saves models/trained_model.joblib
streamlit run app.py

## Reused from midterm
Cleaning (dedupe, text standardising, numeric fixes), target mapping (Used -> Completed; Cancelled/No-Show -> Not Completed),
predictors Facility_Type, User_Type, Purpose, Participants, 80/20 stratified split, DecisionTree(max_depth=5, class_weight=balanced, random_state=42).
Results reproduced: accuracy 0.4125, weighted precision 0.484, recall 0.412, F1 0.430.

## Differences / limitations
- Lead-time chart uses Days_Before_Event (EDA only; the midterm alias search did not match this column, so it is not a model input).
- Accuracy is below the 0.675 majority-class baseline; predictions only prompt human review.
- Synthetic/sample data only. Never commit real student records or passwords.
