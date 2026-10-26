"""Cleaning/validation reused from the midterm notebook (Parts II and IV)."""
import re
import numpy as np
import pandas as pd

TARGET = "Reservation_Outcome"
CAT_FEATURES = ["Facility_Type", "User_Type", "Purpose"]
NUM_FEATURES = ["Participants"]
FEATURES = CAT_FEATURES + NUM_FEATURES          # exact midterm predictors (no leakage fields)
POSITIVE = "Not Completed"                       # class the risk score refers to
STATUS_MAP = {"used": "Completed", "completed": "Completed", "cancelled": "Not Completed",
              "canceled": "Not Completed", "no-show": "Not Completed", "no show": "Not Completed",
              "noshow": "Not Completed"}
LEAKAGE = ["Usage_Status", "Actual_Attendance", "Cancellation_Days_Before", "Reservation_Outcome", "Attendance_Ratio"]

def lead_category(x):
    if pd.isna(x): return np.nan
    return "Last Minute" if x <= 2 else "Short Lead" if x <= 7 else "Medium Lead" if x <= 14 else "Early Booking"

def clean_df(raw: pd.DataFrame):
    """Same steps as the midterm: drop duplicates, standardise text, fix numerics, derive target/lead category."""
    df = raw.copy()
    n0 = len(df)
    df = df.drop_duplicates()
    if "Reservation_ID" in df.columns:
        df = df.drop_duplicates(subset=["Reservation_ID"], keep="first")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].map(lambda x: re.sub(r"\s+", " ", x.strip()).casefold().title() if isinstance(x, str) else x)
    for c in ["Participants", "Actual_Attendance", "Duration_Hours", "Days_Before_Event", "Cancellation_Days_Before"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan
    if "Participants" in df.columns:
        df.loc[df["Participants"] == 0, "Participants"] = np.nan
    if "Usage_Status" in df.columns:
        s = df["Usage_Status"].astype("string").str.strip().str.casefold()
        df[TARGET] = s.map(STATUS_MAP)
    if "Days_Before_Event" in df.columns:
        df["Lead_Time_Category"] = df["Days_Before_Event"].apply(lead_category)
    return df, n0 - len(df)

def validate(raw: pd.DataFrame, known_categories=None):
    """Return (errors, warnings). Errors block use of the file."""
    errors, warnings = [], []
    if raw is None or raw.empty:
        return ["The file is empty or has no rows."], warnings
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        return [f"Missing required column(s): {missing}. Required: {FEATURES}."], warnings
    bad = pd.to_numeric(raw["Participants"], errors="coerce").isna().sum() - raw["Participants"].isna().sum()
    if bad > 0:
        warnings.append(f"{bad} non-numeric Participants value(s) will be treated as missing and imputed.")
    if "Usage_Status" not in raw.columns:
        warnings.append("Usage_Status not found: observed outcomes/EDA by outcome unavailable; scoring still works.")
    else:
        st_ = raw["Usage_Status"].astype("string").str.strip().str.casefold()
        unk = int((~st_.isin(STATUS_MAP.keys())).sum())
        if unk: warnings.append(f"{unk} row(s) have unrecognized Usage_Status and are left unknown.")
    leak = [c for c in LEAKAGE if c in raw.columns]
    if leak:
        warnings.append(f"Columns {leak} are never used as model inputs (leakage prevention).")
    if known_categories:
        for c in CAT_FEATURES:
            new = set(raw[c].dropna().astype(str).str.strip().str.title()) - set(known_categories[c])
            if new: warnings.append(f"{c} has unseen value(s) {sorted(new)}; the model will ignore them (one-hot 'ignore').")
    return errors, warnings

def quality_summary(raw: pd.DataFrame):
    nulls = pd.DataFrame({"missing": raw.isna().sum(), "dtype": raw.dtypes.astype(str)})
    dup_rows = int(raw.duplicated().sum())
    dup_ids = int(raw["Reservation_ID"].duplicated().sum()) if "Reservation_ID" in raw.columns else None
    return nulls, dup_rows, dup_ids
