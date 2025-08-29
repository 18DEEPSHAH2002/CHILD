# app.py
# Streamlit dashboard to analyze "Child Rescue Details" from a Google Sheet
# Author: ChatGPT (GPT-5 Thinking)

import re
import io
import math
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ----------------------
# CONFIG
# ----------------------
SHEET_ID = "13svivZvyrpXZPhApZLcyr4kafCvMcFgC1jIT7Xu64L8"  # provided by user
GID = "0"  # main tab gid
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

st.set_page_config(
    page_title="Child Rescue Analytics",
    page_icon="👧🏽",
    layout="wide",
)

# ----------------------
# THEME (light, clean)
# ----------------------
CUSTOM_CSS = """
<style>
:root { --radius: 16px; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; }
div[data-testid="stMetric"] > div { border-radius: var(--radius); }
.stDataFrame, .stTable { border-radius: var(--radius); overflow: hidden; }
.badge { background:#eef2ff; border:1px solid #c7d2fe; padding:4px 10px; border-radius:999px; font-size:12px; }
.section-title { font-size:1.25rem; font-weight:700; margin: 8px 0 4px 0; }
.subtle { color:#475569; }
.kpi-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:var(--radius); padding:14px; }
.warn { background:#fff7ed; border:1px solid #fdba74; }
.ok { background:#ecfeff; border:1px solid #67e8f9; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------
# HELPERS
# ----------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9_]+", "_", c.strip().lower().replace(" ", "_")).strip("_")
        for c in df.columns
    ]
    return df

AGE_COL_CANDIDATES = ["age", "child_age"]
WEIGHT_COL_CANDIDATES = ["weight", "wt", "weight_kg"]
GENDER_COL_CANDIDATES = ["gender", "sex"]
LOCATION_COL_CANDIDATES = ["rescue_location", "location", "city", "district"]
SCHOOL_COL_CANDIDATES = ["school_enrollment_specify_the_school_name", "school", "school_name", "enrolled_school"]
CLASS_COL_CANDIDATES = ["class", "grade", "standard"]
ATTENDANCE_COL_CANDIDATES = ["monthly_attendance", "attendance", "avg_attendance"]
BIRTH_CERT_COLS = ["birth_certificate", "birth_cert", "birth_cert_status"]
AADHAR_COLS = ["aadhar", "aadhaar", "aadhar_status"]
MEDICAL_COLS = ["medical_check_up", "medical", "medical_status"]
PARENTS_VERIFIED_COLS = ["parents_verified", "guardians_verified", "verification_status"]
DNA_COLS = ["dna", "dna_status"]


def choose_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return None
    def _coerce(v):
        if pd.isna(v):
            return np.nan
        x = str(v).strip().lower()
        if x in {"yes", "y", "true", "1", "done", "completed"}:
            return True
        if x in {"no", "n", "false", "0", "pending", "incomplete"}:
            return False
        return np.nan
    return s.map(_coerce)


def parse_age(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)[\s\-to]+(\d+(?:\.\d+)?)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return round((a + b) / 2.0, 2)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_weight(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return np.nan


@st.cache_data(ttl=300, show_spinner=False)
def load_data(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    df = normalize_columns(df)

    age_col = choose_col(df, AGE_COL_CANDIDATES)
    weight_col = choose_col(df, WEIGHT_COL_CANDIDATES)
    gender_col = choose_col(df, GENDER_COL_CANDIDATES)

    if age_col:
        df[age_col + "_num"] = df[age_col].apply(parse_age)
    if weight_col:
        df[weight_col + "_kg"] = df[weight_col].apply(parse_weight)
    if gender_col:
        df[gender_col] = df[gender_col].astype(str).str.strip().str.title()

    for candidates in [BIRTH_CERT_COLS, AADHAR_COLS, MEDICAL_COLS, PARENTS_VERIFIED_COLS, DNA_COLS]:
        col = choose_col(df, candidates)
        if col:
            df[col + "_bool"] = to_bool_series(df[col])

    att_col = choose_col(df, ATTENDANCE_COL_CANDIDATES)
    if att_col and att_col in df.columns:
        def _att(v):
            if pd.isna(v):
                return np.nan
            s = str(v)
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            return float(m.group(1))
        df[att_col + "_pct"] = df[att_col].apply(_att)

    return df


def kpi_card(label: str, value, help_text: str | None = None):
    with st.container(border=True):
        st.metric(label, value)
        if help_text:
            st.caption(help_text)


# ----------------------
# SIDEBAR
# ----------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Data Source")
st.sidebar.code(CSV_URL, language="text")

if st.sidebar.button("🔄 Refresh data"):
    load_data.clear()

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

with st.spinner("Loading data…"):
    try:
        df = load_data(CSV_URL)
        load_error = None
    except Exception as e:
        df = pd.DataFrame()
        load_error = str(e)

if load_error:
    st.error("Failed to load data from Google Sheets.\n\n" + load_error)
    st.stop()

age_col = choose_col(df, AGE_COL_CANDIDATES)
weight_col = choose_col(df, WEIGHT_COL_CANDIDATES)
gender_col = choose_col(df, GENDER_COL_CANDIDATES)
loc_col = choose_col(df, LOCATION_COL_CANDIDATES)
school_col = choose_col(df, SCHOOL_COL_CANDIDATES)
class_col = choose_col(df, CLASS_COL_CANDIDATES)
att_col = choose_col(df, ATTENDANCE_COL_CANDIDATES)

if gender_col and gender_col in df.columns:
    genders = ["All"] + sorted([g for g in df[gender_col].dropna().unique().tolist() if g])
    sel_gender = st.sidebar.selectbox("Gender", options=genders, index=0)
else:
    sel_gender = "All"

if age_col and age_col + "_num" in df.columns:
    min_age = float(np.nanmin(df[age_col + "_num"])) if not df.empty else 0.0
    max_age = float(np.nanmax(df[age_col + "_num"])) if not df.empty else 20.0
    age_range = st.sidebar.slider("Age range", min_value=0.0, max_value=max(1.0, round(max_age + 0.5, 1)), value=(0.0, max(1.0, max_age)), step=0.5)
else:
    age_range = (0.0, 100.0)

if loc_col and loc_col in df.columns:
    locations = ["All"] + sorted([l for l in df[loc_col].dropna().astype(str).unique().tolist() if l])
    sel_loc = st.sidebar.selectbox("Rescue location", options=locations, index=0)
else:
    sel_loc = "All"

birth_col = choose_col(df, BIRTH_CERT_COLS)
aadhar_col = choose_col(df, AADHAR_COLS)
medical_col = choose_col(df, MEDICAL_COLS)
parents_col = choose_col(df, PARENTS_VERIFIED_COLS)
dna_col = choose_col(df, DNA_COLS)

req_docs = []
if birth_col: req_docs.append((birth_col + "_bool", "Missing Birth Certificate"))
if aadhar_col: req_docs.append((aadhar_col + "_bool", "Missing Aadhar"))
if medical_col: req_docs.append((medical_col + "_bool", "Medical Pending"))
if parents_col: req_docs.append((parents_col + "_bool", "Parents Not Verified"))
if dna_col: req_docs.append((dna_col + "_bool", "DNA Pending"))

with_docs_filter = st.sidebar.toggle("Show only children with pending actions", value=False)

filtered = df.copy()
if gender_col and sel_gender != "All":
    filtered = filtered[filtered[gender_col] == sel_gender]
if age_col and age_col + "_num" in filtered.columns:
    filtered = filtered[filtered[age_col + "_num"].between(age_range[0], age_range[1])]
if loc_col and sel_loc != "All":
    filtered = filtered[filtered[loc_col] == sel_loc]
if with_docs_filter and req_docs:
    mask = np.zeros(len(filtered), dtype=bool)
    for colname, _ in req_docs:
        if colname in filtered.columns:
            mask |= (~filtered[colname].fillna(False))
    filtered = filtered[mask]

# ----------------------
# HEADER
# ----------------------
st.title("👧🏽 Child Rescue Analytics")
st.caption("Interactive dashboard generated from your Google Sheet. Use the sidebar to filter and the tabs below to explore.")

# ----------------------
# KPI ROW
# ----------------------
col1, col2, col3, col4, col5 = st.columns(5)

total_children = len(filtered)
with col1:
    kpi_card("Total children (filtered)", f"{total_children}")

if gender_col and gender_col in filtered.columns:
    male_ct = int((filtered[gender_col] == "Male").sum())
    female_ct = int((filtered[gender_col] == "Female").sum())
    other_ct = int(total_children - male_ct - female_ct)
    with col2:
        kpi_card("Gender split (M/F/Other)", f"{male_ct}/{female_ct}/{other_ct}")
else:
    with col2:
        kpi_card("Gender split", "—")

if age_col and age_col + "_num" in filtered.columns:
    avg_age = round(float(np.nanmean(filtered[age_col + "_num"])) , 2) if total_children else np.nan
    with col3:
        kpi_card("Avg. age", f"{avg_age if not math.isnan(avg_age) else '—'} yrs")
else:
    with col3:
        kpi_card("Avg. age", "—")

if weight_col and weight_col + "_kg" in filtered.columns:
    avg_wt = round(float(np.nanmean(filtered[weight_col + "_kg"])) , 2) if total_children else np.nan
    with col4:
        kpi_card("Avg. weight", f"{avg_wt if not math.isnan(avg_wt) else '—'} kg")
else:
    with col4:
        kpi_card("Avg. weight", "—")

completed_parts = []
possible_parts = 0
for (colname, label) in req_docs:
    if colname in filtered.columns:
        possible_parts += len(filtered)
        completed_parts.append(filtered[colname].fillna(False).sum())
if possible_parts > 0:
    docs_completion = int(round(100 * sum(completed_parts) / possible_parts))
else:
    docs_completion = None
with col5:
    kpi_card("Documentation complete", f"{docs_completion}%" if docs_completion is not None else "—", "Share of YES across doc/medical checks")

# ----------------------
# TABS
# ----------------------

overview_tab, records_tab, gaps_tab = st.tabs([
    "📊 Overview", "📋 Records", "⚠️ Gaps & Actions"
])

with overview_tab:
    st.subheader("Snapshot")
    left, right = st.columns([1,1])

    if gender_col and gender_col in filtered.columns and not filtered.empty:
        gender_counts = (
            filtered.groupby(gender_col).size().reset_index(name="count")
        )
        fig = px.pie(gender_counts, names=gender_col, values="count", title="Gender Distribution")
        left.plotly_chart(fig, use_container_width=True)
    else:
        left.info("Gender column not found.")

    if age_col and age_col + "_num" in filtered.columns and not filtered.empty:
        ages = filtered[age_col + "_num"].dropna()
        if not ages.empty:
            bins = [0,2,5,10,15,20]
            labels = ["0-2","2-5","5-10","10-15","15-20"]
            bands = pd.cut(ages, bins=bins, labels=labels, include_lowest=True)
            age_df = pd.DataFrame({"Age Band": bands}).value_counts().reset_index(name="count")
            age_df = age_df.rename(columns={0:"count"}) if 0 in age_df.columns else age_df
            fig2 = px.bar(age_df, x="Age Band", y="count", title="Age Bands")
            right.plotly_chart(fig2, use_container_width=True)
        else:
            right.info("No numeric age data.")
    else:
        right.info("Age column not found.")

with records_tab:
    st.subheader("Filtered Records")
    st.caption("Tip: Use the column header menu to search & sort. Download below.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name="child_rescue_filtered.csv",
        mime="text/csv",
    )

with gaps_tab:
    st.subheader("Pending Actions")
    if not req_docs:
        st.info("No documentation/medical columns found.")
    else:
        cols = st.columns(2)
        todo_frames = []
        for (colname, label) in req_docs:
            if colname in filtered.columns:
                pending_df = filtered[~filtered[colname].fillna(False)]
                count = len(pending_df)
                with cols[0] if len(todo_frames)%2==0 else cols[1]:
                    with st.container(border=True, key=label):
                        st.markdown(f"**{label}**")
                        st.write(f"{count} child(ren) pending")
                        if count:
                            st.dataframe(pending_df, use_container_width=True, hide_index=True)
                            todo_frames.append(pending_df)
        if not todo_frames:
            st.success("Great! No pending items found for the current filters.")

# ----------------------
# FOOTER / HELP
# ----------------------
with st.expander("ℹ️ How this works / Setup"):
    st.markdown(
        """
        **Data Source**: This app reads your Google Sheet via the CSV export URL. For private sheets, share the sheet as "Anyone with the link - Viewer" or use a service
"""
    
