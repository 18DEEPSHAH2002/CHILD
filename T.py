"""
Rescued Children – Streamlit Dashboard (app.py)
================================================
How to run:
  1) pip install streamlit pandas numpy plotly python-dateutil
  2) streamlit run app.py

Set your Google Sheet below (must be shared as "Anyone with the link can view").
You can also use the file uploader in the sidebar to load a local CSV/Excel.
"""

import io
from datetime import datetime, date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Rescued Children – Dashboard",
    page_icon="🧒",
    layout="wide",
)

# 👉 EDIT THESE
SHEET_ID = "13svivZvyrpXZPhApZLcyr4kafCvMcFgC1jIT7Xu64L8"  # Your Google Sheet ID
GID = 0  # Tab GID (use the number after gid=... in the URL)

# Basic styles (subtle, readable)
st.markdown(
    """
    <style>
      .metric {border: 1px solid #e5e7eb; padding: 12px; border-radius: 16px; background: #f8fafc;}
      .metric h3 {margin: 0 0 4px 0; font-size: 0.9rem; color: #334155;}
      .metric p {margin: 0; font-size: 1.4rem; font-weight: 700; color: #0f172a;}
      .small {color:#475569; font-size: 0.85rem}
      .stDataFrame {border-radius: 12px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# DATA LOADING
# -------------------------
@st.cache_data(show_spinner=True)
def load_from_gsheet(sheet_id: str, gid: int = 0) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    df = pd.read_csv(url)
    return df

@st.cache_data(show_spinner=True)
def load_file(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded)
    elif uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        # Try csv fallback
        return pd.read_csv(uploaded)

# -------------------------
# HELPERS
# -------------------------

def coalesce_cols(df: pd.DataFrame, options: list[str]) -> Optional[str]:
    """Return first existing column from options (case-insensitive match)."""
    lower_map = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    return None


def parse_dates_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)


def make_clickable_link(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    u = url.strip()
    # If the cell contains a bare URL without scheme, attempt to add https://
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u
    name = u.split("/")[-1] or "file"
    return f'<a href="{u}" target="_blank">📎 {name}</a>'


def age_binner(age_val) -> str:
    try:
        a = float(age_val)
    except Exception:
        return "Unknown"
    if a < 7:
        return "0–6"
    elif a < 13:
        return "7–12"
    elif a <= 18:
        return "13–18"
    else:
        return ">18"


# -------------------------
# LOAD DATA (GSHEET or UPLOAD)
# -------------------------
with st.sidebar:
    st.header("🔌 Data Source")
    use_gsheet = st.toggle("Use Google Sheet", value=True)
    sheet_id_input = st.text_input("Google Sheet ID", value=SHEET_ID)
    gid_input = st.number_input("GID (tab id)", value=GID, step=1)
    uploaded = st.file_uploader("…or upload CSV/Excel", type=["csv", "xls", "xlsx"])

if uploaded is not None:
    df_raw = load_file(uploaded)
else:
    if use_gsheet and sheet_id_input:
        try:
            df_raw = load_from_gsheet(sheet_id_input, int(gid_input))
        except Exception as e:
            st.error(f"Could not load Google Sheet: {e}")
            st.stop()
    else:
        st.warning("Provide a Google Sheet ID or upload a file.")
        st.stop()

# Clean columns: strip, unify spaces
df_raw.columns = [str(c).strip() for c in df_raw.columns]

# Detect commonly used columns
COL_OFFICER = coalesce_cols(df_raw, ["Marked to Officer", "Officer", "Nodal Officer", "Assigned Officer"]) or None
COL_PRIORITY = coalesce_cols(df_raw, ["Priority"]) or None
COL_BRANCH = coalesce_cols(df_raw, ["Dealing Branch", "Branch"]) or None
COL_SUBJECT = coalesce_cols(df_raw, ["Subject"]) or None
COL_RECEIVED_FROM = coalesce_cols(df_raw, ["Received From", "Source", "Source of Rescue"]) or None
COL_FILE = coalesce_cols(df_raw, ["File", "Document", "Attachment", "Link"]) or None
COL_ENTRY_DATE = coalesce_cols(df_raw, ["Entry Date", "Date", "Created On"]) or None
COL_STATUS = coalesce_cols(df_raw, ["Status"]) or None
COL_RESPONSE = coalesce_cols(df_raw, ["Response Recieved", "Response Received", "Response", "Reply"]) or None
COL_STATE = coalesce_cols(df_raw, ["State"]) or None
COL_DISTRICT = coalesce_cols(df_raw, ["District"]) or None
COL_GENDER = coalesce_cols(df_raw, ["Gender", "Sex"]) or None
COL_AGE = coalesce_cols(df_raw, ["Age"]) or None

# Parse dates
if COL_ENTRY_DATE:
    df_raw[COL_ENTRY_DATE] = parse_dates_safe(df_raw[COL_ENTRY_DATE])

# Derived columns
if COL_AGE:
    df_raw["Age Band"] = df_raw[COL_AGE].apply(age_binner)

if COL_FILE:
    df_raw["File Link"] = df_raw[COL_FILE].apply(make_clickable_link)

# -------------------------
# SIDEBAR FILTERS
# -------------------------
with st.sidebar:
    st.header("🔎 Filters")
    # Date range
    if COL_ENTRY_DATE:
        min_d = pd.to_datetime(df_raw[COL_ENTRY_DATE]).min()
        max_d = pd.to_datetime(df_raw[COL_ENTRY_DATE]).max()
        if pd.isna(min_d) or pd.isna(max_d):
            date_range = None
        else:
            start, end = st.date_input(
                "Entry Date Range",
                value=(min_d.date(), max_d.date()),
                min_value=min_d.date(),
                max_value=max_d.date(),
            )
            date_range = (pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))
    else:
        date_range = None

    def multiselect_if(col_name: Optional[str], label: str):
        if not col_name:
            return None
        opts = sorted([x for x in df_raw[col_name].dropna().astype(str).unique() if x != ""])
        return st.multiselect(label, options=opts, default=[])

    f_officer = multiselect_if(COL_OFFICER, "Officer")
    f_branch = multiselect_if(COL_BRANCH, "Branch")
    f_priority = multiselect_if(COL_PRIORITY, "Priority")
    f_state = multiselect_if(COL_STATE, "State")
    f_district = multiselect_if(COL_DISTRICT, "District")
    f_status = multiselect_if(COL_STATUS, "Status")

# Apply filters
fdf = df_raw.copy()

if date_range and COL_ENTRY_DATE:
    s, e = date_range
    fdf = fdf[(fdf[COL_ENTRY_DATE] >= s) & (fdf[COL_ENTRY_DATE] < e)]

if COL_OFFICER and f_officer:
    fdf = fdf[fdf[COL_OFFICER].astype(str).isin(f_officer)]
if COL_BRANCH and f_branch:
    fdf = fdf[fdf[COL_BRANCH].astype(str).isin(f_branch)]
if COL_PRIORITY and f_priority:
    fdf = fdf[fdf[COL_PRIORITY].astype(str).isin(f_priority)]
if COL_STATE and f_state:
    fdf = fdf[fdf[COL_STATE].astype(str).isin(f_state)]
if COL_DISTRICT and f_district:
    fdf = fdf[fdf[COL_DISTRICT].astype(str).isin(f_district)]
if COL_STATUS and f_status:
    fdf = fdf[fdf[COL_STATUS].astype(str).isin(f_status)]

# -------------------------
# HEADER
# -------------------------
st.title("🧒 Rescued Children – Monitoring Dashboard")
st.caption("Analyze, filter, and monitor case progress with clean visuals and KPIs.")

# -------------------------
# KPIs
# -------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_cases = len(fdf)
    st.markdown("<div class='metric'><h3>Total Cases</h3><p>{}</p></div>".format(total_cases), unsafe_allow_html=True)

with col2:
    pending = None
    if COL_STATUS:
        pending = (fdf[COL_STATUS].astype(str).str.lower().isin(["pending", "open", "in progress"]).sum())
        st.markdown("<div class='metric'><h3>Pending</h3><p>{}</p></div>".format(int(pending)), unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric'><h3>Pending</h3><p>—</p></div>", unsafe_allow_html=True)

with col3:
    responded_pct = "—"
    if COL_RESPONSE:
        # Treat truthy strings like Yes/Received as responded
        responded = fdf[COL_RESPONSE].astype(str).str.lower().isin(["yes", "received", "true", "y"]).sum()
        responded_pct = f"{(responded / total_cases * 100):.1f}%" if total_cases else "0%"
    st.markdown("<div class='metric'><h3>Response Rate</h3><p>{}</p></div>".format(responded_pct), unsafe_allow_html=True)

with col4:
    missing_files = "—"
    if COL_FILE:
        missing_files = fdf[COL_FILE].isna().sum() + (fdf[COL_FILE].astype(str).str.strip() == "").sum()
    st.markdown("<div class='metric'><h3>Missing Files</h3><p>{}</p></div>".format(missing_files), unsafe_allow_html=True)

# -------------------------
# CHARTS
# -------------------------

def safe_bar(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("No data for this chart.")
        return
    fig = px.bar(df, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)

# 1) Trend: Cases by month
if COL_ENTRY_DATE and not fdf.empty:
    temp = fdf.copy()
    temp["Month"] = temp[COL_ENTRY_DATE].dt.to_period("M").dt.to_timestamp()
    by_month = temp.groupby("Month").size().reset_index(name="Count")
    fig = px.line(by_month, x="Month", y="Count", markers=True, title="Cases Over Time (by Month)")
    st.plotly_chart(fig, use_container_width=True)

# Two-column charts block
c1, c2 = st.columns(2)

with c1:
    # Cases by Officer
    if COL_OFFICER:
        by_officer = fdf.groupby(COL_OFFICER).size().reset_index(name="Count").sort_values("Count", ascending=False)
        safe_bar(by_officer, x=COL_OFFICER, y="Count", title="Cases by Officer")
    else:
        st.info("Officer column not found.")

with c2:
    # Cases by Branch
    if COL_BRANCH:
        by_branch = fdf.groupby(COL_BRANCH).size().reset_index(name="Count").sort_values("Count", ascending=False)
        safe_bar(by_branch, x=COL_BRANCH, y="Count", title="Cases by Branch")
    else:
        st.info("Branch column not found.")

c3, c4 = st.columns(2)

with c3:
    # Priority Pie
    if COL_PRIORITY:
        pri = fdf[COL_PRIORITY].astype(str).replace({"": np.nan}).dropna()
        pri_df = pri.value_counts().reset_index()
        pri_df.columns = ["Priority", "Count"]
        if not pri_df.empty:
            fig = px.pie(pri_df, names="Priority", values="Count", title="Priority Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No priority data.")
    else:
        st.info("Priority column not found.")

with c4:
    # Status Pie
    if COL_STATUS:
        stat = fdf[COL_STATUS].astype(str).replace({"": np.nan}).dropna()
        stat_df = stat.value_counts().reset_index()
        stat_df.columns = ["Status", "Count"]
        if not stat_df.empty:
            fig = px.pie(stat_df, names="Status", values="Count", title="Case Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status data.")
    else:
        st.info("Status column not found.")

c5, c6 = st.columns(2)

with c5:
    # Gender Pie
    if COL_GENDER:
        g = fdf[COL_GENDER].astype(str).replace({"": np.nan}).dropna()
        g_df = g.value_counts().reset_index()
        g_df.columns = ["Gender", "Count"]
        if not g_df.empty:
            fig = px.pie(g_df, names="Gender", values="Count", title="Gender Ratio")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gender data.")
    else:
        st.info("Gender column not found.")

with c6:
    # Age Bands
    if COL_AGE or "Age Band" in fdf.columns:
        if "Age Band" not in fdf.columns:
            fdf["Age Band"] = fdf[COL_AGE].apply(age_binner)
        ab = (
            fdf["Age Band"].astype(str).replace({"": np.nan}).dropna().value_counts().reindex(["0–6", "7–12", "13–18", ">18", "Unknown"], fill_value=0).reset_index()
        )
        ab.columns = ["Age Band", "Count"]
        safe_bar(ab, x="Age Band", y="Count", title="Age Distribution")
    else:
        st.info("Age column not found.")

# -------------------------
# TABLE (with clickable file links if present)
# -------------------------
st.subheader("📋 Records")
show_cols = list(df_raw.columns)
if "File Link" in fdf.columns:
    # Insert the clickable column next to the original file column for visibility
    if COL_FILE and COL_FILE in show_cols:
        insert_at = show_cols.index(COL_FILE) + 1
    else:
        insert_at = len(show_cols)
    if "File Link" not in show_cols:
        show_cols.insert(insert_at, "File Link")

# Allow download of filtered data
csv_data = fdf.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv_data, file_name="rescued_children_filtered.csv", mime="text/csv")

# Render table
if "File Link" in fdf.columns:
    df_display = fdf.copy()
    # Use HTML for the clickable link
    st.write(
        df_display[show_cols]
        .to_html(escape=False, index=False)
        .replace("<table", "<div class='small'>Scroll horizontally for more →</div><table"),
        unsafe_allow_html=True,
    )
else:
    st.dataframe(fdf[show_cols], use_container_width=True)

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("🔍 Quick Insights")
ins = []
if COL_OFFICER and not fdf.empty:
    top_officer = fdf[COL_OFFICER].value_counts().idxmax()
    top_officer_cnt = fdf[COL_OFFICER].value_counts().max()
    ins.append(f"Most assigned officer: **{top_officer}** ({top_officer_cnt} cases)")
if COL_BRANCH and not fdf.empty:
    top_branch = fdf[COL_BRANCH].value_counts().idxmax()
    ins.append(f"Heaviest branch: **{top_branch}**")
if COL_PRIORITY and not fdf.empty:
    high_cnt = (fdf[COL_PRIORITY].astype(str).str.lower() == "high").sum()
    if high_cnt:
        ins.append(f"High-priority cases: **{high_cnt}**")
if COL_STATUS and not fdf.empty:
    pending_cnt = (fdf[COL_STATUS].astype(str).str.lower().isin(["pending", "open", "in progress"]).sum())
    if pending_cnt:
        ins.append(f"Pending/open cases: **{int(pending_cnt)}**")
if COL_FILE and not fdf.empty:
    missing_files_cnt = fdf[COL_FILE].isna().sum() + (fdf[COL_FILE].astype(str).str.strip() == "").sum()
    if missing_files_cnt:
        ins.append(f"Records missing file links: **{int(missing_files_cnt)}**")

if ins:
    for i in ins:
        st.markdown(f"- {i}")
else:
    st.write("No notable insights based on current filters.")

st.caption("Built with ❤️ in Streamlit. Use the sidebar to filter and explore. ")
