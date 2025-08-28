import streamlit as st
import pandas as pd
import numpy as np
import io
from urllib.parse import urlparse, parse_qs
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Universal Google Sheet Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helpers
# -----------------------------
def to_csv_export_url(google_sheet_url: str) -> str:
    """
    Convert an 'edit?gid=' Google Sheet URL into a direct CSV export URL for that gid.
    Works if the sheet is viewable publicly or to anyone with the link.
    """
    if "docs.google.com/spreadsheets" not in google_sheet_url:
        return google_sheet_url  # assume it's already a direct CSV or a normal CSV link

    # Try to extract gid
    parsed = urlparse(google_sheet_url)
    qs = parse_qs(parsed.query)
    gid = None
    if "gid" in qs and qs["gid"]:
        gid = qs["gid"][0]

    # Turn /edit? into /export?format=csv
    # Typical: https://docs.google.com/spreadsheets/d/<ID>/edit?gid=<GID>
    # ->      https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>
    parts = google_sheet_url.split("/edit")
    if len(parts) > 1:
        base = parts[0]
        export = f"{base}/export?format=csv"
        if gid:
            export += f"&gid={gid}"
        return export

    # If no /edit segment, try a simple replacement fallback
    return google_sheet_url.replace("/view", "/export?format=csv")

@st.cache_data(show_spinner=True, ttl=600)
def load_data(sheet_url: str) -> pd.DataFrame:
    csv_url = to_csv_export_url(sheet_url.strip())
    df = pd.read_csv(csv_url, dtype=str)  # read as string first to avoid bad type guesses
    # Clean columns: strip spaces/newlines in headers
    df.columns = [str(c).strip() for c in df.columns]
    # Try smart type inference per column
    df = smart_cast_df(df)
    return df

def smart_cast_series(s: pd.Series) -> pd.Series:
    # Try boolean
    lower_vals = s.dropna().astype(str).str.lower().unique()
    bool_like = set(["true","false","yes","no","y","n","t","f","1","0"])
    if len(lower_vals) > 0 and all(v in bool_like for v in lower_vals):
        return s.astype(str).str.lower().map({
            "true": True, "t": True, "yes": True, "y": True, "1": True,
            "false": False, "f": False, "no": False, "n": False, "0": False
        })

    # Try numeric
    try:
        s_num = pd.to_numeric(s, errors="raise")
        return s_num
    except Exception:
        pass

    # Try datetime (day-first off by default; we’ll allow flexible)
    try:
        s_dt = pd.to_datetime(s, errors="raise", infer_datetime_format=True, utc=False)
        return s_dt
    except Exception:
        pass

    # Fallback to stripped strings
    return s.astype(str).str.strip().replace({"": np.nan})

def smart_cast_df(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        out[col] = smart_cast_series(df[col])
    return pd.DataFrame(out)

def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact profile for each column:
    type, non-null count, missing %, unique, sample values, min/max for numeric/dates, avg length for text.
    """
    rows = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        non_null = s.notna().sum()
        missing = s.isna().mean() * 100.0
        unique = s.nunique(dropna=True)

        sample_vals = ", ".join([repr(x)[:30] for x in s.dropna().unique()[:5]])

        min_val = max_val = None
        avg_len = None
        if pd.api.types.is_numeric_dtype(s):
            min_val = pd.to_numeric(s, errors="coerce").min()
            max_val = pd.to_numeric(s, errors="coerce").max()
        elif pd.api.types.is_datetime64_any_dtype(s):
            min_val = pd.to_datetime(s, errors="coerce").min()
            max_val = pd.to_datetime(s, errors="coerce").max()
        else:
            # text-like
            avg_len = s.dropna().astype(str).apply(len).mean() if non_null else None

        rows.append({
            "Column": c,
            "Type": dtype,
            "Non-Null": int(non_null),
            "Missing %": round(missing, 2),
            "Unique": int(unique),
            "Min": min_val,
            "Max": max_val,
            "Avg Length (if text)": None if avg_len is None else round(avg_len, 2),
            "Sample Values": sample_vals
        })
    return pd.DataFrame(rows)

def infer_date_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

def infer_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def infer_categorical_columns(df: pd.DataFrame, max_unique_ratio=0.2, max_unique_abs=50):
    cats = []
    n = len(df)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cats.append(c)
        else:
            # numeric with small cardinality: treat as categorical
            u = df[c].nunique(dropna=True)
            if u <= max_unique_abs or (n > 0 and u / n <= max_unique_ratio):
                cats.append(c)
    return list(dict.fromkeys(cats))  # preserve order, unique

def download_link(df: pd.DataFrame, filename="filtered_data.csv"):
    csv = df.to_csv(index=False)
    st.download_button("⬇ Download filtered data (CSV)", data=csv, file_name=filename, mime="text/csv")

# -----------------------------
# Sidebar – Data Input
# -----------------------------
st.sidebar.title("🔗 Data Source")
default_url = "https://docs.google.com/spreadsheets/d/13svivZvyrpXZPhApZLcyr4kafCvMcFgC1jIT7Xu64L8/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Google Sheet URL (shareable link)", value=default_url)
st.sidebar.caption("Tip: Make sure your sheet is accessible to 'Anyone with the link' for direct CSV export.")

load_button = st.sidebar.button("Load / Refresh Data", type="primary")

# -----------------------------
# Load data
# -----------------------------
if sheet_url or load_button:
    try:
        df = load_data(sheet_url)
    except Exception as e:
        st.error(f"❌ Could not load data. Check sharing settings or link.\n\nDetails: {e}")
        st.stop()
else:
    st.info("Paste your Google Sheet link in the sidebar, then click *Load / Refresh Data*.")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.title("📊 Google Sheet Explorer – Instant Insight Dashboard")
st.caption("Paste a Google Sheet link in the sidebar and explore everything in one place.")

st.markdown(
    """
    *What you get at a glance*
    - Data overview, missing values, and column profiling  
    - Powerful filters for categorical, numeric, and date columns  
    - Quick summary statistics and distributions  
    - One-click visualizations (histogram, bar, box, scatter)  
    - Correlation heatmap for numeric columns  
    - Pivot builder (drag-like via selectors)  
    - Time-series view if your data has date columns  
    """
)

# -----------------------------
# Schema + Profiling
# -----------------------------
with st.expander("📋 Dataset Snapshot", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Completeness %", f"{round(100*(1 - df.isna().sum().sum()/(df.shape[0]*df.shape[1]+1e-9)),2)}%")

    st.dataframe(df.head(50), use_container_width=True)

prof = basic_profile(df)
with st.expander("🧬 Column Profiling", expanded=False):
    st.dataframe(prof, use_container_width=True)

# -----------------------------
# Build Filters
# -----------------------------
st.subheader("🔎 Filter Your Data")
date_cols = infer_date_columns(df)
num_cols = infer_numeric_columns(df)
cat_cols = infer_categorical_columns(df)

with st.container():
    left, right = st.columns([2, 3])

    with left:
        st.markdown("*Categorical / Boolean Filters*")
        cat_filters = {}
        for c in cat_cols:
            # Skip date/numeric from categorical box if already detected
            if c in date_cols or c in num_cols:
                continue
            uniques = df[c].dropna().unique().tolist()
            if len(uniques) > 2000:
                continue  # avoid too heavy widgets
            chosen = st.multiselect(f"{c}", options=sorted(uniques, key=lambda x: str(x)), default=[])
            if chosen:
                cat_filters[c] = chosen

    with right:
        st.markdown("*Numeric & Date Filters*")

        # Numeric ranges
        num_ranges = {}
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() == 0:
                continue
            min_v = float(np.nanmin(s))
            max_v = float(np.nanmax(s))
            if min_v == max_v:
                continue
            r = st.slider(f"{c} range", min_value=min_v, max_value=max_v, value=(min_v, max_v))
            num_ranges[c] = r

        # Date ranges
        date_ranges = {}
        for c in date_cols:
            s = pd.to_datetime(df[c], errors="coerce")
            smin = s.min()
            smax = s.max()
            if pd.isna(smin) or pd.isna(smax) or smin == smax:
                continue
            dr = st.date_input(f"{c} range", value=(smin.date(), smax.date()))
            if isinstance(dr, tuple) and len(dr) == 2:
                start, end = dr
                date_ranges[c] = (pd.to_datetime(start), pd.to_datetime(end))

# Apply filters
df_filtered = df.copy()

# Apply cat filters
for c, vals in (cat_filters if 'cat_filters' in locals() else {}).items():
    df_filtered = df_filtered[df_filtered[c].isin(vals)]

# Apply numeric ranges
for c, (lo, hi) in (num_ranges if 'num_ranges' in locals() else {}).items():
    s = pd.to_numeric(df_filtered[c], errors="coerce")
    df_filtered = df_filtered[(s >= lo) & (s <= hi)]

# Apply date ranges
for c, (start, end) in (date_ranges if 'date_ranges' in locals() else {}).items():
    s = pd.to_datetime(df_filtered[c], errors="coerce")
    df_filtered = df_filtered[(s >= start) & (s <= end)]

st.success(f"Filters applied. Showing *{len(df_filtered):,} / {len(df):,}* rows.")
st.dataframe(df_filtered.head(1000), use_container_width=True, height=300)
download_link(df_filtered)

# -----------------------------
# Quick Insights
# -----------------------------
st.header("✨ Quick Insights")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Summary", "Distributions", "Correlation", "Scatter", "Box Plots", "Pivot"
])

with tab1:
    st.subheader("📌 Summary Statistics")
    # Numeric summary
    if num_cols:
        st.markdown("*Numeric Columns*")
        st.dataframe(df_filtered[num_cols].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns detected.")

    # Missing by column
    st.markdown("*Missing Values by Column*")
    miss = df_filtered.isna().mean().sort_values(ascending=False) * 100
    miss_df = pd.DataFrame({"Column": miss.index, "Missing %": miss.values})
    st.dataframe(miss_df, use_container_width=True)

with tab2:
    st.subheader("📈 Distributions")
    # Choose a column and auto-plot by type
    dist_col = st.selectbox("Select a column to visualize", options=df.columns)
    if dist_col:
        s = df_filtered[dist_col]
        if pd.api.types.is_numeric_dtype(s):
            fig = px.histogram(df_filtered, x=dist_col, nbins=40)
            st.plotly_chart(fig, use_container_width=True)
        elif pd.api.types.is_datetime64_any_dtype(s):
            # Plot counts by date
            temp = df_filtered.copy()
            temp["_date"] = pd.to_datetime(temp[dist_col], errors="coerce").dt.date
            counts = temp.groupby("_date").size().reset_index(name="Count")
            fig = px.line(counts, x="_date", y="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # categorical bar
            counts = s.value_counts(dropna=False).reset_index()
            counts.columns = [dist_col, "Count"]
            if len(counts) > 50:
                counts = counts.head(50)
            fig = px.bar(counts, x=dist_col, y="Count")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📉 Correlation Heatmap (numeric)")
    if len(num_cols) >= 2:
        corr = df_filtered[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for correlation.")

with tab4:
    st.subheader("🔍 Scatter Explorer (numeric vs numeric)")
    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            xcol = st.selectbox("X-axis", options=num_cols, index=0)
        with c2:
            ycol = st.selectbox("Y-axis", options=[c for c in num_cols if c != xcol], index=0)
        color_by = st.selectbox("Color by (optional)", options=["(none)"] + df.columns.tolist(), index=0)
        if xcol and ycol:
            if color_by == "(none)":
                fig = px.scatter(df_filtered, x=xcol, y=ycol, hover_data=df_filtered.columns)
            else:
                fig = px.scatter(df_filtered, x=xcol, y=ycol, color=df_filtered[color_by].astype(str), hover_data=df_filtered.columns)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns.")

with tab5:
    st.subheader("📦 Box Plots (numeric by category)")
    if num_cols:
        num_for_box = st.selectbox("Numeric column", options=num_cols)
        cat_for_box = st.selectbox("Group by (categorical/date/any)", options=["(none)"] + df.columns.tolist(), index=0)
        if cat_for_box == "(none)":
            fig = px.box(df_filtered, y=num_for_box, points="outliers")
        else:
            fig = px.box(df_filtered, x=df_filtered[cat_for_box].astype(str), y=num_for_box, points="outliers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns for box plots.")

with tab6:
    st.subheader("🧮 Pivot Builder")
    # Choose index, columns, values, aggfunc
    idx = st.multiselect("Index (rows)", options=df.columns)
    cols = st.multiselect("Columns (pivot across)", options=[c for c in df.columns if c not in idx])
    vals = st.multiselect("Values", options=[c for c in df.columns if c not in idx + cols])
    agg = st.selectbox("Aggregation", options=["count","sum","mean","median","min","max","nunique"], index=0)

    if idx and vals:
        agg_map = {v: agg for v in vals}
        try:
            pt = pd.pivot_table(
                df_filtered,
                index=idx,
                columns=cols if cols else None,
                values=vals,
                aggfunc=agg
            )
            # Flatten multiindex columns for display
            if isinstance(pt.columns, pd.MultiIndex):
                pt.columns = [" | ".join([str(x) for x in col]).strip() for col in pt.columns.values]
            st.dataframe(pt.reset_index(), use_container_width=True)
        except Exception as e:
            st.error(f"Pivot failed: {e}")
    else:
        st.info("Pick at least Index and Values to create a pivot.")

# -----------------------------
# Time Series (if any date cols)
# -----------------------------
if date_cols:
    st.header("⏱ Time Series Overview")
    ts_date_col = st.selectbox("Choose date/time column", options=date_cols)
    group_by_col = st.selectbox("Optional: group by", options=["(none)"] + df.columns.tolist(), index=0)
    y_choice = st.selectbox("Y metric", options=["Row Count"] + num_cols, index=0)

    tsdf = df_filtered.copy()
    tsdf["_date"] = pd.to_datetime(tsdf[ts_date_col], errors="coerce").dt.date
    base = tsdf.dropna(subset=["_date"]).groupby("_date")

    if y_choice == "Row Count":
        if group_by_col == "(none)":
            series = base.size().reset_index(name="Count")
            fig = px.line(series, x="_date", y="Count")
        else:
            series = tsdf.dropna(subset=[group_by_col]).groupby(["_date", group_by_col]).size().reset_index(name="Count")
            fig = px.line(series, x="_date", y="Count", color=group_by_col)
    else:
        # numeric aggregation
        if group_by_col == "(none)":
            series = tsdf.groupby("_date")[y_choice].mean(numeric_only=True).reset_index(name=f"{y_choice} (mean)")
            fig = px.line(series, x="_date", y=f"{y_choice} (mean)")
        else:
            series = tsdf.dropna(subset=[group_by_col]).groupby(["_date", group_by_col])[y_choice].mean(numeric_only=True).reset_index(name=f"{y_choice} (mean)")
            fig = px.line(series, x="_date", y=f"{y_choice} (mean)", color=group_by_col)

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footnote
# -----------------------------
st.caption("Built with ❤ in Streamlit. Works with any shareable Google Sheet link.")
