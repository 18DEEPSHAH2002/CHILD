import os
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
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"  # Disable file watcher issue

st.set_page_config(
    page_title="Child Rescue Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# DATA LOADING
# ----------------------
SHEET_ID = "13svivZvyrpXZPhApZLcyr4kafCvMcFgC1jIT7Xu64L8"
SHEET_NAME = "Sheet1"
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data(URL)

if df.empty:
    st.warning("No data found in Google Sheet. Please check the link.")
    st.stop()

# ----------------------
# DASHBOARD
# ----------------------
st.title("📊 Child Rescue Details Dashboard")

st.markdown("This dashboard provides a quick analysis of the uploaded Google Sheet data.")

# Show raw data
with st.expander("🔎 View Raw Data"):
    st.dataframe(df, use_container_width=True)

# ----------------------
# BASIC METRICS
# ----------------------
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Last Updated", time.strftime("%Y-%m-%d %H:%M:%S"))

# ----------------------
# COLUMN-WISE ANALYSIS
# ----------------------
st.subheader("📊 Column-wise Distribution")

for col in df.columns:
    if df[col].dtype == "object":
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "Count"]
        fig = px.bar(vc, x=col, y="Count", title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------
# DOWNLOAD OPTION
# ----------------------
st.subheader("⬇️ Download Processed Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "child_rescue_data.csv", "text/csv")
