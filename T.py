# app.py
# Streamlit dashboard for Child Rescue Data (Only Verification, Age, Gender, Location)

import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(
    page_title="Child Rescue Dashboard",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# LOAD DATA
# ----------------------
@st.cache_data
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/13svivZvyrpXZPhApZLcyr4kafCvMcFgC1jIT7Xu64L8/export?format=csv"
    df = pd.read_csv(sheet_url)

    # Keep only the needed columns
    required_cols = ["Verification", "Age", "Gender", "Location"]
    df = df[[c for c in required_cols if c in df.columns]]

    # Clean Age column if it exists
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    return df

df = load_data()

st.title("📊 Child Rescue Dashboard")
st.markdown("### Overview of Verification, Age, Gender, and Location")

# ----------------------
# METRICS
# ----------------------
col1, col2, col3, col4 = st.columns(4)

if "Verification" in df.columns:
    verified = df["Verification"].value_counts().get("Verified", 0)
    total = len(df)
    col1.metric("✅ Verified", verified)
    col2.metric("❌ Not Verified", total - verified)

if "Age" in df.columns:
    avg_age = round(df["Age"].mean(skipna=True), 1)
    col3.metric("📅 Avg Age", avg_age)

if "Gender" in df.columns:
    male_count = (df["Gender"].str.lower() == "male").sum()
    female_count = (df["Gender"].str.lower() == "female").sum()
    col4.metric("♂️ Males", male_count)

# ----------------------
# CHARTS
# ----------------------
st.markdown("## 📈 Visual Analysis")

# Verification Chart
if "Verification" in df.columns:
    fig = px.pie(df, names="Verification", title="Verification Status Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Age Distribution
if "Age" in df.columns:
    fig = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Gender Chart
if "Gender" in df.columns:
    fig = px.bar(df["Gender"].value_counts().reset_index(),
                 x="index", y="Gender",
                 title="Gender Distribution",
                 labels={"index": "Gender", "Gender": "Count"})
    st.plotly_chart(fig, use_container_width=True)

# Location Chart
if "Location" in df.columns:
    fig = px.bar(df["Location"].value_counts().reset_index(),
                 x="index", y="Location",
                 title="Location Distribution",
                 labels={"index": "Location", "Location": "Count"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# DATA TABLE
# ----------------------
st.markdown("## 📋 Data Preview")
st.dataframe(df)
