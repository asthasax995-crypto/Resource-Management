import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Industrial Workforce Analysis – India",
    layout="wide"
)

st.title("📊 Industrial Workforce Classification Dashboard (India)")

# -------------------------------------------------
# Load data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("merged_industrial_data_cleaned.csv")

df = load_data()

# -------------------------------------------------
# Sidebar Filters
# -------------------------------------------------
st.sidebar.header("🔍 Filter Options")

state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["india_states"].dropna().unique())
)

division = st.sidebar.selectbox(
    "Select Division",
    ["All"] + sorted(df["division"].dropna().unique())
)

industry = st.sidebar.selectbox(
    "Select Industry (NIC)",
    ["All"] + sorted(df["nic_name"].dropna().unique())
)

filtered_df = df.copy()

if state != "All":
    filtered_df = filtered_df[filtered_df["india_states"] == state]

if division != "All":
    filtered_df = filtered_df[filtered_df["division"] == division]

if industry != "All":
    filtered_df = filtered_df[filtered_df["nic_name"] == industry]

# -------------------------------------------------
# KPI Metrics
# -------------------------------------------------
st.subheader("📌 Workforce Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("👥 Total Workers", f"{int(filtered_df['workers'].sum()):,}")
col2.metric("♂ Male Workers", f"{int(filtered_df['male_workers'].sum()):,}")
col3.metric("♀ Female Workers", f"{int(filtered_df['female_workers'].sum()):,}")
col4.metric("🌾 Rural Workers", f"{int(filtered_df['rural_workers'].sum()):,}")
col5.metric("🏙 Urban Workers", f"{int(filtered_df['urban_workers'].sum()):,}")

# -------------------------------------------------
# Charts
# -------------------------------------------------
st.subheader("📈 Visual Analysis")

# Top industries by workforce
industry_df = (
    filtered_df.groupby("nic_name", as_index=False)["workers"]
    .sum()
    .sort_values("workers", ascending=False)
    .head(10)
)

fig1 = px.bar(
    industry_df,
    x="nic_name",
    y="workers",
    title="Top 10 Industries by Workforce",
    labels={"nic_name": "Industry", "workers": "Number of Workers"}
)
st.plotly_chart(fig1, use_container_width=True)

# Gender distribution
gender_df = pd.DataFrame({
    "Gender": ["Male", "Female"],
    "Workers": [
        filtered_df["male_workers"].sum(),
        filtered_df["female_workers"].sum()
    ]
})

fig2 = px.pie(
    gender_df,
    names="Gender",
    values="Workers",
    title="Gender-wise Workforce Distribution"
)
st.plotly_chart(fig2, use_container_width=True)

# Rural vs Urban distribution
area_df = pd.DataFrame({
    "Area": ["Rural", "Urban"],
    "Workers": [
        filtered_df["rural_workers"].sum(),
        filtered_df["urban_workers"].sum()
    ]
})

fig3 = px.bar(
    area_df,
    x="Area",
    y="Workers",
    title="Rural vs Urban Workforce Distribution"
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# Insights
# -------------------------------------------------
st.subheader("🧠 Key Insights")

st.markdown("""
• Industrial and service sectors account for the **largest workforce share**  
• **Male participation** is higher across most industries  
• **Rural workforce dominance** is visible in traditional sectors  
• Urban employment is concentrated in organized and service industries  
""")
