import streamlit as st
import pandas as pd
import pickle
import plotly.express as px  # For nicer charts

# ----------------------------
# Load data and models
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_industrial_data_cleaned.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df

@st.cache_resource
def load_model():
    with open("industry_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, le, vectorizer

df = load_data()
model, label_encoder, vectorizer = load_model()

# ----------------------------
# Page layout
# ----------------------------
st.set_page_config(page_title="Industrial Workforce Dashboard", layout="wide")
st.title("📊 Industrial Workforce Classification Dashboard")
st.markdown(
    "Analysis of Main and Marginal Workers across Industries in India"
)

# ----------------------------
# Key metrics
# ----------------------------
total_workers = df["main_workers_total_persons"].sum() + df["marginal_workers_total_persons"].sum()
total_males = df["main_workers_total_males"].sum() + df["marginal_workers_total_males"].sum()
total_females = df["main_workers_total_females"].sum() + df["marginal_workers_total_females"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Workers", f"{total_workers:,}")
col2.metric("Total Males", f"{total_males:,}")
col3.metric("Total Females", f"{total_females:,}")

st.markdown("---")

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")

# Multi-select States
states = sorted(df["india_states"].unique())
state_input = st.sidebar.multiselect("Select State(s)", options=states, default=states)

filtered_df = df[df["india_states"].isin(state_input)]

# Multi-select Divisions
divisions = sorted(filtered_df["division"].unique())
division_input = st.sidebar.multiselect("Select Division(s)", options=divisions, default=divisions)
filtered_df = filtered_df[filtered_df["division"].isin(division_input)]

# Multi-select NIC Names
nic_names = sorted(filtered_df["nic_name"].unique())
nic_input = st.sidebar.multiselect("Select Industry(s)", options=nic_names, default=nic_names)
filtered_df = filtered_df[filtered_df["nic_name"].isin(nic_input)]

# ----------------------------
# Filtered Data Table
# ----------------------------
st.subheader("Filtered Data")
st.dataframe(filtered_df, height=350)

# ----------------------------
# Industry Prediction
# ----------------------------
st.sidebar.header("Industry Classification Prediction")
industry_input = st.sidebar.text_input("Enter Industry Name")
if st.sidebar.button("Predict Category") and industry_input.strip() != "":
    industry_transformed = vectorizer.transform([industry_input])
    prediction_encoded = model.predict(industry_transformed)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    st.sidebar.success(f"Predicted Division: **{prediction_label}**")

# ----------------------------
# Charts
# ----------------------------
st.subheader("Worker Distribution by Gender")
gender_df = filtered_df[[
    "main_workers_total_males",
    "main_workers_total_females",
    "marginal_workers_total_males",
    "marginal_workers_total_females"
]].sum()
gender_df.index = ["Main Males", "Main Females", "Marginal Males", "Marginal Females"]

fig_gender = px.bar(
    x=gender_df.values,
    y=gender_df.index,
    orientation="h",
    text=gender_df.values,
    labels={'x': 'Number of Workers', 'y': 'Category'},
    color=gender_df.index,
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig_gender.update_layout(showlegend=False, height=350)
st.plotly_chart(fig_gender, use_container_width=True)

# ----------------------------
# Top 10 Industries by Total Workers
# ----------------------------
st.subheader("Top 10 Industries by Total Workers")
filtered_df["total_workers"] = filtered_df["main_workers_total_persons"] + filtered_df["marginal_workers_total_persons"]
top_industries = filtered_df.groupby("nic_name")["total_workers"].sum().sort_values(ascending=False).head(10)

fig_industry = px.bar(
    x=top_industries.values,
    y=top_industries.index,
    orientation="h",
    text=top_industries.values,
    labels={'x': 'Total Workers', 'y': 'Industry'},
    color=top_industries.values,
    color_continuous_scale='Viridis'
)
fig_industry.update_layout(showlegend=False, height=450)
st.plotly_chart(fig_industry, use_container_width=True)
