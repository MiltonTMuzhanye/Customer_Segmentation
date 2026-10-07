from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
)


@st.cache_data
def load_segments():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "batch_segment_predictions.csv"
    )
    return pd.read_csv(path)


@st.cache_data
def load_profiles():
    path = (
        PROJECT_ROOT
        / "reports"
        / "segment_reports"
        / "segment_profiles.csv"
    )
    return pd.read_csv(path)


st.title("Customer Segmentation Dashboard")
st.caption("K-Means customer segmentation and marketing intelligence")

try:
    segments = load_segments()
    profiles = load_profiles()
except Exception as exc:
    st.error(f"Failed to load segmentation data: {exc}")
    st.stop()


st.header("Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{len(segments):,}",
)

col2.metric(
    "Segments",
    f"{segments['Cluster'].nunique()}",
)

champions = (
    segments["Segment"]
    .eq("Champions / VIP")
    .sum()
)

col3.metric(
    "Champions / VIP",
    f"{champions:,}",
)

at_risk = (
    segments["Segment"]
    .eq("At Risk / Needs Attention")
    .sum()
)

col4.metric(
    "At Risk",
    f"{at_risk:,}",
)


st.header("Segment Distribution")

distribution = (
    segments["Segment"]
    .value_counts()
    .rename_axis("Segment")
    .reset_index(name="Customers")
)

distribution["Percentage"] = (
    distribution["Customers"]
    / len(segments)
    * 100
)

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(
        distribution.set_index("Segment")["Customers"]
    )

with col2:
    st.dataframe(
        distribution,
        width="stretch",
        hide_index=True,
    )

st.header("Segment Profiles")

display_profiles = profiles[
    [
        "Cluster",
        "Segment",
        "Customers",
        "Customer_Percentage",
        "Recency",
        "Frequency",
        "Monetary",
        "Churn_Risk",
        "Engagement_Score",
        "Recommended_Action",
    ]
].copy()

st.dataframe(
    display_profiles,
    width="stretch",
    hide_index=True,
)


st.header("Customer Lookup")

customer_id = st.number_input(
    "Enter Customer ID",
    min_value=1,
    step=1,
    value=12347,
)

customer = segments[
    segments["CustomerID"] == customer_id
]

if customer.empty:
    st.warning("Customer ID not found.")
else:
    row = customer.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cluster",
        int(row["Cluster"]),
    )

    col2.metric(
        "Segment",
        row["Segment"],
    )

    col3.metric(
        "Customer ID",
        int(row["CustomerID"]),
    )

    st.info(
        f"Recommended Action: {row['Recommended_Action']}"
    )

with st.expander("View all customer predictions"):
    st.dataframe(
        segments,
        width="stretch",
        hide_index=True,
    )
