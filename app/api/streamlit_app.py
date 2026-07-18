import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .segment-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000/api/v1"

# Authentication
def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        # Simple authentication - replace with proper auth
        if username == "admin" and password == "admin123":
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.authenticated:
    login()
    st.stop()

# Main content
st.sidebar.title("🎯 Customer Segmentation")
st.sidebar.markdown("---")

# Navigation
navigation = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "👤 Customer Lookup", "📈 Segment Analysis", "🎯 Marketing Campaigns", "📊 Segmentation Explorer"]
)

# API Functions
def get_customer_profile(customer_id):
    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/customer/{customer_id}"
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_segment_analysis(segment_name):
    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/segment-analysis/{segment_name}"
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_recommendations(customer_id, n=10):
    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/recommendations/{customer_id}?n_recommendations={n}"
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Dashboard
if navigation == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Customers", "4,339", "+12%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Customer Value", "£1,847", "+8.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Retention Rate", "72%", "+3.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Customers", "2,847", "+5.1%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Segment Distribution
    st.subheader("📊 Segment Distribution")
    
    # Sample data - replace with actual API data
    segment_data = {
        'Segment': ['Champions', 'Loyal', 'Potential', 'At Risk', 'Needs Attention', 'Dormant'],
        'Count': [450, 820, 650, 380, 1200, 839],
        'Revenue': [450000, 620000, 350000, 180000, 420000, 120000],
        'Avg_Value': [1000, 756, 538, 474, 350, 143]
    }
    df_segments = pd.DataFrame(segment_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            df_segments,
            values='Count',
            names='Segment',
            title='Customer Distribution by Segment',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_segments,
            x='Segment',
            y='Count',
            color='Segment',
            title='Segment Size Comparison',
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue Analysis
    st.subheader("💰 Revenue by Segment")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=df_segments['Segment'], y=df_segments['Revenue'], name='Total Revenue'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df_segments['Segment'], y=df_segments['Avg_Value'], 
                  name='Avg. Revenue per Customer', mode='lines+markers'),
        secondary_y=True,
    )
    
    fig.update_layout(title_text="Revenue Analysis by Segment")
    fig.update_xaxes(title_text="Segment")
    fig.update_yaxes(title_text="Total Revenue (£)", secondary_y=False)
    fig.update_yaxes(title_text="Avg. Revenue per Customer (£)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# Customer Lookup
elif navigation == "👤 Customer Lookup":
    st.markdown('<div class="main-header">👤 Customer Lookup</div>', unsafe_allow_html=True)
    
    customer_id = st.text_input("Enter Customer ID", value="12345")
    
    if st.button("Lookup Customer"):
        with st.spinner("Fetching customer data..."):
            profile = get_customer_profile(int(customer_id))
            
            if profile:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.info(f"Customer: {profile['customer_id']}")
                    
                    # Segment badge
                    segment_colors = {
                        'Champions': '#2E86AB',
                        'Loyal': '#A23B72',
                        'Potential': '#F18F01',
                        'At Risk': '#C73E1D',
                        'Needs Attention': '#6A4C93',
                        'Dormant': '#3D3D3D'
                    }
                    
                    segment = profile.get('segment', 'Unknown')
                    color = segment_colors.get(segment, '#6c757d')
                    st.markdown(
                        f'<div class="segment-badge" style="background-color:{color}; color:white;">'
                        f'<span style="font-size:1.2rem;">{segment}</span></div>',
                        unsafe_allow_html=True
                    )
                    
                    st.metric("Lifetime Value", f"£{profile.get('lifetime_value', 0):,.2f}")
                    st.metric("Engagement Score", f"{profile.get('engagement_score', 0):.2f}")
                    st.metric("Churn Risk", f"{profile.get('churn_risk', 0):.1%}")
                
                with col2:
                    st.subheader("RFM Scores")
                    rfm = profile.get('rfm_scores', {})
                    
                    col_r, col_f, col_m = st.columns(3)
                    with col_r:
                        st.metric("Recency", f"{rfm.get('recency', 0)} days")
                    with col_f:
                        st.metric("Frequency", f"{rfm.get('frequency', 0)} orders")
                    with col_m:
                        st.metric("Monetary", f"£{rfm.get('monetary', 0):,.2f}")
                    
                    # Radar chart for RFM
                    categories = ['Recency', 'Frequency', 'Monetary']
                    values = [
                        max(0, 5 - rfm.get('recency_score', 0)),
                        rfm.get('frequency_score', 0),
                        rfm.get('monetary_score', 0)
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        marker=dict(color='#1f77b4')
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 5]
                            )
                        ),
                        title="RFM Radar Chart"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                with st.expander("🎯 Recommendations"):
                    recs = get_recommendations(int(customer_id), 10)
                    if recs:
                        for item in recs.get('recommendations', []):
                            st.write(f"• {item}")
            else:
                st.error("Customer not found")

# Segment Analysis
elif navigation == "📈 Segment Analysis":
    st.markdown('<div class="main-header">📈 Segment Analysis</div>', unsafe_allow_html=True)
    
    segments = ['Champions', 'Loyal', 'Potential', 'At Risk', 'Needs Attention', 'Dormant']
    selected_segment = st.selectbox("Select Segment", segments)
    
    if selected_segment:
        with st.spinner("Analyzing segment..."):
            analysis = get_segment_analysis(selected_segment)
            
            if analysis:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Segment Size", f"{analysis.get('size', 0):,} customers")
                    st.metric("Percentage", f"{analysis.get('percentage', 0):.1f}%")
                    
                    st.subheader("Key Characteristics")
                    characteristics = analysis.get('characteristics', {})
                    for key, value in characteristics.items():
                        st.metric(key.replace('_', ' ').title(), value)
                
                with col2:
                    st.subheader("Segment Insights")
                    insights = analysis.get('insights', [])
                    for insight in insights:
                        st.info(f"💡 {insight}")
                    
                    st.subheader("Recommendations")
                    recommendations = analysis.get('recommendations', [])
                    for rec in recommendations:
                        st.success(f"✅ {rec}")
            else:
                st.warning("Segment analysis not available")

# Marketing Campaigns
elif navigation == "🎯 Marketing Campaigns":
    st.markdown('<div class="main-header">🎯 Marketing Campaigns</div>', unsafe_allow_html=True)
    
    # Campaign creation
    with st.expander("Create New Campaign"):
        st.subheader("New Marketing Campaign")
        
        campaign_name = st.text_input("Campaign Name")
        target_segments = st.multiselect(
            "Target Segments",
            ['Champions', 'Loyal', 'Potential', 'At Risk', 'Needs Attention', 'Dormant']
        )
        campaign_type = st.selectbox(
            "Campaign Type",
            ["Email", "SMS", "Push Notification", "Social Media", "Direct Mail"]
        )
        budget = st.number_input("Budget (£)", min_value=0, value=1000)
        
        if st.button("Create Campaign"):
            st.success(f"Campaign '{campaign_name}' created successfully!")
            st.info(f"Target segments: {', '.join(target_segments)}")
            st.info(f"Budget: £{budget:,}")
    
    # Campaign analytics
    st.subheader("📊 Campaign Performance")
    
    # Sample data
    campaign_data = {
        'Campaign': ['Summer Sale', 'Loyalty Rewards', 'Win-back', 'New Product', 'Holiday'],
        'Segment': ['All', 'Loyal', 'At Risk', 'Potential', 'Champions'],
        'Open Rate': [24.5, 32.1, 18.7, 22.3, 28.9],
        'Click Rate': [8.2, 12.3, 5.1, 7.8, 10.5],
        'Conversion': [3.1, 5.2, 1.8, 2.9, 4.8],
        'Revenue': [45000, 28000, 12000, 35000, 52000]
    }
    df_campaigns = pd.DataFrame(campaign_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Open Rate by Campaign', 'Conversion Rate by Campaign',
                       'Revenue by Campaign', 'Performance by Segment')
    )
    
    fig.add_trace(
        go.Bar(x=df_campaigns['Campaign'], y=df_campaigns['Open Rate'], 
               name='Open Rate', marker_color='#1f77b4'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df_campaigns['Campaign'], y=df_campaigns['Conversion'], 
               name='Conversion', marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=df_campaigns['Campaign'], y=df_campaigns['Revenue'], 
               name='Revenue', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    segment_performance = df_campaigns.groupby('Segment')[['Open Rate', 'Click Rate', 'Conversion']].mean().reset_index()
    fig.add_trace(
        go.Bar(x=segment_performance['Segment'], y=segment_performance['Open Rate'],
               name='Avg Open Rate', marker_color='#d62728'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Segmentation Explorer
elif navigation == "📊 Segmentation Explorer":
    st.markdown('<div class="main-header">📊 Segmentation Explorer</div>', unsafe_allow_html=True)
    
    # Upload data
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} customers with {len(df.columns)} features")
        
        # Preview
        with st.expander("Data Preview"):
            st.dataframe(df.head())
        
        # Run segmentation
        if st.button("Run Segmentation"):
            with st.spinner("Segmenting customers..."):
                # Convert to JSON for API
                customers = df.to_dict('records')
                
                try:
                    response = requests.post(
                        f"{st.session_state.api_base_url}/segment",
                        json=customers
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        st.success(f"Segmented {results['total_customers']} customers")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Segment Distribution")
                            dist_data = pd.DataFrame(
                                list(results['segment_distribution'].items()),
                                columns=['Segment', 'Count']
                            )
                            
                            fig = px.pie(
                                dist_data,
                                values='Count',
                                names='Segment',
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Metrics")
                            st.metric("Total Customers", results['total_customers'])
                            st.metric("Unique Segments", len(results['segment_distribution']))
                    else:
                        st.error("Segmentation failed")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")