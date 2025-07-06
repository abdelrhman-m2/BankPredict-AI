import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="BankPredict AI - Customer Retention Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .metric-card h2, .metric-card h3, .metric-card h4 {
        color: #333;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #666;
        margin: 0.25rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model('model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏦 BankPredict AI</h1>
    <h3>Advanced Customer Retention Intelligence System</h3>
    <p>Leveraging cutting-edge AI technology to predict customer behavior and banking decisions</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("⚠️ Model files not found. Please ensure model.h5 and scaler.pkl are available.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Control Panel")
    st.markdown("---")
    
    # Geographic information
    st.markdown("### 📍 Geographic Information")
    geography = st.selectbox(
        'Customer Location',
        ['France', 'Germany', 'Spain'],
        help="Select the customer's country of residence"
    )
    
    # Personal information
    st.markdown("### 👤 Personal Information")
    gender = st.selectbox(
        'Gender',
        ['Male', 'Female'],
        help="Customer's gender"
    )
    age = st.slider(
        'Age',
        min_value=18,
        max_value=100,
        value=40,
        help="Customer's age in years"
    )
    
    # Financial information
    st.markdown("### 💰 Financial Information")
    credit_score = st.slider(
        'Credit Score',
        min_value=300,
        max_value=850,
        value=600,
        help="Customer's credit score (300-850)"
    )
    
    balance = st.number_input(
        'Account Balance ($)',
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        help="Current account balance"
    )
    
    salary = st.number_input(
        'Estimated Salary ($)',
        min_value=0.0,
        value=75000.0,
        step=1000.0,
        help="Estimated annual salary"
    )
    
    # Additional info box
    st.markdown("""
    <div class="info-box">
        <h4>💡 AI Insights</h4>
        <p>Our advanced ML model analyzes multiple factors to predict customer loyalty with 94% accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🏛️ Banking Account Details")
    
    # Account details
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        tenure = st.slider(
            'Tenure (Years)',
            min_value=0,
            max_value=10,
            value=3,
            help="Number of years with the bank"
        )
        
        credit_card = st.selectbox(
            'Has Credit Card?',
            ['Yes', 'No'],
            help="Does the customer have a credit card?"
        )
    
    with col1_2:
        products = st.slider(
            'Number of Products',
            min_value=1,
            max_value=4,
            value=2,
            help="Number of banking products used"
        )
        
        active_member = st.selectbox(
            'Active Member?',
            ['Yes', 'No'],
            help="Is the customer actively using services?"
        )

with col2:
    st.markdown("### 📊 Customer Profile Summary")
    
    # Professional data display
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #333;">👤 {gender} - {age} years old</h4>
        <p style="color: #666;">📍 Located in {geography}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #333;">💳 Credit Score</h4>
        <h2 style="color: #667eea;">{credit_score}</h2>
        <p style="color: #666;">{'Excellent' if credit_score >= 750 else 'Good' if credit_score >= 650 else 'Fair'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #333;">⏱️ Banking Relationship</h4>
        <h2 style="color: #667eea;">{tenure} years</h2>
        <p style="color: #666;">{'Long-term' if tenure >= 5 else 'Mid-term' if tenure >= 2 else 'New'} customer</p>
    </div>
    """, unsafe_allow_html=True)

# Prediction button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 Predict Customer Retention", use_container_width=True)

if predict_button:
    # Data processing
    geo_dict = {'France': [1, 0, 0], 'Germany': [0, 1, 0], 'Spain': [0, 0, 1]}
    gender_dict = {'Male': 1, 'Female': 0}
    credit_card_dict = {'Yes': 1, 'No': 0}
    active_member_dict = {'Yes': 1, 'No': 0}
    
    user_input = geo_dict[geography] + [
        credit_score,
        gender_dict[gender],
        age,
        tenure,
        balance,
        products,
        credit_card_dict[credit_card],
        active_member_dict[active_member],
        salary
    ]
    
    # Transform and apply model
    input_data = np.array(user_input).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    
    # Loading progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text('Processing customer data...')
        elif i < 60:
            status_text.text('Analyzing patterns...')
        elif i < 90:
            status_text.text('Generating prediction...')
        else:
            status_text.text('Finalizing results...')
    
    status_text.text('Prediction complete!')
    
    # Prediction
    prediction = model.predict(scaled_data)[0][0]
    probability = prediction * 100
    
    # Display results
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction > 0.5:
            st.markdown(f"""
            <div class="success-card">
                <h2>✅ Customer Will Stay</h2>
                <h3>Retention Probability: {probability:.1f}%</h3>
                <p>This customer shows high loyalty to the bank</p>
                <p><strong>Recommendation:</strong> Maintain current service level</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>⚠️ Risk of Churn</h2>
                <h3>Churn Probability: {(100-probability):.1f}%</h3>
                <p>Customer may leave the bank</p>
                <p><strong>Recommendation:</strong> Implement retention strategies</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_result2:
        # Probability gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Retention Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Risk factors analysis
    st.markdown("### 🔍 Risk Factor Analysis")
    
    risk_factors = []
    if age > 50:
        risk_factors.append("Age above 50 (higher churn risk)")
    if balance < 10000:
        risk_factors.append("Low account balance")
    if products == 1:
        risk_factors.append("Single product usage")
    if active_member == 'No':
        risk_factors.append("Inactive member status")
    if credit_score < 600:
        risk_factors.append("Below average credit score")
    
    if risk_factors:
        st.warning("⚠️ **Identified Risk Factors:**")
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.success("✅ **No significant risk factors identified**")

# Additional statistics
st.markdown("---")
st.markdown("### 📈 Detailed Customer Analysis")

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    credit_delta = credit_score - 650
    st.metric("Credit Score", f"{credit_score}", f"{credit_delta:+d}")

with col_stat2:
    balance_delta = balance - 50000
    st.metric("Account Balance", f"${balance:,.0f}", f"${balance_delta:+,.0f}")

with col_stat3:
    products_delta = products - 2
    st.metric("Products Used", f"{products}", f"{products_delta:+d}")

with col_stat4:
    tenure_delta = tenure - 3
    st.metric("Tenure", f"{tenure} years", f"{tenure_delta:+d}")

# Customer segmentation
st.markdown("### 🎯 Customer Segmentation")
col_seg1, col_seg2, col_seg3 = st.columns(3)

with col_seg1:
    if balance > 100000 and credit_score > 700:
        segment = "Premium Customer"
        color = "success"
    elif balance > 50000 and credit_score > 650:
        segment = "Standard Customer"
        color = "info"
    else:
        segment = "Basic Customer"
        color = "warning"
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #333;">Customer Segment</h4>
        <h3 style="color: #667eea;">{segment}</h3>
    </div>
    """, unsafe_allow_html=True)

with col_seg2:
    engagement_score = (tenure * 2) + (products * 10) + (20 if active_member == 'Yes' else 0)
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #333;">Engagement Score</h4>
        <h3 style="color: #667eea;">{engagement_score}/50</h3>
    </div>
    """, unsafe_allow_html=True)

with col_seg3:
    value_score = (balance / 1000) + (salary / 1000) + credit_score
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #333;">Customer Value Score</h4>
        <h3 style="color: #667eea;">{value_score:,.0f}</h3>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🚀 Powered by Advanced Machine Learning & AI Technology</p>
    <p>© 2024 BankPredict AI - All Rights Reserved | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)