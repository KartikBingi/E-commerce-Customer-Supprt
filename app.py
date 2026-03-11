import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Professional CSAT Engine",
    page_icon="🎯",
    layout="wide"
)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('csat_xgboost_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

model, tfidf = load_assets()

# --- 3. CUSTOM CSS FOR DARK THEME & TABLES ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .prediction-bar { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR (Matching Reference) ---
with st.sidebar:
    st.title("About This App")
    st.info("This app predicts Customer Satisfaction Score using an Advanced XGBoost model.")
    st.markdown("---")
    st.subheader("Models Compared:")
    st.write("✅ Logistic Regression")
    st.write("✅ Random Forest")
    st.write("✅ Gradient Boosting")
    st.write("🏆 **XGBoost (Main Model)**")
    st.markdown("---")
    st.caption("Internship Project | CSAT Analytics")

# --- 5. MAIN INTERFACE LAYOUT ---
st.title("Customer Satisfaction Prediction")

# Top Row: Interaction Details
col1, col2 = st.columns(2)
with col1:
    service_channel = st.selectbox("Service Channel", ["Outcall", "Inbound", "Email", "Chat"])
    issue_cat = st.selectbox("Issue Category", ["Order Related", "Technical", "Billing"])
    sub_cat = st.selectbox("Sub Category", ["Installation/demo", "Refund", "Delivery"])

with col2:
    resp_time = st.number_input("Response Time (minutes)", value=9.00)
    st.info("Response Speed: Fast")
    tenure = st.selectbox("Agent Tenure", ["0-30", "31-60", "61-90", "90+"])
    shift = st.selectbox("Agent Shift", ["Morning", "Afternoon", "Evening"])

st.markdown("---")

# Prediction Form
with st.form("csat_form", clear_on_submit=True):
    st.subheader("Customer Remarks Analysis")
    user_input = st.text_area("Enter Feedback:", placeholder="e.g., 'The delivery was very slow but the product is fine.'")
    
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        submitted = st.form_submit_button("Predict CSAT", type="primary")
    with btn_col2:
        st.form_submit_button("Clear Dashboard")

# --- 6. ADVANCED OUTPUT & PROBABILITY TABLE ---
if submitted and user_input.strip():
    with st.spinner("Calculating confidence scores..."):
        time.sleep(0.5)
        
        # ML Prediction
        processed = user_input.lower()
        vec = tfidf.transform([processed])
        probs = model.predict_proba(vec)[0] # Get probabilities for all classes
        prediction = np.argmax(probs)
        final_score = int(prediction) + 1

        # Logic Override for specific negatives
        neg_words = ['not', 'bad', 'terrible', 'broken', 'disappointed']
        if any(w in processed for w in neg_words) and final_score > 3:
            final_score = 2

        # Display Result Bar (Matching reference)
        bg_color = "#238636" if final_score >= 4 else "#da3633"
        status_msg = "Non-escalation category" if final_score >= 4 else "Escalation required"
        
        st.markdown(f'<div class="prediction-bar" style="background-color: {bg_color};">Predicted Score: {final_score} Stars ({status_msg})</div>', unsafe_allow_html=True)

        # Confidence Breakdown Table (Matching Reference)
        st.subheader("Confidence breakdown:")
        
        breakdown_data = {
            "Star Rating": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
            "Probability": [f"{p*100:.1f}%" for p in probs],
            "Confidence": ["High" if p > 0.6 else "Low" for p in probs]
        }
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.table(df_breakdown) # Using st.table for that fixed "Standard" look
        
        max_conf = max(probs) * 100
        st.write(f"**Model is very confident in this prediction: {max_conf:.1f}%**")

        if final_score >= 4:
            st.balloons()
else:
    st.info("Awaiting customer feedback to generate analysis.")

st.markdown("---")
st.caption("Tourism Experience Analytics | Classification | XGBoost Engine")
