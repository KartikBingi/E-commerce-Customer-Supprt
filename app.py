import streamlit as st
import joblib
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="CSAT Analytics Dashboard", layout="wide")

# --- 2. ASSET LOADING (Ensure these match your file names in GitHub) ---
@st.cache_resource
def load_assets():
    model = joblib.load('csat_xgboost_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

model, tfidf = load_assets()

# --- 3. SIDEBAR (Matching your reference image) ---
with st.sidebar:
    st.title("About This App")
    st.write("This app predicts Customer Satisfaction Score using a Deep Learning ANN model.") # Note: You are using XGBoost, but text matches your image
    st.markdown("---")
    st.subheader("Models trained and compared:")
    st.write("1. Logistic Regression")
    st.write("2. Random Forest")
    st.write("3. Gradient Boosting")
    st.write("4. **XGBoost (main model)**")
    st.markdown("---")
    st.caption("CSAT | Internship Project")

# --- 4. MAIN INTERFACE (Multi-column layout from your photo) ---
st.title("Customer Satisfaction Prediction")

# Row 1: Primary Dropdowns
col1, col2 = st.columns(2)
with col1:
    service_channel = st.selectbox("Service Channel", ["Outcall", "Inbound", "Email", "Chat"])
    issue_category = st.selectbox("Issue Category", ["Order Related", "Technical", "Billing"])
    sub_category = st.selectbox("Sub Category", ["Installation/demo", "Refund", "Delivery"])

with col2:
    response_time = st.number_input("Response Time (minutes)", value=9.00)
    st.info("Response Speed: Fast")
    agent_tenure = st.selectbox("Agent Tenure", ["0-30", "31-60", "61-90", "90+"])
    agent_shift = st.selectbox("Agent Shift", ["Morning", "Afternoon", "Evening"])

# Row 2: Staff
col3, col4 = st.columns(2)
with col3:
    supervisor = st.selectbox("Supervisor", ["Austin Johnson", "Sarah Miller"])
with col4:
    manager = st.selectbox("Manager", ["Olivia Tan", "Robert Chen"])

st.markdown("---")

# Row 3: Text Input and Buttons
st.subheader("Analysis Section")
user_input = st.text_area("Enter Customer Remarks:", placeholder="Type here...", key="input_area")

c1, c2 = st.columns([1, 4])
with c1:
    predict_btn = st.button("Predict CSAT", type="primary", use_container_width=True)
with c2:
    if st.button("Clear Text", use_container_width=True):
        st.rerun()

# --- 5. PREDICTION OUTPUT ---
if predict_btn:
    if user_input.strip():
        # Text Processing
        processed_text = user_input.lower()
        vec = tfidf.transform([processed_text])
        prediction = model.predict(vec)[0]
        final_score = int(prediction) + 1
        
        # UI
