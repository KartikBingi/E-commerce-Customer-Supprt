import streamlit as st
import joblib
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CSAT AI Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ASSET LOADING (with Caching) ---
@st.cache_resource
def load_assets():
    # Ensure these files are in your GitHub repo
    model = joblib.load('csat_xgboost_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

try:
    model, tfidf = load_assets()
except Exception as e:
    st.error("⚠️ Model files not found. Please ensure .joblib files are uploaded to GitHub.")

# --- 3. CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextArea textarea { border: 1px solid #4a4a4a; border-radius: 10px; }
    .metric-card {
        background-color: #161b22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Admin Panel")
    st.markdown("---")
    st.write("**Model Engine:** XGBoost 3.2.0")
    st.write("**NLP Pipeline:** TF-IDF Vectorizer")
    st.info("This system uses Natural Language Processing to detect customer sentiment and predict CSAT scores.")

# --- 5. MAIN INTERFACE ---
st.title("🎯 AI-Powered CSAT Prediction")
st.markdown("Analyze customer remarks to generate instant satisfaction metrics.")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    # Use a Form for the "Standard" Clear/Submit behavior
    with st.form("input_form", clear_on_submit=True):
        user_input = st.text_area(
            "Customer Remarks",
            placeholder="Type customer feedback here (e.g., 'The delivery was late and the item is damaged')...",
            height=200
        )
        
        c1, c2 = st.columns([1, 1])
        with c1:
            submit_btn = st.form_submit_button("🚀 Run Analysis")
        with c2:
            # This button will clear the text area because clear_on_submit=True
            clear_btn = st.form_submit_button("🗑️ Clear Dashboard")

with col_right:
    st.markdown("### 📊 Prediction Result")
    
    if submit_btn:
        if user_input.strip():
            with st.spinner("Analyzing sentiment patterns..."):
                time.sleep(0.6) # Aesthetic delay for 'Advanced' feel
                
                # PREDICTION LOGIC
                processed_input = user_input.lower()
                vec = tfidf.transform([processed_input])
                raw_pred = model.predict(vec)[0]
                final_score = int(raw_pred) + 1
                
                # MANUAL ACCURACY CORRECTION
                # Catching negatives that basic TF-IDF models might miss (like "not ok")
                neg_words = ['not', 'bad', 'worst', 'broken', 'terrible', 'disappointed', 'late']
                if any(word in processed_input for word in neg_words) and final_score > 3:
                    final_score = 2

                # VISUAL OUTPUT
                colors
