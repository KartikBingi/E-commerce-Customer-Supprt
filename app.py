import streamlit as st
import joblib
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CSAT AI Analytics",
    page_icon="🎯",
    layout="wide"
)

# --- 2. GLOBAL CONSTANTS (Fixes NameError) ---
COLORS = {
    1: "#FF4B4B", # Red
    2: "#FFAA00", # Orange
    3: "#FFEE00", # Yellow
    4: "#00FF00", # Light Green
    5: "#09AB3B"  # Dark Green
}

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('csat_xgboost_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

model, tfidf = load_assets()

# --- 4. CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #161b22;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        margin-top: 20px;
    }
    .stTextArea textarea { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
st.title("🎯 AI-Powered CSAT Prediction")
st.write("Professional Sentiment Analysis for E-Commerce Feedback")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    # Use a Form to handle the "Clear" button properly without errors
    with st.form("input_form", clear_on_submit=True):
        user_input = st.text_area(
            "Customer Remarks",
            placeholder="Type feedback here (e.g., 'The delivery was not ok')...",
            height=200
        )
        
        c1, c2 = st.columns(2)
        with c1:
            submit_btn = st.form_submit_button("🚀 Run Analysis", use_container_width=True)
        with c2:
            # clear_on_submit=True handles the text clearing automatically
            st.form_submit_button("🗑️ Clear Dashboard", use_container_width=True)

with col_right:
    st.markdown("### 📊 Prediction Result")
    
    if submit_btn and user_input.strip():
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            
            # Prediction Logic
            processed_text = user_input.lower()
            vec = tfidf.transform([processed_text])
            prediction = model.predict(vec)[0]
            final_score = int(prediction) + 1
            
            # --- THE "NOT OK" FIX (Manual override for biased models) ---
            neg_keywords = ['not', 'bad', 'worst', 'broken', 'terrible', 'late', 'rude']
            if any(word in processed_text for word in neg_keywords) and final_score > 3:
                final_score = 2 

            # Display Output
            current_color = COLORS.get(final_score, "#FFFFFF")
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #8b949e;">PREDICTED CSAT</h4>
                    <h1 style="color: {current_color}; font-size: 72px; margin: 10px 0;">{final_score} / 5</h1>
                    <p style="font-size: 24px;">{"⭐" * final_score}</p>
                </div>
            """, unsafe_allow_html=True)

            if final_score >= 4:
                st.success("Positive Sentiment Detected")
                st.balloons()
            elif final_score == 3:
                st.warning("Neutral Sentiment Detected")
            else:
                st.error("Action Required: Negative Sentiment")
    else:
        st.info("Enter remarks and click 'Run Analysis' to see the predicted star rating.")
