import streamlit as st
import joblib
import time

# --- ADVANCED PAGE CONFIG ---
st.set_page_config(page_title="CSAT AI Engine", page_icon="📊", layout="wide")

# Load model and vectorizer
@st.cache_resource # Keeps model in memory for speed
def load_assets():
    model = joblib.load('csat_xgboost_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

model, tfidf = load_assets()

# --- CUSTOM CSS FOR ADVANCED LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextArea textarea { font-size: 1.1rem !important; }
    .score-card { 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center;
        background-color: #1e2130;
        border: 1px solid #3e445b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ CSAT Analytics")
    st.markdown("---")
    st.write("**Model:** XGBoost Classifier")
    st.write("**NLP:** TF-IDF Vectorization")
    st.write("**Dependencies:**", ["streamlit", "xgboost", "joblib"])
    st.info("This engine analyzes customer sentiment to provide actionable satisfaction scores.")

# --- MAIN CONTENT ---
st.title("🚀 Advanced CSAT Prediction Engine")
st.subheader("Analyze Customer Sentiment in Real-Time")

# Use columns for a cleaner layout
col_input, col_output = st.columns([2, 1])

with col_input:
    # Key-linked input for the Clear button functionality
    user_input = st.text_area(
        "Enter Customer Remarks:", 
        placeholder="Type here (e.g., 'The service was slow but the agent was nice' or 'I love this product!')",
        height=200,
        key="remarks_box"
    )
    
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        predict_btn = st.button("🔍 Run AI Prediction", use_container_width=True)
    with btn_col2:
        # Improved Clear functionality using st.rerun()
        if st.button("🗑️ Clear Dashboard", use_container_width=True):
            st.session_state.remarks_box = ""
            st.rerun()

with col_output:
    st.markdown("### Result Visualization")
    if predict_btn and user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.5) # Simulated latency for "Advanced" feel
            
            # Prediction logic
            vec = tfidf.transform([user_input])
            res = model.predict(vec)[0]
            final_score = int(res) + 1 # Shifting labels back
            
            # Dynamic output colors
            colors = {1: "#FF4B4B", 2: "#FFAA00", 3: "#FFEE00", 4: "#00FF00", 5: "#09AB3B"}
            color = colors.get(final_score, "#FFFFFF")
            
            st.markdown(f"""
                <div class="score-card">
                    <h2 style='color: {color};'>Predicted Score</h2>
                    <h1 style='color: {color}; font-size: 4rem;'>{final_score}</h1>
                    <p style='color: {color}; font-weight: bold;'>{"⭐" * final_score}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Success feedback
            if final_score >= 4:
                st.balloons()
    else:
        st.info("Awaiting input to generate prediction.")
