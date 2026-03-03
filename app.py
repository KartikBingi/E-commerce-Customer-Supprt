import streamlit as st
import joblib
import time

# --- 1. ADVANCED PAGE CONFIG ---
st.set_page_config(page_title="AI CSAT Engine", page_icon="📈", layout="wide")

# Load model and vectorizer with caching for speed
@st.cache_resource
def load_assets():
    model = joblib.load('csat_xgboost_model.joblib')
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    return model, tfidf

model, tfidf = load_assets()

# --- 2. FIX FOR THE CLEAR BUTTON (THE CALLBACK) ---
def reset_dashboard():
    # This resets the key BEFORE the widget is created
    st.session_state["remarks_box"] = ""

# --- 3. SIDEBAR DETAILS ---
with st.sidebar:
    st.title("📊 Project Analytics")
    st.markdown("---")
    st.write("**Model:** XGBoost Classifier")
    st.write("**NLP:** TF-IDF Vectorizer")
    st.info("This engine predicts Customer Satisfaction based on text sentiment.")

# --- 4. MAIN INTERFACE ---
st.title("🚀 Advanced CSAT Prediction Engine")

col_left, col_right = st.columns([2, 1])

with col_left:
    # Text Input - The 'key' matches the reset function above
    user_input = st.text_area(
        "Enter Customer Remarks:", 
        placeholder="e.g., 'The product quality is poor and delivery was late.'",
        height=200,
        key="remarks_box"
    )

    btn_1, btn_2 = st.columns(2)
    with btn_1:
        predict_clicked = st.button("🔍 Run AI Analysis", use_container_width=True, type="primary")
    with btn_2:
        # We use on_click to call our fix function
        st.button("🗑️ Clear Dashboard", on_click=reset_dashboard, use_container_width=True)

with col_right:
    st.subheader("AI Result")
    if predict_clicked and user_input.strip():
        with st.spinner("Processing Sentiment..."):
            time.sleep(0.4) # Aesthetic delay
            
            # Prediction Logic
            vec = tfidf.transform([user_input.lower()])
            res = model.predict(vec)[0]
            final_score = int(res) + 1
            
            # UI Visualization
            colors = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "💎"}
            rating_text = colors.get(final_score, "⚪")
            
            st.metric(label="Predicted CSAT", value=f"{final_score} / 5")
            st.write(f"Rating: {rating_text * final_score}")
            
            if final_score <= 2:
                st.error("Action Required: Low Satisfaction Detected")
            elif final_score >= 4:
                st.success("High Satisfaction: Positive Feedback")
                st.balloons()
    else:
        st.info("Enter text and click 'Run Analysis' to see results.")
