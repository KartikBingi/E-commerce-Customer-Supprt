import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('csat_xgboost_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

st.title("Customer Satisfaction Prediction")

# --- FIX: Initialize session state for the text input ---
if 'user_remarks' not in st.session_state:
    st.session_state.user_remarks = ""

def clear_text():
    st.session_state.user_remarks = ""

# --- UI Layout ---
# The 'value' is linked to session state so it can be cleared
user_input = st.text_area("Enter Customer Remarks:", value=st.session_state.user_remarks, key="remarks_input")

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Predict CSAT"):
        # Update session state with what's currently in the box
        st.session_state.user_remarks = user_input
        
        if st.session_state.user_remarks.strip():
            # Process prediction
            vectorized_input = tfidf.transform([st.session_state.user_remarks])
            prediction = model.predict(vectorized_input)[0]
            final_score = int(prediction) + 1
            st.success(f"Predicted CSAT Score: {final_score} Stars")
        else:
            st.warning("Please enter actual remarks first!")

with col2:
    # This button now calls the clear_text function
    if st.button("Clear Text", on_click=clear_text):
        st.rerun()
