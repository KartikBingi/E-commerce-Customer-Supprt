import streamlit as st
import joblib

# Load your model and vectorizer
model = joblib.load('csat_xgboost_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# --- SIDEBAR ---
with st.sidebar:
    st.title("About the Project")
    st.info("This app uses an XGBoost model to predict Customer Satisfaction (CSAT) scores based on text remarks.")
    st.write("📊 **Model:** XGBoost Classifier")
    st.write("📖 **Feature Extraction:** TF-IDF")

# --- MAIN INTERFACE ---
st.title("Customer Satisfaction Prediction")

# Create a container for text input
user_input = st.text_area("Enter Customer Remarks:", key="remarks_input")

col1, col2 = st.columns([1, 5])

with col1:
    predict_btn = st.button("Predict CSAT")

with col2:
    # Adding a clear button (this simply re-runs the script and clears state)
    if st.button("Clear Text"):
        st.rerun()

if predict_btn:
    if user_input.strip():
        # Transform and Predict
        vectorized_input = tfidf.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        
        # Add +1 because labels were 0-4
        final_score = int(prediction) + 1
        
        st.success(f"Predicted CSAT Score: {final_score} Stars")
    else:
        st.warning("Please enter some remarks before predicting.")
