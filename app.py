import streamlit as st
import joblib

# 1. Load the files (Make sure these names match your downloads exactly)
model = joblib.load('csat_xgboost_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

st.title("Customer Satisfaction Prediction")
user_input = st.text_area("Enter Customer Remarks:")

if st.button("Predict CSAT"):
    if user_input:
        # 2. Transform the text using the loaded vectorizer
        vectorized_input = tfidf.transform([user_input])
        
        # 3. Predict
        prediction = model.predict(vectorized_input)[0]
        
        # 4. IMPORTANT: Add +1 because we shifted labels to 0-4 for training
        final_score = prediction + 1
        
        st.success(f"Predicted CSAT Score: {final_score} Stars")
    else:
        st.warning("Please enter some text first!")
