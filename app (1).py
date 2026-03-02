import streamlit as st
import joblib

st.title("🌟 CSAT Prediction Dashboard")
user_input = st.text_area("Enter Customer Remark:")

if st.button("Analyze Sentiment"):
    model = joblib.load('csat_xgboost_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    vec = vectorizer.transform([user_input])
    res = model.predict(vec)
    
    st.metric("Predicted Score", f"{res[0]} Stars")
    if res[0] <= 2:
        st.error(" High Risk: Dissatisfied Customer")
    else:
        st.success(" Positive Sentiment Detected")