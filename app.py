import streamlit as st
import joblib

user_input = st.text_area("Enter Customer Remark:")

if st.button("Analyze Sentiment"):
    # Everything below this line MUST be indented (4 spaces or 1 tab)
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)
    st.write(f"Predicted CSAT: {prediction[0]}")
