import streamlit as st
import pickle
import re

# Page Configuration & Header
st.title("AI Customer Feedback System")
st.write("This application predicts review sentiment and categorizes complaints into clusters.")

# Load Saved Assets
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# User Input Interface
user_review = st.text_area("Write customer review here:")

if st.button("Analyze Review"):
    if user_review.strip() != "":
        # Preprocessing
        text = user_review.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        # Transformation & Scaling
        vec = tfidf.transform([text])
        vec_scaled = scaler.transform(vec)

        # Sentiment Prediction
        prediction = model.predict(vec_scaled)[0]
        prediction_proba = model.predict_proba(vec_scaled)[0]

        if prediction == 1:
            st.success(f"Result: **Positive Review** 😀 (Confidence: {prediction_proba[1]*100:.2f}%) ")
        else:
            st.error(f"Result: **Negative Review / Complaint** 🙁 (Confidence: {prediction_proba[0]*100:.2f}%) ")
            cluster = kmeans.predict(vec_scaled)[0]
            st.write(f"Complaint Group / Cluster Number: **{cluster}**")
    else:
        st.write("Please enter a review first.")
