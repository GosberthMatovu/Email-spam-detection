import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model and vectorizer
model = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (optional)
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
    return text

# Modern CSS styling
css = """
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 8px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
    border: none;
}

.stTextArea textarea {
    border-radius: 8px;
    padding: 10px;
    font-size: 16px;
    border: 2px solid #ccc;
}

.stErrorMessage, .stSuccessMessage {
    font-size: 18px;
    padding: 10px;
    border-radius: 8px;
}

.stWarning {
    font-size: 18px;
    padding: 10px;
    border-radius: 8px;
    background-color: #f0ad4e;
    color: #fff;
}
</style>
"""

# Streamlit app with modern styling
st.title("EMAIL ✉️ SPAM DETECTION WITH CHART  VISUALIZATION")
st.write("Enter the email text below to check if it's spam or not.")
st.markdown(css, unsafe_allow_html=True)

# Text area for email input
email_input = st.text_area("Email Text", height=150)

# Prediction button and visualization
if st.button("Predict"):
    if email_input:
        email_processed = preprocess_text(email_input)
        email_vectorized = vectorizer.transform([email_processed])
        prediction = model.predict(email_vectorized)[0]
        
        # Display prediction result
        if prediction == 'spam':
            st.error("Warning: This email is SPAM.")
        else:
            st.success("This email is NOT SPAM.")
        
        # Count occurrences of spam and non-spam
        predictions = model.predict(vectorizer.transform([email_processed]))
        spam_count = sum(predictions == "spam")
        ham_count = sum(predictions == "ham")
        
        # Visualization: Bar chart showing distribution of spam and non-spam emails
        df = pd.DataFrame({'Category': ['Spam', 'Not Spam'], 'Count': [spam_count, ham_count]})
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Category', y='Count', data=df, palette='viridis')
        st.pyplot(plt)
    else:
        st.warning("Please enter some email text to classify.")
