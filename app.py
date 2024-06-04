import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import sklearn

nltk.data.path.append(os.path.expanduser('~/nltk_data'))
# Initialize PorterStemmer
ps = PorterStemmer()


# Preprocessing function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# Load the trained model
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


# Streamlit app
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon=":bird:", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        background-color: #e8f0fe;
    }
    .stButton button {
        background-color: #1da1f2;
        color: white;
    }
    .stButton button:hover {
        background-color: #0d8ce2;
    }
    </style>
    """, unsafe_allow_html=True)
# Header and Description
st.title('🐦 Twitter Sentiment Analysis')
st.write("Welcome to the Twitter Sentiment Analysis app! Enter the tweet text below and click 'Analyze' to see the sentiment.")

# Text area for user input
user_input = st.text_area("Enter the tweet text:")

# Analyze button
if st.button('Analyze'):
    if user_input.strip():
        # Preprocess the input
        processed_input = stemming(user_input)

        # Transform input using the vectorizer
        vectorized_input = vectorizer.transform([processed_input])

        # Predict sentiment
        prediction = model.predict(vectorized_input)

        # Display result
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        st.markdown(f'**Sentiment:** {sentiment}')
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Developed by [Basanta Dharala](https://www.linkedin.com/in/basanta-dharala) | [GitHub](https://github.com/basantadharala)", unsafe_allow_html=True)