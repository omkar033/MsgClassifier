import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import joblib

nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model
model = joblib.load(r'C:\Users\omkar\Downloads\Spit_hacks-main\Spit_hacks-main\sms-spam-classifier\spam_detectmodel.pkl')

# Initialize the Porter Stemmer
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    tokens = []
    for token in text:
        if token.isalnum():  # Remove special characters
            tokens.append(token)
    text = tokens[:]
    tokens.clear()
    for token in text:
        if token not in stopwords.words('english') and token not in string.punctuation:  # Remove stopwords and punctuation
            tokens.append(token)
    text = tokens[:]
    tokens.clear()
    for token in text:
        tokens.append(ps.stem(token))  # Stem the words
    return " ".join(tokens)  # Return preprocessed text as a string

from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TfidfVectorizer

def classify_message(message):
    # Load the TfidfVectorizer
    tfidf_vectorizer = joblib.load(r'C:\Users\omkar\Downloads\Spit_hacks-main\Spit_hacks-main\sms-spam-classifier\vectorizer.pkl')
    
    # Preprocess Input
    preprocessed_input = preprocess_text(message)
    
    # Transform preprocessed input using the loaded TfidfVectorizer
    input_features = tfidf_vectorizer.transform([preprocessed_input])
    
    # Use the Trained Model to classify the message
    prediction = model.predict(input_features)[0]
    return prediction


# Streamlit UI
st.title('SMS Spam Classifier')

user_input = st.text_area("Enter your message:", height=200) 
if st.button('Classify'):
    prediction = classify_message(user_input)
    if prediction == 0:
        st.write("The message is classified as 'Safe message.")
    else:
        st.write("The message is classified as 'potential Scam'.")

