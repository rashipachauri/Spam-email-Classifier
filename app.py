import streamlit as st
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# === STEP 1: Train & save model if not found ===
if not (os.path.exists('vectorizer.pkl') and os.path.exists('model.pkl')):
    st.warning("Training model...")

    # ✅ Load dataset (make sure spam.csv is present)
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['transformed_text'] = df['text'].apply(transform_text)

    # Vectorize and train
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['transformed_text'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    # Save to pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    st.success("Model trained and saved.")
else:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

# === STEP 2: Streamlit Interface ===
st.title("📧 Email / SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Output
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")
