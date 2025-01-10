import streamlit as st
import pickle
import string
import numpy
import pandas

import nltk
nltk.download('punkt')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in y]

    return " ".join(y)

tridf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Classifier")

# Input field for user to type email text
input_sms = st.text_area("Enter the email/SMS")
if st.button('Predict'):
# Preprocess the input email
    transformed_sms = transform_text(input_sms)

# Vectorize the transformed text
    vector_input = tridf.transform([transformed_sms])

# Predict the label (spam or not spam)
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
