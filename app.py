import streamlit as st
import pickle
nltk.download('punkt')
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]
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
    ps = PorterStemmer()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load the model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Classifier')

input_sms = st.text_area('Enter the text to classify', height=200)

if st.button('Classify'):
    mod_sms = transform_text(input_sms)
    vect_sms = tfidf.transform([mod_sms])
    result = model.predict(vect_sms)[0]
    print(result)
    if result == 1:
        st.write('This is a SPAM message')
    else:
        st.write('This is NOT A SPAM message')
