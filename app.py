import numpy as np
import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string

porter_stem = PorterStemmer()

def stemming(message):
  stemmed_content = re.sub(r'[^\w\s]', '', message) 
  stemmed_content = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>+', '', stemmed_content)  
  stemmed_content = re.sub(r'\w*\d\w*', '', stemmed_content)
  stemmed_content = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', stemmed_content)  
  stemmed_content = stemmed_content.lower().split()
  stemmed_content = [porter_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
  stemmed_content = " ".join(stemmed_content)  

  return stemmed_content

loaded_model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('feature_extraction.pkl','rb'))


st.title('Spam Mail Predictor')

mail_text = st.text_area('Enter message')

if st.button('Predict'):
  transformed_text = stemming(mail_text)

  tfid_vectorized = tfidf.transform([transformed_text])

  result = loaded_model.predict(tfid_vectorized)[0]

  if result == 1:
    st.header('This is a Spam Mail')
  else:
    st.header('This is a Ham Mail (Not Spam)')  # Improved wording for clarity
