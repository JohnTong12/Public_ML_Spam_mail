# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:38:13 2024

@author: JT
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:30:19 2024

@author: JT
"""

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
  """
  Preprocesses the message text for better spam detection:

  - Removes non-alphanumeric characters
  - Removes URLs and HTML tags
  - Removes digits
  - Removes punctuation
  - Removes newlines
  - Removes words containing alphanumeric and digits (e.g., "ph0n3")
  - Converts to lowercase
  - Performs stemming using PorterStemmer
  - Removes stop words

  Args:
      message (str): The message text to be preprocessed.

  Returns:
      str: The preprocessed message text.
  """

  stemmed_content = re.sub(r'[^\w\s]', '', message)  # Remove non-alphanumeric characters
  stemmed_content = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>+', '', stemmed_content)  # Remove URLs, HTML tags, and digits
  stemmed_content = re.sub(r'\w*\d\w*', '', stemmed_content)
  stemmed_content = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', stemmed_content)  # Remove punctuation
  stemmed_content = stemmed_content.lower().split()
  stemmed_content = [porter_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
  stemmed_content = " ".join(stemmed_content)  # Join stemmed words

  return stemmed_content

# Loading the model and TF-IDF vectorizer (assuming they exist in your file structure)
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
