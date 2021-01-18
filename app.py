import streamlit as st
import pandas as pd
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model():
	return tf.keras.models.load_model('model.h5')

@st.cache
def load_tokenizer():
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer

def make_prediction(text):
	text_tok = tokenizer.texts_to_sequences([text])
	text_pad = pad_sequences(text_tok, maxlen=189) #189 is maxlen in training
	return model.predict(text_pad)[0][0]

model = load_model()
tokenizer = load_tokenizer()

# sidebar
st.sidebar.header("Parameters")
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5)

# Main Page
st.title('Spam Detection Using RNN')
st.write("""
Detail analysis can be found in notebook: [Link](https://github.com/hendraronaldi/tensorflow_projects/blob/main/TF2.0%20NLP%20Spam%20Detection%20RNN%20and%20CNN.ipynb)
""")
st.write("""
Dataset: [Link](https://lazyprogrammer.me/course_files/spam.csv)
""")

text = st.text_area('Text to analyze', "", height=5)

st.subheader('Prediction')
try:
	pred = make_prediction(text)
	if pred < threshold:
		st.write('Not a spam')
	else:
		st.write('Spam detected!!')
	st.write(f'Score: {pred}')
except:
	st.write('Please input text')