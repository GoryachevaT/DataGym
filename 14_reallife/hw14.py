from flask import Flask
from flask import request, jsonify
import base64
from bs4 import BeautifulSoup
from pymystem3 import Mystem
import re
import nltk
import pickle
import numpy as np


nltk.download('punkt')
morph = Mystem()

def text_to_sent(t):
    text = BeautifulSoup(t).text.lower()
    tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    raw_sentences = tokenizer.tokenize(text.strip())
    raw_sentences = [x.split(';') for x in raw_sentences]
    raw_sentences = sum(raw_sentences, [])
    sentences = [morph.lemmatize(x) for x in raw_sentences]
    return [[y for y in x if re.match('[а-яёa-z0-9]', y)]
            for x in sentences]

def text_to_vec(words, model):
    index2word_set = set(model.wv.index2word)
    text_vec = np.zeros((300,), dtype="float32")
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words = n_words + 1
            text_vec = np.add(text_vec, model.wv[word])
    
    if n_words != 0:
        text_vec /= n_words
    return text_vec


wv_model = pickle.load(open('w2v_model.pickle', 'rb'))
rf = pickle.load(open('rf_model.pickle', 'rb'))


app = Flask(__name__)


@app.route("/predict",  methods=['POST'])
def predict():
    
    text = request.form.get('text')
    s = text_to_sent(text)
    s = sum(s, [])
    vec = text_to_vec(s, wv_model)

    resp = {
        'predict': rf.predict(vec.reshape((1, vec.shape[0])))[0],
    }
    
    return jsonify(resp)

app.run(host='0.0.0.0', port=11010)