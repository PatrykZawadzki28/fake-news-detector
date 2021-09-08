import pickle

import gensim
import joblib
import nltk
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup


def get_page_text(url):
    html_page = requests.get(url).content
    soup = BeautifulSoup(html_page, 'html.parser')
    # add page / article title concatenation with text
    whitelist = ['p', 'strong', 'em', 'b', 'u', 'i', 'h1', 'h2', 'h3']
    out = ""

    # for a in soup.find_all('article'):
    for t in soup.find_all(text=True):
        if t.parent.name in whitelist:
            out += '{} '.format(t)

    escape = ['\r', '\n', '\t', '\xa0']

    for e in escape:
        out = out.replace(e, '')

    return out


def get_lstm_simple_prediction(text):
    model = tf.keras.models.load_model('./models/fake-news.model')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=10000,
        lower=True
    )
    test_one_tokens = tokenizer.texts_to_sequences(text)
    test_one_x = tf.keras.preprocessing.sequence.pad_sequences(test_one_tokens, maxlen=128)
    predict_y = (model.predict(test_one_x) >= 0.5).astype("int")
    print(predict_y)
    return int(predict_y[0][0])  # check if this is the correct result


def get_nlp_prediction(text):
    model = joblib.load('models/fake-news-model.sav')
    vectorizer = joblib.load('models/vectorizer.pkl')
    test_x = vectorizer.transform([text])
    predict_y = model.predict(test_x)
    print(predict_y)
    return int(predict_y[0])  # check if this is the correct result

def get_lstm_prediction(text):
    X = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tmp = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=700)

    model = tf.keras.models.load_model('models/model-LSTM.model')
    y_pred = (model.predict(X) > 0.5).astype("int")
    return int(y_pred[0])