from flask import Flask, json, request
import tensorflow as tf

from utils import get_page_text, get_lstm_simple_prediction, get_nlp_prediction, get_lstm_prediction
from flask_cors import CORS

api = Flask(__name__)
CORS(api)


@api.route('/')
def root():
    return api.send_static_file('index.html')


@api.route('/lstm_simple', methods=['POST'])
def get_lstm_simple():
    data = json.loads(request.data)
    print(data)

    if data['text']:
        text = data['text']
    else:
        text = get_page_text(data['url'])

    print(text)
    response = get_lstm_simple_prediction(text)
    return json.dumps({'answer': response})


@api.route('/nlp', methods=['POST'])
def get_nlp():
    data = json.loads(request.data)
    print(data)

    if data['text']:
        text = data['text']
    else:
        text = get_page_text(data['url'])

    print(text)
    response = get_nlp_prediction(text)
    return json.dumps({'answer': response})


@api.route('/lstm', methods=['POST'])
def get_lstm():
    pageUrl = json.loads(request.data)
    print(pageUrl)
    text = get_page_text(pageUrl)
    print(text)
    # Fit and transform the training data.
    model = tf.keras.models.load_model('models/model-LSTM')
    # vectorizer = joblib.load('models/vectorizer.pkl')
    # X_test = vectorizer.transform([text])
    # Ypredict = model.predict(X_test)
    # print(Ypredict)
    # return json.dumps({'answer': bool(Ypredict[0])})
    return 1


@api.route('/detect', methods=['POST'])
def get_detect():
    data = json.loads(request.data)
    algorithms = {
        'lstm_simple': None,
        'lstm': None,
        'nlp': None
    }

    print(data)

    if data['text']:
        text = data['text']
    else:
        text = get_page_text(data['url'])

    print(text)

    if data['algorithms']['lstm_simple']:
        algorithms['lstm_simple'] = get_lstm_simple_prediction(text)
    if data['algorithms']['nlp']:
        algorithms['nlp'] = get_nlp_prediction(text)
    if data['algorithms']['lstm']:
        algorithms['lstm'] = get_lstm_prediction(text)

    print(algorithms)
    return json.dumps({'answer': algorithms})


if __name__ == '__main__':
    api.run()
