# Data analysis and manipulation tool
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix

# Constants

EPOCHS = 10  # controls the number of complete passes through dataset
BATCH_SIZE = 128  # controls the number of training samples to work through

MAX_LEN = 128
NUM_WORDS = 10000

NUM_OF_TEST_SAMPLES = 1000

EMBEDDING_DIM = 64
H1 = 24

THRESH = 0.5

# Prepare data to use
train_data = pd.read_csv('./databases/fake-news/train.csv')
test_data = pd.read_csv('./databases/fake-news/test.csv')

train_data.loc[train_data["text"].isnull(), "text"] = ""
test_data.loc[test_data["text"].isnull(), "text"] = ""

print(train_data.shape, test_data.shape, train_data.head())

train_data.head()

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=10000,
    lower=True
)

# create mapping
tokenizer.fit_on_texts(train_data["text"].values)

tokenizer.word_index

# create tokens 
train_tokens = tokenizer.texts_to_sequences(train_data["text"])
test_tokens = tokenizer.texts_to_sequences(test_data["text"])

# tokens sample
train_X = keras.preprocessing.sequence.pad_sequences(train_tokens, maxlen=MAX_LEN)
test_X = keras.preprocessing.sequence.pad_sequences(test_tokens, maxlen=MAX_LEN)
train_Y = train_data["label"].values

# model
model = keras.Sequential()
model.add(keras.layers.Embedding(tokenizer.num_words, EMBEDDING_DIM))  # , batch_size=batch_size
model.add(keras.layers.LSTM(H1, return_sequences=True))
model.add(keras.layers.Dropout(0.7))
model.add(keras.layers.GRU(32))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.18, shuffle=True)
# zwraca 4 obiekty 
test_y = model.predict(test_X) > THRESH
test_data["label"] = test_y.astype(int)
test_data[["id", "label"]].to_csv("submission.csv", index=False)

submit_data = pd.read_csv('./databases/fake-news/submit.csv')

submission_data = pd.read_csv('./databases/submission.csv')

model.save('models/fake-news-LSTM-simple.model')


tn, fp, fn, tp = confusion_matrix(submit_data['label'], submission_data['label']).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)

Accuracy = (tn + tp) * 100 / (tp + tn + fp + fn)
print("Accuracy {:0.2f}%".format(Accuracy))

