# warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd
import nltk

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print(os.getcwd())

fake = pd.read_csv("/Users/patryk/Documents/projects/stack/szkola/AI /databases/fake-news-sorted/Fake.csv")

# Counting by Subjects
for key, count in fake.subject.value_counts().iteritems():
    print(f"{key}:\t{count}")

# Getting Total Rows
print(f"Total Records:\t{fake.shape[0]}")

# plt.figure(figsize=(8,5))
# sns.countplot("subject", data=fake)
# plt.show()

real = pd.read_csv("/Users/patryk/Documents/projects/stack/szkola/AI /databases/fake-news-sorted/True.csv")

# First Creating list of index that do not have publication part
unknown_publishers = []
for index, row in enumerate(real.text.values):
    try:
        record = row.split(" -", maxsplit=1)
        # if no text part is present, following will give error
        record[1]
        # if len of piblication part is greater than 260
        # following will give error, ensuring no text having "-" in between is counted
        assert (len(record[0]) < 260)
    except:
        unknown_publishers.append(index)

# Thus we have list of indices where publisher is not mentioned
# lets check
real.iloc[unknown_publishers].text
# true, they do not have text like "WASHINGTON (Reuters)"

# Seperating Publication info, from actual text
publisher = []
tmp_text = []
for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        # Add unknown of publisher not mentioned
        tmp_text.append(row)

        publisher.append("Unknown")
        continue
    record = row.split(" -", maxsplit=1)
    publisher.append(record[0])
    tmp_text.append(record[1])

# Replace existing text column with new text
# add seperate column for publication info
real["publisher"] = publisher
real["text"] = tmp_text

del publisher, tmp_text, record, unknown_publishers

# checking for rows with empty text like row:8970
[index for index, text in enumerate(real.text.values) if str(text).strip() == '']
# seems only one :)

# dropping this record
real = real.drop(8970, axis=0)

empty_fake_index = [index for index, text in enumerate(fake.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(empty_fake_index)}")
fake.iloc[empty_fake_index].tail()

# Getting Total Rows
print(f"Total Records:\t{real.shape[0]}")

# Counting by Subjects
for key, count in real.subject.value_counts().iteritems():
    print(f"{key}:\t{count}")

# Adding class Information
real["class"] = 1
fake["class"] = 0

# Combining Title and Text
real["text"] = real["title"] + " " + real["text"]
fake["text"] = fake["title"] + " " + fake["text"]

# Subject is diffrent for real and fake thus dropping it
# Aldo dropping Date, title and Publication Info of real
real = real.drop(["subject", "date", "title", "publisher"], axis=1)
fake = fake.drop(["subject", "date", "title"], axis=1)

# Combining both into new dataframe
data = real.append(fake, ignore_index=True)
del real, fake

y = data["class"].values
# Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

del data

import gensim

# Dimension of vectors we are generating
EMBEDDING_DIM = 100

# Creating Word Vectors by Word2Vec Method (takes time...)
w2v_model = gensim.models.Word2Vec(sentences=X, vector_size=EMBEDDING_DIM, window=5, min_count=1)

# vocab size
print(len(w2v_model.wv.vocab))

# We have now represented each of 122248 words by a 100dim vector.

# Tokenizing Text -> Repsesenting each word by a number
# Mapping of orginal word to number is preserved in word_index property of tokenizer

# Tokenized applies basic processing like changing it yo lower case, explicitely setting that as False
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
for word, num in word_index.items():
    print(f"{word} -> {num}")
    if num == 10:
        break

    # For determining size of input...

# Its heavily skewed. There are news with 5000 words? Lets truncate these outliers :)

nos = np.array([len(x) for x in X])
# Out of 48k news, 44k have less than 700 words

# Lets keep all news to 700, add padding to news with less than 700 words and truncating long ones
maxlen = 700

# Making all news of size maxlen defined above
X = pad_sequences(X, maxlen=maxlen)

# Adding 1 because of reserved 0 index
# Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.
# Thus our vocab size inceeases by 1
# Lets keep all news to 700, add padding to news with less than 700 words and truncating long ones
maxlen = 700

# Making all news of size maxlen defined above
vocab_size = len(tokenizer.word_index) + 1


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


# Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix(w2v_model, word_index)

# Defining Neural Network
model = Sequential()
# Non-trainable embeddidng layer
model.add(
    Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
# LSTM
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

del embedding_vectors

model.summary()


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train, validation_split=0.3, epochs=6)

# Prediction is in probability of news being real, so converting into classes
# Class 0 (Fake) if predicted prob < 0.5, else class 1 (Real)
y_pred = (model.predict(X_test) >= 0.5).astype("int")

model.save('models/fake-news-LSTM.model')

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
