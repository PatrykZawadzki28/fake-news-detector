# Machine learning

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Modelling Algorithms
from sklearn.linear_model import LogisticRegression

# Modelling Helpers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import joblib

# Computations
import itertools

# Visualization
import matplotlib.pyplot as plt

train = pd.read_csv("./databases/fake-news/train.csv")
test = pd.read_csv("./databases/fake-news/test.csv")
submit = pd.read_csv("./databases/fake-news/submit.csv")

train.head()

print(f"Train Shape : {train.shape}")
print(f"Test Shape : {test.shape}")
print(f"Submit Shape : {submit.shape}")

train.info()  # 20800 entries

train.dtypes.value_counts()  # 3 objects, 2 int64

# create column with all the data available
test = test.fillna(' ')
train = train.fillna(' ')

test['total'] = test['title'] + ' ' + test['author'] + ' ' + test['text']
train['total'] = train['title'] + ' ' + train['author'] + ' ' + train['text']

# check if new column is there
train.info()
train.head()

# VECTORIZATION
# Dividing the training set by using train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['total'], train.label, test_size=0.20, random_state=0)

print(train['total'])
# Map words or phrases to vector of real numbers

# 1 option:
# Initialize the `count_vectorizer`
count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
# Fit and transform the training data.
count_train = count_vectorizer.fit_transform(X_train)
# Transform the test set
count_test = count_vectorizer.transform(X_test)

# 2 option:
# #Initialize the `tfidf_vectorizer`
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
# #Fit and transform the training data
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# #Transform the test set
# tfidf_test = tfidf_vectorizer.transform(X_test)

## second part

# 1 option:
# # Multinomial Naive Bayes
# nb_classifier = MultinomialNB(alpha=0.1)
# nb_classifier.fit(count_train, y_train)
# pred_nb_count = nb_classifier.predict(count_test)
# cm = metrics.confusion_matrix(y_test, pred_nb_count, labels=[0, 1])

# 2 option:
# Logistic regression
logreg = LogisticRegression(C=1e5)
logreg.fit(count_train, y_train)
pred_logreg_count = logreg.predict(count_test)
acc_logreg_count = metrics.accuracy_score(y_test, pred_logreg_count)
print(acc_logreg_count)
cm3 = metrics.confusion_matrix(y_test, pred_logreg_count, labels=[0, 1])

joblib.dump(logreg, 'models/fake-news-model.pkl')
joblib.dump(count_vectorizer, 'models/vectorizer.pkl')

# Creating a function that outputs a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm3, classes=['TRUE', 'FAKE'],
                      title='Confusion matrix for a Logistic Regression with Count Vectorizer')

