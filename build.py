import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# load dataset
x = pd.read_csv('/Users/mengyicui/Documents/coding/heavywater/shuffled-full-set-hashed.csv', names=["doclabel", "words"])
# x = xx[:10]
x_words = x.words
y_label = x.doclabel

# preprocess data with tf-idf
X_train, X_test, y_train, y_test = train_test_split(x_words, y_label)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# extract features from words
vector = TfidfVectorizer()
X_train_tfidf = vector.fit_transform(X_train.values.astype(str))
X_test_tfidf = vector.transform(X_test.values.astype(str))

# train model
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train.values.astype(str))
predictions = clf.predict(X_test_tfidf)
print np.mean(predictions == y_test)


