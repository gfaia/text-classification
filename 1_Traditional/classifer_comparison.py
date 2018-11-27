"""
  gfaia - gutianfeigtf@163.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')

import argparse
import helper
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def data_loader():
  """Load and label the data from raw files."""
  pos_examples = [s.decode("utf-8", "ignore").strip() 
    for s in list(open(helper.mr_pos_data, mode="rb").readlines())]
  neg_examples = [s.decode("utf-8", "ignore").strip() 
    for s in list(open(helper.mr_neg_data, mode="rb").readlines())]

  pos_nums, neg_nums = len(pos_examples), len(neg_examples)

  x = pos_examples + neg_examples
  x = [helper.clean_str(sentence) for sentence in x]

  pos_labels = [1 for _ in range(pos_nums)]
  neg_labels = [0 for _ in range(neg_nums)]
  y = pos_labels + neg_labels

  return np.array(x), np.array(y)


def tfidf(x_train, x_dev):
  """Use tf-idf to extract the feature of sentence."""
  tfidf = TfidfVectorizer(min_df=2, ngram_range=(1,2))
  tfidf.fit(x_train)

  x_train_tf = tfidf.transform(x_train)
  x_dev_tf = tfidf.transform(x_dev)

  return x_train_tf, x_dev_tf


def main():
  # Data preprocessing stage.
  x, y = data_loader()
  x_train, y_train, x_dev, y_dev = helper.split_train_dev(x, y)
  x_train_tf, x_dev_tf = tfidf(x_train, x_dev)

  # Of those algorithms, the knn need very high computation cost because of the large-dim feature.
  # The problem illustrated above also call `the curse of dimensionality`.
  classifiers = {
    # "K-Nearest Neighbors": KNeighborsClassifier(7),
    "Linear SVC": LinearSVC(),
    "Linear Classify with SGD": SGDClassifier(tol=1e-3),
    "Logistic Regression": LogisticRegression(),
    "MLP": MLPClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0, n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Gaussian Naive Bayes": GaussianNB()
  }

  for nam, cla in classifiers.items():
    cla.fit(x_train_tf.toarray(), y_train)
    predicted = cla.predict(x_dev_tf.toarray())
    print("{0} - Accuracy: {1:.4f}".format(nam, metrics.accuracy_score(predicted, y_dev)))
    print(metrics.classification_report(predicted, y_dev))
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  main()
