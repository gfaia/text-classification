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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.decomposition import PCA
from classifer_comparison import data_loader


def main():
  """Investigate the effect of dimensionality reduction (DR)."""
  x, y = data_loader()

  dims = [1000, 500, 100]
  for d in dims:
    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1,2))
    tfidf.fit(x)
    x_tf = tfidf.transform(x).toarray()
    pca = PCA(n_components=d)

    x_train_tf, y_train, x_dev_tf, y_dev = helper.split_train_dev(x_pca, y)
    svc = LinearSVC()
    svc.fit(x_train_tf, y_train)
    predicted = svc.predict(x_dev_tf)
    print("Accuracy: {0:.4f}".format(metrics.accuracy_score(predicted, y_dev)))
    print(metrics.classification_report(predicted, y_dev))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  main()

