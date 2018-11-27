# Text Classification

This is a survey on deep learning models for text classification and we will continue to update. Text Classification has been the most competed NLP task in kaggle and other similar competitions. Although many traditional proposals based on traditional algorithms achieve good results, the community also wish to improve the classification performance further. 


## Environment Settings

In this survey, we pay more attention to models' details, not datasets (domain), so trying a small dataset named [*Movie Reviews*](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz). In the recent, we only use the *Tensorflow* to implement our models, and the training process run on a *GTX1080* gpu single machine.

1. scikit-learn==0.19.1
2. tensorflow-gpu==1.10.1
3. tensorboard==1.10.0


## Experimental Results

### 1. Traditional algorithms

Before deep learning, we first to compare some of the traditional learning algorithms, and the best classifier is linear SVC training with SGD. 

| Algorithms               | Accuracy  |
|------------------------- |:--------: |
| K-Nearest Neighbors      | 68.95     |
| Linear SVC               | 78.80     |
| Linear Classify with SGD | **79.01** |
| Logistic Regression      | 76.64     |
| MLP                      | 77.77     |
| Decision Tree            | 61.07     |
| Random Forest            | 71.86     |
| AdaBoost                 | 66.89     |
| Gaussian Naive Bayes     | 68.95     |

### 2. KimCNN

KimCNN is the basic CNN model used to the task of text classification. In the paper, Kim (2014) proposes four different model variations, in this survey, we only investigate the top two proposals *CNN-rand* and *CNN-static*. For CNN-static, [*Word2vec*](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) is used as the pre-trained vectors.

| Algorithms               | Accuracy  |
|------------------------- |:--------: |
| CNN-rand                 | 77.77     |
| CNN-static               | 82.36     |

* the best results are seleted here.


## References

1. [Convolutional Neural Networks for Sentence Classification, Yoon Kim (2014)](https://www.aclweb.org/anthology/D14-1181)
