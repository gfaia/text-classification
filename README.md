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

KimCNN is the basic CNN model used to the task of text classification. In the paper, *Kim (2014)* proposes four different model variations, in this survey, we only investigate the top three proposals *CNN-rand*, *CNN-static* and *CNN-non-static*. For CNN-static, [*Word2vec*](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) is used as the pre-trained vectors.

| Algorithms               | Accuracy  |
|------------------------- |:--------: |
| CNN-rand                 | 77.02     |
| CNN-static               | 81.14     |
| CNN-non-static           | 81.50     |

* all results are CV score. In the experiment of CNN-non-static, the `FLAGS.weight_decay` should be small. The paper *Zhang and Wallace (2015)* give a more detail guide to setting hyperparameters.

### 3. FastText

*Joulin et al.* propose a very simple baseline model for text classification. This algorithm just calculates the mean of word embeddings of the document, and seem the mean vector as the feature map of the document. 

| Algorithms | Accuracy |
|------------|:--------:|
| FastText   | 79.64    |

### 4. Recurrent CNN

To address the limitation of conventional window-based neural networks, *Lai et al.* proposes a *Recurrent Convolutional Neural Network (RCNN)*, which concatenates Bi-LSTM and directional layers. Here, we compare some of the variants of RNN, and the model with the attention mechanism () performs the best among all of the approaches. 

| Algorithms           | Accuracy |
|--------------        |:--------:|
| Recurrent NN         | 79.64    |
| Recurrent CNN        | 81.52    |
| Recurrent NN + Atten | 81.99    |


## References

1. Convolutional Neural Networks for Sentence Classification, [Yoon Kim (2014)](https://www.aclweb.org/anthology/D14-1181)

2. A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
Neural Networks for Sentence Classification, [Zhang and Wallace (2015)](https://arxiv.org/pdf/1510.03820.pdf)

3. Bag of Tricks for Efficient Text Classification, [Joulin et al.](https://arxiv.org/pdf/1607.01759.pdf) 

4. Recurrent Convolutional Neural Networks for Text Classification, [Lai et al.](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

5. Attention-Based Bidirectional Long Short-Term Memory Networks for
Relation Classification, [Zhou et al.](http://www.aclweb.org/anthology/P16-2034)