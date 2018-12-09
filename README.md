# Text Classification

This is a survey on deep learning models for text classification and we will continue to update. Text Classification has been the most competed NLP task in kaggle and other similar competitions. Although many traditional proposals based on traditional algorithms achieve good results, the community also wish to improve the classification performance further. 


## Environment Settings

In the experiment, we use the Tensorflow to implement the proposed deep-based models, and the training process runs on a GTX1080 GPU single machine.

1. scikit-learn==0.19.1
2. tensorflow-gpu==1.10.1
3. tensorboard==1.10.0

In this survey, we select two small-scale text classification datasets. 

* *MR* [**Movie Reviews**](http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz) Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews. The number of total examples is 10662.
* *AG* [**AG’s news**](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) As a task of English news categorization, each example in the dataset can be classified into 4 classes. #Train 120k #7.6k

## Experimental Results

### 1. Traditional algorithms

Before deep learning, we first to compare some of the traditional learning algorithms, and the TFIDF (term-frequency inverse-document-frequency) is used to extract the feature with fixed length from the text. Because of the continuous type, we pay more attention to linear models. 

### 2. KimCNN

KimCNN is the basic CNN model used to the task of text classification. In the paper, Kim (2014) proposes four different model variations, in this survey, we only investigate the top three proposals CNN-rand, CNN-static, and CNN-non-static. For CNN-static, [Word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) is used as the pre-trained vectors. All results are CV score. The paper Zhang and Wallace (2015) give more detail guide about setting hyperparameters. 

### 4. Recurrent CNN

To address the limitation of conventional window-based neural networks, Lai et al. propose a Recurrent Convolutional Neural Network (RCNN), which concatenates Bi-LSTM and directional layers. Here, we compare some of the variants of RNN, including the original version and model with the attention mechanism (Zhou et al.) performs the best among all of the approaches. 

### 3. FastText

Joulin et al. propose a very simple baseline model for text classification. This algorithm just calculates the mean of word embeddings of the document, and seem the mean vector as the feature map of the document. 

### 4. Char CNN

Different from word-level models, Zhang et al.(CharCNN) introduce the design of character-level ConvNets for text classification. Conneau et al.(VDCNN) point that the deeper models perform better and able to learn hierarchical representations of the whole sentences.

### 5. Deep Pyramid CNN (DPCNN)

Based on the shallow CNN, DPCNN, proposed by Johnson and Zhang, is a deeper CNN model. They fixed the number of filters in each convolutional layer, and introduce the identity map. 

### Accuracy

| Algorithms               |  MR   |  AG   |   
|------------------------- |:-----:|:-----:|
| Logistic Regression      | 76.64 |  -    |
| MLP                      | 77.77 |  -    |
| Linear SVC               | 78.80 |  -    |
| CNN-rand                 | 77.02 |  -    |
| CNN-static               | 81.14 | 92.29 |
| CNN-non-static           | 81.50 |  -    |
| Recurrent NN             | 79.64 |  -    |
| Recurrent CNN            | 81.52 | 91.86 |
| Recurrent NN + Atten     | 81.99 |  -    |
| FastText                 | 79.64 | 92.01 |
| CharCNN                  |  -    | 89.06 |
| DPCNN                    | 78.45 | 92.16 |

## References

1. Convolutional Neural Networks for Sentence Classification, [Yoon Kim (2014)](https://www.aclweb.org/anthology/D14-1181)

2. A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
Neural Networks for Sentence Classification, [Zhang and Wallace (2015)](https://arxiv.org/pdf/1510.03820.pdf)

3. Bag of Tricks for Efficient Text Classification, [Joulin et al.](https://arxiv.org/pdf/1607.01759.pdf) 

4. Recurrent Convolutional Neural Networks for Text Classification, [Lai et al.](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

5. Attention-Based Bidirectional Long Short-Term Memory Networks for
Relation Classification, [Zhou et al.](http://www.aclweb.org/anthology/P16-2034)

6. Character-level Convolutional Networks for Text Classification, [Zhang et al.](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)

7. Very Deep Convolutional Networks for Text Classification, [Conneau et al.](http://www.aclweb.org/anthology/E17-1104)

8. Deep Pyramid Convolutional Neural Networks for Text Categorization, [Johnson and Zhang](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)