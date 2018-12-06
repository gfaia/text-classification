"""
  Common utils for data preprocessing.
  gfaia - gutianfeigtf@163.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import csv
import numpy as np
import tensorflow as tf
import gensim
from os.path import join


data_dir = join(os.path.dirname(__file__), "data")

# Movie Review
mr_data_dir = join(data_dir, "rt-polaritydata")
mr_pos_data = join(mr_data_dir, "rt-polarity.pos")
mr_neg_data = join(mr_data_dir, "rt-polarity.neg")
mr_word2vec = join(mr_data_dir, "word2vec.txt")

# AG’s news corpus
ag_data_dir = join(data_dir, "ag_news_csv")
ag_train = join(ag_data_dir, "train.csv")
ag_test = join(ag_data_dir, "test.csv")
ag_word2vec = join(ag_data_dir, "word2vec.txt")


def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from `https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py`
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()


def detect_dir_is_existed(dir):
  """Detect whether the directory existed."""
  if tf.gfile.Exists(dir):
    tf.gfile.DeleteRecursively(dir)
  else:
    tf.gfile.MakeDirs(dir)


def split_train_dev(x, y, split_rate=0.1):
  """Split the original data into train and dev part."""
  n_samples = len(y)
  shuffle_indices = np.random.permutation(n_samples)
  x, y = x[shuffle_indices], y[shuffle_indices]
  dev_index = int(n_samples * split_rate)
  return x[dev_index:], y[dev_index:], x[:dev_index], y[:dev_index]


def generate_batch(data, labels, batch_size=128):
  """Generate the training batch with sampling."""
  n_samples = data.shape[0]
  selected_samples = np.random.choice(n_samples, batch_size)
  return data[selected_samples], labels[selected_samples]


def generate_batches(data, labels, batch_size=128):
  """Output the next batch of the dataset."""
  n_samples = len(data)

  shuffle_indices = np.random.permutation(n_samples)
  data, labels = data[shuffle_indices], labels[shuffle_indices]

  n_batches = n_samples // batch_size + 1
  
  def _generate_batch():

    for i in range(n_batches):
      data_batch = data[i * batch_size: (i + 1) *  batch_size]
      label_batch = labels[i * batch_size: (i + 1) *  batch_size]

      yield data_batch, label_batch

  batches = [(xt, yt) for xt, yt in _generate_batch()]
  return batches


def next_batch(data, labels, batch_size=128):
  """Output the next batch of the dataset."""
  n_samples = len(data)

  shuffle_indices = np.random.permutation(n_samples)
  data, labels = data[shuffle_indices], labels[shuffle_indices]

  n_batches = n_samples // batch_size + 1

  for i in range(n_batches):
    data_batch = data[i * batch_size: (i + 1) *  batch_size]
    label_batch = labels[i * batch_size: (i + 1) *  batch_size]
    
    yield data_batch, label_batch


def filter_vectors_from_word2vec(dataset_vocab, dataset_vectors):
  """To reduce the time on the data loading, we per-extract the word embeddings used in the
   dataset form word2vec.
  
  Args:
    dataset_vocab: a set of words appeared in text dataset, set type.
  """
  word2vec_file = join(data_dir, "word2vec/GoogleNews-vectors-negative300.bin")
  word2vec_vocab_file = join(data_dir, "word2vec/word2vec_vocab.txt")
  word2vec_vocab = set()

  word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
  word2vec_vocab = word2vec.vocab.keys()

  # Save the word2vec vocab file
  if not tf.gfile.Exists(word2vec_vocab_file):
    with open(word2vec_vocab_file, 'w' , encoding='utf-8') as f:
      for w in word2vec_vocab:
        f.write(w + "\n")

  vocab = dataset_vocab.intersection(word2vec_vocab)
  with open(dataset_vectors, 'w', encoding='utf-8') as f:
    f.write(str(len(vocab)) + ' 300\n')
    for w in vocab:
      f.write(w + ' ' + ' '.join(map(str, word2vec[w])) + '\n')


def build_embeddings(word_index, embedding_size, word2vec_file):
  # create a static word embeddings
  # NOTE: The special token <PAD> set as index 0 and <UNK> set as 1 by convention.
  # this version come from
  # https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

  embeddings = np.zeros(shape=(len(word_index), embedding_size), dtype='float32')
  vectors = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file)

  for w in word_index.keys():
    if w in vectors.vocab:
      embeddings[word_index[w]] = vectors[w]
    else:
      if w != '<PAD>':
        embeddings[word_index[w]] = np.random.uniform(-0.25, 0.25, embedding_size)
  return embeddings


def _mr_data_loader():
  """Load the data of movie reviews."""
  pos_examples = [s.decode("utf-8", "ignore").strip() 
                  for s in list(open(mr_pos_data, mode="rb").readlines())]
  neg_examples = [s.decode("utf-8", "ignore").strip() 
                  for s in list(open(mr_neg_data, mode="rb").readlines())]

  pos_nums, neg_nums = len(pos_examples), len(neg_examples)
  texts = pos_examples + neg_examples
  texts = [clean_str(text) for text in texts]

  pos_labels = [1 for _ in range(pos_nums)]
  neg_labels = [0 for _ in range(neg_nums)]
  labels = pos_labels + neg_labels

  return texts, np.array(labels)


def mr_data_loader(seq_len, is_rand=False, char_level=False, embedding_size=300):
  """Data preprocessing of movie review.
  
  Args:
    is_rand: decide whether use word embeddings.
    seq_len: the final length of text sequence.
    embedding_size: the size of word embeddings, default set as 300.
  """
  texts, labels = _mr_data_loader()
  n_labels = 2

  tp = TextPreprocessing(seq_len=seq_len, char_level=char_level)
  tp.fit_on_texts(texts)
  x = tp.texts_to_sequences(texts)
  x_train, y_train, x_test, y_test = split_train_dev(x, labels)
  vocab_size = tp.vocab_size
  embeddings = None

  if not is_rand:
    embeddings = build_embeddings(tp.word_index, embedding_size, mr_word2vec)

  return x_train, y_train, x_test, y_test, embeddings, vocab_size, n_labels


def _ag_data_loader(filename):
  """Load the data AG's news corpus."""
  texts, labels = [], []
  with open(filename, 'r', encoding='utf-8') as f:
    rdr = csv.reader(f, delimiter=',', quotechar='"')
    for row in rdr:
      labels.append(int(row[0]))
      text = clean_str(" ".join(row[1:]))
      texts.append(text)
  return texts, np.array(labels)


def ag_data_loader(seq_len, is_rand=False, char_level=False, embedding_size=300):
  """Data preprocessing of AG’s news corpus.
  
  Args:
    seq_len: the final length of text sequence.
  """
  train_texts, train_labels = _ag_data_loader(ag_train)
  test_texts, test_labels = _ag_data_loader(ag_test)
  all_texts = train_texts + test_texts
  y_train = train_labels - 1
  y_test = test_labels - 1
  n_labels = 4

  tp = TextPreprocessing(seq_len=seq_len, char_level=char_level)
  tp.fit_on_texts(all_texts)
  x_train = tp.texts_to_sequences(train_texts)
  x_test = tp.texts_to_sequences(test_texts)
  vocab_size = tp.vocab_size
  embeddings = None

  if not is_rand:
    embeddings = build_embeddings(tp.word_index, embedding_size, ag_word2vec)

  return x_train, y_train, x_test, y_test, embeddings, vocab_size, n_labels


def data_loader(seq_len, dataset='MR', is_rand=False, char_level=False, embedding_size=300):
  """Collect all data loader functions."""
  data_loaders = {'MR': mr_data_loader, 
                  'AG': ag_data_loader}
  return data_loaders[dataset](seq_len, is_rand, char_level, embedding_size)


class TextPreprocessing(object):
  """This class used to text preprocessing. 
    A light-preprocessor for text.
  
  Args:
    min_df: the minimum frequency which the count of token should bigger than.
      default set as 1, i.e. the total token appeared in corpus.
    max_df: the maximum frequency which the count of token should smaller than. 
    lower: whether to convert the texts to lowercase, boolen type.
    split: separator for word splitting, str type.
    char_level: whether treat each char as token. boolen type.
    filters: a string where each element is a character that will be filtered from the texts.
    oov_token: whether set the out of vocabulary.
    seq_len: the final length of text sequence.
  """
  def __init__(self, 
               min_df=1, max_df=None, lower=True, split=' ', char_level=False, 
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
               oov_token=True, seq_len=None, **kwargs):
    
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    self.min_df = min_df
    self.max_df = max_df
    self.lower = lower
    self.split = split
    self.char_level = char_level
    self.filters = filters
    self.oov_token = oov_token
    self.seq_len = seq_len

    self.word_index = dict()
    self.index_word = dict()
    self.word_counts = dict()
    self.vocab_size = 0

  def _text_to_word_sequence(self, text):
    """Convert a string to the word sequence.
    
    Args:
      text: a string, str type.
    """
    if self.lower:
      text = text.lower()

    for c in self.filters:
      text = text.replace(c, self.split)

    seq = text.split(self.split)

    return [i for i in seq if i]

  def fit_on_texts(self, texts):
    """Updates internal vocabulary based on a list of texts.
    
    Args:
      texts: a list of string, list type
    """
    max_seq_len = 0
    for text in texts:
      if self.char_level:
        seq = text.lower()
      else:
        seq = self._text_to_word_sequence(text)

      for w in seq:
        if w in self.word_counts:
          self.word_counts[w] += 1
        else:
          self.word_counts[w] = 1

      if len(seq) > max_seq_len:
        max_seq_len = len(seq)

    if self.seq_len is None:
      self.seq_len = max_seq_len

    if not self.max_df:
      self.max_df = max(self.word_counts.values())

    # NOTE: The special token <PAD> set as index 0 and <UNK> set as 1 by convention.
    self.word_index['<PAD>'] = 0
    if self.oov_token:
      self.word_index['<UNK>'] = 1

    wcounts = self.word_counts.items()
    for w, c in wcounts:
      if self.min_df <= c <= self.max_df:
        self.word_index[w] = len(self.word_index)

    self.index_word = dict(zip(self.word_index.values(), self.word_index.keys()))
    self.vocab_size = len(self.index_word)

  def get_vocab(self):
    """Get the vocabulary of dataset."""
    if len(self.word_index) == 0:
      raise Exception("Have not preprocess the dataset.")
    else:
      return set(self.word_index.keys())

  def _pad_seq(self, seq):
    """Modify the length of sequence as self.seq_len.
    
    Args:
      seq: a list of word index, list type.
    """
    padded_seq = []
    if len(seq) > self.seq_len:
      padded_seq = seq[:self.seq_len]
    else:
      padded_seq = seq + [0 for i in range(self.seq_len - len(seq))]

    return padded_seq

  def texts_to_sequences(self, texts):
    """Convert texts to sequences.
    
    Args:
      texts: a list of string, list type.
    """
    return np.array(list(self.texts_to_sequences_genator(texts)))

  def texts_to_sequences_genator(self, texts):
    """Updates the parameters of text dataset.

    Not change the order of datasets.
    
    Args:
      texts: a list of string, list type.
    """
    for text in texts:
      if self.char_level:
        seq = text.lower()
      else:
        seq = self._text_to_word_sequence(text)

      vect = []
      for w in seq:
        if w in self.word_index.keys():
          vect.append(self.word_index[w])
        else:
          if self.oov_token:
            vect.append(1)

      vect = self._pad_seq(vect)

      yield vect
