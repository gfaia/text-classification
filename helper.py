"""
  Common utils for data preprocessing.
  gfaia - gutianfeigtf@163.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
import tensorflow as tf
import gensim
from os.path import join


PAD_INDEX = 0
UNK_INDEX = 1
FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

data_dir = join(os.path.dirname(__file__), "data")

# the settings of data dir
mr_data_dir = join(data_dir, "rt-polaritydata")
mr_pos_data = join(mr_data_dir, "rt-polarity.pos")
mr_neg_data = join(mr_data_dir, "rt-polarity.neg")
mr_word2vec = join(mr_data_dir, "word2vec.txt")


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
  return string.strip().lower()


def detect_dir_is_existed(dir):
  """Detect whether the directory existed."""
  if tf.gfile.Exists(dir):
    tf.gfile.DeleteRecursively(dir)
  else:
    tf.gfile.MakeDirs(dir)


def split_train_dev(x, y, split_rate=0.1):
  """Split the original data into train and dev part."""
  n_samples = len(y)
  # np.random.seed(40)
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


def mr_data_preprocess(is_rand, seq_len, embedding_size=300):
  """Data preprocessing of movie review.
  Args:
    is_rand: decide whether use word embeddings.
    seq_len: the final length of text sequence.
    embedding_size: the size of word embeddings, default set as 300.
  """
  pos_examples = [s.decode("utf-8", "ignore").strip() 
                  for s in list(open(mr_pos_data, mode="rb").readlines())]
  neg_examples = [s.decode("utf-8", "ignore").strip() 
                  for s in list(open(mr_neg_data, mode="rb").readlines())]
  pos_nums, neg_nums = len(pos_examples), len(neg_examples)
  texts = pos_examples + neg_examples
  x = [clean_str(text) for text in texts]
  pos_y = [[0, 1] for _ in range(pos_nums)]
  neg_y = [[1, 0] for _ in range(neg_nums)]
  y = pos_y + neg_y

  tp = TextPreprocessing(seq_len=seq_len)
  x = tp.convert_texts_to_seqs(x)
  vocab_size = tp.vocab_size
  embeddings = None

  if not is_rand:
    # create a static word embeddings
    # the index 0 reversed to <PAD> and 1 reversed to <UNK>
    # this version come from
    # https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    vocab = tp.word_index

    embeddings = np.zeros(shape=(vocab_size, embedding_size), dtype='float32')
    vectors = gensim.models.KeyedVectors.load_word2vec_format(mr_vectors)

    for w in vocab.keys():
      if w in vectors.vocab:
        embeddings[vocab[w]] = vectors[w]
      else:
        if w != '<PAD>':
          embeddings[vocab[w]] = np.random.uniform(-0.25, 0.25, embedding_size)

  x_train, y_train, x_dev, y_dev = split_train_dev(np.array(x), np.array(y))
  return x_train, y_train, x_dev, y_dev, np.array(embeddings), vocab_size


class TextPreprocessing(object):
  """This class used to text preprocessing."""

  def __init__(self, min_df=1, max_df=None, lower=True, split=' ', filters=FILTERS, 
               char_level=False, oov=True, n_samples=0, seq_len=None, **kwargs):
    
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    self.min_df = min_df
    self.max_df = max_df
    self.lower = lower
    self.split = split
    self.filters = filters
    self.char_level = char_level
    self.n_samples = n_samples
    self.seq_len = seq_len

    self.word_index = dict()
    self.index_word = dict()
    self.word_counts = dict()

    # We limit the some Special Words. <PAD> 0, <UNK> 1
    self.word_index['<PAD>'] = PAD_INDEX
    self.word_index['<UNK>'] = UNK_INDEX
    self.vocab_size = 0

  def _convert_text_to_word_seq(self, text):
    """Convert a string to the word sequence.
    
    Args:
      text: a string, str type.
    """
    if self.lower:
      text = text.lower()

    for c in self.filters:
      text = text.replace(c, self.split)

    # text = re.sub(r"\s{2,}", " ", text)
    seq = text.split(self.split)

    return [i for i in seq if i]

  def _pad_seq(self, seq):
    """Modify the length of sequence as self.seq_len.
    
    Args:
      seq: a list of word index, list type.
    """
    padded_seq = []
    if len(seq) > self.seq_len:
      padded_seq = seq[:self.seq_len]
    else:
      padded_seq = seq + [PAD_INDEX for i in range(self.seq_len - len(seq))]

    return padded_seq

  def convert_texts_to_seqs(self, texts):
    """Updates the parameters of text dataset.

    Not change the order of datasets.
    
    Args:
      texts: a list of string, list type.
    """
    seqs = []
    for text in texts:
      if isinstance(text, str):
        seq = self._convert_text_to_word_seq(text)
        seqs.append(seq)
      else:
        raise("The type of text must be str.")

      for w in seq:
        if w in self.word_counts:
          self.word_counts[w] += 1
        else:
          self.word_counts[w] = 1

    if not self.max_df:
      self.max_df = max(self.word_counts.values())

    wcounts = self.word_counts.items()
    for w, c in wcounts:
      if self.min_df <= c <= self.max_df:
        self.word_index[w] = len(self.word_index)

    self.index_word = dict(zip(self.word_index.values(), self.word_index.keys()))
    self.vocab_size = len(self.index_word)

    seqids = [[self.word_index[w] if w in self.word_index.keys() else UNK_INDEX for w in seq] 
              for seq in seqs]

    if not self.seq_len:
      self.seq_len = max([len(s) for s in seqids])

    seqids = [self._pad_seq(seqid) for seqid in seqids]

    return seqids

  def get_vocab(self):
    """Get the vocabulary of dataset."""
    if len(self.word_index) == 0:
      raise Exception("Have not preprocess the dataset.")
    else:
      return set(self.word_index.keys())


# if __name__ == "__main__":
  # word2vec = gensim.models.KeyedVectors.load_word2vec_format(mr_word2vec, binary=True)
