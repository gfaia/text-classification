"""
  gfaia - gutianfeigtf@163.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')

import os
import argparse
from tqdm import tqdm
import time
import tensorflow as tf
import numpy as np
import helper
from fasttext_model import FastText


def lengths_matrix(x, embedding_size):
  """Return the matrix of lengths values. batch_size * embedding_size
  
  case: [batch_size, embedding_size] = [3, 3]
  [[2,2,2],[5,5,5],[3,3,3]]

  Args:
    x: the data matrix, ndarray type
    embedding_size: the size of embedding, int type
  """
  lengths = np.count_nonzero(x, axis=1)
  lengths = [[l for i in range(embedding_size)] for l in lengths]
  return lengths


def main(unused_argv):
  # Movie dataset process.
  x_train, y_train, x_dev, y_dev, embeddings, vocab_size \
    = helper.mr_data_preprocess(True, FLAGS.seq_len)

  model = FastText(
    num_classes=FLAGS.num_classes, seq_len=FLAGS.seq_len, embedding_size=FLAGS.embedding_size,
    weight_decay=FLAGS.weight_decay, init_lr=FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, 
    decay_rate=FLAGS.decay_rate, vocab_size=vocab_size)

  sess = tf.InteractiveSession()
  tf.summary.scalar('loss', model.loss)
  tf.summary.scalar("accuracy", model.accuracy)
  helper.detect_dir_is_existed(FLAGS.log_dir)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tests')
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  tf.global_variables_initializer().run()

  # model train
  for e in range(FLAGS.epochs):
    print("----- Epoch {}/{} -----".format(e + 1, FLAGS.epochs))
    for xt, yt in tqdm(helper.generate_batches(x_train, y_train, FLAGS.batch_size), 
                       desc="Training"):
      xt_lengths = lengths_matrix(xt, FLAGS.embedding_size)
      _, i = sess.run([model.optimization, model.add_global], 
                      feed_dict={ model.inputs: xt, model.labels: yt, 
                                  model.lengths: xt_lengths, 
                                  model.dropout_rate: FLAGS.dropout_rate})

    dev_lengths = lengths_matrix(x_dev, FLAGS.embedding_size)
    summary, acc, loss, lr = sess.run([merged, model.accuracy, model.loss, model.learning_rate], 
                                      feed_dict={ model.inputs: x_dev, model.labels: y_dev, 
                                                  model.lengths: dev_lengths, 
                                                  model.dropout_rate: 1})

    current = time.asctime(time.localtime(time.time()))
    print("""{0} Step {1:5} Learning rate: {2:.6f} Losss: {3:.4f} Accuracy: {4:.4f}"""
          .format(current, i, lr, loss, acc))
    test_writer.add_summary(summary, i)

  save_path = saver.save(sess, FLAGS.model_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Initial learning rate.')
  parser.add_argument('--num_classes', type=int, default=2, 
                      help='the number of classes.')
  parser.add_argument('--embedding_size', type=int, default=300, 
                      help='The size of embedding.')
  parser.add_argument('--batch_size', type=int, default=128, 
                      help='The size of batch.')
  parser.add_argument('--seq_len', type=int, default=60, 
                      help='The length of sequence.')
  parser.add_argument('--dropout_rate', type=float, default=0.5, 
                      help='The probability rate of dropout')
  parser.add_argument('--weight_decay', type=float, default=2e-6, 
                      help='The rate of weight decay.')
  parser.add_argument('--decay_steps', type=int, default=2000, 
                      help='The period of decay.')
  parser.add_argument('--decay_rate', type=float, default=0.65, 
                      help='The rate of decay.')
  parser.add_argument('--log_dir', type=str, default="logs/fasttext_mr",
                      help='Summaries logs directory')
  parser.add_argument('--model_dir', type=str, 
                      default="models/fasttext_mr.ckpt",
                      help='The path to save model.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()

