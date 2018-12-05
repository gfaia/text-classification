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
import helper


class CharCNN(object):

  def __init__(self, 
               seq_len, num_classes, alphabet_size, 
               weight_decay, init_lr, decay_steps, decay_rate):

    # convolution layers: feature_map, filter_size, max_pool_size
    # Zhang et al. ZhangCNN model settings.
    self.conv_layers = [(256, 7, 3), (256, 7, 3), (256, 3, None), (256, 3, None), 
                        (256, 3, None), (256, 3, 3)]
    self.fc_layers = [1024, 1024]

    # parameters init
    self.seq_len = seq_len
    self.num_classes = num_classes
    self.alphabet_size = alphabet_size

    # weight decay
    self.weight_decay = weight_decay
    self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, global_step=self.global_step, 
                                                    decay_steps=decay_steps, decay_rate=decay_rate)
    
    self.model(), self.loss_acc(), self.train_op()

  def conv_max_layer(self, index, inputs, layer_size):

    with tf.name_scope("%s-conv-maxpool" % index):
      feature_map, filter_size, max_pool_size = layer_size
      filter_width = int(inputs.get_shape()[2])
      filter_shape = [filter_size, filter_width, 1, feature_map]
      
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
      b = tf.Variable(tf.constant(0.1, shape=[feature_map], name='bias'))
      conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

      if max_pool_size:
        pooled = tf.nn.max_pool(h, 
                                ksize=[1, max_pool_size, 1, 1], 
                                strides=[1, max_pool_size, 1, 1],
                                padding='VALID', name='pool')
        outputs = tf.transpose(pooled, [0, 1, 3, 2], name='outputs%s' % index)
      else:
        outputs = tf.transpose(h, [0, 1, 3, 2], name='outputs%s' % index)

    return outputs

  def linear_layer(self, index, inputs, outputs_size):
    
    with tf.name_scope('%s-linear-layer' % index):
      inputs_size = int(inputs.get_shape()[1])
      W = tf.Variable(tf.truncated_normal([inputs_size, outputs_size], stddev=0.1), name='W')
      b = tf.Variable(tf.constant(0.1, shape=[outputs_size]), name='bias')
      outputs = tf.nn.xw_plus_b(inputs, W, b)
      # outputs = tf.nn.dropout(outputs, self.dropout_rate)
    
    return outputs

  def model(self):
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.onehot_inputs = tf.one_hot(self.inputs, self.alphabet_size)
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.onehot_labels = tf.one_hot(self.labels, self.num_classes)

    convs = tf.expand_dims(self.onehot_inputs, -1)

    for i, layer_size in enumerate(self.conv_layers):
      convs = self.conv_max_layer(i, convs, layer_size)

    with tf.name_scope("fully-connected"):
      fc_size = int(convs.get_shape()[1]) * int(convs.get_shape()[2])
      fc_layer = tf.reshape(convs, [-1, fc_size])
      fc = tf.nn.dropout(fc_layer, self.dropout_rate)

    # for i, layer_size in enumerate(self.fc_layers):
    #   fc = self.linear_layer(i, fc, layer_size)

    with tf.name_scope('inference'):
      # W = tf.Variable(tf.truncated_normal([self.fc_layers[-1], self.num_classes], 
      #                                     stddev=0.1), name='W')
      W = tf.Variable(tf.truncated_normal([fc_size, self.num_classes], stddev=0.1), name='W')
      b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='bias')
      self.logits = tf.nn.xw_plus_b(fc, W, b, name='logits')
      self.predictions = tf.argmax(self.logits, 1, name='predictions')

  def loss_acc(self):

    with tf.name_scope("loss"):
      # losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.onehot_labels, 
                                                       logits=self.logits)
      
      # exculde the biases parameters
      self.loss = tf.add(tf.reduce_mean(losses), 
        self.weight_decay * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]))

    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.onehot_labels, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

  def train_op(self):
    
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) 
    #  for grad, var in grads_and_vars]
    self.optimization = optimizer.apply_gradients(grads_and_vars)
