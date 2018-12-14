"""
  gfaia - gutianfeigtf@163.com

A simple implementation of Deep Pyramid Convolutional Neural Networks. (Johnson and Zhang)
"""
import tensorflow as tf


# def conv_layer(inputs, filter_size, in_channels, out_channels, stride_size, name):
#   """A 1-D convolutional layer. Pre-activations op before convolution."""
#   with tf.name_scope(name):
#     W = tf.Variable(tf.truncated_normal([filter_size, in_channels, out_channels], 
#                                         stddev=0.1), name='W')
#     b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name='bias')
#     pre_activation = tf.nn.relu(inputs, name='pre-activation')
#     conv = tf.nn.conv1d(pre_activation, W, stride=stride_size, padding='SAME', name='conv')
#     outputs = tf.nn.bias_add(conv, b)

#   return outputs


def conv_layer(inputs, filter_size, in_channels, out_channels, stride_size, name):
  """A 1-D convolutional layer. Pre-activations op before convolution."""
  with tf.name_scope(name):
    W = tf.Variable(tf.truncated_normal([filter_size, in_channels, out_channels], 
                                        stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name='bias')
    conv = tf.nn.conv1d(inputs, W, stride=stride_size, padding='SAME', name='conv')
    outputs = tf.nn.relu(tf.nn.bias_add(conv, b))

  return outputs


def conv_block(inputs, index, filter_size, n_filters, stride_size, pool_size, pool_stride):
  """The block containing the conv and max pool layers."""
  with tf.name_scope('block-%s' % index):
    pooled = tf.layers.max_pooling1d(inputs, pool_size, pool_stride, padding='SAME')

    conv1 = conv_layer(pooled, filter_size, n_filters, n_filters, stride_size, 
                       'block-%s-conv1' % index)
    conv2 = conv_layer(conv1, filter_size, n_filters, n_filters, stride_size,
                       'block-%s-conv2' % index)

    # add identify map
    outputs = pooled + conv2

    return outputs


class DPCNN(object):
  """The cnn model of text classification."""
  def __init__(self, 
               num_classes, seq_len, embedding_size, vocab_size,
               weight_decay, init_lr, decay_steps, decay_rate, 
               is_rand=False, is_finetuned=False, embeddings=None):

    # conv filter size, filters num, stride size
    self.conv_layer = (3, 250, 1)
    self.pool_layer = (3, 2)
    self.num_blocks = 3

    # parameters init
    self.num_classes = num_classes
    self.seq_len = seq_len
    self.embedding_size = embedding_size
    self.vocab_size = vocab_size

    self.is_rand = is_rand
    self.is_finetuned = is_finetuned
    self.embeddings = embeddings

    # weight decay
    self.weight_decay = weight_decay
    self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, global_step=self.global_step, 
                                                    decay_steps=decay_steps, decay_rate=decay_rate)

    self.model(), self.loss_acc(), self.train_op()

  def model(self):
    # word-embeddings + multi-layer cnn + fc
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.onehot_labels = tf.one_hot(self.labels, self.num_classes)
    filter_size, n_filters, stride_size = self.conv_layer
    pool_size, pool_stride = self.pool_layer

    # words embeddings
    with tf.device("/gpu:0"):
      if self.is_rand:
        W = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], 
                                            stddev=0.1), name="W")
      else:
        W = tf.get_variable(name='W', trainable=self.is_finetuned,
                            shape=[self.vocab_size, self.embedding_size], 
                            initializer=tf.constant_initializer(self.embeddings))

    embedded_chars = tf.nn.embedding_lookup(W, self.inputs)
    conv = conv_layer(embedded_chars, filter_size, self.embedding_size, n_filters, stride_size, 
                      'init-conv')
    
    # repeat the conv block 
    for i in range(self.num_blocks):
      conv = conv_block(conv, i, filter_size, n_filters, stride_size, pool_size, pool_stride)

    nums_feature = int(conv.get_shape()[1]) * int(conv.get_shape()[2])
    feature_flat = tf.reshape(conv, [-1, nums_feature])
    feature_drop = tf.nn.dropout(feature_flat, self.dropout_rate)

    with tf.name_scope("inference"):
      W = tf.Variable(tf.truncated_normal(
        [nums_feature, self.num_classes], stddev=0.1), name='W')
      b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='bias')
      self.logits = tf.nn.xw_plus_b(feature_drop, W, b, name='logits')

  def loss_acc(self):

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.onehot_labels, 
                                                       logits=self.logits)
      
      # exculde the biases parameters
      self.loss = tf.add(tf.reduce_mean(losses), self.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]))

    with tf.name_scope("accuracy"):
      self.predictions = tf.argmax(self.logits, 1, name='predictions')
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.onehot_labels, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

  def train_op(self):
    
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) 
    #  for grad, var in grads_and_vars]
    self.optimization = optimizer.apply_gradients(grads_and_vars)
