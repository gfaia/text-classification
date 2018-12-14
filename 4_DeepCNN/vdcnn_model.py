"""
  gfaia - gutianfeigtf@163.com

Very Deep Convolutional Networks. Conneau et al.
"""
import tensorflow as tf


def conv_layer(inputs, filter_size, in_channels, out_channels, stride_size, is_training, name):
  """A 1-D convolutional layer."""
  with tf.name_scope(name):
    W = tf.Variable(tf.truncated_normal([filter_size, in_channels, out_channels], 
                                        stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name='bias')
    conv = tf.nn.conv1d(inputs, W, stride=stride_size, padding='SAME', name='conv')
    conv = tf.layers.batch_normalization(conv, training=is_training)
    outputs = tf.nn.relu(tf.nn.bias_add(conv, b))

  return outputs


class VDCNN(object):
  """The cnn model of text classification."""
  def __init__(self, 
               num_classes, seq_len, embedding_size, vocab_size,
               weight_decay, init_lr, decay_steps, decay_rate):

    # convs settings, n_filters
    self.conv_layers = [64, 128, 256]

    # parameters init
    self.num_classes = num_classes
    self.seq_len = seq_len
    self.embedding_size = embedding_size
    self.vocab_size = vocab_size

    # weight decay
    self.weight_decay = weight_decay
    self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, global_step=self.global_step, 
                                                    decay_steps=decay_steps, decay_rate=decay_rate)

    self.model(), self.loss_acc(), self.train_op()

  def model(self):
    # char-embeddings + multi-layer convs + fc
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.onehot_labels = tf.one_hot(self.labels, self.num_classes)
    self.is_training = tf.placeholder(tf.bool)

    # Lookup table 16
    look_up_table = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], 
                                                    stddev=0.1), name='look-up-table')
    embedded_chars = tf.nn.embedding_lookup(look_up_table, self.inputs)
    conv = conv_layer(embedded_chars, 3, self.embedding_size, 64, 1, self.is_training, 
                      name='temp-conv')

    # repeat the conv block
    for i, cl in enumerate(self.conv_layers):
      prior_filter = int(conv.get_shape()[2])

      with tf.name_scope('block%s-0' % i):
        conv = conv_layer(conv, 3, prior_filter, cl, 1, self.is_training, 'conv1')
        conv = conv_layer(conv, 3, cl, cl, 1, self.is_training, 'conv2')

      with tf.name_scope('block%s-0' % i):
        conv = conv_layer(conv, 3, cl, cl, 1, self.is_training, 'conv1')
        conv = conv_layer(conv, 3, cl, cl, 1, self.is_training, 'conv2')

      conv = tf.layers.max_pooling1d(conv, 3, 2, padding='SAME')

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
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      grads_and_vars = optimizer.compute_gradients(self.loss)
      # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) 
      #  for grad, var in grads_and_vars]
      self.optimization = optimizer.apply_gradients(grads_and_vars)
