import tensorflow as tf
from tensorflow.contrib import rnn


class TextCNNAndLSTM(object):
  """the model based on the rnn or lstm."""
  def __init__(self, 
               num_classes, seq_len, embedding_size, rnn_size, vocab_size, 
               weight_decay, init_lr, decay_steps, decay_rate,  
               is_rand=True, is_finetuned=False, embeddings=None):

    # model settings
    self.filter_sizes = [3, 4, 5]
    self.num_filters = 100
    self.rnn_size = rnn_size

    self.num_classes = num_classes
    self.seq_len = seq_len
    self.embedding_size = embedding_size
    self.vocab_size = vocab_size
    self.is_rand = is_rand
    self.is_finetuned = is_finetuned
    self.embeddings = embeddings

    self.weight_decay = weight_decay
    self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, 
                                                    global_step=self.global_step, 
                                                    decay_steps=decay_steps, 
                                                    decay_rate=decay_rate)

    self.model(), self.loss_acc(), self.train_op()

  def cnn_model(self, embedded_chars_expanded):
    """CNN model"""

    pooled_outputs = []
    for i, filter_size in enumerate(self.filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        W = tf.Variable(tf.truncated_normal(
          [filter_size, self.embedding_size, 1, self.num_filters], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='bias')
        conv = tf.nn.conv2d(embedded_chars_expanded, W, 
                            strides=[1, 1, 1, 1], padding='VALID', name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(h, 
                                ksize=[1, self.seq_len - filter_size + 1, 1, 1], 
                                strides=[1, 1, 1, 1], padding='VALID', name='pool') 
        pooled_outputs.append(pooled)

    num_filters_total = self.num_filters * len(self.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    outputs_mean_dim = int(h_pool_flat.shape[1])
    
    return h_pool_flat, outputs_mean_dim

  def lstm_model(self, embedded_chars):
    """Bi-LSTM model."""
    lstm_fw_cell = rnn.BasicLSTMCell(self.rnn_size)
    lstm_bw_cell = rnn.BasicLSTMCell(self.rnn_size)
    if self.dropout_rate is not None:
      lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_rate)
      lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_rate)

    outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
      lstm_fw_cell, lstm_bw_cell, embedded_chars, dtype=tf.float32)

    # forward, reverse direction lstm outputs
    forward_outputs, reverse_outputs = outputs
    # concat the forward, reverse, original
    final_outputs = tf.concat([forward_outputs, embedded_chars, reverse_outputs], 2)
    # reduce_mean max pooling
    feature = tf.reduce_max(final_outputs, axis=1)
    # feature = tf.nn.dropout(feature, self.dropout_rate)
    # feature = tf.reduce_mean(final_outputs, axis=1)
    outputs_mean_dim = int(feature.shape[1])

    return feature, outputs_mean_dim

  def feature_fusion_(self, features):
    """Feature fusion by simply concating."""
    representation = tf.concat(features, 1)
    representation = tf.nn.dropout(representation, self.dropout_rate)
    return representation

  def model(self):
    # 1. inputs
    # 2. word embeddings
    # 3. CNN model | Bi-LSTM model
    # 4. fully-connected layer
    # 5. loss + train operation
    
    # model inputs
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.onehot_labels = tf.one_hot(self.labels, self.num_classes)

    # words embeddings
    # with tf.device("/cpu:0"):
    with tf.device("/gpu:0"):
      if self.is_rand:
        W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], 
                                          -1.0, 1.0), name="W")
      else:
        W = tf.get_variable(name='W', trainable=self.is_finetuned,
                            shape=[self.vocab_size, self.embedding_size], 
                            initializer=tf.constant_initializer(self.embeddings))

    embedded_chars = tf.nn.embedding_lookup(W, self.inputs)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    
    # CNN model
    cnn_feature, cnn_dim = self.cnn_model(embedded_chars_expanded)
    # Bi-LSTM model 
    lstm_feature, lstm_dim = self.lstm_model(embedded_chars)

    features = [cnn_feature, lstm_feature]
    feature_fusion = self.feature_fusion_(features)
    feature_dim = int(feature_fusion.shape[1])

    # fully-connected layer
    with tf.name_scope("inference"):
      w = tf.Variable(tf.truncated_normal(shape=[feature_dim, self.num_classes], 
                                          mean=0, stddev=0.1), name="w")
      b = tf.Variable(tf.constant(value=0.1, shape=[self.num_classes]), name="bias")
      self.logits = tf.add(tf.matmul(feature_fusion, w), b)

  def loss_acc(self):
    """the loss and accuracy of model"""
    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.onehot_labels, 
                                                       logits=self.logits)
      
      self.loss = tf.add(tf.reduce_mean(losses), self.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]))

    with tf.name_scope("accuracy"):
      self.predictions = tf.argmax(self.logits, 1, name='predictions')
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.onehot_labels, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

  def train_op(self):
    
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    self.optimization = optimizer.apply_gradients(grads_and_vars)
