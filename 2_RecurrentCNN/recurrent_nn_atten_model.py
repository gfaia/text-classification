"""
  gfaia - gutianfeigtf@163.com
"""
import tensorflow as tf
from tensorflow.contrib import rnn


class RecurrentNNAtten(object):
  """Use the attention on the rnn model."""
  def __init__(self, 
               num_classes, embedding_size, vocab_size, rnn_size, seq_len,
               weight_decay, init_lr, decay_steps, decay_rate, 
               is_rand=True, is_finetuned=True, embeddings=None):

    self.num_classes = num_classes
    self.seq_len = seq_len
    self.embedding_size = embedding_size
    self.weight_decay = weight_decay
    self.vocab_size = vocab_size
    self.is_rand = is_rand
    self.embeddings = embeddings
    self.rnn_size = rnn_size
    self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

    # weight decay
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, 
                                                    global_step=self.global_step, 
                                                    decay_steps=decay_steps, 
                                                    decay_rate=decay_rate)

    self.model(), self.loss_acc(), self.train_op()

  def attention(self, hidden_state, hidden_size):
    """utilize the attention to outputs of rnn model.
    Args:
      hidden_state: tensor type, [batch_size, seq_len, hidden_size]
      hidden_size: the size of hidden_state, int type
    Return:
      representation: tensor type, [batch_size, hidden_size]
    """
    with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
      W_w = tf.get_variable("W_w", shape=[hidden_size, hidden_size], 
                            initializer=tf.random_normal_initializer(stddev=0.1))
      b_w = tf.get_variable("bias_w", shape=[hidden_size])
      u_w = tf.get_variable("u_w", shape=[hidden_size])

    hidden_state_ = tf.stack(hidden_state, axis=1)
    hidden_state_2 = tf.reshape(hidden_state_, shape=[-1, hidden_size])

    # u = tanh(Wh + b)
    hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_2, W_w) + b_w)
    hidden_representation = tf.reshape(hidden_representation, 
                                       shape=[-1, self.seq_len, hidden_size])
  
    # context attention exp / sum(exp)
    hidden_state_context = tf.multiply(hidden_representation, u_w)
    attention_logits = tf.reduce_sum(hidden_state_context, axis=2)
    attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
  
    # get possibility distribution for each word in the sentence
    p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
    p_attention = tf.expand_dims(p_attention, axis=2)

    # s = sum(a * h)
    representation = tf.multiply(p_attention, hidden_state_)
    representation = tf.reduce_sum(representation, axis=1)

    return representation

  def model(self):
    # word embeddings -> Bidirectional LSTM -> Attention -> logits
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.onehot_labels = tf.one_hot(self.labels, self.num_classes)

    # word embeddings, option: is_or_not random
    with tf.device('/gpu:0'):
      if self.is_rand:
        W = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], 
                                            stddev=0.1), name="W")
      else:
        W = tf.get_variable(name='W', trainable=False, 
                            shape=[self.vocab_size, self.embedding_size], 
                            initializer=tf.constant_initializer(self.embeddings))
    
    # embedding words
    embedded = tf.nn.embedding_lookup(W, self.inputs, "embedded")

    # bi-lstm
    lstm_fw_cell = rnn.BasicLSTMCell(self.rnn_size)
    lstm_bw_cell = rnn.BasicLSTMCell(self.rnn_size)
    if self.dropout_rate is not None:
      lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_rate)
      lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_rate)

    outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, 
                                                         embedded, dtype=tf.float32)
    hidden_state = tf.concat(outputs, axis=2)
    hidden_state = tf.split(hidden_state, self.seq_len, axis=1)
    # list type seq_Len * [batch_size, rnn_size * 2]
    hidden_state_squeezed = [tf.squeeze(x, axis=1) for x in hidden_state]
    
    # use attention to replace the max operation.
    feature = self.attention(hidden_state_squeezed, self.rnn_size * 2)
    outputs_mean_dim = int(feature.shape[1])

    # inference layer
    with tf.name_scope("inference"):
      w = tf.Variable(tf.truncated_normal(shape=[outputs_mean_dim, self.num_classes], 
                                          mean=0, stddev=0.1), name="w")
      b = tf.Variable(tf.constant(value=1.0, shape=[self.num_classes]), name="bias")
      self.logits = tf.add(tf.matmul(feature, w), b)
      self.predictions = tf.argmax(self.logits, 1, name='predictions')

  def loss_acc(self):
    """the loss and accuracy of model"""
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot_labels, 
                                                       logits=self.logits)
      # exculde the bias parameters
      self.loss = tf.add(tf.reduce_mean(losses), 
        self.weight_decay * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]))

    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.onehot_labels, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

  def train_op(self):
    
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    self.optimization = optimizer.apply_gradients(grads_and_vars)
