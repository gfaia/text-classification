"""FastText without <pad> - average words embedding as the feature."""
import tensorflow as tf


class FastText(object):
  """FastText: Facebook's simple model for text classification."""
  def __init__(self, num_classes, seq_len, embedding_size, weight_decay, init_lr, decay_steps,
               decay_rate, vocab_size):

    self.num_classes = num_classes
    self.seq_len = seq_len
    self.embedding_size = embedding_size
    self.weight_decay = weight_decay
    self.vocab_size = vocab_size
    self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

    # Learning rate decay.
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, 
                                                    global_step=self.global_step, 
                                                    decay_steps=decay_steps, 
                                                    decay_rate=decay_rate)

    self.model(), self.loss_acc(), self.train_op()

  def model(self):
    
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name='labels')
    # the length of each sentence.
    self.lengths = tf.placeholder(tf.float32, [None, self.embedding_size], name='lengths')

    with tf.device("/gpu:0"):
      W = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], 
                                          stddev=0.1), name="W")      
    embedded_chars = tf.nn.embedding_lookup(W, self.inputs)
    embedded_sum = tf.reduce_sum(embedded_chars, axis=1)
    feature = embedded_sum / self.lengths

    with tf.name_scope("inference"):
      W = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_classes], 
                                          stddev=0.1), name='W')
      b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='bias')
      self.logits = tf.matmul(feature, W) + b
      self.predictions = tf.argmax(self.logits, 1, name='predictions')

  def loss_acc(self):

    with tf.name_scope("loss"):
      # losses = tf.nn.softmax_cross_entropy_with_logits(
      #   labels=self.labels, logits=self.logits)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
      
      # exculde the biases parameters
      self.loss = tf.add(tf.reduce_mean(losses), 
        self.weight_decay * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]))

    with tf.name_scope("accuracy"):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

  def train_op(self):
    
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) 
    # for grad, var in grads_and_vars]
    self.optimization = optimizer.apply_gradients(grads_and_vars)
