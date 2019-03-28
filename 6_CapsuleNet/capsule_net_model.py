"""
  gfaia - gutianfeigtf@163.com
"""
import tensorflow as tf


def squash(x, axis=-1):
  """Squash function"""
  s_squared_norm = tf.math.reduce_sum(tf.square(x), axis, keepdims=True)
  scale = tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
  return x / scale


class CapsuleLayer(object):

  def __init__(self, 
               num_capsule, 
               dim_capsule,
               routings=3,
               activation=None):
  
    self.num_capsule = num_capsule
    self.dim_capsule = dim_capsule
    self.routings = routings

    if activation:
      self.activation = activation
    else:
      self.activation = squash

  def __call__(self, u_vecs, kernel_size=1, stride_size=1):
    """Return capsule layer."""
      
    # First conv layer, [batch_size, seq_len, num_capsule * dim_capsule]
    u_hat_vecs = tf.contrib.layers.conv1d(u_vecs, self.num_capsule * self.dim_capsule,
                                          kernel_size, stride_size, padding="VALID",
                                          activation_fn=tf.nn.relu)

    seq_len = u_hat_vecs.get_shape().as_list()[1]
    # [batch_size, seq_len, num_capsule, dim_capsule]
    u_hat_vecs = tf.reshape(u_hat_vecs, [-1, seq_len, self.num_capsule, self.dim_capsule])
    # [batch_size, num_capsule, seq_len, dim_capsule]
    u_hat_vecs = tf.transpose(u_hat_vecs, (0, 2, 1, 3))
    b = tf.zeros_like(u_hat_vecs[:, :, :, 0])

    for i in range(self.routings):
      # [batch_size, seq_len, num_capsule]
      b = tf.transpose(b, (0, 2, 1))
      # [batch_size, seq_len, num_capsule]
      c = tf.nn.softmax(b)
      # [batch_size, num_capsule, seq_len]
      c = tf.transpose(c, (0, 2, 1))
      # [batch_size, num_capsule, seq_len]
      b = tf.transpose(b, (0, 2, 1))
      # [batch_size, num_capsule, dim_capsule]
      outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

      if i < self.routings - 1:
        b += tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])

    return outputs


class CapsuleNet(object):
  """The capsule network for text classification."""
  def __init__(self, 
               num_classes, seq_len, embedding_size, vocab_size,
               weight_decay, init_lr, decay_steps, decay_rate, 
               num_capsule, dim_capsule,
               is_rand=False, is_finetuned=False, embeddings=None):
    
    # capsule parameters
    self.num_capsule = num_capsule
    self.dim_capsule = dim_capsule

    # parameters
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
    # word embeddings -> capsule net -> fc
    self.inputs = tf.placeholder(tf.int32, [None, self.seq_len], name='inputs')
    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.onehot_labels = tf.one_hot(self.labels, self.num_classes)

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

    # capsule net
    with tf.name_scope("capsule"):
      capsule_layer = CapsuleLayer(num_capsule=self.num_capsule, dim_capsule=self.dim_capsule)
      capsule_output = capsule_layer(embedded_chars)
      flatten = tf.reshape(capsule_output, [-1, self.num_capsule * self.dim_capsule])

    # output logits
    self.logits = tf.layers.dense(flatten, self.num_classes, 
                                  kernel_initializer=tf.keras.initializers.glorot_normal())

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
    self.optimization = optimizer.apply_gradients(grads_and_vars)
