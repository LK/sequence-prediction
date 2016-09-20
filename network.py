import tensorflow as tf
import numpy as np
from generators import *

class Network(object):
  """ Represents a binary sequence prediction network. """

  def __init__(self, generator, hidden_size=5, seq_length=100):
    self.generator = generator
    self.hidden_size = hidden_size
    self.seq_length = seq_length

    # Initialize and reshape input/target placeholders
    self.x = tf.placeholder(tf.float32, shape=(seq_length), name='x')
    self.y = tf.placeholder(tf.float32, shape=(seq_length), name='y')

    self.x = tf.reshape(self.x, (seq_length, 1, 1))
    self.y = tf.reshape(self.y, (seq_length, 1))

    # Set up LSTM and output layer
    self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)

    self.weights = tf.Variable(tf.random_normal([self.hidden_size, 1]))
    self.biases = tf.Variable(tf.random_normal([1]))

    lstm_output, _ = tf.nn.dynamic_rnn(self.lstm, self.x,
      sequence_length=[self.seq_length], dtype=tf.float32, time_major=True)
    lstm_output = tf.reshape(lstm_output, [seq_length, hidden_size])
    self.predict = tf.tanh(tf.matmul(lstm_output, self.weights) + self.biases)

    # Set up the cost function
    predicted_distribution = self._convert_to_distribution(self.predict)
    target_distribution = self._convert_to_distribution(self.y)
    self.cost = -tf.reduce_sum(
      target_distribution * tf.log(predicted_distribution))

  def _convert_to_distribution(self, tensor):
    """ Converts an n x 1 tensor of values in [-1, 1] to a n x 2 tensor of
    probability distributions (for -1/1). """
    positive_probability = (tensor + 1) / 2
    negative_probability = 1 - positive_probability

    return tf.concat(1, [positive_probability, negative_probability])

  def train(self, steps=100000, learning_rate=1e-2):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                        .minimize(self.cost)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for step in range(steps):
        actual = self.generator.next(self.seq_length+1)

        sess.run(optimizer, feed_dict={'x:0': actual[:-1], 'y:0': actual[1:]})

        if step % 10 == 0:
          print 'Step %d: %f' % (step, sess.run(self.cost,
            feed_dict={'x:0': actual[:-1], 'y:0': actual[1:]}))
          # print sess.run(self.predict,
          #   feed_dict={'x:0': actual[:-1], 'y:0': actual[1:]})

def main():
  network = Network(AlternatingGenerator())
  network.train()

if __name__ == '__main__':
  main()
