import tensorflow as tf
import numpy as np
from generators import *

class Network(object):
  """ Represents a binary sequence prediction network. """

  def __init__(self, generator, hidden_size=30, seq_length=1000):
    self.generator = generator
    self.hidden_size = hidden_size
    self.seq_length = seq_length

    # Initialize and reshape input/target placeholders
    self.x = tf.placeholder(tf.float32, shape=(seq_length+10), name='x')
    self.y = tf.placeholder(tf.float32, shape=(seq_length, 2), name='y')

    self.x = tf.reshape(self.x, (seq_length+10, 1, 1))

    # Set up LSTM and output layer
    self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)

    self.weights = tf.Variable(tf.random_normal([self.hidden_size, 2]))
    self.biases = tf.Variable(tf.random_normal([2]))

    lstm_output, _ = tf.nn.dynamic_rnn(self.lstm, self.x,
      sequence_length=[self.seq_length], dtype=tf.float32, time_major=True)
    lstm_output = tf.reshape(lstm_output[10:, :, :], [seq_length, hidden_size])
    linear_output = tf.tanh(tf.matmul(lstm_output, self.weights) + self.biases)
    self.predict = tf.nn.softmax(linear_output)

    # Set up the cost function
    self.cost = -tf.reduce_sum(self.y * tf.log(self.predict))

    # TensorBoard summaries
    tf.scalar_summary('cost', self.cost)
    tf.histogram_summary('probability', self.predict[:,0])

  def train(self, steps=100000, initial_learning_rate=1e-3):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 200, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                        .minimize(self.cost, global_step=global_step)

    with tf.Session() as sess:
      self.sess = sess

      summaries = tf.merge_all_summaries()
      summary_writer = tf.train.SummaryWriter('/tmp/sequence-prediction', sess.graph)

      sess.run(tf.initialize_all_variables())

      for step in range(steps):
        self.generator.reset()
        actual = self.generator.next(self.seq_length+10)

        summary, _ = sess.run([summaries, optimizer], feed_dict={'x:0': actual[0], 'y:0': actual[1][10:]})
        summary_writer.add_summary(summary, step)

        # if step % 100 == 0:
        #   print sess.run(self.predict,
        #     feed_dict={'x:0': actual[0][:-1], 'y:0': actual[1][1:]})
        #   print actual[0][:-1]

        if step % 10 == 0:
          print 'Step %d: %f' % (step, sess.run(self.cost,
            feed_dict={'x:0': actual[0], 'y:0': actual[1][10:]}))

def main():
  network = Network(DeterministicStateMachineGenerator(10))
  network.train()

if __name__ == '__main__':
  main()
