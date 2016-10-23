import tensorflow as tf
import numpy as np
from generators import *

class Network(object):
  """ Represents a binary sequence prediction network. """

  def __init__(self, generator, hidden_size=30, seq_length=1000, context=10):
    """ Initialize a new network with `hidden_size` LSTM units, training on
    sequences of length `seq_length`. `context` represents the number of samples
    to feed to the model before sampling predictions (should be equal to n). """

    self.generator = generator
    self.hidden_size = hidden_size
    self.seq_length = seq_length
    self.context = context

    # Initialize and reshape input/target placeholders
    self.input_length = seq_length + self.context
    self.x = tf.placeholder(tf.float32, shape=(self.input_length), name='x')
    self.y = tf.placeholder(tf.float32, shape=(seq_length, 2), name='y')

    self.x = tf.reshape(self.x, (self.input_length, 1, 1))

    # Set up LSTM and output layer
    self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)

    self.weights = tf.Variable(tf.random_normal([self.hidden_size, 2]))
    self.biases = tf.Variable(tf.random_normal([2]))

    lstm_output, _ = tf.nn.dynamic_rnn(self.lstm, self.x,
      sequence_length=[self.seq_length], dtype=tf.float32, time_major=True)
    lstm_output = tf.reshape(lstm_output[self.context:, :, :],
        [seq_length, hidden_size])
    linear_output = tf.tanh(tf.matmul(lstm_output, self.weights) + self.biases)
    self.predict = tf.nn.softmax(linear_output)

    # Set up the cost function
    self.cost = -tf.reduce_sum(self.y * tf.log(self.predict))

    # TensorBoard summaries
    tf.scalar_summary('cost', self.cost)
    tf.histogram_summary('probability', self.predict[:,0])

  def train(self, steps=250, initial_learning_rate=1e-3, summary_dir=None,
            model_dir=None):
    """ Trains the network with the given parameters, returning the accuracy."""
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
        global_step, 100, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)            \
                        .minimize(self.cost, global_step=global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
      summaries = tf.merge_all_summaries()
      summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)

      sess.run(tf.initialize_all_variables())

      for step in range(steps):
        self.generator.reset()
        actual = self.generator.next(self.input_length+1)

        summary, _ = sess.run([summaries, optimizer],
            feed_dict={'x:0': actual[0][:-1], 'y:0': actual[1][self.context+1:]})
        summary_writer.add_summary(summary, step)

        if step % 25 == 0:
          print 'Step %d: %f' % (step, sess.run(self.cost,
            feed_dict={'x:0': actual[0][:-1], 'y:0': actual[1][self.context+1:]}))

      # Compute the accuracy
      test = self.generator.next(self.input_length+1)

      predicted = sess.run(self.predict,
          feed_dict={'x:0': test[0][:-1], 'y:0': test[1][self.context+1:]})

      correct = 0
      for i in range(len(predicted)):
        if (predicted[i][0] > 0.5 and test[1][self.context+i+1][0] > 0.5) or   \
           (predicted[i][0] < 0.5 and test[1][self.context+i+1][0] < 0.5):
          correct += 1

      save_path = saver.save(sess, model_dir)
      return correct / float(self.seq_length)

  def evaluate(self, model_path):
    self.generator.reset()

    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, model_path)

      test = self.generator.next(self.input_length+1)
      predicted = sess.run(self.predict,
          feed_dict={'x:0': test[0][:-1], 'y:0': test[1][self.context+1:]})
      errors = []
      for i in range(self.seq_length):
        if (predicted[i][0] < 0.5 and test[1][self.context+1+i][0] > 0.5) or   \
           (predicted[i][0] > 0.5 and test[1][self.context+1+i][0] < 0.5):
          errors.append(i)

      return errors

def main():
  network = Network(DeterministicStateMachineGenerator(10))
  network.train(summary_dir='/tmp/sequence-prediction')

if __name__ == '__main__':
  main()

