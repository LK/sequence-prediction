import numpy as np
from generators import *
from network import Network
import tensorflow as tf

def main():
  transition_matrix = np.fromfile('/tmp/sequence-prediction.gen', dtype=np.int).reshape([2 ** 10, 2])
  generator = DeterministicStateMachineGenerator(10, transition_matrix=transition_matrix
    )
  network = Network(generator)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, '/tmp/sequence-prediction.ckpt')
    actual = generator.next(1010)
    print str(sess.run(network.cost,
            feed_dict={'x:0': actual[0], 'y:0': actual[1][10:]}))

    predicted = sess.run(network.predict, feed_dict={'x:0': actual[0], 'y:0': actual[1][10:]})
    correct = 0
    incorrect = 0
    for i in range(len(predicted)):
      if predicted[i][0] > 0.5 and actual[0][10+i] == -1:
        correct += 1
      else:
        incorrect += 1

    print correct
    print incorrect


if __name__ == '__main__':
  main()
