import argparse
from generators import DeterministicStateMachineGenerator
from glob import glob
import matplotlib.pyplot as plt
from network import Network
import numpy as np
import tensorflow as tf
import os

def generate_generators(n_vals, replicas, output_directory):
  """ Generate generators and save them in `output_directory`. Will read an
  array of n values from `n_vals`, generating `replicas` generators for each
  n value. Creates with deterministic state machine generators. """
  filename_format = os.path.abspath(output_directory) + '/%d-%d.gen'
  for n in n_vals:
    for i in range(replicas):
      generator = DeterministicStateMachineGenerator(n)
      generator.transition_matrix.tofile(filename_format % (n, i))

def load_generator(generator_path):
  transition_matrix = np.fromfile(generator_path, dtype=np.int)
  num_states = transition_matrix.shape[0] / 2
  n = np.log2(num_states)
  transition_matrix = transition_matrix.reshape([num_states, 2])
  return DeterministicStateMachineGenerator(n,
      transition_matrix=transition_matrix)

def evaluate(generators_path, models_path):
  models = sorted(glob(os.path.join(models_path, '*.gen')))
  errors = [[-1] for i in range(10)]

  for model_path in models:
    tf.reset_default_graph()
    model_name = os.path.split(model_path)[1]
    generator = load_generator(os.path.join(generators_path, model_name))
    network = Network(generator, context=generator.n.astype(np.int))
    error = []
    for i in range(10):
      error += network.evaluate(model_path)

    if len(error) < len(errors[generator.n.astype(np.int) - 1]) or             \
       errors[generator.n.astype(np.int) - 1] == [-1]:
      errors[generator.n.astype(np.int) - 1] = error

  plt.figure(1)
  prev_plot = plt.subplot(5, 2, 1)
  for i in range(len(errors)):
    plt.subplot(5, 2, i+1, sharex=prev_plot, sharey=prev_plot)
    if errors[i] == [-1]:
      errors[i] = []
    plt.hist(errors[i], bins=1000 / 5)
    plt.xlabel('Position')
    plt.ylabel('Errors')
    plt.title('n = ' + str(i+1))

  plt.show()

def main():
  parser = argparse.ArgumentParser(description='Train and evaluate sequence '  \
    'prediction network.')
  parser.add_argument('--generator-path', action='store', dest='generator_path',
                      help='The path to the generator directory.')
  parser.add_argument('--summary-path', action='store', dest='summary_path',
                      help='The path to save TensorBoard summary files to.')
  parser.add_argument('--model-path', action='store', dest='model_path',
                      help='The path to save TensorFlow models to.')
  parser.add_argument('--evaluate-only', action='store_true',
                      dest='evaluate_only', help='Skip training the models.')
  args = parser.parse_args()

  if args.evaluate_only:
    evaluate(args.generator_path, args.model_path)
    return

  print 'Checking for existing generators...'
  if len(os.listdir(args.generator_path)) > 0:
    print 'Found -- skipping generation'
  else:
    print '-=-=-=- GENERATING GENERATORS -=-=-=-'
    print 'n: 1 <= n <= 10 (5 replicas each)'
    generate_generators(range(1, 11), 5, args.generator_path)
    print 'Generators saved in %s' % args.generator_path

  generators = sorted(glob(os.path.join(args.generator_path, '*.gen')))
  for generator_path in generators:
    generator = load_generator(generator_path)

    print '-=-=-=- TRAINING ON %s (n = %d) -=-=-=-' % (generator_path,
        generator.n)
    tf.reset_default_graph()

    generator_filename = os.path.split(generator_path)[1]
    summary_path = os.path.abspath(args.summary_path) + '/' + generator_filename
    model_path = os.path.abspath(args.model_path) + '/' + generator_filename

    network = Network(generator, context=generator.n.astype(np.int))
    acc = network.train(summary_dir=summary_path, model_dir=model_path)
    print '###### ACCURACY: %f' % acc

  evaluate(args.generator_path, args.model_path)

if __name__ == '__main__':
  main()
