import numpy as np

class Generator(object):
	""" Abstract class representing any generator that can spit out a binary
	sequence. """

	def next(self, n=1):
		""" Returns an array of the next `n` bits of the sequence. """
		raise NotImplementedError

class AlternatingGenerator(object):
	""" Generates an alternating sequence of -1/1, starting at a random state. """

	def __init__(self, initial=1.0):
		""" Initialized an AlternatingGenerator with the given initial state. """
		self.state = initial

	def next(self, n=1):
		retval = []
		for i in range(n):
			self.state = self.state * -1
			retval.append(self.state)

		return retval

class DeterministicStateMachineGenerator(object):
	""" Generate sequences using a state machine with transition probabilities of
	either 0 or 1, in which each state is represented by the last n outputs. """

	def __init__(self, n, transition_matrix=None):
		""" Initialize a new DeterministicStateMachineGenerator with each state
		representing thelast `n` bits. A transition matrix (np.array) can be passed
		in with dimensions (2^n, 2), or a random one will be generated. """
		self.n = n

		expected_shape = (2 ** n, 2)
		if transition_matrix != None:
			assert transition_matrix.shape == expected_shape, ('Transition matrix ',
				' must be of dimensions (%d, %d), not (%d, %d)') % (expected_shape[0],
				expected_shape[1], n.shape[0], n.shape[1])
				# TODO: Assert transition probabilities are 0 or 1

		if transition_matrix == None:
			transition_matrix = self.generate_transition_matrix(expected_shape)

		self.transition_matrix = transition_matrix

		self.reset()

	def reset(self):
		self.state = list(np.random.randint(2, size=(self.n,1)))
		self.state = [-1 if i == 0 else 1 for i in self.state]

	def generate_transition_matrix(self, shape):
		""" Generates a random transition matrix of the given dimensions. shape[1]
		must be 2. """
		assert shape[1] == 2, ('Tried to generate transition matrix with %d ',
			'columns, 2 required') % shape[1]

		col0 = np.random.randint(2, size=(shape[0],1))
		col1 = 1 - col0
		return np.concatenate((col0, col1), axis=1)

	def next(self, n=1):
		""" Samples the next n items in the sequence, returning a tuple containing
		an array of the sampled values and an array of the distributions to select
		that value.
		"""
		samples = []
		distributions = []

		for i in range(n):
			# Compute the index into the transition matrix from the currents state by
			# interpreting state as binary number
			state_string = ''.join(['0' if i == -1 else '1' for i in self.state])
			idx = int(state_string, 2)

			self.state.pop(0)
			if np.random.rand() <= self.transition_matrix[idx, 0]:
				self.state.append(-1)
			else:
				self.state.append(1)

			samples.append(self.state[-1])
			distributions.append(self.transition_matrix[idx])

		return (samples, distributions)

class StateMachineGenerator(object):
	""" Generate sequences using a state machine, in which each state is
	represented by the last n outputs. """

	def __init__(self, n, transition_matrix=None):
		""" Initialize a new DeterministicStateMachineGenerator with each state
		representing thelast `n` bits. A transition matrix (np.array) can be passed
		in with dimensions (2^n, 2), or a random one will be generated. """
		self.n = n

		expected_shape = (n ** 2 + 1, 2)
		if transition_matrix:
			assert transition_matrix.shape == expected_shape, ('Transition matrix ',
				' must be of dimensions (%d, %d), not (%d, %d)') % (expected_shape[0],
				expected_shape[1], n.shape[0], n.shape[1])

		if not transition_matrix:
			transition_matrix = self.generate_transition_matrix(expected_shape)

		self.transition_matrix = transition_matrix

		self.reset()

	def reset(self):
		self.state = list(np.random.randint(2, size=(self.n,1)))
		self.state = [-1 if i == 0 else 1 for i in self.state]

	def generate_transition_matrix(self, shape):
		""" Generates a random transition matrix of the given dimensions. shape[1]
		must be 2. """
		assert shape[1] == 2, ('Tried to generate transition matrix with %d ',
			'columns, 2 required') % shape[1]

		col0 = np.random.rand(shape[0], 1)
		col1 = 1 - col0
		return np.concatenate((col0, col1), axis=1)

	def next(self, n=1):
		""" Samples the next n items in the sequence, returning a tuple containing
		an array of the sampled values and an array of the distributions to select
		that value.
		"""
		samples = []
		distributions = []

		for i in range(n):
			# Compute the index into the transition matrix from the currents state by
			# interpreting state as binary number
			state_string = ''.join(['0' if i == -1 else '1' for i in self.state])
			idx = int(state_string, 2)

			self.state.pop(0)
			if np.random.rand() <= self.transition_matrix[idx, 0]:
				self.state.append(-1)
			else:
				self.state.append(1)

			samples.append(self.state[-1])
			distributions.append(self.transition_matrix[idx])

		return (samples, distributions)

if __name__ == '__main__':
	g = StateMachineGenerator(2)
	print g.next(100)

	print g.transition_matrix
