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

class StateMachineGenerator(object):
	""" Generate sequences using a state machine, in which each state is
	represented by the last n outputs. """

	def __init__(self, n, transition_matrix=None):
		""" Initialize a new StateMachineGenerator with each state representing the
		last `n` bits. A transition matrix (np.array) can be passed in with
		dimensions (2^n, 2), or a random one will be generated. """
		expected_shape = (n ** 2, 2)
		assert n.shape == expected_shape, ('Transition matrix must be of dimensions'
			' (%d, %d), not (%d, %d)') % (expected_shape[0], expected_shape[1],
			n.shape[0], n.shape[1])

		if not transition_matrix:
			transition_matrix = self.generate_transition_matrix(expected_shape)

		self.transition_matrix = transition_matrix

		self.state = [1] * n # Is this a valid starting state?

	def generate_transition_matrix(self, shape):
		""" Generates a random transition matrix of the given dimensions. shape[1]
		must be 2. """
		assert shape[1] == 2, ('Tried to generate transition matrix with %d ',
			'columns, 2 required') % shape[1]

		col0 = np.random.rand(shape[0])
		col1 = 1 - col0
		return np.concatenate((col0, col1), axis=1)

	def next(self, n=1):
		retval = []

		for i in range(n):
			# Get the index of the state in the transition matrix by interpreting
			# state string as a binary number
			idx = int(self.state, 2)

			self.state.pop()
			if np.random.rand() <= self.transition_matrix[idx, 0]:
				self.state.append(0)
			else:
				self.state.append(1)

			retval.append(self.state)

		return retval
