from trivial_nn import TrivialNN
import sys, os
import numpy as np

if __name__ == '__main__':
	def dummy_function(x):
		def func(x):
			return((x - 5) * (x - 5))
		return(x[1] - func(x[0]))

	def make_data(num):
		"""
		This is the data preparation method.
		"""
		x = np.random.random((2, num)) * 10
		y = []
		for i in range(num):
			v = 1 if dummy_function(x[:,i]) > 0 else 0
			y.append(v)
		y = np.array(y).reshape(1, num)
		return([x, y])

	def usage(msg = ''):
		if msg != '':
			print("\n*** ERROR: " + msg + "\n")
		msg = "Usage:\n"
		msg += "\tpython3 " + os.path.basename(sys.argv[0]) + " <train_set_size> <test_set_size> <number_of_training_iterations>\n"
		quit(msg)

	try:
		num_train = int(sys.argv[1])
		num_test = int(sys.argv[2])
		num_iters = int(sys.argv[3])
		train_set = make_data(num_train)
		test_set = make_data(num_test)
	except (IndexError, ValueError) as e:
		usage(str(e))

	TrivialNN([2, 5, 1], train_set, test_set).begin_training(num_iters)