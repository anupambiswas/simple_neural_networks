import sys, os, re
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x): return(1.0 / (1 + np.exp(-x)))

def sigmoid_der(x):
	s = sigmoid(x)
	return(s * (1 - s))

class TrivialNN:
	def __init__(self, layer_dims, training_set, test_set = None):
		self.layer_dims = layer_dims
		self.A = [None for i in range(len(layer_dims))]
		self.Z = [None for i in range(len(layer_dims))]
		self.dA = [None for i in range(len(layer_dims))]
		self.dZ = [None for i in range(len(layer_dims))]
		self.A[0] = training_set[0]
		self.Y = training_set[1]
		self.m = training_set[0].shape[1]
		self.L = len(layer_dims) - 1
		self.costs = []
		self.A_test = None
		self.Y_test = None
		if test_set != None:
			self.A_test = test_set[0]
			self.Y_test = test_set[1]
		plt.ion()


	def initialize_parameters(self):
		self.dw = [None for i in range(len(self.layer_dims))]
		self.db = [None for i in range(len(self.layer_dims))]
		self.w = [None]
		for l in range(1, len(self.layer_dims)):
			nl, nlm1 = self.layer_dims[l], self.layer_dims[l - 1]
			self.w.append(np.random.random((nl, nlm1)))
		self.b = [None]
		for l in range(1, len(self.layer_dims)):
			nl = self.layer_dims[l]
			self.b.append(np.zeros((nl, 1)))

	def forward_propagate(self):
		for l in range(1, len(self.layer_dims)):
			self.Z[l] = np.dot(self.w[l], self.A[l - 1]) + self.b[l]
			self.A[l] = sigmoid(self.Z[l])

	def backward_propagate(self):
		rng = list(range(1, self.L))
		rng.reverse()
		self.dA[self.L] = 2.0 * (self.A[self.L] - self.Y)
		self.dZ[self.L] = self.dA[self.L] * sigmoid_der(self.Z[self.L])
		for l in rng:
			self.dA[l] = np.dot(self.w[l + 1].T, self.dZ[l + 1])
			self.dZ[l] = self.dA[l] * sigmoid_der(self.Z[l])

	def compute_gradients(self):
		for l in range(1, self.L + 1):
			self.dw[l] = np.dot(self.dZ[l], self.A[l - 1].T) / self.m
			self.db[l] = np.sum(self.dZ[l], axis = 1, keepdims = True) / self.m

	def perform_gradient_descent(self, learning_rate = 0.001):
		for l in range(1, self.L + 1):
			self.w[l] = self.w[l] - self.dw[l] * learning_rate
			self.b[l] = self.b[l] - self.db[l] * learning_rate

	def compute_cost(self, do_plot = False):
		cost = np.sum(np.power(self.Y - self.A[self.L], 2.0)) / self.m
		print(cost)
		self.costs.append(cost)
		if do_plot:
			plt.cla()
			plt.plot(self.costs)
			plt.pause(0.001)

	def test(self):
		if not isinstance(self.A_test, np.ndarray):
			return(None)
		a = self.A_test
		for l in range(1, len(self.layer_dims)):
			z = np.dot(self.w[l], a) + self.b[l]
			a = sigmoid(z)
		return(a)


	def compute_training_accuracy(self, tol = 0.1):
		self.forward_propagate()
		res = np.abs(self.A[self.L] - self.Y)
		s = 0
		for i in range(len(res[0])):
			e = res[0][i]
			print("\tdifference " + str(e) + " " + str(self.A[self.L][0][i]) + " " + str(self.Y[0][i]))
			if e < tol:
				s += 1
		print("Train accuracy:", float(s) * 100.0 / res.shape[1], s, res.shape[1])

	def compute_test_accuracy(self, tol = 0.1):
		a = self.test()
		if not isinstance(a, np.ndarray):
			print("*** No test set available.")
			return(None)

		res = np.abs(a - self.Y_test)
		s = 0
		for i in range(len(res[0])):
			e = res[0][i]
			#print("\tdifference " + str(e) + " " + str(self.A[self.L][0][i]) + " " + str(self.Y[0][i]))
			if e < tol:
				s += 1
		print("Test accuracy:", float(s) * 100.0 / res.shape[1], s, res.shape[1])

	def begin_training(self, num_iters):
		self.initialize_parameters()
		for it in range(num_iters):
			self.forward_propagate()
			self.backward_propagate()
			self.compute_gradients()
			self.perform_gradient_descent()
			if it % 100 == 0:
				self.compute_cost(True)
		for l in range(1, len(self.layer_dims)):
			print('layer:', l, 'minw', np.min(self.w[l]), 'maxw', np.max(self.w[l]), 'minb', np.min(self.b[l]), 'maxb', np.max(self.b[l]))

		self.compute_training_accuracy()
		self.compute_test_accuracy()