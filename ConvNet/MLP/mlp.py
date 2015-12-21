#Alexis Langlois
'''
Algorithme du perceptron multicouche
'''

import numpy as np

#Activation
def logistic(x):
	return 1/(1 + np.exp(-x))

def logistic_dx(x):
	return logistic(x)*(1-logistic(x))

def tanh(x):
	return np.tanh(x)

def tanh_dx(x):
	return 1.0 - np.tanh(x)**2
	

#MLP

class MultilayerPerceptron:

	def __init__(self, layers, activation='tanh'):
		if activation == 'logistic':
			self.activation = logistic
			self.activation_dx = logistic_dx
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_dx = tanh_dx
		self.weights = []
		for i in range(1, len(layers) - 1):
			self.weights.append(0.5 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 0.25)
		self.weights.append(0.5 * np.random.random((layers[i] + 1, layers[i + 1])) - 0.25)


	#Training					
	def train(self, X, y, learning_rate=0.1, epochs=10000):
		X = np.atleast_2d(X)
		inputs_and_bias = np.ones([X.shape[0], X.shape[1]+1])
		inputs_and_bias[:, 0:-1] = X 
		X = inputs_and_bias
		y = np.array(y)
		for epoch in range(epochs):
			read_id = np.random.randint(X.shape[0])
			read = [X[read_id]]
			for layer_id in range(len(self.weights)):
				read.append(self.activation(np.dot(read[layer_id], self.weights[layer_id])))
			error = y[read_id] - read[-1]
			deltas = [error * self.activation_dx(read[-1])]
			for layer_id in range(len(read) - 2, 0, -1):
				deltas.append(deltas[-1].dot(self.weights[layer_id].T)*self.activation_dx(read[layer_id]))
			deltas.reverse()
			for layer_id in range(len(self.weights)):
				node = np.atleast_2d(read[layer_id])
				delta = np.atleast_2d(deltas[layer_id])
				self.weights[layer_id] += learning_rate * node.T.dot(delta)


	#Prediction
	def prediction(self, x):
		x = np.array(x)
		example_and_bias = np.ones(x.shape[0]+1)
		example_and_bias[0:-1] = x
		predictions = example_and_bias
		for layer_id in range(0, len(self.weights)):
			predictions = self.activation(np.dot(predictions, self.weights[layer_id]))
		return predictions		