import os
import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.updates import adam
from lasagne.layers import get_all_params
from lasagne.init import GlorotUniform
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

class ConvolutionalNetwork:
	def __init__(self):
		self.network = NeuralNet(
			layers=[
				('input', InputLayer),
				('conv2d1', Conv2DLayer),
				('maxpool1', MaxPool2DLayer),
				('dropout1', DropoutLayer),
				('dense', DenseLayer),
				('dropout2', DropoutLayer),
				('output', DenseLayer),
			],
			input_shape=(None, 1, 8, 8),
			conv2d1_num_filters=32,
			conv2d1_filter_size=(3, 3),
			conv2d1_nonlinearity=rectify,
			conv2d1_W=GlorotUniform(),  
			maxpool1_pool_size=(2, 2), 
			dropout1_p=0.5,    
			dense_num_units=256,
			dense_nonlinearity=rectify,    
			dropout2_p=0.5,    
			output_nonlinearity=softmax,
			output_num_units=22,
			update=nesterov_momentum,
			update_learning_rate=0.01,
			update_momentum=0.9,
			max_epochs=5000,
			verbose=1,
		)

	def regularization_objective(self, layers, lambda1=0., lambda2=0., *args, **kwargs):
		# default loss
		losses = objective(layers, *args, **kwargs)
		# get the layers' weights, but only those that should be regularized
		# (i.e. not the biases)
		weights = get_all_params(layers[-1], regularizable=True)
		# sum of absolute weights for L1
		sum_abs_weights = sum([abs(w).sum() for w in weights])
		# sum of squared weights for L2
		sum_squared_weights = sum([(w ** 2).sum() for w in weights])
		# add weights to regular loss
		losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
		return losses

	def train(self, X, y):
		self.fit_net = self.network.fit(X, y)

	def predict(self, X_test):
		return self.fit_net.predict(X_test)
		