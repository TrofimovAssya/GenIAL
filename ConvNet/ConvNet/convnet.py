#Alexis Langlois
'''
Implementation d'un reseau de neurones a convolution.
(API Lasagne)
'''

import os
import numpy as np
import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv1DLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool1DLayer
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

#Rectification du learning rate a la fin d'un epoch
class AdjustLearningRate(object):
	def __init__(self, start=0.05, stop=0.001):
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
		epoch = train_history[-1]['epoch']
		new_value = np.cast['float32'](self.ls[epoch - 1])
		getattr(nn, 'update_learning_rate').set_value(new_value)

#Appel de la classe NeuralNet
class ConvolutionalNetwork:
	def __init__(self):
		self.network = NeuralNet(
			layers=[
				('input', InputLayer),
				('conv2d1', Conv2DLayer),
				('maxpool1', MaxPool2DLayer),
				('dropout1', DropoutLayer),
				('dense', DenseLayer),
				('dense2', DenseLayer),
				('dense3', DenseLayer),
				('dropout2', DropoutLayer),
				('output', DenseLayer),
			],
			input_shape=(None, 1, 8, 8),
			conv2d1_num_filters=32,
			conv2d1_filter_size=(3, 3),
			conv2d1_nonlinearity=rectify,
			conv2d1_W=GlorotUniform(),  
			maxpool1_pool_size=(2, 2), 
			dropout1_p=0.1,    
			dense_num_units=256,
			dense_nonlinearity=rectify, 
			dense2_num_units=256,
			dense2_nonlinearity=rectify,    
			dense3_num_units=256,
			dense3_nonlinearity=rectify,  
			dropout2_p=0.1,    
			output_nonlinearity=softmax,
			output_num_units=25,
			update=nesterov_momentum,
			update_learning_rate=theano.shared(np.cast['float32'](0.05)),
			update_momentum=theano.shared(np.cast['float32'](0.9)),
			max_epochs=20,
			verbose=1,
			on_epoch_finished=[AdjustLearningRate(start=0.05, stop=0.0001)],
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

	#Entrainement
	def train(self, X, y):
		self.fit_net = self.network.fit(X, y)

	#Prediction
	def predict(self, X_test):
		return self.fit_net.predict(X_test)	