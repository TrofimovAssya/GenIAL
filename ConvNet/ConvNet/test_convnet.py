#Alexis Langlois
'''
Fichier de test pour le reseau de neurones a convolution.
'''

import numpy as np

from sklearn.utils import shuffle
from convnet import ConvolutionalNetwork
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#Train dataset
X = np.loadtxt('train_data')
y = np.loadtxt('train_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Data reshape
X = X.reshape(-1, 1, 8, 8)
y = np.asarray(y)
y = y.astype(np.int32)


#Instanciation
convnet = ConvolutionalNetwork()


#Training
convnet.train(X, y)


#Test dataset
X = np.loadtxt('test_data')
y = np.loadtxt('test_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Data reshape
X = X.reshape(-1, 1, 8, 8)
y = np.asarray(y)
y = y.astype(np.int32)


#Predictions
predictions = convnet.predict(X)


#Report
print classification_report(y, predictions)
print 'Accuracy: ' + str(accuracy_score(y, predictions))