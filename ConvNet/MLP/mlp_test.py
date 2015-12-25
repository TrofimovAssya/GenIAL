#Alexis Langlois
'''
Fichier de test pour le perceptron multicouche
'''

import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from mlp import MultilayerPerceptron


#Train dataset
X = np.loadtxt('../data/train_data')
y = np.loadtxt('../data/train_label')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#instantiation
mlp = MultilayerPerceptron([100])


#Training data information
print 'Total Train Examples: ' + str(X.shape[0])


#Training
mlp.train(X,y,epochs=8)


#Test dataset
X = np.loadtxt('../data/test_data')
y = np.loadtxt('../data/test_label')
X, y = shuffle(X, y)


#Test data information
print 'Total Test Examples: ' + str(X.shape[0])


#Data normalization
X -= X.min()
X /= X.max()


#Predictions
predictions = []
for i in range(X.shape[0]):
    output = mlp.prediction(X[i])
    predictions.append(np.argmax(output))


#Results
print classification_report(y,predictions)
print 'Accuracy: ' + str(accuracy_score(y, predictions))