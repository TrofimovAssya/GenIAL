#Alexis Langlois
'''
Fichier de test pour le perceptron multicouche v.2
'''

import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from multilayer_perceptron  import MultilayerPerceptronClassifier


#Train dataset
X = np.loadtxt('../train_data')
y = np.loadtxt('../train_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Instanciation
mlp = MultilayerPerceptronClassifier(activation='relu', hidden_layer_sizes = (100,), max_iter = 8)


#Training
mlp.fit(X, y)


#Test dataset
X = np.loadtxt('../test_data')
y = np.loadtxt('../test_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Predictions
preds = mlp.predict(X)


#Report
print classification_report(y, preds)
print 'Accuracy: ' + str(accuracy_score(y, preds))