#Alexis Langlois
'''
Fichier de test pour l'algorithme DecisionTree.
'''

import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tree import DecisionTree


#Train dataset
X = np.loadtxt('train_data')
y = np.loadtxt('train_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Instanciation
tree = DecisionTree()


#Training
tree.train(X_train, y_train)


#Test dataset
X = np.loadtxt('test_data')
y = np.loadtxt('test_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Predictions
predictions = tree.predict(X_test)


#Report
print classification_report(y, predictions)
print 'Accuracy: ' + str(accuracy_score(tags, preds))