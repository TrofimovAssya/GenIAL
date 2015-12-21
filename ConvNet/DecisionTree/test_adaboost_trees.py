#Alexis Langlois
'''
Fichier de test pour l'algorithme Adaboost avec arbres de décision (@nbTrees).
'''

import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from adaboost_trees import AdaboostTrees


#Trees
nbTrees = 20


#Train dataset
X = np.loadtxt('train_data')
y = np.loadtxt('train_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Instanciation
forest = AdaboostTrees(nbTrees)


#Training
forest.train(X, y)


#Test dataset
X = np.loadtxt('test_data')
y = np.loadtxt('test_labels')
X, y = shuffle(X, y)


#Data normalization
X -= X.min()
X /= X.max()


#Predictions
predictions = forest.predict(X)


#Report
print classification_report(y, predicted)
print 'Accuracy: ' + str(accuracy_score(tags, preds))