#Alexis Langlois
'''
Simple appel des classes AdaBoostClassifier et DecisionTreeClassifier de scikit-learn.
'''

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class AdaboostTrees:

	def __init__(self, nb_estimators):
		self.forest = AdaBoostClassifier(
			DecisionTreeClassifier(max_depth=2),
			n_estimators=nb_estimators,
			learning_rate=1.5,
			algorithm="SAMME")
	
	def train(self, X_train, y_train):
		self.forest.fit(X_train, y_train)

	def predict(self, X_test):
		return self.forest.predict(X_test)