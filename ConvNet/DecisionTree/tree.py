#Alexis Langlois
'''
Simple appel de la classe DecisionTreeClassifier de scikit-learn.
'''

from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

	def __init__(self):
		self.tree = DecisionTreeClassifier()
	
	def train(self, X_train, y_train):
		self.tree.fit(X_train, y_train)

	def predict(self, X_test):
		return self.tree.predict(X_test)